from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .final_interaction_components import (
    ModalityProjector,
    PairRelationEncoder,
    RelationConditionedOperator,
    RelationStateBilinearConditioner,
    RelationCrossAttention,
    RelationGroundedRetriever,
    StateUpdate,
    canonicalize_physical_edges,
    incoming_degree,
    leave_one_out_context,
)


VARIANTS = ("full", "no_context", "shared_relation", "static_execution", "operator_off")
GRADIENT_GROUPS = (
    "projector", "stage1_pair", "stage1_context", "stage2_conditioner",
    "stage2_operator", "state_update", "fusion",
)


class Model(nn.Module):
    """Frozen two-stage relation interpretation and semantic execution model."""

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        input_dim = int(data_info["input_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.text_dim + self.visual_dim != input_dim:
            raise ValueError("final_interaction requires ordered text and visual features")
        mc = cfg.model
        self.hidden_dim = int(mc.hidden_dim)
        self.relation_dim = int(mc.relation_dim)
        self.operator_rank = int(mc.operator_rank)
        self.conditioner_rank = int(mc.conditioner_rank)
        self.num_interaction_steps = int(mc.num_interaction_steps)
        fixed = (self.hidden_dim, self.relation_dim, self.operator_rank,
                 self.conditioner_rank, self.num_interaction_steps)
        if fixed != (256, 64, 32, 32, 2):
            raise ValueError("final_interaction freezes dimensions 256/64, ranks 32/32, and two steps")
        self.variant = str(mc.variant).strip().lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"model.variant must be one of {VARIANTS}")
        self.dropout_p = float(mc.dropout)
        self.relation_dropout = float(mc.relation_dropout)
        self.edge_chunk_size = int(mc.get("edge_chunk_size", 16384))
        if self.edge_chunk_size < 1:
            raise ValueError("model.edge_chunk_size must be positive")
        relation_heads = int(mc.relation.num_heads)
        retrieval_heads = int(mc.base_relation_retrieval.num_heads)
        if (relation_heads, retrieval_heads) != (2, 2):
            raise ValueError("final_interaction fixes relation and retrieval attention to two heads")

        # Stage I: independent modality streams, ordered pair evidence, and
        # leave-one-out recipient context.
        self.text_projector = ModalityProjector(self.text_dim, self.hidden_dim, self.dropout_p)
        self.visual_projector = ModalityProjector(self.visual_dim, self.hidden_dim, self.dropout_p)
        self.text_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.pair_text_encoder = PairRelationEncoder(self.relation_dim, self.relation_dropout)
        self.pair_visual_encoder = PairRelationEncoder(self.relation_dim, self.relation_dropout)
        self.relation_text_cross_attention = RelationCrossAttention(
            self.relation_dim, relation_heads, self.relation_dropout
        )
        self.relation_visual_cross_attention = RelationCrossAttention(
            self.relation_dim, relation_heads, self.relation_dropout
        )
        self.no_context_text = nn.Parameter(torch.empty(self.relation_dim))
        self.no_context_visual = nn.Parameter(torch.empty(self.relation_dim))
        nn.init.normal_(self.no_context_text, mean=0.0, std=0.02)
        nn.init.normal_(self.no_context_visual, mean=0.0, std=0.02)

        # Stage II: grounded relation retrieval, absolute bilinear conditioning,
        # a low-rank semantic operator, shared across the two update steps.
        self.static_query_text = nn.Parameter(torch.empty(self.relation_dim))
        self.static_query_visual = nn.Parameter(torch.empty(self.relation_dim))
        nn.init.normal_(self.static_query_text, mean=0.0, std=0.02)
        nn.init.normal_(self.static_query_visual, mean=0.0, std=0.02)
        self.text_retriever = RelationGroundedRetriever(
            self.relation_dim, retrieval_heads, self.relation_dropout
        )
        self.visual_retriever = RelationGroundedRetriever(
            self.relation_dim, retrieval_heads, self.relation_dropout
        )
        self.text_conditioner = RelationStateBilinearConditioner(
            self.relation_dim, self.hidden_dim, self.conditioner_rank
        )
        self.visual_conditioner = RelationStateBilinearConditioner(
            self.relation_dim, self.hidden_dim, self.conditioner_rank
        )
        self.text_operator = RelationConditionedOperator(
            self.hidden_dim, self.relation_dim, self.operator_rank
        )
        self.visual_operator = RelationConditionedOperator(
            self.hidden_dim, self.relation_dim, self.operator_rank
        )
        self.text_state_update = StateUpdate(self.hidden_dim, self.dropout_p)
        self.visual_state_update = StateUpdate(self.hidden_dim, self.dropout_p)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.GELU(), nn.Dropout(self.dropout_p),
        )
        self.out_dim = self.hidden_dim

        if self.variant == "shared_relation":
            self.shared_relation_text = nn.Parameter(torch.empty(self.relation_dim))
            self.shared_relation_visual = nn.Parameter(torch.empty(self.relation_dim))
            nn.init.normal_(self.shared_relation_text, mean=0.0, std=0.02)
            nn.init.normal_(self.shared_relation_visual, mean=0.0, std=0.02)

        self._cached_edge_input = None
        self._cached_edges = None
        self._cached_degree = None
        self._gradient_path = str(mc.get("gradient_trace_path", ""))
        self._epoch = 0
        self._epoch_open = False
        self._grad_sums = [0.0] * len(GRADIENT_GROUPS)
        self._grad_numels = [0] * len(GRADIENT_GROUPS)
        self._active_names = self._resolve_active_parameter_names()
        self._parameter_groups = {name: self._group_for_parameter(name) for name, _ in self.named_parameters()}
        self._install_gradient_hooks()

    def _group_for_parameter(self, name: str) -> str | None:
        if name.startswith(("text_projector.", "visual_projector.")):
            return "projector"
        if name.startswith(("text_relation_projection.", "visual_relation_projection.",
                            "pair_text_encoder.", "pair_visual_encoder.")):
            return "stage1_pair"
        if name.startswith(("relation_text_cross_attention.", "relation_visual_cross_attention.")) or name in (
            "no_context_text", "no_context_visual", "shared_relation_text", "shared_relation_visual",
        ):
            return "stage1_context"
        if name.startswith(("text_retriever.", "visual_retriever.",
                            "text_conditioner.", "visual_conditioner.")) or name in (
            "static_query_text", "static_query_visual",
        ):
            return "stage2_conditioner"
        if name.startswith(("text_operator.", "visual_operator.")):
            return "stage2_operator"
        if name.startswith(("text_state_update.", "visual_state_update.")):
            return "state_update"
        if name.startswith("fusion."):
            return "fusion"
        return None

    def _resolve_active_parameter_names(self) -> set[str]:
        active = {name for name, _ in self.named_parameters()}
        stage1 = (
            "text_relation_projection.", "visual_relation_projection.",
            "pair_text_encoder.", "pair_visual_encoder.",
            "relation_text_cross_attention.", "relation_visual_cross_attention.",
        )
        if self.variant == "shared_relation":
            active = {name for name in active if not name.startswith(stage1)
                      and name not in {"no_context_text", "no_context_visual"}}
        if self.variant == "static_execution":
            active = {name for name in active if not name.startswith((
                "text_conditioner.", "visual_conditioner.",
            ))}
        if self.variant == "operator_off":
            inactive_prefixes = stage1 + (
                "text_retriever.", "visual_retriever.",
                "text_conditioner.", "visual_conditioner.",
                "text_operator.down.", "visual_operator.down.",
                "text_operator.modulation.", "visual_operator.modulation.",
                "text_operator.up.", "visual_operator.up.",
            )
            active = {name for name in active if not name.startswith(inactive_prefixes)
                      and name not in {"static_query_text", "static_query_visual",
                                       "no_context_text", "no_context_visual"}}
        return active

    def active_parameter_names(self) -> set[str]:
        return set(self._active_names)

    def _install_gradient_hooks(self) -> None:
        for name, parameter in self.named_parameters():
            group = self._parameter_groups[name]
            if name not in self._active_names or group is None:
                continue
            idx = GRADIENT_GROUPS.index(group)

            def collect(grad, group_idx=idx):
                self._grad_sums[group_idx] += float(grad.detach().float().square().sum().item())
                self._grad_numels[group_idx] += int(grad.numel())
                return grad

            parameter.register_hook(collect)

    def _append_gradient_rows(self, rows: list[dict[str, Any]]) -> None:
        if not self._gradient_path:
            return
        path = Path(self._gradient_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["epoch", "group", "gradient_rms", "gradient_numel"],
                lineterminator="\n",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def flush_gradient_trace(self) -> None:
        if not self._epoch_open:
            return
        rows = []
        for idx, group in enumerate(GRADIENT_GROUPS):
            n = self._grad_numels[idx]
            rows.append({
                "epoch": self._epoch,
                "group": group,
                "gradient_rms": math.sqrt(self._grad_sums[idx] / n) if n else 0.0,
                "gradient_numel": n,
            })
        self._append_gradient_rows(rows)
        self._epoch_open = False

    def set_epoch(self, epoch: int) -> None:
        self.flush_gradient_trace()
        self._epoch = int(epoch)
        self._grad_sums = [0.0] * len(GRADIENT_GROUPS)
        self._grad_numels = [0] * len(GRADIENT_GROUPS)
        self._epoch_open = True

    def _graph(self, edge_index, num_nodes: int, device: torch.device):
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        elif edge_index.device != device:
            edge_index = edge_index.to(device)
        if self._cached_edge_input is not edge_index or self._cached_edges is None:
            self._cached_edges = canonicalize_physical_edges(edge_index, num_nodes).to(device)
            self._cached_degree = incoming_degree(self._cached_edges[1], num_nodes)
            self._cached_edge_input = edge_index
        return self._cached_edges, self._cached_degree

    @staticmethod
    def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(left, right, dim=-1, eps=1e-8)

    def _shuffle_context(self, context_text, context_visual, src, dst, degree):
        edge_targets = dst.detach().cpu().tolist()
        edge_sources = src.detach().cpu().tolist()
        node_degrees = degree.detach().cpu().tolist()
        edges_by_target: dict[int, list[int]] = {}
        for edge_pos, target in enumerate(edge_targets):
            edges_by_target.setdefault(int(target), []).append(edge_pos)
        recipients: dict[tuple[str, int], list[int]] = {}
        for target in edges_by_target:
            d = int(node_degrees[target])
            if d > 1:
                recipients.setdefault((f"{d}", d), []).append(target)
        donor_for = list(range(len(edge_targets)))
        for (_, d), targets in recipients.items():
            targets = sorted(targets)
            if len(targets) < 2:
                continue
            for idx, target in enumerate(targets):
                donor = targets[(idx + 1) % len(targets)]
                receiver_edges = sorted(edges_by_target[target], key=lambda pos: edge_sources[pos])
                donor_edges = sorted(edges_by_target[donor], key=lambda pos: edge_sources[pos])
                if len(receiver_edges) != d or len(donor_edges) != d:
                    raise RuntimeError("degree-matched context shuffle found inconsistent recipient groups")
                for receiver_pos, donor_pos in zip(receiver_edges, donor_edges):
                    donor_for[receiver_pos] = donor_pos
        donor_index = torch.tensor(donor_for, dtype=torch.long, device=src.device)
        return context_text[donor_index], context_visual[donor_index]

    def _pair_and_context(self, h_text, h_visual, src, dst, context_shuffle, null_context, diagnostics):
        u_text = self.text_relation_projection(h_text)
        u_visual = self.visual_relation_projection(h_visual)
        text_cosine = self._cosine(u_text[dst], u_text[src])
        visual_cosine = self._cosine(u_visual[dst], u_visual[src])
        pair_t, pair_v = [], []
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            pair_t.append(self.pair_text_encoder(
                u_text[dst[start:end]], u_text[src[start:end]],
                text_cosine[start:end], visual_cosine[start:end],
            ))
            pair_v.append(self.pair_visual_encoder(
                u_visual[dst[start:end]], u_visual[src[start:end]],
                text_cosine[start:end], visual_cosine[start:end],
            ))
        empty = h_text.new_empty((0, self.relation_dim))
        pair_text = torch.cat(pair_t) if pair_t else empty
        pair_visual = torch.cat(pair_v) if pair_v else empty
        context_text, _, degree = leave_one_out_context(
            u_text, src, dst, h_text.size(0), self.no_context_text
        )
        context_visual, _, _ = leave_one_out_context(
            u_visual, src, dst, h_visual.size(0), self.no_context_visual
        )
        if null_context or self.variant == "no_context":
            context_text = self.no_context_text.unsqueeze(0).expand(src.numel(), -1)
            context_visual = self.no_context_visual.unsqueeze(0).expand(src.numel(), -1)
        if context_shuffle:
            context_text, context_visual = self._shuffle_context(
                context_text, context_visual, src, dst, degree
            )
        relation_t, relation_v, attn_t, attn_v = [], [], [], []
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            keys_t = torch.stack((pair_visual[start:end], context_text[start:end], context_visual[start:end]), 1)
            keys_v = torch.stack((pair_text[start:end], context_text[start:end], context_visual[start:end]), 1)
            r_t, w_t = self.relation_text_cross_attention(pair_text[start:end], keys_t, diagnostics)
            r_v, w_v = self.relation_visual_cross_attention(pair_visual[start:end], keys_v, diagnostics)
            relation_t.append(r_t)
            relation_v.append(r_v)
            if diagnostics:
                attn_t.append(w_t)
                attn_v.append(w_v)
        cat = lambda xs, empty_value: torch.cat(xs, 0) if xs else empty_value
        return {
            "U_text": u_text, "U_visual": u_visual,
            "pair_text": pair_text, "pair_visual": pair_visual,
            "compatibility_text": text_cosine, "compatibility_visual": visual_cosine,
            "compatibility_abs_difference": (text_cosine - visual_cosine).abs(),
            "context_text": context_text, "context_visual": context_visual,
            "relation_text": cat(relation_t, empty), "relation_visual": cat(relation_v, empty),
            "stage1_attention_text": cat(attn_t, h_text.new_empty((0, 3))) if diagnostics else empty,
            "stage1_attention_visual": cat(attn_v, h_text.new_empty((0, 3))) if diagnostics else empty,
        }

    def _relation_memory(self, h_text, h_visual, src, dst, context_shuffle, null_context, diagnostics):
        if self.variant == "shared_relation":
            slots = torch.stack((self.shared_relation_text, self.shared_relation_visual), dim=0)
            memory = slots.unsqueeze(0).expand(src.numel(), -1, -1)
            empty = h_text.new_empty((src.numel(), self.relation_dim))
            evidence = {
                "U_text": h_text.new_empty((h_text.size(0), self.relation_dim)),
                "U_visual": h_visual.new_empty((h_visual.size(0), self.relation_dim)),
                "pair_text": empty, "pair_visual": empty,
                "compatibility_text": h_text.new_empty((src.numel(),)),
                "compatibility_visual": h_text.new_empty((src.numel(),)),
                "compatibility_abs_difference": h_text.new_empty((src.numel(),)),
                "context_text": empty, "context_visual": empty,
                "relation_text": memory[:, 0], "relation_visual": memory[:, 1],
                "stage1_attention_text": h_text.new_empty((src.numel(), 3)),
                "stage1_attention_visual": h_text.new_empty((src.numel(), 3)),
            }
        else:
            evidence = self._pair_and_context(
                h_text, h_visual, src, dst, context_shuffle, null_context, diagnostics
            )
            memory = torch.stack((evidence["relation_text"], evidence["relation_visual"]), dim=1)
        return memory, evidence

    def _base_relation_code(self, modality: str, memory: torch.Tensor):
        if modality == "text":
            query, retriever = self.static_query_text, self.text_retriever
        elif modality == "visual":
            query, retriever = self.static_query_visual, self.visual_retriever
        else:
            raise ValueError("modality must be 'text' or 'visual'")
        if memory.size(0) == 0:
            return memory.new_empty((0, self.relation_dim))
        query = query.to(memory).expand(memory.size(0), -1)
        chunks = []
        for start in range(0, memory.size(0), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, memory.size(0))
            chunks.append(retriever(query[start:end], memory[start:end])[0])
        return torch.cat(chunks, dim=0)

    def _shuffle_relation_memory(self, memory, dst):
        positions_by_target: dict[int, list[int]] = {}
        for edge_pos, target in enumerate(dst.detach().cpu().tolist()):
            positions_by_target.setdefault(int(target), []).append(edge_pos)
        permutation = list(range(memory.size(0)))
        for positions in positions_by_target.values():
            if len(positions) > 1:
                for idx, pos in enumerate(positions):
                    permutation[pos] = positions[(idx + 1) % len(positions)]
        index = torch.tensor(permutation, dtype=torch.long, device=memory.device)
        return memory[index]

    def _execution_step(self, state, memory, base_relation_code, src, dst, modality,
                        operator_enabled, capture, step):
        aggregate = state.new_zeros((state.size(0), self.hidden_dim))
        codes, deltas_code, modulations, delta_messages, base_messages = [], [], [], [], []
        conditioner = self.text_conditioner if modality == "text" else self.visual_conditioner
        operator = self.text_operator if modality == "text" else self.visual_operator
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            r = base_relation_code[start:end]
            target_state = state[dst[start:end]]
            if self.variant == "static_execution":
                code = r
                delta_code = torch.zeros_like(r)
            else:
                code, delta_code, _, _, _ = conditioner(r, target_state)
            source_state = state[src[start:end]]
            base, delta, modulation, message = operator(source_state, code, enabled=operator_enabled)
            aggregate.index_add_(0, dst[start:end], message)
            if capture:
                codes.append(code)
                deltas_code.append(delta_code)
                modulations.append(modulation)
                delta_messages.append(delta)
                base_messages.append(base)
        degree = incoming_degree(dst, state.size(0)).to(state.dtype).clamp_min_(1)
        aggregate = aggregate / degree.unsqueeze(-1)
        updater = self.text_state_update if modality == "text" else self.visual_state_update
        next_state = updater(state, aggregate)
        empty_rel = state.new_empty((0, self.relation_dim))
        empty_rank = state.new_empty((0, self.operator_rank))
        empty_hidden = state.new_empty((0, self.hidden_dim))
        details = {
            "execution_code": torch.cat(codes) if codes else empty_rel,
            "delta_execution_code": torch.cat(deltas_code) if deltas_code else empty_rel,
            "modulation": torch.cat(modulations) if modulations else empty_rank,
            "delta_message": torch.cat(delta_messages) if delta_messages else empty_hidden,
            "base_message": torch.cat(base_messages) if base_messages else empty_hidden,
        }
        return next_state, details

    def _encode(self, x, edge_index, *, diagnostics=False, relation_shuffle=False,
                context_shuffle=False, null_context=False, operator_enabled=True):
        if not self.training:
            self.flush_gradient_trace()
        x_text, x_visual = x[:, :self.text_dim], x[:, self.text_dim:]
        h0_text = self.text_projector(x_text)
        h0_visual = self.visual_projector(x_visual)
        edges, degree = self._graph(edge_index, x.size(0), x.device)
        src, dst = edges
        memory, evidence = self._relation_memory(
            h0_text, h0_visual, src, dst, context_shuffle, null_context, diagnostics
        )
        if relation_shuffle and src.numel():
            memory = self._shuffle_relation_memory(memory, dst)
        base_text = self._base_relation_code("text", memory)
        base_visual = self._base_relation_code("visual", memory)
        state_text, state_visual = h0_text, h0_visual
        states_text, states_visual = [state_text], [state_visual]
        detail_text, detail_visual = [], []
        for step in range(self.num_interaction_steps):
            state_text, text_details = self._execution_step(
                state_text, memory, base_text, src, dst, "text", operator_enabled, diagnostics, step
            )
            state_visual, visual_details = self._execution_step(
                state_visual, memory, base_visual, src, dst, "visual", operator_enabled, diagnostics, step
            )
            states_text.append(state_text)
            states_visual.append(state_visual)
            if diagnostics:
                detail_text.append(text_details)
                detail_visual.append(visual_details)
        fused = self.fusion(torch.cat((state_text, state_visual), dim=-1))
        values: dict[str, Any] = {
            "H0_text": h0_text, "H0_visual": h0_visual,
            "H1_text": states_text[1], "H1_visual": states_visual[1],
            "H2_text": states_text[2], "H2_visual": states_visual[2],
            "final_text": state_text, "final_visual": state_visual, "fused_z": fused,
            "canonical_edge_index": edges, "degree": degree, "relation_memory": memory,
            "base_relation_text": base_text, "base_relation_visual": base_visual,
            **evidence,
        }
        if diagnostics:
            for step in range(self.num_interaction_steps):
                for modality, detail_list in (("text", detail_text), ("visual", detail_visual)):
                    for key, value in detail_list[step].items():
                        values[f"{key}_{modality}_step{step}"] = value
            for key in ("execution_code", "delta_execution_code", "modulation", "delta_message", "base_message"):
                values[key] = {
                    f"{modality}_step{step}": values[f"{key}_{modality}_step{step}"]
                    for modality in ("text", "visual") for step in range(self.num_interaction_steps)
                }
        return fused, values

    def forward(self, x, edge_index=None, *, relation_shuffle=False, context_shuffle=False,
                null_context=False, operator_enabled=True):
        z, _ = self._encode(
            x, edge_index, diagnostics=False, relation_shuffle=relation_shuffle,
            context_shuffle=context_shuffle, null_context=null_context,
            operator_enabled=operator_enabled,
        )
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(self, x, edge_index=None, *, relation_shuffle=False, context_shuffle=False,
                null_context=False, operator_enabled=True, **_kwargs):
        _, values = self._encode(
            x, edge_index, diagnostics=True, relation_shuffle=relation_shuffle,
            context_shuffle=context_shuffle, null_context=null_context,
            operator_enabled=operator_enabled,
        )
        return values

    @torch.no_grad()
    def inference(self, x, edge_index=None, device=None, batch_size=65536):
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device) if edge_index is not None else None
        return self.forward(x, edge_index)[0].detach().cpu()


__all__ = ["Model", "VARIANTS", "GRADIENT_GROUPS"]
