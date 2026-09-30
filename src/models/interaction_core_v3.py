from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_core_v3_components import (
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


VARIANTS = (
    "context_static", "context_attn_dynamic",
    "context_bilinear_absolute", "context_bilinear_delta",
)
TRACE_GROUPS = (
    "projector_text", "projector_visual", "stage1_pair", "stage1_context_attention",
    "stage2_retrieval", "stage2_conditioner_relation", "stage2_conditioner_state",
    "stage2_conditioner_output", "stage2_modulation_Wa", "stage2_operator_down",
    "stage2_operator_up", "state_update", "fusion",
)
MAX_TRACE_EPOCHS = 512


class Model(nn.Module):
    """M0-Core v3: relation–state conditioner audit over the frozen v2 interface."""

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        input_dim = int(data_info["input_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.text_dim + self.visual_dim != input_dim:
            raise ValueError("interaction_core_v2 requires ordered text and visual features")
        mc = cfg.model
        self.hidden_dim = int(mc.hidden_dim)
        self.relation_dim = int(mc.relation_dim)
        self.operator_rank = int(mc.operator_rank)
        self.conditioner_rank = int(mc.conditioner_rank)
        self.num_interaction_steps = int(mc.num_interaction_steps)
        if (self.relation_dim, self.operator_rank, self.conditioner_rank, self.num_interaction_steps) != (64, 32, 32, 2):
            raise ValueError("M0-Core v3 fixes relation_dim=64, both ranks=32, and two steps")
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
        dynamic_heads = int(mc.attn_dynamic.num_heads)
        if dynamic_heads != retrieval_heads:
            raise ValueError("v3 shares the v2 grounded retriever across base and dynamic queries; head counts must match")

        # Stage I: two modality-specific streams and the frozen v1 relation path.
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
        for token in (self.no_context_text, self.no_context_visual):
            nn.init.normal_(token, mean=0.0, std=0.02)

        # Stage II: one static base relation code, then exactly one of four
        # execution-code formulations. The retrievers remain purely grounded.
        self.static_query_text = nn.Parameter(torch.empty(self.relation_dim))
        self.static_query_visual = nn.Parameter(torch.empty(self.relation_dim))
        nn.init.normal_(self.static_query_text, mean=0.0, std=0.02)
        nn.init.normal_(self.static_query_visual, mean=0.0, std=0.02)
        self.text_target_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_target_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.text_dynamic_embedding = nn.Parameter(torch.empty(self.relation_dim))
        self.visual_dynamic_embedding = nn.Parameter(torch.empty(self.relation_dim))
        nn.init.normal_(self.text_dynamic_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.visual_dynamic_embedding, mean=0.0, std=0.02)
        self.text_query_norm = nn.LayerNorm(self.relation_dim)
        self.visual_query_norm = nn.LayerNorm(self.relation_dim)
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

        self.register_buffer("trace_epoch", torch.full((MAX_TRACE_EPOCHS,), -1, dtype=torch.long))
        self.register_buffer("gradient_rms_trace", torch.zeros(MAX_TRACE_EPOCHS, len(TRACE_GROUPS)))
        self.register_buffer("gradient_numel_trace", torch.zeros(MAX_TRACE_EPOCHS, len(TRACE_GROUPS), dtype=torch.long))
        self.register_buffer("parameter_norm_trace", torch.zeros(MAX_TRACE_EPOCHS, len(TRACE_GROUPS)))
        self.register_buffer("operator_up_norm_trace", torch.zeros(MAX_TRACE_EPOCHS, 2))
        self.register_buffer("conditioner_output_norm_trace", torch.zeros(MAX_TRACE_EPOCHS, 2))
        self._epoch = 0
        self._epoch_open = False
        self._grad_sums = [0.0] * len(TRACE_GROUPS)
        self._grad_numels = [0] * len(TRACE_GROUPS)
        self._trace_paths = {
            "gradient": str(mc.get("gradient_trace_path", "")),
            "operator": str(mc.get("operator_trace_path", "")),
        }
        self._install_gradient_hooks()
        self._cached_edge_input = None
        self._cached_edges = None
        self._cached_degree = None

    def _group_for_parameter(self, name: str) -> str | None:
        if name.startswith("text_projector."):
            return "projector_text"
        if name.startswith("visual_projector."):
            return "projector_visual"
        if name.startswith(("text_relation_projection.", "visual_relation_projection.",
                            "pair_text_encoder.", "pair_visual_encoder.")):
            return "stage1_pair"
        if name.startswith(("relation_text_cross_attention.", "relation_visual_cross_attention.")) or name in (
            "no_context_text", "no_context_visual",
        ):
            return "stage1_context_attention"
        if name.startswith(("text_target_projection.", "visual_target_projection.",
                            "text_retriever.", "visual_retriever.",
                            "text_query_norm.", "visual_query_norm.")) or name in (
            "static_query_text", "static_query_visual", "text_dynamic_embedding", "visual_dynamic_embedding",
        ):
            return "stage2_retrieval"
        if name.startswith(("text_conditioner.relation.", "visual_conditioner.relation.")):
            return "stage2_conditioner_relation"
        if name.startswith(("text_conditioner.state.", "visual_conditioner.state.")):
            return "stage2_conditioner_state"
        if name.startswith(("text_conditioner.output.", "visual_conditioner.output.")):
            return "stage2_conditioner_output"
        if name.startswith(("text_operator.modulation.", "visual_operator.modulation.")):
            return "stage2_modulation_Wa"
        if name.startswith(("text_operator.message.", "visual_operator.message.",
                            "text_operator.down.", "visual_operator.down.")):
            return "stage2_operator_down"
        if name.startswith(("text_operator.up.", "visual_operator.up.")):
            return "stage2_operator_up"
        if name.startswith(("text_state_update.", "visual_state_update.")):
            return "state_update"
        if name.startswith("fusion."):
            return "fusion"
        return None

    def _install_gradient_hooks(self) -> None:
        self._parameter_groups = {name: self._group_for_parameter(name) for name, _ in self.named_parameters()}
        for name, parameter in self.named_parameters():
            group = self._parameter_groups[name]
            if group is None or not parameter.requires_grad:
                continue
            group_idx = TRACE_GROUPS.index(group)

            def collect(grad, idx=group_idx):
                self._grad_sums[idx] += float(grad.detach().float().square().sum().item())
                self._grad_numels[idx] += int(grad.numel())
                return grad

            parameter.register_hook(collect)

    def set_epoch(self, epoch: int) -> None:
        if self._epoch_open:
            self._flush_epoch_trace()
        self._epoch = int(epoch)
        self._grad_sums = [0.0] * len(TRACE_GROUPS)
        self._grad_numels = [0] * len(TRACE_GROUPS)
        self._epoch_open = True

    def _append_trace(self, path_string: str, fieldnames: list[str], rows: list[dict]) -> None:
        if not path_string:
            return
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _flush_epoch_trace(self) -> None:
        if not self._epoch_open:
            return
        epoch = self._epoch
        if 1 <= epoch <= MAX_TRACE_EPOCHS:
            self.trace_epoch[epoch - 1] = epoch
            grad_rms, grad_numel, param_norm = [], [], []
            for idx, group in enumerate(TRACE_GROUPS):
                n = self._grad_numels[idx]
                grad_numel.append(n)
                grad_rms.append(math.sqrt(self._grad_sums[idx] / n) if n else 0.0)
                sq_norm, nparam = 0.0, 0
                for name, parameter in self.named_parameters():
                    if self._parameter_groups.get(name) == group:
                        value = parameter.detach().float()
                        sq_norm += float(value.square().sum().item())
                        nparam += value.numel()
                param_norm.append(math.sqrt(sq_norm) if nparam else 0.0)
                self.gradient_rms_trace[epoch - 1, idx] = grad_rms[-1]
                self.gradient_numel_trace[epoch - 1, idx] = n
                self.parameter_norm_trace[epoch - 1, idx] = param_norm[-1]
            up_norms = [float(self.text_operator.up.weight.detach().float().norm().item()),
                        float(self.visual_operator.up.weight.detach().float().norm().item())]
            self.operator_up_norm_trace[epoch - 1] = torch.tensor(up_norms, device=self.operator_up_norm_trace.device)
            wc_norms = [float(self.text_conditioner.output.weight.detach().float().norm().item()),
                        float(self.visual_conditioner.output.weight.detach().float().norm().item())]
            self.conditioner_output_norm_trace[epoch - 1] = torch.tensor(wc_norms, device=self.conditioner_output_norm_trace.device)
            common = {"epoch": epoch}
            self._append_trace(
                self._trace_paths["gradient"],
                ["epoch", "group", "gradient_rms_preclip", "gradient_numel", "parameter_l2_norm"],
                [{**common, "group": group, "gradient_rms_preclip": grad_rms[idx],
                  "gradient_numel": grad_numel[idx], "parameter_l2_norm": param_norm[idx]}
                 for idx, group in enumerate(TRACE_GROUPS)],
            )
            self._append_trace(
                self._trace_paths["operator"],
                ["epoch", "modality", "W_up_frobenius_norm", "W_c_frobenius_norm"],
                [{"epoch": epoch, "modality": modality, "W_up_frobenius_norm": up_norms[i],
                  "W_c_frobenius_norm": wc_norms[i]} for i, modality in enumerate(("text", "visual"))],
            )
        self._epoch_open = False

    def _graph(self, edge_index, num_nodes, device):
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
    def _cosine(left, right):
        return F.cosine_similarity(left, right, dim=-1, eps=1e-8)

    def _shuffle_context(self, context_text, context_visual, src, dst, degree):
        """Rotate paired LOO contexts across distinct recipients within degree buckets."""
        edge_targets = dst.detach().cpu().tolist()
        edge_sources = src.detach().cpu().tolist()
        node_degrees = degree.detach().cpu().tolist()
        bucket_ranges = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"),
                         (5, 8, "5-8"), (9, 16, "9-16"), (17, math.inf, "17+"))
        def bucket(value: int) -> str:
            return next(name for low, high, name in bucket_ranges if low <= value <= high)
        edges_by_target: dict[int, list[int]] = {}
        for edge_pos, target in enumerate(edge_targets):
            edges_by_target.setdefault(int(target), []).append(edge_pos)
        recipients: dict[tuple[str, int], list[int]] = {}
        for target, positions in edges_by_target.items():
            d = int(node_degrees[target])
            if d > 1:
                recipients.setdefault((bucket(d), d), []).append(target)
        donor_for = list(range(len(edge_targets)))
        for (_bucket, d), targets in recipients.items():
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

    def _pair_and_context(self, h_text, h_visual, src, dst, context_shuffle, diagnostics):
        u_text = self.text_relation_projection(h_text)
        u_visual = self.visual_relation_projection(h_visual)
        text_cosine = self._cosine(u_text[dst], u_text[src])
        visual_cosine = self._cosine(u_visual[dst], u_visual[src])
        pair_t, pair_v = [], []
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            pair_t.append(self.pair_text_encoder(u_text[dst[start:end]], u_text[src[start:end]],
                                                  text_cosine[start:end], visual_cosine[start:end]))
            pair_v.append(self.pair_visual_encoder(u_visual[dst[start:end]], u_visual[src[start:end]],
                                                    text_cosine[start:end], visual_cosine[start:end]))
        empty = h_text.new_empty((0, self.relation_dim))
        pair_text = torch.cat(pair_t) if pair_t else empty
        pair_visual = torch.cat(pair_v) if pair_v else empty
        context_text, _, degree = leave_one_out_context(
            u_text, src, dst, h_text.size(0), self.no_context_text
        )
        context_visual, _, _ = leave_one_out_context(
            u_visual, src, dst, h_visual.size(0), self.no_context_visual
        )
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
            relation_t.append(r_t); relation_v.append(r_v)
            if diagnostics:
                attn_t.append(w_t); attn_v.append(w_v)
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

    def _relation_memory(self, h_text, h_visual, src, dst, context_shuffle, diagnostics):
        evidence = self._pair_and_context(h_text, h_visual, src, dst, context_shuffle, diagnostics)
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

    def build_execution_query(self, modality: str, target: torch.Tensor) -> torch.Tensor:
        if modality == "text":
            projection, embedding, norm = self.text_target_projection, self.text_dynamic_embedding, self.text_query_norm
        elif modality == "visual":
            projection, embedding, norm = self.visual_target_projection, self.visual_dynamic_embedding, self.visual_query_norm
        else:
            raise ValueError("modality must be 'text' or 'visual'")
        return norm(projection(target) + embedding)

    def _execution_step(self, state, initial_state, memory, base_relation_code, src, dst, modality,
                        operator_enabled, capture, step, conditioner_enabled=True,
                        step1_dynamic_off=False, frozen_query=False):
        aggregate = state.new_zeros((state.size(0), self.hidden_dim))
        code_parts, delta_code_parts, mod_parts, attn_parts = [], [], [], []
        latent_parts, ur_parts, uh_parts = [], [], []
        base_norms, delta_norms, ratios, cosines, rotations = [], [], [], [], []
        scale_parts, message_rotation_parts, delta_message_parts = [], [], []
        retriever = self.text_retriever if modality == "text" else self.visual_retriever
        # v2 used one pure relation-grounded retriever for both static and
        # state-conditioned queries; sharing it keeps the comparison controlled.
        dynamic_retriever = retriever
        conditioner = self.text_conditioner if modality == "text" else self.visual_conditioner
        operator = self.text_operator if modality == "text" else self.visual_operator
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            r = base_relation_code[start:end]
            target_state = initial_state[dst[start:end]] if frozen_query else state[dst[start:end]]
            if self.variant == "context_static":
                code = r
                delta_code = torch.zeros_like(r)
                latent = r.new_zeros((r.size(0), conditioner.output.in_features))
                ur, uh = latent, latent
                weights = r.new_empty((r.size(0), 0))
            elif self.variant == "context_attn_dynamic":
                query = self.build_execution_query(modality, target_state)
                code, weights = dynamic_retriever(query, memory[start:end], need_weights=capture)
                delta_code = code - r
                latent = r.new_zeros((r.size(0), conditioner.output.in_features))
                ur, uh = latent, latent
            else:
                if self.variant == "context_bilinear_delta":
                    condition = target_state - initial_state[dst[start:end]]
                else:
                    condition = target_state
                code, delta_code, ur, uh, latent = conditioner(r, condition)
                if not conditioner_enabled or (step1_dynamic_off and step == 1):
                    delta_code = torch.zeros_like(delta_code)
                    code = r
                weights = r.new_empty((r.size(0), 0))
            source_state = state[src[start:end]]
            base, delta, modulation, message = operator(source_state, code, enabled=operator_enabled)
            aggregate.index_add_(0, dst[start:end], message)
            if capture:
                code_parts.append(code); delta_code_parts.append(delta_code)
                mod_parts.append(modulation); attn_parts.append(weights)
                latent_parts.append(latent); ur_parts.append(ur); uh_parts.append(uh)
                base_norm = base.norm(dim=-1); delta_norm = delta.norm(dim=-1)
                cos_delta = F.cosine_similarity(base, delta, dim=-1, eps=1e-8)
                cos_delta = torch.where(delta_norm > 1e-12, cos_delta, torch.full_like(cos_delta, float("nan")))
                cos_message = F.cosine_similarity(base, message, dim=-1, eps=1e-8)
                angle = torch.acos(cos_message.clamp(-1.0, 1.0)) * (180.0 / math.pi)
                base_norms.append(base_norm); delta_norms.append(delta_norm)
                ratios.append(delta_norm / (base_norm + torch.finfo(state.dtype).eps))
                cosines.append(cos_delta); rotations.append(angle)
                scale_parts.append(message.norm(dim=-1) / (base_norm + torch.finfo(state.dtype).eps))
                message_rotation_parts.append(1.0 - cos_message)
                delta_message_parts.append(delta)
        degree = incoming_degree(dst, state.size(0)).to(state.dtype).clamp_min_(1)
        aggregate = aggregate / degree.unsqueeze(-1)
        updater = self.text_state_update if modality == "text" else self.visual_state_update
        next_state = updater(state, aggregate)
        empty_rel = state.new_empty((0, self.relation_dim))
        empty_rank = state.new_empty((0, int(self.text_conditioner.output.in_features)))
        empty_scalar = state.new_empty((0,))
        details = {
            "execution_code": torch.cat(code_parts) if code_parts else empty_rel,
            "delta_execution_code": torch.cat(delta_code_parts) if delta_code_parts else empty_rel,
            "modulation": torch.cat(mod_parts) if mod_parts else empty_rank,
            "conditioner_latent": torch.cat(latent_parts) if latent_parts else empty_rank,
            "conditioner_u_r": torch.cat(ur_parts) if ur_parts else empty_rank,
            "conditioner_u_h": torch.cat(uh_parts) if uh_parts else empty_rank,
            "execution_attention": torch.cat(attn_parts) if attn_parts else state.new_empty((0, 2)),
            "delta_message": torch.cat(delta_message_parts) if delta_message_parts else state.new_empty((0, self.hidden_dim)),
            "base_message_norm": torch.cat(base_norms) if base_norms else empty_scalar,
            "delta_message_norm": torch.cat(delta_norms) if delta_norms else empty_scalar,
            "operator_deviation_ratio": torch.cat(ratios) if ratios else empty_scalar,
            "cosine_base_delta": torch.cat(cosines) if cosines else empty_scalar,
            "message_rotation_degrees": torch.cat(rotations) if rotations else empty_scalar,
            "message_scale": torch.cat(scale_parts) if scale_parts else empty_scalar,
            "message_rotation": torch.cat(message_rotation_parts) if message_rotation_parts else empty_scalar,
        }
        return next_state, details

    def _encode(self, x, edge_index, *, operator_enabled=True, diagnostics=False,
                relation_shuffle=False, context_shuffle=False, conditioner_enabled=True,
                step1_dynamic_off=False, frozen_query=False):
        if not self.training:
            self._flush_epoch_trace()
        x_t, x_v = x[:, :self.text_dim], x[:, self.text_dim:]
        h0_t, h0_v = self.text_projector(x_t), self.visual_projector(x_v)
        edges, degree = self._graph(edge_index, x.size(0), x.device)
        src, dst = edges
        memory, evidence = self._relation_memory(h0_t, h0_v, src, dst, context_shuffle, diagnostics)
        if relation_shuffle and src.numel():
            positions_by_target: dict[int, list[int]] = {}
            for edge_pos, target in enumerate(dst.detach().cpu().tolist()):
                positions_by_target.setdefault(int(target), []).append(edge_pos)
            permutation = list(range(src.numel()))
            for positions in positions_by_target.values():
                if len(positions) > 1:
                    for idx, pos in enumerate(positions):
                        permutation[pos] = positions[(idx + 1) % len(positions)]
            memory = memory[torch.tensor(permutation, dtype=torch.long, device=memory.device)]
        base_t = self._base_relation_code("text", memory)
        base_v = self._base_relation_code("visual", memory)
        h_t, h_v = h0_t, h0_v
        states_t, states_v = [h_t], [h_v]
        details_t, details_v = [], []
        for step in range(self.num_interaction_steps):
            h_t, detail_t = self._execution_step(
                h_t, h0_t, memory, base_t, src, dst, "text", operator_enabled, diagnostics,
                step, conditioner_enabled, step1_dynamic_off, frozen_query and step == 1,
            )
            h_v, detail_v = self._execution_step(
                h_v, h0_v, memory, base_v, src, dst, "visual", operator_enabled, diagnostics,
                step, conditioner_enabled, step1_dynamic_off, frozen_query and step == 1,
            )
            states_t.append(h_t); states_v.append(h_v)
            if diagnostics:
                details_t.append(detail_t); details_v.append(detail_v)
        fused = self.fusion(torch.cat((h_t, h_v), -1))
        values: dict[str, Any] = {
            "H0_text": h0_t, "H0_visual": h0_v,
            "H1_text": states_t[1], "H1_visual": states_v[1],
            "final_text": h_t, "final_visual": h_v, "fused_z": fused,
            "canonical_edge_index": edges, "degree": degree, "relation_memory": memory,
            "base_relation_text": base_t, "base_relation_visual": base_v,
            **evidence,
        }
        if diagnostics:
            for step in range(self.num_interaction_steps):
                for modality, details in (("text", details_t), ("visual", details_v)):
                    for key, value in details[step].items():
                        values[f"{key}_{modality}_step{step}"] = value
            for key in ("execution_code", "delta_execution_code", "modulation", "conditioner_latent",
                        "conditioner_u_r", "conditioner_u_h", "execution_attention",
                        "delta_message", "operator_deviation_ratio", "base_message_norm",
                        "delta_message_norm", "cosine_base_delta", "message_rotation_degrees",
                        "message_scale", "message_rotation"):
                values[key] = {f"{m}_step{s}": values[f"{key}_{m}_step{s}"]
                               for m in ("text", "visual") for s in range(self.num_interaction_steps)}
        return fused, values

    def forward(self, x, edge_index=None, *, operator_enabled=True, relation_shuffle=False,
                context_shuffle=False, conditioner_enabled=True, step1_dynamic_off=False,
                frozen_query=False):
        z, _ = self._encode(
            x, edge_index, operator_enabled=operator_enabled, diagnostics=False,
            relation_shuffle=relation_shuffle, context_shuffle=context_shuffle,
            conditioner_enabled=conditioner_enabled, step1_dynamic_off=step1_dynamic_off,
            frozen_query=frozen_query,
        )
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(self, x, edge_index=None, *, operator_enabled=True, relation_shuffle=False,
                context_shuffle=False, conditioner_enabled=True, step1_dynamic_off=False,
                frozen_query=False, **_kwargs):
        _, values = self._encode(
            x, edge_index, operator_enabled=operator_enabled, diagnostics=True,
            relation_shuffle=relation_shuffle, context_shuffle=context_shuffle,
            conditioner_enabled=conditioner_enabled, step1_dynamic_off=step1_dynamic_off,
            frozen_query=frozen_query,
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

    def active_parameter_names(self) -> set[str]:
        return {name for name, _ in self.named_parameters() if self._group_for_parameter(name) is not None}


__all__ = ["Model", "VARIANTS", "TRACE_GROUPS"]
