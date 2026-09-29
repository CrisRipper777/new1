from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_core_components import (
    ConditionalLowRankOperator,
    CrossAttentionBlock,
    ExecutionCrossAttention,
    ModalityProjector,
    PairRelationEncoder,
    StateUpdate,
    canonicalize_physical_edges,
    incoming_degree,
    leave_one_out_context,
)


VALID_VARIANTS = {
    "global_dynamic", "pair_dynamic", "context_static", "context_dynamic"
}


class Model(nn.Module):
    """M0-Core: contextual relation interpretation followed by semantic execution."""

    def __init__(self, cfg, data_info: dict[str, Any]) -> None:
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        input_dim = int(data_info["input_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.text_dim + self.visual_dim != input_dim:
            raise ValueError("interaction_core requires ordered text and visual input features")

        model_cfg = cfg.model
        self.hidden_dim = int(model_cfg.hidden_dim)
        self.relation_dim = int(model_cfg.relation_dim)
        self.operator_rank = int(model_cfg.operator_rank)
        self.num_interaction_steps = int(model_cfg.num_interaction_steps)
        if self.relation_dim != 64:
            raise ValueError("M0-Core fixes relation_dim=64")
        if self.operator_rank != 32:
            raise ValueError("M0-Core fixes operator_rank=32")
        if self.num_interaction_steps != 2:
            raise ValueError("M0-Core fixes num_interaction_steps=2")
        self.variant = str(model_cfg.variant).strip().lower()
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"model.variant must be one of {sorted(VALID_VARIANTS)}")
        self.dropout_p = float(model_cfg.dropout)
        self.relation_dropout = float(model_cfg.relation_dropout)
        self.edge_chunk_size = int(model_cfg.get("edge_chunk_size", 16384))
        if self.edge_chunk_size < 1:
            raise ValueError("model.edge_chunk_size must be positive")
        relation_heads = int(model_cfg.relation.num_heads)
        execution_heads = int(model_cfg.execution.num_heads)
        zero_init_up = bool(model_cfg.execution.zero_init_up)

        # Independent modality streams; fusion happens only after terminal states.
        self.text_projector = ModalityProjector(self.text_dim, self.hidden_dim, self.dropout_p)
        self.visual_projector = ModalityProjector(self.visual_dim, self.hidden_dim, self.dropout_p)
        self.text_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)

        # The pair/context relation encoder structure is shared by every variant.
        self.pair_text_encoder = PairRelationEncoder(self.relation_dim, self.relation_dropout)
        self.pair_visual_encoder = PairRelationEncoder(self.relation_dim, self.relation_dropout)
        self.relation_text_cross_attention = CrossAttentionBlock(
            self.relation_dim, relation_heads, self.relation_dropout
        )
        self.relation_visual_cross_attention = CrossAttentionBlock(
            self.relation_dim, relation_heads, self.relation_dropout
        )
        self.no_context_text = nn.Parameter(torch.empty(self.relation_dim))
        self.no_context_visual = nn.Parameter(torch.empty(self.relation_dim))
        self.null_context_text = nn.Parameter(torch.empty(self.relation_dim))
        self.null_context_visual = nn.Parameter(torch.empty(self.relation_dim))
        self.global_relation_text = nn.Parameter(torch.empty(self.relation_dim))
        self.global_relation_visual = nn.Parameter(torch.empty(self.relation_dim))
        for token in (
            self.no_context_text,
            self.no_context_visual,
            self.null_context_text,
            self.null_context_visual,
            self.global_relation_text,
            self.global_relation_visual,
        ):
            nn.init.normal_(token, mean=0.0, std=0.02)

        # Shared stage-II parameters across the two steps, independent by modality.
        self.text_target_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.text_source_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_target_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_source_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.text_dynamic_embedding = nn.Parameter(torch.empty(self.relation_dim))
        self.visual_dynamic_embedding = nn.Parameter(torch.empty(self.relation_dim))
        self.static_query_text = nn.Parameter(torch.empty(self.relation_dim))
        self.static_query_visual = nn.Parameter(torch.empty(self.relation_dim))
        for query in (
            self.text_dynamic_embedding,
            self.visual_dynamic_embedding,
            self.static_query_text,
            self.static_query_visual,
        ):
            nn.init.normal_(query, mean=0.0, std=0.02)
        self.text_query_norm = nn.LayerNorm(self.relation_dim)
        self.visual_query_norm = nn.LayerNorm(self.relation_dim)
        self.text_execution_attention = ExecutionCrossAttention(
            self.relation_dim, execution_heads, self.relation_dropout
        )
        self.visual_execution_attention = ExecutionCrossAttention(
            self.relation_dim, execution_heads, self.relation_dropout
        )
        self.text_operator = ConditionalLowRankOperator(
            self.hidden_dim, self.relation_dim, self.operator_rank, zero_init_up
        )
        self.visual_operator = ConditionalLowRankOperator(
            self.hidden_dim, self.relation_dim, self.operator_rank, zero_init_up
        )
        self.text_state_update = StateUpdate(self.hidden_dim, self.dropout_p)
        self.visual_state_update = StateUpdate(self.hidden_dim, self.dropout_p)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
        )
        self.out_dim = self.hidden_dim
        self._cached_edge_input = None
        self._cached_edges = None
        self._cached_degree = None

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(
                f"x must be [N,{self.text_dim + self.visual_dim}], got {tuple(x.shape)}"
            )
        return x[:, : self.text_dim], x[:, self.text_dim :]

    def _graph(self, edge_index: torch.Tensor | None, num_nodes: int, device):
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

    def _pair_and_context(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        context_mode: str,
    ) -> dict[str, torch.Tensor]:
        u_text = self.text_relation_projection(h_text)
        u_visual = self.visual_relation_projection(h_visual)
        text_cosine = self._cosine(u_text[dst], u_text[src])
        visual_cosine = self._cosine(u_visual[dst], u_visual[src])
        pair_text_parts: list[torch.Tensor] = []
        pair_visual_parts: list[torch.Tensor] = []
        empty = h_text.new_empty((0, self.relation_dim))
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            pair_text_parts.append(
                self.pair_text_encoder(
                    u_text[dst[start:end]], u_text[src[start:end]],
                    text_cosine[start:end], visual_cosine[start:end],
                )
            )
            pair_visual_parts.append(
                self.pair_visual_encoder(
                    u_visual[dst[start:end]], u_visual[src[start:end]],
                    text_cosine[start:end], visual_cosine[start:end],
                )
            )
        pair_text = torch.cat(pair_text_parts, dim=0) if pair_text_parts else empty
        pair_visual = torch.cat(pair_visual_parts, dim=0) if pair_visual_parts else empty

        if context_mode == "full":
            context_text, _, _ = leave_one_out_context(
                u_text, src, dst, h_text.size(0), self.no_context_text
            )
            context_visual, _, _ = leave_one_out_context(
                u_visual, src, dst, h_visual.size(0), self.no_context_visual
            )
        elif context_mode == "null":
            context_text = self.null_context_text.to(h_text).expand(src.numel(), -1)
            context_visual = self.null_context_visual.to(h_visual).expand(src.numel(), -1)
        else:
            raise ValueError("context_mode must be 'full' or 'null'")

        relation_text_parts: list[torch.Tensor] = []
        relation_visual_parts: list[torch.Tensor] = []
        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            key_value = torch.stack(
                (pair_visual[start:end], context_text[start:end], context_visual[start:end]), dim=1
            )
            relation_text_parts.append(
                self.relation_text_cross_attention(pair_text[start:end], key_value)
            )
            reverse_key_value = torch.stack(
                (pair_text[start:end], context_text[start:end], context_visual[start:end]), dim=1
            )
            relation_visual_parts.append(
                self.relation_visual_cross_attention(pair_visual[start:end], reverse_key_value)
            )
        relation_text = torch.cat(relation_text_parts, dim=0) if relation_text_parts else empty
        relation_visual = torch.cat(relation_visual_parts, dim=0) if relation_visual_parts else empty
        return {
            "U_text": u_text,
            "U_visual": u_visual,
            "pair_text": pair_text,
            "pair_visual": pair_visual,
            "compatibility_text": text_cosine,
            "compatibility_visual": visual_cosine,
            "compatibility_abs_difference": (text_cosine - visual_cosine).abs(),
            "context_text": context_text,
            "context_visual": context_visual,
            "relation_text": relation_text,
            "relation_visual": relation_visual,
        }

    def _relation_memory(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        context_mode: str,
        relation_mode: str,
        diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.variant == "global_dynamic":
            relation_text = self.global_relation_text.to(h_text).expand(src.numel(), -1)
            relation_visual = self.global_relation_visual.to(h_visual).expand(src.numel(), -1)
            if diagnostics:
                # Diagnostic-only endpoint/context evidence; never used by this variant's forward.
                evidence = self._pair_and_context(h_text, h_visual, src, dst, "full")
            else:
                empty = h_text.new_empty((0, self.relation_dim))
                evidence = {
                    key: empty
                    for key in ("pair_text", "pair_visual", "context_text", "context_visual")
                }
        else:
            selected_context = "null" if self.variant == "pair_dynamic" else context_mode
            evidence = self._pair_and_context(h_text, h_visual, src, dst, selected_context)
            relation_text = evidence["relation_text"]
            relation_visual = evidence["relation_visual"]

        if relation_mode == "mean" and relation_text.numel() > 0:
            relation_text = relation_text.mean(dim=0, keepdim=True).expand_as(relation_text)
            relation_visual = relation_visual.mean(dim=0, keepdim=True).expand_as(relation_visual)
        elif relation_mode != "specific":
            raise ValueError("relation_mode must be 'specific' or 'mean'")
        evidence["relation_text"] = relation_text
        evidence["relation_visual"] = relation_visual
        return torch.stack((relation_text, relation_visual), dim=1), evidence

    def build_execution_query(
        self, modality: str, target: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        if self.variant == "context_static":
            query = self.static_query_text if modality == "text" else self.static_query_visual
            return query.to(target).reshape(1, -1).expand(target.size(0), -1)
        if modality == "text":
            return self.text_query_norm(
                self.text_target_projection(target)
                + self.text_source_projection(source)
                + self.text_dynamic_embedding
            )
        if modality == "visual":
            return self.visual_query_norm(
                self.visual_target_projection(target)
                + self.visual_source_projection(source)
                + self.visual_dynamic_embedding
            )
        raise ValueError("modality must be 'text' or 'visual'")

    def _execution_step(
        self,
        state: torch.Tensor,
        relation_memory: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        modality: str,
        operator_enabled: bool,
        capture: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aggregate = state.new_zeros((state.size(0), self.hidden_dim))
        code_parts: list[torch.Tensor] = []
        modulation_parts: list[torch.Tensor] = []
        base_norm_parts: list[torch.Tensor] = []
        delta_norm_parts: list[torch.Tensor] = []
        ratio_parts: list[torch.Tensor] = []
        attention = (
            self.text_execution_attention if modality == "text" else self.visual_execution_attention
        )
        operator = self.text_operator if modality == "text" else self.visual_operator

        for start in range(0, src.numel(), self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, src.numel())
            source_state = state[src[start:end]]
            target_state = state[dst[start:end]]
            query = self.build_execution_query(modality, target_state, source_state)
            execution_code = attention(query, relation_memory[start:end])
            base, delta, modulation = operator(source_state, execution_code, enabled=operator_enabled)
            aggregate.index_add_(0, dst[start:end], base + delta)
            if capture:
                code_parts.append(execution_code)
                modulation_parts.append(modulation)
                base_norm = base.norm(dim=-1)
                delta_norm = delta.norm(dim=-1)
                base_norm_parts.append(base_norm)
                delta_norm_parts.append(delta_norm)
                ratio_parts.append(delta_norm / (base_norm + torch.finfo(state.dtype).eps))

        degree = incoming_degree(dst, state.size(0)).to(state.dtype).clamp_min_(1)
        aggregate = aggregate / degree.unsqueeze(-1)
        update = self.text_state_update if modality == "text" else self.visual_state_update
        next_state = update(state, aggregate)
        empty_code = state.new_empty((0, self.relation_dim))
        empty_modulation = state.new_empty((0, self.operator_rank))
        details = {
            "execution_code": torch.cat(code_parts, dim=0) if code_parts else empty_code,
            "modulation": torch.cat(modulation_parts, dim=0) if modulation_parts else empty_modulation,
            "base_message_norm": torch.cat(base_norm_parts, dim=0) if base_norm_parts else state.new_empty((0,)),
            "delta_message_norm": torch.cat(delta_norm_parts, dim=0) if delta_norm_parts else state.new_empty((0,)),
            "operator_deviation_ratio": torch.cat(ratio_parts, dim=0) if ratio_parts else state.new_empty((0,)),
        }
        return next_state, details

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        context_mode: str = "full",
        relation_mode: str = "specific",
        operator_enabled: bool = True,
        diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x_text, x_visual = self._split_features(x)
        h0_text = self.text_projector(x_text)
        h0_visual = self.visual_projector(x_visual)
        edges, degree = self._graph(edge_index, x.size(0), x.device)
        src, dst = edges
        # Relation memory is built once from intrinsic H0 and reused by both steps.
        relation_memory, evidence = self._relation_memory(
            h0_text, h0_visual, src, dst, context_mode, relation_mode, diagnostics
        )
        h_text, h_visual = h0_text, h0_visual
        step_details: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []
        for _ in range(self.num_interaction_steps):
            h_text, text_details = self._execution_step(
                h_text, relation_memory, src, dst, "text", operator_enabled, diagnostics
            )
            h_visual, visual_details = self._execution_step(
                h_visual, relation_memory, src, dst, "visual", operator_enabled, diagnostics
            )
            if diagnostics:
                step_details.append((text_details, visual_details))

        fused = self.fusion(torch.cat((h_text, h_visual), dim=-1))
        values: dict[str, Any] = {
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "final_text": h_text,
            "final_visual": h_visual,
            "fused_z": fused,
            "canonical_edge_index": edges,
            "degree": degree,
            "relation_memory": relation_memory,
            **evidence,
        }
        if diagnostics:
            for step, (text_values, visual_values) in enumerate(step_details):
                for modality, detail in (("text", text_values), ("visual", visual_values)):
                    for key in (
                        "execution_code", "modulation", "base_message_norm",
                        "delta_message_norm", "operator_deviation_ratio",
                    ):
                        values[f"{key}_{modality}_step{step}"] = detail[key]
            ratio_keys = [
                f"operator_deviation_ratio_{modality}_step{step}"
                for modality in ("text", "visual")
                for step in range(self.num_interaction_steps)
            ]
            values["base_message_norm"] = {
                f"{modality}_step{step}": values[f"base_message_norm_{modality}_step{step}"]
                for modality in ("text", "visual")
                for step in range(self.num_interaction_steps)
            }
            values["delta_message_norm"] = {
                f"{modality}_step{step}": values[f"delta_message_norm_{modality}_step{step}"]
                for modality in ("text", "visual")
                for step in range(self.num_interaction_steps)
            }
            values["operator_deviation_ratio"] = {
                key.removeprefix("operator_deviation_ratio_"): values[key]
                for key in ratio_keys
            }
        return fused, values

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        *,
        context_mode: str = "full",
        relation_mode: str = "specific",
        operator_enabled: bool = True,
    ):
        z, _ = self._encode(
            x, edge_index, context_mode, relation_mode, operator_enabled, diagnostics=False
        )
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        *,
        context_mode: str = "full",
        relation_mode: str = "specific",
        operator_enabled: bool = True,
        **_kwargs,
    ) -> dict[str, Any]:
        """Materialize edge-level diagnostics only for explicit analysis calls."""
        _, values = self._encode(
            x, edge_index, context_mode, relation_mode, operator_enabled, diagnostics=True
        )
        return values

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        # Exact recipient LOO context requires the complete graph.
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self.forward(x, edge_index)
        return z.detach().cpu()

    def active_parameter_names(self) -> set[str]:
        """Normal-forward parameters, excluding modules used only by interventions."""
        prefixes = {
            "text_projector", "visual_projector", "text_state_update", "visual_state_update",
            "fusion", "text_execution_attention", "visual_execution_attention",
            "text_operator", "visual_operator",
        }
        exact: set[str] = set()
        if self.variant == "global_dynamic":
            exact.update({"global_relation_text", "global_relation_visual"})
            prefixes.update({
                "text_target_projection", "text_source_projection", "visual_target_projection",
                "visual_source_projection", "text_query_norm", "visual_query_norm",
            })
            exact.update({"text_dynamic_embedding", "visual_dynamic_embedding"})
        else:
            prefixes.update({
                "text_relation_projection", "visual_relation_projection", "pair_text_encoder",
                "pair_visual_encoder", "relation_text_cross_attention",
                "relation_visual_cross_attention",
            })
            if self.variant == "pair_dynamic":
                exact.update({"null_context_text", "null_context_visual"})
            else:
                exact.update({"no_context_text", "no_context_visual"})
            if self.variant == "context_static":
                exact.update({"static_query_text", "static_query_visual"})
            else:
                prefixes.update({
                    "text_target_projection", "text_source_projection", "visual_target_projection",
                    "visual_source_projection", "text_query_norm", "visual_query_norm",
                })
                exact.update({"text_dynamic_embedding", "visual_dynamic_embedding"})
        return {
            name
            for name, _ in self.named_parameters()
            if name in exact or any(name.startswith(prefix + ".") for prefix in prefixes)
        }
