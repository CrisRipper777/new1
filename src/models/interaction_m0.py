from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_components import (
    CompatibilityEvidenceEncoder,
    PairEvidenceEncoder,
    TypedRelationMixer,
    build_incoming_mean_operator,
    canonicalize_physical_edges,
    incoming_mean,
    leave_one_out_context,
)


VALID_VARIANTS = {"generic", "pair", "pair_compat", "pair_compat_context"}


class ModalityProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Model(nn.Module):
    """M0-S1 contextual relation interpretation with a generic executor."""

    def __init__(self, cfg, data_info: dict) -> None:
        super().__init__()
        self.text_dim = int(data_info.get("text_dim", 0))
        self.visual_dim = int(data_info.get("visual_dim", 0))
        input_dim = int(data_info["input_dim"])
        if self.text_dim <= 0 or self.visual_dim <= 0 or self.text_dim + self.visual_dim != input_dim:
            raise ValueError(
                "interaction_m0 requires positive text_dim and visual_dim whose sum equals input_dim"
            )
        self.hidden_dim = int(cfg.model.hidden_dim)
        self.relation_dim = int(cfg.model.relation_dim)
        self.num_interaction_steps = int(cfg.model.num_interaction_steps)
        if self.num_interaction_steps != 2:
            raise ValueError("M0-S1 fixes num_interaction_steps=2")
        if str(cfg.model.get("context", {}).get("type", "mean_loo")) != "mean_loo":
            raise ValueError("M0-S1 fixes context.type=mean_loo")
        self.variant = str(cfg.model.get("stage1_variant", "generic")).strip().lower()
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"stage1_variant must be one of {sorted(VALID_VARIANTS)}")
        self.dropout = float(cfg.model.dropout)
        self.relation_dropout = float(cfg.model.relation_dropout)
        self.edge_chunk_size = int(cfg.model.get("edge_chunk_size", 32768))
        if self.edge_chunk_size < 1:
            raise ValueError("edge_chunk_size must be positive")

        self.text_projector = ModalityProjector(self.text_dim, self.hidden_dim, self.dropout)
        self.visual_projector = ModalityProjector(self.visual_dim, self.hidden_dim, self.dropout)
        self.text_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)
        self.visual_relation_projection = nn.Linear(self.hidden_dim, self.relation_dim, bias=False)

        # All P/PC/PCC presets instantiate the same encoders, null tokens, and mixer.
        self.pair_text_encoder = PairEvidenceEncoder(self.relation_dim, self.relation_dropout)
        self.pair_visual_encoder = PairEvidenceEncoder(self.relation_dim, self.relation_dropout)
        self.compatibility_encoder = CompatibilityEvidenceEncoder(self.relation_dim)
        self.no_context_text = nn.Parameter(torch.empty(self.relation_dim))
        self.no_context_visual = nn.Parameter(torch.empty(self.relation_dim))
        self.null_compatibility = nn.Parameter(torch.empty(self.relation_dim))
        self.null_context_text = nn.Parameter(torch.empty(self.relation_dim))
        self.null_context_visual = nn.Parameter(torch.empty(self.relation_dim))
        for token in (
            self.no_context_text,
            self.no_context_visual,
            self.null_compatibility,
            self.null_context_text,
            self.null_context_visual,
        ):
            nn.init.normal_(token, mean=0.0, std=0.02)
        mixer_cfg = cfg.model.relation_mixer
        self.relation_mixer = TypedRelationMixer(
            self.relation_dim,
            num_heads=int(mixer_cfg.num_heads),
            ffn_ratio=int(mixer_cfg.ffn_ratio),
            dropout=self.relation_dropout,
        )

        # Independent modality execution streams; each set of weights is shared by both steps.
        self.text_message_source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.visual_message_source = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.text_message_relation = nn.Linear(self.relation_dim, self.hidden_dim, bias=False)
        self.visual_message_relation = nn.Linear(self.relation_dim, self.hidden_dim, bias=False)
        self.text_message_update = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.visual_message_update = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.text_state_norm = nn.LayerNorm(self.hidden_dim)
        self.visual_state_norm = nn.LayerNorm(self.hidden_dim)
        self.text_update_dropout = nn.Dropout(self.dropout)
        self.visual_update_dropout = nn.Dropout(self.dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.out_dim = self.hidden_dim

        self._cached_edge_input = None
        self._cached_canonical_edges = None
        self._cached_adjacency = None
        self._cached_degree = None

    def _split_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.size(1) != self.text_dim + self.visual_dim:
            raise ValueError(
                f"x must have shape [N, {self.text_dim + self.visual_dim}], got {tuple(x.shape)}"
            )
        x_text = x[:, : self.text_dim]
        x_visual = x[:, self.text_dim : self.text_dim + self.visual_dim]
        return x_text, x_visual

    def _graph(self, edge_index: torch.Tensor | None, num_nodes: int, dtype: torch.dtype, device):
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        elif edge_index.device != device:
            edge_index = edge_index.to(device)
        if self._cached_edge_input is not edge_index or self._cached_adjacency is None:
            canonical = canonicalize_physical_edges(edge_index, num_nodes).to(device)
            src, dst = canonical
            adjacency, degree = build_incoming_mean_operator(src, dst, num_nodes, dtype)
            self._cached_edge_input = edge_index
            self._cached_canonical_edges = canonical
            self._cached_adjacency = adjacency
            self._cached_degree = degree
        return (
            self._cached_canonical_edges,
            self._cached_adjacency,
            self._cached_degree,
        )

    def _raw_evidence(
        self,
        u_text: torch.Tensor,
        u_visual: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        degree: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pair_text = self.pair_text_encoder(u_text[dst], u_text[src])
        pair_visual = self.pair_visual_encoder(u_visual[dst], u_visual[src])
        score_text = F.cosine_similarity(u_text[dst], u_text[src], dim=-1, eps=1e-8)
        score_visual = F.cosine_similarity(u_visual[dst], u_visual[src], dim=-1, eps=1e-8)
        compatibility, score_difference = self.compatibility_encoder(score_text, score_visual)
        context_text, defined, _ = leave_one_out_context(
            u_text, src, dst, u_text.size(0), self.no_context_text
        )
        context_visual, _, _ = leave_one_out_context(
            u_visual, src, dst, u_visual.size(0), self.no_context_visual
        )
        return {
            "pair_text": pair_text,
            "pair_visual": pair_visual,
            "compatibility": compatibility,
            "compatibility_token": compatibility,
            "compatibility_text": score_text,
            "compatibility_visual": score_visual,
            "compatibility_abs_difference": score_difference,
            "context_text": context_text,
            "context_visual": context_visual,
            "context_defined_mask": defined,
            "degree": degree,
        }

    def _mix_chunk(
        self,
        pair_text: torch.Tensor,
        pair_visual: torch.Tensor,
        compatibility: torch.Tensor,
        context_text: torch.Tensor,
        context_visual: torch.Tensor,
        need_weights: bool,
        include_context: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        count = pair_text.size(0)
        if self.variant == "pair":
            compatibility = self.null_compatibility.to(pair_text).expand(count, -1)
            context_text = self.null_context_text.to(pair_text).expand(count, -1)
            context_visual = self.null_context_visual.to(pair_text).expand(count, -1)
        elif self.variant == "pair_compat":
            context_text = self.null_context_text.to(pair_text).expand(count, -1)
            context_visual = self.null_context_visual.to(pair_text).expand(count, -1)
        elif self.variant == "pair_compat_context" and include_context is False:
            context_text = self.null_context_text.to(pair_text).expand(count, -1)
            context_visual = self.null_context_visual.to(pair_text).expand(count, -1)
        tokens = torch.stack(
            (pair_text, pair_visual, compatibility, context_text, context_visual), dim=1
        )
        return self.relation_mixer(tokens, need_weights=need_weights)

    def _interpret(
        self,
        raw: dict[str, torch.Tensor],
        need_attention: bool,
        keep_details: bool,
    ) -> dict[str, torch.Tensor | None]:
        pair_text = raw["pair_text"]
        pair_visual = raw["pair_visual"]
        compatibility = raw["compatibility"]
        context_text = raw["context_text"]
        context_visual = raw["context_visual"]
        edge_count = pair_text.size(0)
        pieces_text = []
        pieces_visual = []
        pieces_attention = []
        pieces_pc_text = []
        pieces_pc_visual = []
        for start in range(0, edge_count, self.edge_chunk_size):
            end = min(start + self.edge_chunk_size, edge_count)
            relation, attention = self._mix_chunk(
                pair_text[start:end],
                pair_visual[start:end],
                compatibility[start:end],
                context_text[start:end],
                context_visual[start:end],
                need_weights=need_attention,
            )
            pieces_text.append(relation[:, 0])
            pieces_visual.append(relation[:, 1])
            if attention is not None:
                pieces_attention.append(attention)
            if self.variant == "pair_compat_context" and keep_details:
                pc_relation, _ = self._mix_chunk(
                    pair_text[start:end],
                    pair_visual[start:end],
                    compatibility[start:end],
                    context_text[start:end],
                    context_visual[start:end],
                    need_weights=False,
                    include_context=False,
                )
                pieces_pc_text.append(pc_relation[:, 0])
                pieces_pc_visual.append(pc_relation[:, 1])
        device = pair_text.device
        empty = pair_text.new_empty((0, self.relation_dim))
        relation_text = torch.cat(pieces_text, dim=0) if pieces_text else empty
        relation_visual = torch.cat(pieces_visual, dim=0) if pieces_visual else empty
        relation_attention = torch.cat(pieces_attention, dim=0) if pieces_attention else None
        result: dict[str, torch.Tensor | None] = {
            "relation_text": relation_text,
            "relation_visual": relation_visual,
            "relation_attention": relation_attention,
            "pair_compat_relation_text": torch.cat(pieces_pc_text, dim=0) if pieces_pc_text else None,
            "pair_compat_relation_visual": torch.cat(pieces_pc_visual, dim=0) if pieces_pc_visual else None,
        }
        return result

    def _zero_relation(self, num_edges: int, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_zeros((int(num_edges), self.relation_dim))

    def _interaction_step(
        self,
        h_text: torch.Tensor,
        h_visual: torch.Tensor,
        relation_text: torch.Tensor,
        relation_visual: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        adjacency: torch.Tensor,
        degree: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Since W is linear and bias-free, W(mean_j H_j) equals mean_j W(H_j).
        # Relation messages are likewise mean-aggregated in relation space before W_r.
        mean_text = torch.sparse.mm(adjacency, h_text)
        mean_visual = torch.sparse.mm(adjacency, h_visual)
        mean_relation_text = incoming_mean(relation_text, dst, h_text.size(0))
        mean_relation_visual = incoming_mean(relation_visual, dst, h_visual.size(0))
        message_text = self.text_message_source(mean_text) + self.text_message_relation(mean_relation_text)
        message_visual = self.visual_message_source(mean_visual) + self.visual_message_relation(mean_relation_visual)
        next_text = self.text_state_norm(
            h_text + self.text_update_dropout(self.text_message_update(F.gelu(message_text)))
        )
        next_visual = self.visual_state_norm(
            h_visual + self.visual_update_dropout(self.visual_message_update(F.gelu(message_visual)))
        )
        return next_text, next_visual

    def _encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        need_attention: bool,
        keep_details: bool,
    ) -> dict[str, torch.Tensor | None]:
        x_text, x_visual = self._split_features(x)
        h0_text = self.text_projector(x_text)
        h0_visual = self.visual_projector(x_visual)
        graph, adjacency, degree = self._graph(edge_index, x.size(0), x.dtype, x.device)
        src, dst = graph
        u_text = self.text_relation_projection(h0_text)
        u_visual = self.visual_relation_projection(h0_visual)
        raw = self._raw_evidence(u_text, u_visual, src, dst, degree) if self.variant != "generic" or keep_details else None

        if self.variant == "generic":
            relation_text = self._zero_relation(src.numel(), h0_text)
            relation_visual = self._zero_relation(src.numel(), h0_text)
            interpreted: dict[str, torch.Tensor | None] = {
                "relation_text": relation_text,
                "relation_visual": relation_visual,
                "relation_attention": None,
                "pair_compat_relation_text": None,
                "pair_compat_relation_visual": None,
            }
        else:
            assert raw is not None
            interpreted = self._interpret(raw, need_attention, keep_details)
            relation_text = interpreted["relation_text"]
            relation_visual = interpreted["relation_visual"]
            assert relation_text is not None and relation_visual is not None

        h_text, h_visual = h0_text, h0_visual
        for _ in range(self.num_interaction_steps):
            h_text, h_visual = self._interaction_step(
                h_text, h_visual, relation_text, relation_visual, src, dst, adjacency, degree
            )
        z = self.fusion(torch.cat((h_text, h_visual), dim=-1))

        result: dict[str, torch.Tensor | None] = {
            "H0_text": h0_text,
            "H0_visual": h0_visual,
            "U_text": u_text,
            "U_visual": u_visual,
            "canonical_edge_index": graph,
            "degree": degree,
            "final_text": h_text,
            "final_visual": h_visual,
            "fused_z": z,
            **interpreted,
        }
        if raw is not None:
            result.update(raw)

        if keep_details and raw is not None and self.variant != "generic":
            source_text = self.text_message_source(h0_text[src])
            source_visual = self.visual_message_source(h0_visual[src])
            edge_rel_text = self.text_message_relation(relation_text)
            edge_rel_visual = self.visual_message_relation(relation_visual)
            eps = torch.finfo(h0_text.dtype).eps
            result["relation_message_ratio_text"] = edge_rel_text.norm(dim=-1) / (source_text.norm(dim=-1) + eps)
            result["relation_message_ratio_visual"] = edge_rel_visual.norm(dim=-1) / (source_visual.norm(dim=-1) + eps)
            result["context_correction_text"] = None
            result["context_correction_visual"] = None
            if self.variant == "pair_compat_context":
                pc_text = interpreted["pair_compat_relation_text"]
                pc_visual = interpreted["pair_compat_relation_visual"]
                assert pc_text is not None and pc_visual is not None
                result["context_correction_text"] = (relation_text - pc_text).norm(dim=-1)
                result["context_correction_visual"] = (relation_visual - pc_visual).norm(dim=-1)
        elif keep_details:
            result["relation_message_ratio_text"] = h0_text.new_zeros(src.numel())
            result["relation_message_ratio_visual"] = h0_text.new_zeros(src.numel())
            result["context_correction_text"] = None
            result["context_correction_visual"] = None
        return result

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None):
        values = self._encode(x, edge_index, need_attention=False, keep_details=False)
        z = values["fused_z"]
        assert z is not None
        return z, None, None, z.new_tensor(0.0), {}

    def analyze(self, x: torch.Tensor, edge_index: torch.Tensor | None = None, **_kwargs):
        """Return relation evidence, states, contribution, and optional attention diagnostics."""
        return self._encode(x, edge_index, need_attention=True, keep_details=True)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        device: torch.device | None = None,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        # This encoder requires the whole physical graph for exact leave-one-out contexts.
        self.eval()
        if device is None:
            device = next(self.parameters()).device
        edge_index = edge_index.to(device) if edge_index is not None else None
        z, _, _, _, _ = self.forward(x.to(device), edge_index)
        return z.detach().cpu()
