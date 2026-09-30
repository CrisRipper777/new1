from __future__ import annotations

import torch
import torch.nn as nn


HISTORY_TOKEN_LABELS = ("T0", "T1", "T2", "V0", "V1", "V2")


class IncomingMeanStd(nn.Module):
    """Differentiable per-recipient mean and population std over incoming edges."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)

    def forward(self, edge_values: torch.Tensor, edge_targets: torch.Tensor, num_nodes: int):
        if edge_values.ndim != 2 or edge_targets.ndim != 1:
            raise ValueError("edge_values must be [E,D] and edge_targets must be [E]")
        if edge_values.size(0) != edge_targets.numel():
            raise ValueError("edge_values and edge_targets must have the same edge count")
        if edge_targets.dtype != torch.long:
            raise ValueError("edge_targets must be torch.long")
        if edge_targets.numel() and (int(edge_targets.min()) < 0 or int(edge_targets.max()) >= num_nodes):
            raise ValueError("edge target index is outside num_nodes")

        width = edge_values.size(-1)
        sums = edge_values.new_zeros((num_nodes, width))
        squares = edge_values.new_zeros((num_nodes, width))
        counts = torch.bincount(edge_targets, minlength=num_nodes).to(edge_values.dtype)
        if edge_targets.numel():
            sums = sums.index_add(0, edge_targets, edge_values)
            edge_mean = sums / counts.clamp_min(1).unsqueeze(-1)
            centered = edge_values - edge_mean.index_select(0, edge_targets)
            squares = squares.index_add(0, edge_targets, centered.square())
        mean = sums / counts.clamp_min(1).unsqueeze(-1)
        std = torch.sqrt(squares / counts.clamp_min(1).unsqueeze(-1) + self.eps)
        present = (counts > 0).to(edge_values.dtype).unsqueeze(-1)
        return mean * present, std * present


class HistoryTokenEncoder(nn.Module):
    """Encode intrinsic and successive relation-conditioned interaction states."""

    def __init__(self, hidden_dim: int = 256, operation_profile_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.operation_profile_dim = int(operation_profile_dim)
        self.W_H = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_D = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.W_O = nn.Linear(self.operation_profile_dim, self.hidden_dim, bias=False)
        self.modality_text = nn.Parameter(torch.empty(self.hidden_dim))
        self.modality_visual = nn.Parameter(torch.empty(self.hidden_dim))
        self.stage_0 = nn.Parameter(torch.empty(self.hidden_dim))
        self.stage_1 = nn.Parameter(torch.empty(self.hidden_dim))
        self.stage_2 = nn.Parameter(torch.empty(self.hidden_dim))
        self.NO_OPERATION_TEXT = nn.Parameter(torch.empty(self.operation_profile_dim))
        self.NO_OPERATION_VISUAL = nn.Parameter(torch.empty(self.operation_profile_dim))
        self.NULL_OPERATION_TEXT = nn.Parameter(torch.empty(self.operation_profile_dim))
        self.NULL_OPERATION_VISUAL = nn.Parameter(torch.empty(self.operation_profile_dim))
        self.norm = nn.LayerNorm(self.hidden_dim)
        for parameter in (
            self.modality_text, self.modality_visual, self.stage_0, self.stage_1, self.stage_2,
            self.NO_OPERATION_TEXT, self.NO_OPERATION_VISUAL,
            self.NULL_OPERATION_TEXT, self.NULL_OPERATION_VISUAL,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.02)

    def null_operation(self, modality: str) -> torch.Tensor:
        if modality == "text":
            return self.NULL_OPERATION_TEXT
        if modality == "visual":
            return self.NULL_OPERATION_VISUAL
        raise ValueError("modality must be text or visual")

    def no_operation(self, modality: str) -> torch.Tensor:
        if modality == "text":
            return self.NO_OPERATION_TEXT
        if modality == "visual":
            return self.NO_OPERATION_VISUAL
        raise ValueError("modality must be text or visual")

    def modality_embedding(self, modality: str) -> torch.Tensor:
        if modality == "text":
            return self.modality_text
        if modality == "visual":
            return self.modality_visual
        raise ValueError("modality must be text or visual")

    def forward_modality(
        self,
        modality: str,
        states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        operation_profiles: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        h0, h1, h2 = states
        p0, p1 = operation_profiles
        n = h0.size(0)
        modality_embedding = self.modality_embedding(modality)
        zero_transition = torch.zeros_like(h0)
        no_operation = self.no_operation(modality).expand(n, -1)
        t0 = self.norm(
            self.W_H(h0) + self.W_D(zero_transition) + self.W_O(no_operation)
            + modality_embedding + self.stage_0
        )
        d1, d2 = h1 - h0, h2 - h1
        t1 = self.norm(
            self.W_H(h1) + self.W_D(d1) + self.W_O(p0)
            + modality_embedding + self.stage_1
        )
        t2 = self.norm(
            self.W_H(h2) + self.W_D(d2) + self.W_O(p1)
            + modality_embedding + self.stage_2
        )
        return torch.stack((t0, t1, t2), dim=1)

    def forward(
        self,
        text_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        visual_states: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        text_profiles: tuple[torch.Tensor, torch.Tensor],
        visual_profiles: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        text = self.forward_modality("text", text_states, text_profiles)
        visual = self.forward_modality("visual", visual_states, visual_profiles)
        # Stable token order: T0,T1,T2,V0,V1,V2.
        return torch.cat((text, visual), dim=1)


class HistoryReadout(nn.Module):
    """Single-query, single-layer cross-attention with no query residual."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, history: torch.Tensor, *, need_weights: bool, drop_token: int | None = None):
        if query.ndim != 2 or history.ndim != 3 or history.size(1) != 6:
            raise ValueError("expected query [N,H] and ordered history [N,6,H]")
        mask = None
        if drop_token is not None:
            if not 0 <= int(drop_token) < 6:
                raise ValueError("drop_token must be in [0,5]")
            mask = torch.zeros((history.size(0), history.size(1)), dtype=torch.bool, device=history.device)
            mask[:, int(drop_token)] = True
        attended, weights = self.attention(
            query=query.unsqueeze(1), key=history, value=history,
            key_padding_mask=mask, need_weights=need_weights, average_attn_weights=False,
        )
        representation = self.norm(attended[:, 0])
        if need_weights:
            return representation, weights[:, :, 0, :]
        return representation, None


class HistoryReadoutQuery(nn.Module):
    """Build a node query from intrinsic semantics and relation environment."""

    def __init__(self, hidden_dim: int = 256, relation_environment_dim: int = 128) -> None:
        super().__init__()
        self.global_readout_query = nn.Parameter(torch.empty(hidden_dim))
        self.W_PT = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_PV = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_RT = nn.Linear(relation_environment_dim, hidden_dim, bias=False)
        self.W_RV = nn.Linear(relation_environment_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.NULL_REL_ENV_TEXT = nn.Parameter(torch.empty(relation_environment_dim))
        self.NULL_REL_ENV_VISUAL = nn.Parameter(torch.empty(relation_environment_dim))
        nn.init.normal_(self.global_readout_query, mean=0.0, std=0.02)
        nn.init.normal_(self.NULL_REL_ENV_TEXT, mean=0.0, std=0.02)
        nn.init.normal_(self.NULL_REL_ENV_VISUAL, mean=0.0, std=0.02)

    def forward(
        self,
        intrinsic_text: torch.Tensor,
        intrinsic_visual: torch.Tensor,
        relation_env_text: torch.Tensor,
        relation_env_visual: torch.Tensor,
        *,
        node_conditioning: bool = True,
        relation_environment: bool = True,
        global_only: bool = False,
    ) -> torch.Tensor:
        n = intrinsic_text.size(0)
        query = self.global_readout_query.unsqueeze(0).expand(n, -1)
        if not global_only:
            if node_conditioning:
                query = query + self.W_PT(intrinsic_text) + self.W_PV(intrinsic_visual)
            if relation_environment:
                query = query + self.W_RT(relation_env_text) + self.W_RV(relation_env_visual)
        return self.norm(query)
