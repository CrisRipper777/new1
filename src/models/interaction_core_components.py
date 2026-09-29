from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the already tested physical-graph utilities from M0-S1 without modifying it.
from .interaction_components import (
    canonicalize_physical_edges,
    incoming_degree,
    incoming_mean,
    leave_one_out_context,
)


def pair_relation_features(
    target: torch.Tensor,
    source: torch.Tensor,
    text_cosine: torch.Tensor,
    visual_cosine: torch.Tensor,
) -> torch.Tensor:
    """Ordered endpoint evidence plus explicit cross-modal compatibility."""
    if target.shape != source.shape:
        raise ValueError("target and source features must have identical shapes")
    if text_cosine.shape != visual_cosine.shape or text_cosine.shape != target.shape[:-1]:
        raise ValueError("cosine scores must align with endpoint pairs")
    compatibility = (text_cosine - visual_cosine).abs()
    scalars = torch.stack((text_cosine, visual_cosine, compatibility), dim=-1)
    return torch.cat(
        (target, source, (target - source).abs(), target * source, scalars), dim=-1
    )


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


class PairRelationEncoder(nn.Module):
    """Encode [U_i,U_j,|U_i-U_j|,U_i*U_j,s_T,s_V,|s_T-s_V|]."""

    def __init__(self, relation_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        dim = int(relation_dim)
        self.input_dim = 4 * dim + 3
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        text_cosine: torch.Tensor,
        visual_cosine: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(
            pair_relation_features(target, source, text_cosine, visual_cosine)
        )


class CrossAttentionBlock(nn.Module):
    """One residual cross-attention block for one query per physical edge."""

    def __init__(self, relation_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        dim = int(relation_dim)
        self.attention = nn.MultiheadAttention(
            dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or key_value.ndim != 3:
            raise ValueError("expected query [E,D] and key_value [E,S,D]")
        if query.size(0) == 0:
            return query
        attended, _ = self.attention(
            query.unsqueeze(1), key_value, key_value, need_weights=False
        )
        return self.norm(query + self.dropout(attended[:, 0]))


class ExecutionCrossAttention(nn.Module):
    def __init__(self, relation_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.block = CrossAttentionBlock(relation_dim, num_heads, dropout)

    def forward(self, query: torch.Tensor, relation_memory: torch.Tensor) -> torch.Tensor:
        return self.block(query, relation_memory)


class ConditionalLowRankOperator(nn.Module):
    """Relation-conditioned low-rank deviation around a source base message."""

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        operator_rank: int,
        zero_init_up: bool = True,
    ) -> None:
        super().__init__()
        self.message = nn.Linear(int(hidden_dim), int(hidden_dim), bias=False)
        self.down = nn.Linear(int(hidden_dim), int(operator_rank), bias=False)
        self.modulation = nn.Linear(int(relation_dim), int(operator_rank), bias=True)
        self.up = nn.Linear(int(operator_rank), int(hidden_dim), bias=True)
        if zero_init_up:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(
        self,
        source_state: torch.Tensor,
        execution_code: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = self.message(source_state)
        modulation = torch.tanh(self.modulation(execution_code))
        delta = (
            self.up(modulation * self.down(base)) if enabled else torch.zeros_like(base)
        )
        return base, delta, modulation


class StateUpdate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, state: torch.Tensor, aggregate: torch.Tensor) -> torch.Tensor:
        return self.norm(state + self.dropout(self.update(F.gelu(aggregate))))


__all__ = [
    "ConditionalLowRankOperator", "CrossAttentionBlock", "ExecutionCrossAttention",
    "ModalityProjector", "PairRelationEncoder", "StateUpdate",
    "canonicalize_physical_edges", "incoming_degree", "incoming_mean",
    "leave_one_out_context", "pair_relation_features",
]
