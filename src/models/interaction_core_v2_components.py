from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interaction_components import (
    canonicalize_physical_edges,
    incoming_degree,
    leave_one_out_context,
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


def pair_relation_features(
    target: torch.Tensor,
    source: torch.Tensor,
    text_cosine: torch.Tensor,
    visual_cosine: torch.Tensor,
) -> torch.Tensor:
    if target.shape != source.shape:
        raise ValueError("target and source features must have identical shapes")
    if text_cosine.shape != visual_cosine.shape or text_cosine.shape != target.shape[:-1]:
        raise ValueError("compatibility scores must align with endpoint pairs")
    difference = (text_cosine - visual_cosine).abs()
    return torch.cat(
        (target, source, (target - source).abs(), target * source,
         torch.stack((text_cosine, visual_cosine, difference), dim=-1)),
        dim=-1,
    )


class PairRelationEncoder(nn.Module):
    def __init__(self, relation_dim: int, dropout: float) -> None:
        super().__init__()
        dim = int(relation_dim)
        self.network = nn.Sequential(
            nn.Linear(4 * dim + 3, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
        )

    def forward(self, target, source, text_cosine, visual_cosine):
        return self.network(pair_relation_features(target, source, text_cosine, visual_cosine))


class RelationCrossAttention(nn.Module):
    """Stage-I LOO contextual relation fusion, retaining v1's residual block."""

    def __init__(self, relation_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            int(relation_dim), int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(relation_dim))

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, need_weights=False):
        if query.size(0) == 0:
            empty = query.new_empty((0, key_value.size(1)))
            return query, empty if need_weights else None
        attended, weights = self.attention(
            query.unsqueeze(1), key_value, key_value,
            need_weights=bool(need_weights), average_attn_weights=False,
        )
        output = self.norm(query + self.dropout(attended[:, 0]))
        if weights is not None:
            weights = weights.mean(dim=1).squeeze(1)
        return output, weights


class RelationGroundedRetriever(nn.Module):
    """Read relation memory into a code with no query residual or affine bias.

    With an all-zero memory, the bias-free attention output and fixed LayerNorm
    are exactly zero, independent of the recipient query.
    """

    def __init__(self, relation_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            int(relation_dim), int(num_heads), dropout=float(dropout),
            batch_first=True, bias=False, add_bias_kv=False, add_zero_attn=False,
        )
        self.output_norm = nn.LayerNorm(int(relation_dim), elementwise_affine=False)

    def forward(
        self,
        query: torch.Tensor,
        relation_memory: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if query.ndim != 2 or relation_memory.ndim != 3:
            raise ValueError("expected query [E,D] and relation_memory [E,S,D]")
        if query.size(0) != relation_memory.size(0):
            raise ValueError("query and relation_memory edge counts differ")
        if query.size(0) == 0:
            weights = query.new_empty((0, relation_memory.size(1))) if need_weights else None
            return query, weights
        attended, weights = self.attention(
            query.unsqueeze(1), relation_memory, relation_memory,
            need_weights=bool(need_weights), average_attn_weights=False,
        )
        code = self.output_norm(attended[:, 0])
        if weights is not None:
            weights = weights.mean(dim=1).squeeze(1)
        return code, weights


class RelationConditionedOperator(nn.Module):
    """Transform source content using a relation-grounded low-rank operation."""

    def __init__(self, hidden_dim: int, relation_dim: int, rank: int) -> None:
        super().__init__()
        hidden_dim, rank = int(hidden_dim), int(rank)
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.modulation = nn.Linear(int(relation_dim), rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source_state: torch.Tensor, execution_code: torch.Tensor, enabled=True):
        base = self.message(source_state)
        modulation = torch.tanh(self.modulation(execution_code))
        delta = self.up(modulation * self.down(base)) if enabled else torch.zeros_like(base)
        return base, delta, modulation, base + delta


class StateUpdate(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, state: torch.Tensor, aggregate: torch.Tensor) -> torch.Tensor:
        return self.norm(state + self.dropout(self.update(F.gelu(aggregate))))


__all__ = [
    "ModalityProjector", "PairRelationEncoder", "RelationCrossAttention",
    "RelationGroundedRetriever", "RelationConditionedOperator", "StateUpdate",
    "canonicalize_physical_edges", "incoming_degree", "leave_one_out_context",
]
