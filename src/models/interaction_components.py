from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected


def canonicalize_physical_edges(edge_index: torch.Tensor | None, num_nodes: int) -> torch.Tensor:
    """Return unique, loop-free directed arcs for an undirected physical graph.

    The returned convention is always ``src=edge_index[0]`` and
    ``dst=edge_index[1]``; an arc ``j -> i`` therefore has ``src=j, dst=i``.
    """
    if edge_index is None:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    edge_index = edge_index.to(dtype=torch.long).contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=int(num_nodes))
    edge_index = coalesce(edge_index, num_nodes=int(num_nodes))
    return edge_index.contiguous()


def incoming_degree(dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    return torch.bincount(dst, minlength=int(num_nodes))


def incoming_mean(edge_values: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean-aggregate edge values by destination; empty destinations stay zero."""
    out = edge_values.new_zeros((int(num_nodes),) + tuple(edge_values.shape[1:]))
    if dst.numel() == 0:
        return out
    out.index_add_(0, dst, edge_values)
    degree = incoming_degree(dst, num_nodes).to(dtype=edge_values.dtype).clamp_min_(1)
    shape = (int(num_nodes),) + (1,) * (edge_values.ndim - 1)
    return out / degree.view(shape)


def build_incoming_mean_operator(
    src: torch.Tensor, dst: torch.Tensor, num_nodes: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse matrix A with A[i,j]=1/degree(i) for each physical arc j -> i."""
    degree = incoming_degree(dst, num_nodes)
    if src.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=src.device)
        values = torch.empty((0,), dtype=dtype, device=src.device)
    else:
        indices = torch.stack((dst, src), dim=0)
        values = degree[dst].to(dtype=dtype).reciprocal()
    adjacency = torch.sparse_coo_tensor(
        indices, values, (int(num_nodes), int(num_nodes)), dtype=dtype, device=src.device
    ).coalesce()
    return adjacency, degree


def ordered_pair_features(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Ordered endpoint features for j -> i: [U_i, U_j, |U_i-U_j|, U_i*U_j]."""
    if target.shape != source.shape:
        raise ValueError("target and source relation features must have the same shape")
    return torch.cat((target, source, (target - source).abs(), target * source), dim=-1)


def leave_one_out_context(
    relation_features: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    no_context: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact recipient-side mean context excluding the current source neighbor.

    Returns edge-aligned context, a degree>1 mask, and node incoming degree.
    Degree-one edges receive the supplied learnable no-context vector.
    """
    degree = incoming_degree(dst, num_nodes)
    defined = degree[dst] > 1
    if src.numel() == 0:
        return relation_features.new_empty((0, relation_features.size(-1))), defined, degree
    sums = relation_features.new_zeros((int(num_nodes), relation_features.size(-1)))
    sums.index_add_(0, dst, relation_features[src])
    denom = (degree[dst] - 1).to(dtype=relation_features.dtype).clamp_min_(1)
    context = (sums[dst] - relation_features[src]) / denom.unsqueeze(-1)
    context = torch.where(defined.unsqueeze(-1), context, no_context.reshape(1, -1))
    return context, defined, degree


class PairEvidenceEncoder(nn.Module):
    def __init__(self, relation_dim: int, dropout: float) -> None:
        super().__init__()
        dim = int(relation_dim)
        self.network = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
        )

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        return self.network(ordered_pair_features(target, source))


class CompatibilityEvidenceEncoder(nn.Module):
    def __init__(self, relation_dim: int) -> None:
        super().__init__()
        dim = int(relation_dim)
        self.network = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self, text_score: torch.Tensor, visual_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference = (text_score - visual_score).abs()
        features = torch.stack((text_score, visual_score, difference), dim=-1)
        return self.network(features), difference


class TypedRelationMixer(nn.Module):
    """One small shared typed-token Transformer-style relation mixer."""

    def __init__(
        self,
        relation_dim: int,
        num_heads: int = 2,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dim = int(relation_dim)
        if dim % int(num_heads) != 0:
            raise ValueError("relation_dim must be divisible by relation_mixer.num_heads")
        self.type_embedding = nn.Parameter(torch.empty(5, dim))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
        self.attention = nn.MultiheadAttention(
            dim, int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * int(ffn_ratio)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * int(ffn_ratio), dim),
        )
        self.ffn_dropout = nn.Dropout(float(dropout))
        self.norm2 = nn.LayerNorm(dim)

    def forward(
        self, tokens: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if tokens.ndim != 3 or tokens.size(1) != 5:
            raise ValueError(f"typed relation tokens must have shape [E, 5, D], got {tuple(tokens.shape)}")
        if tokens.size(0) == 0:
            empty_attention = tokens.new_empty((0, 5, 5)) if need_weights else None
            return tokens, empty_attention
        typed = tokens + self.type_embedding.unsqueeze(0)
        attended, weights = self.attention(
            typed,
            typed,
            typed,
            need_weights=bool(need_weights),
            average_attn_weights=False,
        )
        hidden = self.norm1(typed + self.attention_dropout(attended))
        output = self.norm2(hidden + self.ffn_dropout(self.ffn(hidden)))
        if weights is not None:
            weights = weights.mean(dim=1)
        return output, weights
