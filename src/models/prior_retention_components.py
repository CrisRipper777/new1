from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorRetainingStateUpdate(nn.Module):
    """v3 StateUpdate with an optional handle to the exact injected residual.

    Parameter names and shapes match v3.StateUpdate. The update projection,
    single dropout call, residual addition, and LayerNorm are unchanged.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.update = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.capture_residual = False
        self._last_residual: torch.Tensor | None = None

    def forward(self, state: torch.Tensor, aggregate: torch.Tensor) -> torch.Tensor:
        residual = self.dropout(self.update(F.gelu(aggregate)))
        self._last_residual = residual if self.capture_residual else None
        return self.norm(state + residual)

    def take_residual(self) -> torch.Tensor:
        if self._last_residual is None:
            raise RuntimeError("StateUpdate residual capture was not enabled for this call")
        residual = self._last_residual
        self._last_residual = None
        return residual


class ContextAdapter(nn.Module):
    """One shallow modality-specific adapter with a zero-initialized output."""

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.input = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.output(self.dropout(self.activation(self.norm(self.input(context)))))


class FeatureContextGate(nn.Module):
    """Independent per-feature context allocation initialized to 0.5."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, prior: torch.Tensor, enrichment: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(torch.cat((prior, enrichment), dim=-1)))


__all__ = ["PriorRetainingStateUpdate", "ContextAdapter", "FeatureContextGate"]
