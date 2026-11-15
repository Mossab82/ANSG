"""Dialect-specific rank-r adapters with compatibility-aware schema gating.

The recoverable form of thesis Eq. 4.25 is:
    g_d = sigmoid(v_d^T [h; s_ctx]) * 1[d is compatible].

This module applies the weighted low-rank deltas to its hidden-state input.
Where the adapter is inserted into the pretrained transformer remains a
separate unresolved choice (C-DAL-1); this bridge currently calls it after
CSAM and does not claim the historical q/v or fused-QKV mapping.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import DIALECTS


class DialectalAdaptation(nn.Module):
    """Shared-base dialect adapter bank with per-token soft gates.

    dialect_ids contains a known DIA label id or -1 for unknown. A known id
    restricts the bank to that adapter; -1 leaves the compatibility mask in
    control. No label is inferred or replaced with MSA.
    """

    def __init__(self, hidden_size: int, schema_dim: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        if min(hidden_size, schema_dim, rank) <= 0:
            raise ValueError("hidden_size, schema_dim, and rank must be positive")
        self.hidden_size = hidden_size
        self.schema_dim = schema_dim
        self.num_dialects = len(DIALECTS)
        self.down = nn.Parameter(torch.empty(self.num_dialects, hidden_size, rank))
        self.up = nn.Parameter(torch.zeros(self.num_dialects, rank, hidden_size))
        nn.init.normal_(self.down, std=0.02)
        self.scale = alpha / rank
        self.gate = nn.Linear(hidden_size + schema_dim, self.num_dialects)

    def gate_values(
        self,
        hidden: torch.Tensor,
        schema_context: torch.Tensor,
        dialect_ids: torch.Tensor | None = None,
        compatibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [batch, time, dialect] sigmoid gates masked by compatibility."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden must have shape [batch, time, hidden_size]")
        batch, time, _ = hidden.shape
        if schema_context.ndim == 2:
            if schema_context.shape != (batch, self.schema_dim):
                raise ValueError("schema_context must have shape [batch, schema_dim]")
            context = schema_context[:, None, :].expand(-1, time, -1)
        elif schema_context.ndim == 3:
            if schema_context.shape != (batch, time, self.schema_dim):
                raise ValueError("schema_context must have shape [batch, time, schema_dim]")
            context = schema_context
        else:
            raise ValueError("schema_context must be rank 2 or rank 3")

        compatible = torch.ones(
            (batch, time, self.num_dialects), dtype=torch.bool, device=hidden.device
        )
        if compatibility_mask is not None:
            if compatibility_mask.dtype != torch.bool:
                raise ValueError("compatibility_mask must be a boolean tensor")
            if compatibility_mask.shape == (batch, self.num_dialects):
                compatible = compatibility_mask[:, None, :].expand(-1, time, -1)
            elif compatibility_mask.shape == (batch, time, self.num_dialects):
                compatible = compatibility_mask
            else:
                raise ValueError(
                    "compatibility_mask must have shape [batch, dialect] "
                    "or [batch, time, dialect]"
                )
            compatible = compatible.to(device=hidden.device)

        if dialect_ids is not None:
            if dialect_ids.ndim != 1 or dialect_ids.shape[0] != batch:
                raise ValueError("dialect_ids must have shape [batch]")
            dialect_ids = dialect_ids.to(device=hidden.device, dtype=torch.long)
            if ((dialect_ids < -1) | (dialect_ids >= self.num_dialects)).any():
                raise ValueError(
                    f"dialect_ids must be -1 (unknown) or 0..{self.num_dialects - 1}"
                )
            known = dialect_ids.ge(0)
            selected = torch.nn.functional.one_hot(
                dialect_ids.clamp_min(0), self.num_dialects
            ).bool()
            selected_allowed = compatible.gather(
                2, dialect_ids.clamp_min(0)[:, None, None].expand(-1, time, 1)
            ).squeeze(-1)
            if (known[:, None] & ~selected_allowed).any():
                raise ValueError("a known dialect is excluded by compatibility_mask")
            known_mask = selected[:, None, :].expand(-1, time, -1)
            compatible = compatible & torch.where(
                known[:, None, None], known_mask, torch.ones_like(known_mask)
            )

        scores = torch.sigmoid(self.gate(torch.cat((hidden, context), dim=-1)))
        return scores * compatible.to(dtype=scores.dtype)

    def forward(
        self,
        hidden: torch.Tensor,
        schema_context: torch.Tensor,
        dialect_ids: torch.Tensor | None = None,
        compatibility_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gates = self.gate_values(hidden, schema_context, dialect_ids, compatibility_mask)
        down = torch.einsum("btd,ndr->bntr", hidden, self.down)
        up = torch.einsum("bntr,nrd->bntd", down, self.up)
        delta = torch.einsum("btn,bntd->btd", gates, up)
        return hidden + self.scale * delta
