"""Dialect-specific rank-r adapters with schema-conditioned soft gating."""
import torch
from torch import nn
from .config import DIALECTS

class DialectalAdaptation(nn.Module):
    def __init__(self, hidden_size, schema_dim, rank=8, alpha=16.0):
        super().__init__()
        self.down = nn.Parameter(torch.empty(len(DIALECTS), hidden_size, rank))
        self.up = nn.Parameter(torch.zeros(len(DIALECTS), rank, hidden_size))
        nn.init.normal_(self.down, std=0.02)
        self.scale = alpha / rank
        self.gate = nn.Sequential(nn.Linear(schema_dim, hidden_size), nn.Tanh(),
                                  nn.Linear(hidden_size, len(DIALECTS)))

    def forward(self, hidden, schema_summary, dialect_ids=None):
        """dialect_ids=None learns a soft mixture; otherwise injects an explicit dialect.

        Issue #11: an entry of -1 marks an UNKNOWN dialect for that row; such rows
        use the soft gate instead of being forced to a dialect (no label invented).
        Batches whose ids are all known behave exactly as before.
        """
        probs = torch.softmax(self.gate(schema_summary), dim=-1)
        if dialect_ids is not None:
            if dialect_ids.ndim != 1 or dialect_ids.shape[0] != hidden.shape[0]:
                raise ValueError("dialect_ids shape must be [batch]")
            dialect_ids = dialect_ids.long()
            if ((dialect_ids < -1) | (dialect_ids >= len(DIALECTS))).any():
                raise ValueError(f"dialect_ids must be -1 (unknown) or 0..{len(DIALECTS) - 1}")
            known = dialect_ids.ge(0)
            one_hot = torch.nn.functional.one_hot(dialect_ids.clamp_min(0), len(DIALECTS)).to(hidden.dtype)
            probs = torch.where(known[:, None], one_hot, probs)
        down = torch.einsum('btd,ndr->bntr', hidden, self.down)
        up = torch.einsum('bntr,nrd->bntd', down, self.up)
        return hidden + self.scale * torch.einsum('bn,bntd->btd', probs, up)
