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
        """dialect_ids=None learns a soft mixture; otherwise injects an explicit dialect."""
        probs = torch.softmax(self.gate(schema_summary), dim=-1)
        if dialect_ids is not None:
            if dialect_ids.ndim != 1 or dialect_ids.shape[0] != hidden.shape[0]:
                raise ValueError("dialect_ids shape must be [batch]")
            probs = torch.nn.functional.one_hot(dialect_ids.long(), len(DIALECTS)).to(hidden.dtype)
        down = torch.einsum('btd,ndr->bntr', hidden, self.down)
        up = torch.einsum('bntr,nrd->bntd', down, self.up)
        return hidden + self.scale * torch.einsum('bn,bntd->btd', probs, up)
