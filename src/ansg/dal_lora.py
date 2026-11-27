"""Dialectal Adaptation Layers (DAL), following dissertation Eqs. 4.24-4.25 and §4.4.2.

    W_d = W_frozen + B_d A_d,   B_d ∈ R^{d×r}, A_d ∈ R^{r×d}, r = 8                 (4.24)
    g_d = σ( v_dᵀ [h ; s_ctx] ) · 𝟙[s_active compatible with d]                     (4.25)
    y   = W_frozen x + g_d · (α/r) · B_d A_d x

for d ∈ {MSA, EGY, LEV, GULF, MAG}. §4.4.1 counts 2·r·d = 12,288 adapter
parameters per layer and dialect, i.e. one d×d matrix per layer. With d = 768,
12 layers and 5 dialects this gives 737,280 adapter parameters; the unit test
checks that this module has exactly that many.

Choices the dissertation leaves open, declared here:

* The adapted matrix is the attention output projection ``c_proj`` (the only
  d×d projection in a GPT-2 block).
* ``h`` is the input of that projection (the head-mixed attention output),
  token by token.
* ``s_ctx`` is the activation-weighted mean of the active schema embeddings
  from CSAM: Σ w_s c_s / Σ w_s (zero when nothing is active).
* Each example states its target dialect (``dialect_ids``); the gate decides,
  per token, how strongly that dialect's adapter applies.
* LoRA scaling α/r with α = 16, and B_d initialised to zero, so the model
  starts exactly at the backbone.
* Compatibility (§4.4.2): MSA is always compatible. A colloquial adapter is
  blocked where a *formal* schema (wisdom transmission) is active, unless the
  token lies in a dialogue span (``dialogue_mask``), which follows the
  dissertation's "MSA for framing, dialect for embedded dialogue".
"""

from __future__ import annotations

import torch
from torch import nn

DIALECTS = ("msa", "egyptian", "levantine", "gulf", "maghrebi")


class DialectalAdaptation(nn.Module):
    def __init__(self, n_layer: int = 12, d_model: int = 768, rank: int = 8, alpha: float = 16.0,
                 formal_schema_index: list[int] | None = None):
        super().__init__()
        self.n_layer, self.d_model, self.rank = n_layer, d_model, rank
        self.scale = alpha / rank
        n_d = len(DIALECTS)
        self.A = nn.Parameter(torch.randn(n_layer, n_d, rank, d_model) / d_model ** 0.5)
        self.B = nn.Parameter(torch.zeros(n_layer, n_d, d_model, rank))
        # gate vector v_d over [h ; s_ctx], shared across layers (not counted in §4.4.1's adapter total)
        self.gate = nn.Parameter(torch.zeros(n_d, 2 * d_model))
        self.gate_bias = nn.Parameter(torch.full((n_d,), 2.0))   # start mostly open
        self.formal = list(formal_schema_index or [])

    def adapter_parameter_count(self) -> int:
        return self.A.numel() + self.B.numel()

    @staticmethod
    def schema_context(schema_embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """s_ctx = Σ w_s c_s / Σ w_s, [B, d]."""
        total = weights.sum(dim=1, keepdim=True)
        return (weights @ schema_embeddings) / total.clamp_min(1e-8) * (total > 0)

    def compatible(self, dialect_ids: torch.Tensor, weights: torch.Tensor, seq_len: int,
                   dialogue_mask: torch.Tensor | None) -> torch.Tensor:
        """§4.4.2 indicator, [B, T] in {0, 1}."""
        bsz = dialect_ids.shape[0]
        formal_active = (weights[:, self.formal] > 0).any(dim=1) if self.formal else torch.zeros(
            bsz, dtype=torch.bool, device=weights.device)
        in_dialogue = dialogue_mask.bool() if dialogue_mask is not None else torch.zeros(
            bsz, seq_len, dtype=torch.bool, device=weights.device)
        is_msa = (dialect_ids == DIALECTS.index("msa"))[:, None]
        ok = is_msa | in_dialogue | ~formal_active[:, None]
        return ok.float()

    def delta_fn(self, dialect_ids: torch.Tensor, schema_embeddings: torch.Tensor, weights: torch.Tensor,
                 dialogue_mask: torch.Tensor | None = None):
        """Adapter for ``GPT2WithAttentionBias(delta_fn=...)``."""
        s_ctx = self.schema_context(schema_embeddings, weights)                       # [B, d]
        v = self.gate[dialect_ids]                                                     # [B, 2d]
        c = self.gate_bias[dialect_ids]                                                # [B]

        def fn(layer: int, h: torch.Tensor) -> torch.Tensor:
            t = h.shape[1]
            logits = h @ v[:, : self.d_model, None] + (s_ctx * v[:, self.d_model:]).sum(-1)[:, None, None] \
                + c[:, None, None]
            g = torch.sigmoid(logits).squeeze(-1) * self.compatible(dialect_ids, weights, t, dialogue_mask)  # [B,T]
            A = self.A[layer, dialect_ids]                                             # [B, r, d]
            B = self.B[layer, dialect_ids]                                             # [B, d, r]
            low = torch.einsum("btd,brd->btr", h, A)
            return self.scale * g[..., None] * torch.einsum("btr,bdr->btd", low, B)

        return fn


# Schemas treated as formal wisdom transmission (§4.4.2). Editorial choice for the seed CKG.
FORMAL_SCHEMAS = ("virtue:wisdom", "schema:الحكمة")


def formal_schema_indices(schema_ids: list[str]) -> list[int]:
    """Indices of the formal schemas in the CSAM schema order (see ``load_schema_embeddings``)."""
    return [i for i, sid in enumerate(schema_ids) if sid in FORMAL_SCHEMAS]
