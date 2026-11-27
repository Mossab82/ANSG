"""CSAM cultural attention bias, following dissertation Eqs. 4.8, 4.9, 4.12, 4.13 and 4.16-4.20.

    c_s            = W_proj · g_s                                            (4.8)
    w_s            = cos(e_context, c_s),  schema s active if w_s > τ        (4.20)
    Rel_i(s, j, k) = σ( v_{s,i}ᵀ [pos_j ; pos_k ; c_s] )                      (4.13)
    H_i[j, k]      = Σ_{s active} w_s · Rel^H_i(s, j, k)                       (4.12)
    R_i[j, k]      = Σ_{s active} w_s · Rel^R_i(s, j, k)                       (4.12, same form)
    B_i            = β H_i   for heritage heads,  γ R_i  for rhetorical heads   (4.9)
    L              = L_LM + λ_align · L_cultural + λ_schema · L_schema        (4.16)

``g_s`` is the graph embedding of schema s (OWL2Vec*-style, see
``ansg.knowledge.graph_embeddings``). The dissertation leaves some details
open; the choices made here are declared, not taken from the dissertation:

* ``e_context`` is the mean input-token embedding (``wte``) of the sequence,
  so that schema activation does not depend on the biased forward pass.
* ``R_i`` uses the same parameterisation as ``H_i`` with its own parameters.
* The dialectal heads (8-11) receive no bias: Eq. 4.9 defines only βH and γR.
* Each biased layer has its own relevance parameters; ``layers`` selects which.
* Eq. 4.19's predictor is ŷ_s = σ(u_sᵀ e_context + b_s).

Because ``v_{s,i}ᵀ[pos_j; pos_k; c_s] = a_{s,i}·pos_j + b_{s,i}·pos_k + d_{s,i}·c_s``,
the relevance logits are computed as an outer sum, O(S·T) per head instead of
O(S·T²·3d).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as F

HEAD_GROUPS = {"heritage": (0, 4), "rhetorical": (4, 8), "dialectal": (8, 12)}


@dataclass
class CSAMSettings:
    n_head: int = 12
    d_model: int = 768
    graph_dim: int = 300
    beta: float = 0.347   # §4.3.4
    gamma: float = 0.283  # §4.3.4
    tau: float = 0.5      # Eq. 4.20
    layers: tuple[int, ...] = field(default_factory=lambda: tuple(range(12)))

    def __post_init__(self):
        if self.n_head != 12:
            raise ValueError("the dissertation's CSAM uses 12 heads split 4/4/4")


class CulturalSchemaBias(nn.Module):
    def __init__(self, settings: CSAMSettings, graph_embeddings: torch.Tensor):
        super().__init__()
        s = settings
        self.settings = s
        n_schema = graph_embeddings.shape[0]
        if graph_embeddings.shape[1] != s.graph_dim:
            raise ValueError("graph embedding width must equal settings.graph_dim")
        self.register_buffer("graph_embeddings", graph_embeddings.float())
        self.w_proj = nn.Linear(s.graph_dim, s.d_model, bias=False)                     # Eq. 4.8
        self.layer_index = {layer: n for n, layer in enumerate(s.layers)}
        n_layers = len(s.layers)
        # v_{s,i} split into (a, b, d) for the 4 heritage and 4 rhetorical heads, per biased layer
        shape = (n_layers, 2, 4, n_schema, s.d_model)
        self.v_pos_j = nn.Parameter(torch.randn(shape) * 0.02)
        self.v_pos_k = nn.Parameter(torch.randn(shape) * 0.02)
        self.v_schema = nn.Parameter(torch.randn(shape) * 0.02)
        self.v_const = nn.Parameter(torch.full((n_layers, 2, 4, n_schema), -2.0))  # start near zero bias
        self.schema_head = nn.Linear(s.d_model, n_schema)                              # Eq. 4.19

    def schema_embeddings(self) -> torch.Tensor:
        return self.w_proj(self.graph_embeddings)                                      # [S, d]

    def activation(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. 4.20. ``context`` [B, d] -> weights w [B, S] (zero where inactive) and the active mask."""
        sim = F.cosine_similarity(context[:, None, :], self.schema_embeddings()[None], dim=-1)
        active = sim > self.settings.tau
        return sim * active, active

    def relevance_logits(self, layer: int, positions: torch.Tensor, schemas: torch.Tensor | None = None) -> torch.Tensor:
        """Eq. 4.13 pre-sigmoid, [2 groups, 4 heads, S, T, T] (or only the given schema indices)."""
        n = self.layer_index[layer]
        idx = schemas if schemas is not None else torch.arange(self.graph_embeddings.shape[0], device=positions.device)
        c = self.schema_embeddings()[idx]                                              # [S, d]
        a = torch.einsum("gisd,td->gist", self.v_pos_j[n][:, :, idx], positions)       # pos_j term
        b = torch.einsum("gisd,td->gist", self.v_pos_k[n][:, :, idx], positions)       # pos_k term
        d = torch.einsum("gisd,sd->gis", self.v_schema[n][:, :, idx], c) + self.v_const[n][:, :, idx]
        return a[..., :, None] + b[..., None, :] + d[..., None, None]

    def group_matrices(self, layer: int, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Eq. 4.12: [B, 2, 4, T, T] with H (group 0) and R (group 1).

        Only schemas active in some batch element are evaluated; inactive ones
        contribute w_s = 0 to the sum anyway, and skipping them bounds memory.
        """
        used = (weights != 0).any(dim=0).nonzero().flatten()
        t = positions.shape[0]
        if used.numel() == 0:
            return weights.new_zeros(weights.shape[0], 2, 4, t, t)
        rel = torch.sigmoid(self.relevance_logits(layer, positions, used))              # [2,4,S',T,T]
        return torch.einsum("bs,gistu->bgitu", weights[:, used], rel)

    def bias(self, layer: int, positions: torch.Tensor, weights: torch.Tensor) -> torch.Tensor | None:
        """Eq. 4.9: per-head bias [B, 12, T, T]; None for layers without CSAM."""
        if layer not in self.layer_index:
            return None
        hr = self.group_matrices(layer, positions, weights)
        bsz, t = hr.shape[0], hr.shape[-1]
        out = hr.new_zeros(bsz, self.settings.n_head, t, t)
        out[:, 0:4] = self.settings.beta * hr[:, 0]
        out[:, 4:8] = self.settings.gamma * hr[:, 1]
        return out

    def bias_fn(self, positions_table: torch.Tensor, weights: torch.Tensor):
        """Adapter for ``GPT2WithAttentionBias``: (layer, T) -> bias."""
        return lambda layer, t: self.bias(layer, positions_table[:t], weights)


def cultural_alignment_loss(h_matrix: torch.Tensor, pos_pairs: torch.Tensor, neg_pairs: torch.Tensor) -> torch.Tensor:
    """Eq. 4.18 on one heritage matrix [B, T, T]; pairs are LongTensors [N, 3] of (batch, j, k)."""
    pos = h_matrix[pos_pairs[:, 0], pos_pairs[:, 1], pos_pairs[:, 2]]
    neg = h_matrix[neg_pairs[:, 0], neg_pairs[:, 1], neg_pairs[:, 2]]
    return -(F.logsigmoid(pos).sum() + F.logsigmoid(-neg).sum())


def schema_prediction_loss(module: CulturalSchemaBias, context: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. 4.19: multi-label BCE; ``labels`` [B, S] in {0, 1}."""
    return F.binary_cross_entropy_with_logits(module.schema_head(context), labels.float(), reduction="sum")


def csam_objective(lm_loss, align_loss, schema_loss, lambda_align: float, lambda_schema: float):
    """Eq. 4.16. The dissertation does not state λ values; they are training settings here."""
    return lm_loss + lambda_align * align_loss + lambda_schema * schema_loss
