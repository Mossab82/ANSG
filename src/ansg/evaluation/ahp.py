"""Analytic Hierarchy Process (Saaty 1980) — principal-eigenvector weights and CR.

The dissertation derives every expert weight set with AHP (Ch. 4 §4.5.2,
App. A.11, App. A.23). This module recomputes weights from a pairwise matrix so
that reported weights can be *checked* against the printed matrices. It is a
standard method, not a thesis-specific invention.

Random-index values are Saaty's (n = 1..10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SAATY_RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32,
                      8: 1.41, 9: 1.45, 10: 1.49}


@dataclass(frozen=True)
class AHPResult:
    weights: tuple[float, ...]
    lambda_max: float
    consistency_index: float
    consistency_ratio: float
    reciprocity_max_error: float  # max |a_ij * a_ji - 1| — printed matrices are rounded


def ahp_weights(matrix: Sequence[Sequence[float]], iterations: int = 10_000,
                tol: float = 1e-14) -> AHPResult:
    """Principal right eigenvector (power iteration), λ_max, CI and CR."""
    n = len(matrix)
    if n == 0 or any(len(r) != n for r in matrix):
        raise ValueError("pairwise matrix must be square and non-empty")
    if any(v <= 0 for r in matrix for v in r):
        raise ValueError("pairwise entries must be positive")
    w = [1.0 / n] * n
    for _ in range(iterations):
        v = [sum(matrix[i][j] * w[j] for j in range(n)) for i in range(n)]
        s = sum(v)
        nw = [x / s for x in v]
        if max(abs(a - b) for a, b in zip(nw, w)) < tol:
            w = nw
            break
        w = nw
    aw = [sum(matrix[i][j] * w[j] for j in range(n)) for i in range(n)]
    lam = sum(aw[i] / w[i] for i in range(n)) / n
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = SAATY_RANDOM_INDEX.get(n)
    if ri is None:
        raise ValueError(f"no Saaty random index for n={n}")
    cr = ci / ri if ri else 0.0
    recip = max(abs(matrix[i][j] * matrix[j][i] - 1.0) for i in range(n) for j in range(n))
    return AHPResult(tuple(w), lam, ci, cr, recip)
