"""Root–pattern cultural weight — dissertation Eq. (4.2), Ch. 4 §4.2.1.

    w_c(r, p) = α · f_classical(r, p) / max_{r',p'} f_classical(r', p')
              + (1 − α) · w_expert(r, p),            α = 0.6

* ``f_classical`` — frequency of the root–pattern pair in ANHC's classical
  (pre-1800) stratum. Requires a morphological analyser over the corpus
  (Ch. 4 names Stanford Parser + AraMorph 2.0; Ch. 5 names CAMeL Tools —
  SPECIFICATION_CONFLICTS C-MORPH-1). Supplied by the caller as counts.
* ``w_expert`` — expert salience on a 5-point Likert scale "normalized to
  [0, 1]" (12 experts, κ = 0.81). The mapping from 1–5 to [0, 1] is not stated;
  :func:`likert_to_unit` uses the min–max rule the thesis applies elsewhere
  (Eq. A.1 without ×100). Pairs without an expert rating are an error, not 0.

Worked examples in §4.2.1: kuttāb 0.6·0.73 + 0.4·0.95 = 0.818 (Table 4.4 prints
0.9; the text prints "0.82 ≈ 0.9"); kitāb 0.6·0.81 + 0.4·0.55 = 0.706 (0.7).
"""

from __future__ import annotations

from typing import Mapping

RootPattern = tuple[str, str]
THESIS_ALPHA = 0.6


def likert_to_unit(score: float, lo: float = 1.0, hi: float = 5.0) -> float:
    if not lo <= score <= hi:
        raise ValueError(f"Likert score {score} outside [{lo}, {hi}]")
    return (score - lo) / (hi - lo)


def cultural_weights(
    classical_counts: Mapping[RootPattern, int],
    expert_unit_scores: Mapping[RootPattern, float],
    alpha: float = THESIS_ALPHA,
) -> dict[RootPattern, float]:
    """Compute w_c for every pair that has both a corpus count and an expert score."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if not classical_counts:
        raise ValueError("classical_counts is empty")
    fmax = max(classical_counts.values())
    if fmax <= 0:
        raise ValueError("maximum classical frequency must be positive")
    missing = [k for k in classical_counts if k not in expert_unit_scores]
    if missing:
        raise KeyError(f"no expert score for {len(missing)} pair(s), e.g. {missing[:3]}")
    out = {}
    for key, f in classical_counts.items():
        w_exp = expert_unit_scores[key]
        if not 0.0 <= w_exp <= 1.0:
            raise ValueError(f"expert score for {key} must be in [0, 1]")
        out[key] = alpha * f / fmax + (1 - alpha) * w_exp
    return out
