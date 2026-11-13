"""Statistical procedures that the dissertation specifies completely.

Each function cites the section that defines it. Where the thesis names a
procedure but leaves a choice open (e.g. which Cohen's d variant), the choice
is an explicit argument with no silent default that pretends to be the thesis'.

Pure Python; deterministic given ``seed``.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Literal, Sequence


def gap_closure(system: float, baseline: float, human: float) -> float:
    """Eq. (6.1): (ANSG − Baseline) / (Human − Baseline)."""
    if human == baseline:
        raise ZeroDivisionError("human ceiling equals baseline")
    return (system - baseline) / (human - baseline)


def normalize_scale(raw: float, scale_min: float, scale_max: float) -> float:
    """Eq. (A.1), App. A.10.2: (Raw − min)/(max − min) × 100."""
    if scale_max <= scale_min:
        raise ValueError("scale_max must exceed scale_min")
    return (raw - scale_min) / (scale_max - scale_min) * 100.0


def bonferroni_alpha(alpha: float, n_comparisons: int) -> float:
    """Bonferroni-adjusted per-test α (Ch. 6 §6.2.1 uses 9 comparisons → 0.0056;
    App. A.12 Table A.11 states 6 comparisons — see SPECIFICATION_CONFLICTS C-EVAL-3)."""
    if n_comparisons < 1:
        raise ValueError("n_comparisons must be >= 1")
    return alpha / n_comparisons


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def _var(x: Sequence[float]) -> float:
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - 1)


def cohens_d(a: Sequence[float], b: Sequence[float],
             variant: Literal["pooled", "paired_dz"]) -> float:
    """Cohen's d. The thesis reports d with paired t-tests (Ch. 6 §6.1.2) but never
    states the variant; the caller must choose."""
    if variant == "pooled":
        na, nb = len(a), len(b)
        sp = math.sqrt(((na - 1) * _var(a) + (nb - 1) * _var(b)) / (na + nb - 2))
        return (_mean(a) - _mean(b)) / sp
    if variant == "paired_dz":
        if len(a) != len(b):
            raise ValueError("paired d requires equal-length samples")
        diff = [x - y for x, y in zip(a, b)]
        return _mean(diff) / math.sqrt(_var(diff))
    raise ValueError(f"unknown variant {variant!r}")


def cohens_kappa(r1: Sequence[object], r2: Sequence[object]) -> float:
    """Unweighted Cohen's κ for two raters (Ch. 3 §3.5.4)."""
    if len(r1) != len(r2) or not r1:
        raise ValueError("ratings must be non-empty and of equal length")
    n = len(r1)
    po = sum(x == y for x, y in zip(r1, r2)) / n
    c1, c2 = Counter(r1), Counter(r2)
    pe = sum(c1[k] * c2[k] for k in set(c1) | set(c2)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(table: Sequence[Sequence[int]]) -> float:
    """Fleiss' κ; ``table[i][j]`` = raters assigning item i to category j (Ch. 3 §3.5.4,
    3 raters per segment)."""
    n_items = len(table)
    m = sum(table[0])
    if n_items == 0 or m < 2 or any(sum(r) != m for r in table):
        raise ValueError("every item needs the same number (>=2) of ratings")
    k = len(table[0])
    p_j = [sum(r[j] for r in table) / (n_items * m) for j in range(k)]
    p_i = [(sum(c * c for c in r) - m) / (m * (m - 1)) for r in table]
    p_bar = _mean(p_i)
    pe = sum(p * p for p in p_j)
    return 1.0 if pe == 1 else (p_bar - pe) / (1 - pe)


def bootstrap_ci(data: Sequence[object], statistic: Callable[[Sequence[object]], float],
                 n_resamples: int, seed: int, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI (Ch. 3 §3.6.2: 1,000 resamples, 2.5/97.5 percentiles).

    App. A.12 specifies BCa with 10,000 resamples instead (C-EVAL-4); BCa is not
    implemented here."""
    rng = random.Random(seed)
    n = len(data)
    stats = sorted(statistic([data[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo_i = int(math.floor((1 - level) / 2 * n_resamples))
    hi_i = int(math.ceil((1 + level) / 2 * n_resamples)) - 1
    return stats[lo_i], stats[min(hi_i, n_resamples - 1)]
