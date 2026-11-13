"""Tests that re-derive numbers printed in the dissertation from its own formulas.

Where the printed number does NOT follow from the printed inputs, the test asserts
the recomputed value and names the discrepancy (see docs/SPECIFICATION_CONFLICTS.md).
"""
import pytest

from ansg.evaluation.ahp import ahp_weights
from ansg.evaluation.statistics import (
    bonferroni_alpha, bootstrap_ci, cohens_d, cohens_kappa, fleiss_kappa, gap_closure,
    normalize_scale,
)
from ansg.knowledge.cultural_weight import cultural_weights, likert_to_unit

# App. A.11, Table A.10 (Cultural, Linguistic, Coherence, Diversity)
A10 = [[1, 1.17, 1.40, 3.50], [0.86, 1, 1.20, 3.00], [0.71, 0.83, 1, 2.50], [0.29, 0.33, 0.40, 1]]
# App. A.23, Table A.20 (Rhetorical, Virtue, Structure, Register)
A20 = [[1, 1.2, 1.8, 3.5], [0.83, 1, 1.5, 3.0], [0.56, 0.67, 1, 2.5], [0.29, 0.33, 0.40, 1]]


def test_ahp_table_a10_reproduces_stated_weights():
    r = ahp_weights(A10)
    assert [round(w, 2) for w in r.weights] == [0.35, 0.30, 0.25, 0.10]
    # Stated CR = 0.023; the printed matrix gives CR ~ 0.0001 (C-REWARD-4).
    assert r.consistency_ratio < 0.01


def test_ahp_table_a20_does_not_reproduce_stated_weights():
    r = ahp_weights(A20)
    # Stated: 0.35 / 0.30 / 0.25 / 0.10 with CR = 0.078 (C-REWARD-4).
    assert [round(w, 2) for w in r.weights] == [0.37, 0.31, 0.22, 0.10]
    assert r.consistency_ratio < 0.01


def test_ahp_rejects_bad_input():
    with pytest.raises(ValueError):
        ahp_weights([[1, 2], [0.5]])
    with pytest.raises(ValueError):
        ahp_weights([[1, -1], [1, 1]])


def test_gap_closure_eq_6_1():
    assert gap_closure(0.92, 0.68, 0.94) == pytest.approx(0.923, abs=1e-3)


def test_score_normalization_eq_a1_worked_example():
    assert normalize_scale(3.8, 1, 5) == pytest.approx(70.0)


def test_bonferroni_ch6_value():
    assert round(bonferroni_alpha(0.05, 9), 4) == 0.0056


def test_cultural_weight_eq_4_2_worked_examples():
    # Construct counts so that normalised frequencies equal the thesis' f values.
    counts = {("k-t-b", "fu''al"): 73, ("k-t-b", "fi'al"): 81, ("max", "x"): 100}
    expert = {("k-t-b", "fu''al"): 0.95, ("k-t-b", "fi'al"): 0.55, ("max", "x"): 0.0}
    w = cultural_weights(counts, expert)
    assert w[("k-t-b", "fu''al")] == pytest.approx(0.818)  # thesis Table 4.4 prints 0.9
    assert w[("k-t-b", "fi'al")] == pytest.approx(0.706)   # thesis prints 0.71 ~ 0.7
    assert likert_to_unit(5) == 1.0 and likert_to_unit(1) == 0.0


def test_cultural_weight_requires_expert_scores():
    with pytest.raises(KeyError):
        cultural_weights({("r", "p"): 3}, {})


def test_kappas_textbook_values():
    assert cohens_kappa(list("aabb"), list("aabb")) == 1.0
    assert cohens_kappa(list("abab"), list("aabb")) == pytest.approx(0.0)
    # Fleiss (1971) style toy table: perfect agreement
    assert fleiss_kappa([[3, 0], [0, 3], [3, 0]]) == 1.0


def test_cohens_d_variant_must_be_explicit():
    a, b = [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.5]
    assert cohens_d(a, b, "pooled") > 0
    assert cohens_d(a, b, "paired_dz") > 0
    with pytest.raises(ValueError):
        cohens_d(a, b, "hedges")  # type: ignore[arg-type]


def test_bootstrap_deterministic():
    data = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    a = bootstrap_ci(data, mean, 1000, seed=0)
    assert a == bootstrap_ci(data, mean, 1000, seed=0)
    assert a[0] <= 0.7 <= a[1]
