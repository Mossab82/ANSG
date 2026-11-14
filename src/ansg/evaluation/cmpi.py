"""CMPI as defined in the dissertation — the implementation reference for ANSG.

The dissertation defines the Cultural Motif Preservation Index as an AHP-weighted
composite of four dimensions (§5.1 p.131; repeated in §8.7.4 p.223; "composite
metric (0-1 scale)", App. A.1 Table A.2 p.268)::

    CMPI = 0.35 * rhetorical_devices + 0.25 * dialectal_appropriateness
         + 0.25 * narrative_structure + 0.15 * moral_framing

:func:`ansg_cmpi` implements exactly this equation for dimension scores in
[0, 1]. The dissertation does **not** specify how each dimension is scored from
a text, so nothing here scores text: the caller supplies the four dimension
scores and must state where they came from.

CULTURA ("Cultural Matching Performance Index"), CST-Onto ("Cultural Moral
Precision Index") and MM-GTA ("Cross-dialectal Morphological Preservation
Index") use the same acronym for different metrics, recorded in
``configs/evaluation/cmpi_variants.yaml``. The two printed paper formulas are
reproduced *as printed* (:func:`cultura_cmpi_as_printed`,
:func:`mm_gta_cmpi_as_printed`) only so that their stated properties can be
checked; they are never converted to, compared with or substituted for the ANSG
CMPI.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
VARIANTS_CONFIG = ROOT / "configs" / "evaluation" / "cmpi_variants.yaml"

# §5.1 p.131 and §8.7.4 p.223 (AHP-derived).
ANSG_CMPI_WEIGHTS: dict[str, float] = {
    "rhetorical_devices": 0.35,
    "dialectal_appropriateness": 0.25,
    "narrative_structure": 0.25,
    "moral_framing": 0.15,
}
ANSG_CMPI_REFERENCE = "Dissertation §5.1 p.131; §8.7.4 p.223; App. A.1 Table A.2 p.268"
DIMENSION_SCORERS_SPECIFIED = False   # the thesis names the dimensions but gives no scoring procedure


def ansg_cmpi(rhetorical_devices: float, dialectal_appropriateness: float,
              narrative_structure: float, moral_framing: float) -> float:
    """The dissertation's CMPI equation applied to four dimension scores in [0, 1]."""
    scores = {"rhetorical_devices": rhetorical_devices, "dialectal_appropriateness": dialectal_appropriateness,
              "narrative_structure": narrative_structure, "moral_framing": moral_framing}
    for name, value in scores.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"CMPI dimension {name!r} must be in [0, 1], got {value!r}")
    return sum(ANSG_CMPI_WEIGHTS[k] * float(v) for k, v in scores.items())


def ansg_cmpi_from_mapping(scores: Mapping[str, float], *, provenance: str) -> dict[str, Any]:
    """Compute ANSG CMPI from named dimension scores and record their provenance.

    ``provenance`` must describe where the dimension scores came from (for
    example a named human-annotation protocol); the dissertation specifies no
    automatic scorer, so the result is never a reproduction of its CMPI values."""
    if not provenance or not provenance.strip():
        raise ValueError("provenance of the dimension scores is required")
    missing = [k for k in ANSG_CMPI_WEIGHTS if k not in scores]
    extra = [k for k in scores if k not in ANSG_CMPI_WEIGHTS]
    if missing or extra:
        raise KeyError(f"CMPI needs exactly {list(ANSG_CMPI_WEIGHTS)}; missing {missing}, unexpected {extra}")
    return {"metric": "CMPI (ANSG dissertation definition)", "value": ansg_cmpi(**{k: scores[k] for k in ANSG_CMPI_WEIGHTS}),
            "weights": dict(ANSG_CMPI_WEIGHTS), "reference": ANSG_CMPI_REFERENCE,
            "dimension_scores_provenance": provenance, "dimension_scorers_specified_in_thesis": False,
            "reproduces_thesis_result": False}


def load_cmpi_variants(path: str | Path | None = None) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(Path(path or VARIANTS_CONFIG).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping) or data.get("schema") != "ansg.cmpi_variants/1":
        raise ValueError("not an ansg.cmpi_variants/1 file")
    return dict(data)


def cultura_cmpi_as_printed(scores_0_to_10, weights) -> float:
    """CULTURA paper (PRIMA 2025, pp. 314-315), exactly as printed:
    CMPI(S) = (1/K) * sum_i w_i * (s_i / 10), sum_i w_i = 1. No correction applied."""
    scores, weights = list(scores_0_to_10), list(weights)
    if len(scores) != len(weights) or not scores:
        raise ValueError("one weight per criterion is required")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("the printed formula requires weights summing to 1")
    if any(not 0 <= s <= 10 for s in scores):
        raise ValueError("judge scores are on a 0-10 scale")
    return sum(w * (s / 10.0) for w, s in zip(weights, scores)) / len(scores)


def cultura_printed_upper_bound(weights) -> float:
    """Maximum of the printed CULTURA formula (all judge scores = 10)."""
    weights = list(weights)
    return cultura_cmpi_as_printed([10] * len(weights), weights)


def mm_gta_cmpi_as_printed(p_morph: float, r_morph: float) -> float:
    """MM-GTA paper, as printed: CMPI = 2 * P_morph * R_morph / (P_morph + R_morph)."""
    if not (0 <= p_morph <= 1 and 0 <= r_morph <= 1):
        raise ValueError("precision and recall must be in [0, 1]")
    return 0.0 if p_morph + r_morph == 0 else 2 * p_morph * r_morph / (p_morph + r_morph)


SEPARATE_METRICS = ("cultura", "cst_onto", "mm_gta", "ai_cinema_cpm")


def dimension_scorer_report(data: Mapping[str, Any]) -> dict[str, Any]:
    """Per ANSG CMPI dimension: whether the dissertation specifies a text scorer, and what is missing."""
    dims = (data.get("ansg_thesis") or {}).get("dimension_scorers") or {}
    rows = {name: {"scorer_specified": bool((dims.get(name) or {}).get("scorer_specified")),
                   "thesis_describes": (dims.get(name) or {}).get("thesis_describes"),
                   "related_mechanisms": list((dims.get(name) or {}).get("related_mechanisms") or []),
                   "missing": (dims.get(name) or {}).get("missing")}
            for name in ANSG_CMPI_WEIGHTS}
    computable = all(r["scorer_specified"] for r in rows.values())
    return {"dimensions": rows, "cmpi_computable_from_text": computable,
            "cmpi_computable_from_supplied_dimension_scores": True,
            "blocking": [f"{k}: {v['missing']}" for k, v in rows.items() if not v["scorer_specified"]]}


def validate_cmpi_variants(data: Mapping[str, Any]) -> list[str]:
    """Consistency of the variants file with this module and its separation rules."""
    problems: list[str] = []
    ansg = data.get("ansg_thesis") or {}
    if dict(ansg.get("weights") or {}) != ANSG_CMPI_WEIGHTS:
        problems.append(f"ansg_thesis weights {ansg.get('weights')} differ from ANSG_CMPI_WEIGHTS")
    if not any(s.get("page") == 131 for s in ansg.get("sources") or []):
        problems.append("ansg_thesis must cite §5.1 p.131")
    if not ansg.get("terminology_conflict"):
        problems.append("ansg_thesis: the CMPI naming conflict must be recorded")
    dims = ansg.get("dimension_scorers") or {}
    if set(dims) != set(ANSG_CMPI_WEIGHTS):
        problems.append(f"ansg_thesis.dimension_scorers must cover exactly {list(ANSG_CMPI_WEIGHTS)}")
    for name, d in dims.items():
        if not isinstance((d or {}).get("scorer_specified"), bool):
            problems.append(f"dimension {name}: scorer_specified must be true or false")
        elif not d["scorer_specified"] and not d.get("missing"):
            problems.append(f"dimension {name}: an unspecified scorer must state what is missing")
    if dims and all((d or {}).get("scorer_specified") for d in dims.values()) != DIMENSION_SCORERS_SPECIFIED:
        problems.append("DIMENSION_SCORERS_SPECIFIED disagrees with the variants file")
    names = {ansg.get("name")}
    for key in SEPARATE_METRICS:
        v = data.get(key) or {}
        if v.get("status") != "different_metric":
            problems.append(f"{key}: status must be 'different_metric'")
        if "weights" in v or "equation" in v:
            problems.append(f"{key}: must not carry ANSG-style weights/equation keys")
        src = v.get("formula_source") or {}
        if not src.get("kind") or not src.get("pages"):
            problems.append(f"{key}: formula_source with kind and pages is required")
        if not v.get("formula") or v.get("formula") == "pending":
            problems.append(f"{key}: formula (or an explicit statement that none is printed) is required")
        if not v.get("name_in_paper") or v.get("name_in_paper") in names:
            problems.append(f"{key}: name_in_paper must be given and distinct from the other CMPI names")
        names.add(v.get("name_in_paper"))
        if not v.get("conflicts_with_dissertation"):
            problems.append(f"{key}: differences from the dissertation must be recorded")
    cultura = data.get("cultura") or {}
    weights = [c.get("weight") for c in cultura.get("criteria") or []]
    if len(weights) != cultura.get("K") or not weights:
        problems.append("cultura: K must equal the number of weighted criteria")
    else:
        try:
            bound = cultura_printed_upper_bound(weights)
        except ValueError as exc:
            problems.append(f"cultura: {exc}")
        else:
            reported = [r["value"] for r in cultura.get("results_reported") or [] if isinstance(r.get("value"), (int, float))]
            if any(r > bound for r in reported):
                if not cultura.get("printed_formula_conflicts"):
                    problems.append(f"cultura: printed formula bound {bound:g} is below a reported value {reported}; "
                                    "the conflict must be recorded")
                if not cultura.get("author_decision_options"):
                    problems.append("cultura: the options for the author's decision must be listed")
    eq = cultura.get("printed_equation_in_thesis") or {}
    if eq.get("is_cmpi") is not False or not eq.get("apparent_conflicts"):
        problems.append("cultura: printed Eq. 4.1 must be marked is_cmpi: false with its apparent conflicts recorded")
    return problems
