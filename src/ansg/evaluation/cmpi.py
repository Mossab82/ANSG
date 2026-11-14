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

CULTURA, CST-Onto and MM-GTA use metrics of the same name defined in their own
papers, on other test sets and protocols. They are recorded separately in
``configs/evaluation/cmpi_variants.yaml`` and are never computed, converted or
compared here.
"""
from __future__ import annotations

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


def validate_cmpi_variants(data: Mapping[str, Any]) -> list[str]:
    """Consistency of the variants file with this module and its separation rules."""
    problems: list[str] = []
    ansg = data.get("ansg_thesis") or {}
    if dict(ansg.get("weights") or {}) != ANSG_CMPI_WEIGHTS:
        problems.append(f"ansg_thesis weights {ansg.get('weights')} differ from ANSG_CMPI_WEIGHTS")
    if not any(s.get("page") == 131 for s in ansg.get("sources") or []):
        problems.append("ansg_thesis must cite §5.1 p.131")
    for key in ("cultura", "cst_onto", "mm_gta"):
        v = data.get(key) or {}
        if v.get("status") != "different_metric":
            problems.append(f"{key}: status must be 'different_metric'")
        if v.get("paper_available_in_repository") is False and v.get("formula") != "pending":
            problems.append(f"{key}: formula must stay 'pending' while the paper is unavailable")
        if "weights" in v or "equation" in v:
            problems.append(f"{key}: must not carry ANSG-style weights/equation")
    eq = ((data.get("cultura") or {}).get("printed_equation_in_thesis") or {})
    if eq.get("is_cmpi") is not False or not eq.get("apparent_conflicts"):
        problems.append("cultura: printed Eq. 4.1 must be marked is_cmpi: false with its apparent conflicts recorded")
    return problems
