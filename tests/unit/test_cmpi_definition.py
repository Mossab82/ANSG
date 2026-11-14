"""CMPI: the dissertation's definition is the ANSG reference; same-name metrics stay separate."""
import pytest

from ansg.evaluation import ansg_cmpi as lazy_ansg_cmpi
from ansg.evaluation.cmpi import (
    ANSG_CMPI_WEIGHTS, DIMENSION_SCORERS_SPECIFIED, ansg_cmpi, ansg_cmpi_from_mapping, cultura_cmpi_as_printed,
    cultura_printed_upper_bound, load_cmpi_variants, mm_gta_cmpi_as_printed, validate_cmpi_variants,
)
from ansg.utils.config import load_yaml_config


def test_weights_are_the_thesis_ahp_weights():
    assert ANSG_CMPI_WEIGHTS == {"rhetorical_devices": 0.35, "dialectal_appropriateness": 0.25,
                                 "narrative_structure": 0.25, "moral_framing": 0.15}
    assert sum(ANSG_CMPI_WEIGHTS.values()) == pytest.approx(1.0)
    assert DIMENSION_SCORERS_SPECIFIED is False


def test_equation_and_range():
    assert ansg_cmpi(1, 1, 1, 1) == pytest.approx(1.0)
    assert ansg_cmpi(0, 0, 0, 0) == 0.0
    assert ansg_cmpi(1, 0, 0, 0) == pytest.approx(0.35)
    assert ansg_cmpi(0, 0, 0, 1) == pytest.approx(0.15)
    assert lazy_ansg_cmpi is ansg_cmpi
    with pytest.raises(ValueError):
        ansg_cmpi(1.2, 0, 0, 0)


def test_mapping_api_requires_provenance_and_exact_dimensions():
    scores = {"rhetorical_devices": 0.5, "dialectal_appropriateness": 0.5, "narrative_structure": 0.5, "moral_framing": 0.5}
    out = ansg_cmpi_from_mapping(scores, provenance="synthetic test values")
    assert out["value"] == pytest.approx(0.5) and out["reproduces_thesis_result"] is False
    assert out["dimension_scorers_specified_in_thesis"] is False
    with pytest.raises(ValueError):
        ansg_cmpi_from_mapping(scores, provenance=" ")
    with pytest.raises(KeyError):
        ansg_cmpi_from_mapping({**scores, "formulaic": 0.5}, provenance="x")


def test_variants_file_is_consistent_and_keeps_metrics_apart():
    data = load_cmpi_variants()
    assert validate_cmpi_variants(data) == []
    ansg = data["ansg_thesis"]
    assert {s["page"] for s in ansg["sources"]} >= {131, 223, 268}
    assert ansg["terminology_conflict"]
    names = {ansg["name"]}
    for key in ("cultura", "cst_onto", "mm_gta"):
        v = data[key]
        assert v["status"] == "different_metric" and v["formula"] != "pending"
        assert v["formula_source"]["kind"] == "owner_extraction"
        assert "weights" not in v and "equation" not in v
        assert v["name_in_paper"] not in names
        names.add(v["name_in_paper"])
    assert data["cultura"]["name_in_paper"] == "Cultural Matching Performance Index"
    assert data["cst_onto"]["name_in_paper"] == "Cultural Moral Precision Index"
    assert data["mm_gta"]["name_in_paper"] == "Cross-dialectal Morphological Preservation Index"


def test_cultura_printed_formula_and_its_recorded_conflict():
    cultura = load_cmpi_variants()["cultura"]
    weights = [c["weight"] for c in cultura["criteria"]]
    assert weights == [0.25, 0.25, 0.25, 0.15, 0.10] and cultura["K"] == 5
    assert cultura["formula_source"]["pages"] == "314-315"
    assert cultura_printed_upper_bound(weights) == pytest.approx(0.2)
    assert any(r["value"] == 0.82 for r in cultura["results_reported"] if isinstance(r["value"], float))
    assert any("0.2" in c and "0.82" in c for c in cultura["printed_formula_conflicts"])
    # computed exactly as printed, never "corrected"
    assert cultura_cmpi_as_printed([8, 8, 8, 8, 8], weights) == pytest.approx(0.8 / 5)


def test_cst_onto_paper_vs_dissertation_labels_recorded():
    cst = load_cmpi_variants()["cst_onto"]
    reported = {r["metric"]: r["value"] for r in cst["results_reported"]}
    assert reported == {"CRAFT": 0.89, "CMPI (Cultural Moral Precision Index)": 0.78}
    assert any("CulturalScore" in c and "0.89" in c for c in cst["conflicts_with_dissertation"])


def test_mm_gta_is_a_morphological_f1():
    mm = load_cmpi_variants()["mm_gta"]
    assert mm["formula"] == "CMPI = 2 * P_morph * R_morph / (P_morph + R_morph)"
    assert mm_gta_cmpi_as_printed(0.8, 0.8) == pytest.approx(0.8)
    assert mm_gta_cmpi_as_printed(0.0, 0.0) == 0.0
    assert "not the MM-GTA paper's metric" in mm["results_reported"][0]["note"]


def test_cultura_printed_equation_is_not_cmpi_and_its_conflicts_are_recorded():
    eq = load_cmpi_variants()["cultura"]["printed_equation_in_thesis"]
    assert eq["id"] == "Eq. 4.1" and eq["page"] == 87 and eq["is_cmpi"] is False
    assert len(eq["apparent_conflicts"]) == 3
    assert eq["paper_gating_equation_check"] == "pending"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["ansg_thesis"]["weights"].__setitem__("moral_framing", 0.2), "differ"),
    (lambda d: d["cultura"].__setitem__("formula", "pending"), "formula"),
    (lambda d: d["mm_gta"].__setitem__("weights", {"a": 1.0}), "ANSG-style"),
    (lambda d: d["cst_onto"].__setitem__("name_in_paper", "Cultural Motif Preservation Index"), "distinct"),
    (lambda d: d["cultura"].__setitem__("printed_formula_conflicts", []), "conflict must be recorded"),
    (lambda d: d["mm_gta"].__setitem__("conflicts_with_dissertation", []), "differences from the dissertation"),
    (lambda d: d["cultura"]["printed_equation_in_thesis"].__setitem__("apparent_conflicts", []), "apparent conflicts"),
])
def test_validation_catches_mixing(mutate, message):
    import copy
    data = copy.deepcopy(load_cmpi_variants())
    mutate(data)
    assert any(message in p for p in validate_cmpi_variants(data))


def test_craft_registry_records_the_equation_but_still_refuses_text():
    cfg = load_yaml_config("configs/evaluation/craft.yaml")
    cmpi = next(m for m in cfg["metrics"] if m["name"] == "CMPI")
    assert cmpi["equation"].startswith("CMPI = 0.35*rhetorical_devices")
    assert cmpi["status"] == "incomplete" and "p.131" in cmpi["thesis_reference"]
