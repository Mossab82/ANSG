"""CMPI: the dissertation's definition is the ANSG reference; same-name metrics stay separate."""
import pytest

from ansg.evaluation import ansg_cmpi as lazy_ansg_cmpi
from ansg.evaluation.cmpi import (
    ANSG_CMPI_WEIGHTS, DIMENSION_SCORERS_SPECIFIED, ansg_cmpi, ansg_cmpi_from_mapping, load_cmpi_variants,
    validate_cmpi_variants,
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
    for key in ("cultura", "cst_onto", "mm_gta"):
        v = data[key]
        assert v["status"] == "different_metric" and v["formula"] == "pending"
        assert v["paper_available_in_repository"] is False
        assert "weights" not in v and "equation" not in v
    # MM-GTA's 0.78 is recorded as the thesis's re-evaluation, not the paper's metric
    assert "not the MM-GTA paper's metric" in data["mm_gta"]["results_reported_in_thesis"][0]["note"]
    # CST-Onto's 0.89 is CulturalScore, not CMPI
    assert data["cst_onto"]["results_reported_in_thesis"][0]["metric"] == "CulturalScore"


def test_cultura_printed_equation_is_not_cmpi_and_its_conflicts_are_recorded():
    eq = load_cmpi_variants()["cultura"]["printed_equation_in_thesis"]
    assert eq["id"] == "Eq. 4.1" and eq["page"] == 87 and eq["is_cmpi"] is False
    assert len(eq["apparent_conflicts"]) == 3
    assert eq["paper_equation_check"] == "pending"


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["ansg_thesis"]["weights"].__setitem__("moral_framing", 0.2), "differ"),
    (lambda d: d["cultura"].__setitem__("formula", "0.5*x"), "pending"),
    (lambda d: d["mm_gta"].__setitem__("weights", {"a": 1.0}), "ANSG-style"),
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
