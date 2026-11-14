"""The decisions register agrees with the code and configuration it governs."""
from pathlib import Path

import yaml

from ansg.evaluation.cmpi import ANSG_CMPI_WEIGHTS, load_cmpi_variants
from ansg.reward_spec import load_reward_spec

ROOT = Path(__file__).resolve().parents[2]
REG = yaml.safe_load((ROOT / "configs/specification/owner_decisions.yaml").read_text(encoding="utf-8"))


def _decided(prefix):
    return next(d for d in REG["decided"] if d["id"].startswith(prefix))


def test_d1_weights_match_the_implementation():
    assert _decided("D1")["weights"] == ANSG_CMPI_WEIGHTS


def test_d3_single_record_holds_the_cultura_options():
    cultura = load_cmpi_variants()["cultura"]
    assert cultura["author_decision_options"] and cultura["printed_formula_conflicts"]
    assert cultura["formula"].startswith("CMPI(S) = (1/K)")
    assert any(r.get("value") == 0.82 for r in cultura["results_reported"])
    awaiting = {a["id"]: a for a in REG["awaiting_previous_decision"]}
    assert awaiting["cultura-0.2-vs-0.82"]["status"] == "not_re_raised"


def test_d4_no_variant_selected():
    spec = load_reward_spec()
    assert spec.selected_variant is None and spec.data["selected_ppo_variant"] is None
    assert spec.variant("ch5_eq_5_1_s5_6_3")["kl_penalty"]["placement"] == "unspecified"


def test_d5_resources_rights_and_training_stay_blocked():
    ckg = yaml.safe_load((ROOT / "configs/knowledge/ckg.yaml").read_text(encoding="utf-8"))
    for res in ("ckg", "cst_onto"):
        assert set(ckg["rights"][res].values()) == {"pending"}
    profile = yaml.safe_load((ROOT / "configs/training/ansg_direct_profile.yaml").read_text(encoding="utf-8"))
    assert profile["training_ready"] is False and profile["training_start_permitted"] is False


def test_register_sections_are_disjoint():
    ids = [x["id"] for key in ("decided", "awaiting_previous_decision", "needs_decision", "awaiting_inputs") for x in REG[key]]
    assert len(ids) == len(set(ids))
