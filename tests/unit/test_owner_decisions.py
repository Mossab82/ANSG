"""The decisions register agrees with the code and configuration it governs."""
from pathlib import Path

import yaml

from ansg.evaluation.cmpi import ANSG_CMPI_WEIGHTS, load_cmpi_variants
from ansg.reward_spec import load_reward_spec

ROOT = Path(__file__).resolve().parents[2]
REG = yaml.safe_load((ROOT / "configs/specification/owner_decisions.yaml").read_text(encoding="utf-8"))
SCHED = yaml.safe_load((ROOT / "configs/specification/training_schedule.yaml").read_text(encoding="utf-8"))


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


def test_d6_reference_reward_is_eq_4_26_to_4_29_with_printed_weights():
    spec = load_reward_spec()
    assert spec.selected_variant == "ch4_eq_4_26_to_4_29"
    nodes = {n: {t.name: t.weight for t in ts} for n, ts in spec.nodes("ch4_eq_4_26_to_4_29").items()}
    stated = REG["stated_values"]["reward_reference"]
    for node in ("R_total", "R_cult", "R_ling", "R_coh"):
        assert nodes[node] == stated[node]
    alternatives = set(spec.data["selection"]["reward"]["conflicting_alternatives"])
    assert alternatives == {vid for vid, v in spec.variants.items() if v.get("role") == "conflicting_alternative"}
    # sub-components without a text measurement stay blocked (D9), not decided
    assert {"virtue_coherence", "coreference", "temporal", "causal"} <= {b["component"] for b in spec.blockers("ch4_eq_4_26_to_4_29")}


def test_d7_ppo_values_and_appendix_conflict():
    spec = load_reward_spec()
    assert spec.data["selected_ppo_variant"] == "ch5_s5_6_3_table_5_9"
    p = spec.ppo_variants["ch5_s5_6_3_table_5_9"]
    stated = REG["stated_values"]["ppo"]["values"]
    assert {k: p[k] for k in stated} == stated
    conflict = spec.data["selection"]["ppo"]["conflicts"][0]
    assert (conflict["setting"], conflict["value"], conflict["body_value"]) == ("learning_rate", 1.0e-6, 1.0e-5)
    assert spec.ppo_variants["appendix_a3_listing_a2"]["learning_rate"] == 1.0e-6     # kept, not overwritten
    kw = spec.selected_ppo_kwargs()["kwargs"]
    assert kw == {"clip_eps": 0.2, "value_coef": 0.5, "max_grad_norm": 0.5, "entropy_coef": 0.01}


def test_d8_normalisation_and_kl_are_operational_gaps_not_questions():
    gaps = load_reward_spec().data["operational_gaps"]
    assert gaps["perplexity_normalisation"]["status"] == "required_input"
    assert gaps["kl_placement"]["status"] == "implementation_choice_to_pin_before_training"
    assert gaps["kl_placement"]["coefficient"] == 0.01
    assert "needs_decision" not in REG


def test_d10_training_schedule():
    stages = SCHED["stages"]
    assert [s["id"] for s in stages] == ["supervised_pretraining", "adversarial_refinement", "ppo"]
    assert [tuple(s["epochs"]) for s in stages] == [(1, 30), (31, 65), (66, 100)]
    assert "70.4M" in stages[0]["data"] and "23M" in stages[0]["data"]
    assert "50K generated samples per epoch" in stages[1]["data"]
    assert "128" in stages[2]["data"]
    assert load_reward_spec().ppo_variants[stages[2]["ppo_variant"]]["rollout_batch"] == 128
    assert SCHED["execution"]["status"] == "blocked"


def test_d5_resources_rights_and_training_stay_blocked():
    ckg = yaml.safe_load((ROOT / "configs/knowledge/ckg.yaml").read_text(encoding="utf-8"))
    for res in ("ckg", "cst_onto"):
        assert set(ckg["rights"][res].values()) == {"pending"}
    profile = yaml.safe_load((ROOT / "configs/training/ansg_direct_profile.yaml").read_text(encoding="utf-8"))
    assert profile["training_ready"] is False and profile["training_start_permitted"] is False


def test_register_sections_are_disjoint():
    ids = [x["id"] for key in ("decided", "conflicts_recorded", "operational_gaps", "awaiting_previous_decision")
           for x in REG[key]]
    assert len(ids) == len(set(ids))
