"""Reward / RL variants: transcription, arithmetic, refusal of undefined components."""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

from ansg.reward_spec import (
    DEFAULT_SPEC, RewardSpec, UndefinedComponentError, load_reward_spec,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spec():
    return load_reward_spec()


def _cli():
    s = importlib.util.spec_from_file_location("audit_reward_rl_spec", ROOT / "scripts" / "audit_reward_rl_spec.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_spec_is_valid_and_selects_nothing(spec):
    assert spec.validate() == []
    assert spec.selected_variant is None
    assert spec.data["selected_ppo_variant"] is None


def test_every_variant_is_blocked_and_cmpi_is_undefined(spec):
    assert spec.components["CMPI"]["status"] == "undefined"
    for vid in spec.variants:
        assert spec.blockers(vid), f"{vid} unexpectedly computable"
    for vid in ("ch4_eq_4_26_to_4_29", "ch4_table_4_1_and_s4_1_1", "ch5_eq_5_1_s5_6_3"):
        assert "CMPI" in {b["component"] for b in spec.blockers(vid)}


def test_evaluate_refuses_undefined_components_by_default(spec):
    values = {c: 0.5 for c in spec.leaf_components("ch4_table_4_1_and_s4_1_1")}
    with pytest.raises(UndefinedComponentError, match="CMPI"):
        spec.evaluate("ch4_table_4_1_and_s4_1_1", values)
    res = spec.evaluate("ch4_table_4_1_and_s4_1_1", values, allow_undefined=True)
    assert res["reproduces_thesis_result"] is False
    assert set(res["caller_supplied_undefined"]) == {"CMPI", "PCI"}


def test_worked_examples_recomputed_from_printed_inputs(spec):
    ex = {e["id"]: e["rows"] for e in spec.worked_examples()}
    eq430 = ex["eq_4_30"]
    assert eq430["R_cult"]["recomputed"] == pytest.approx(0.860)        # printed 0.867 (C-REWARD-6)
    assert eq430["R_ling"]["recomputed"] == pytest.approx(0.785)        # printed 0.784
    assert eq430["R_coh"]["recomputed"] == pytest.approx(0.950)
    assert eq430["R_total"]["recomputed"] == pytest.approx(0.85175)
    assert eq430["R_total_from_printed_partials"]["recomputed"] == pytest.approx(0.85455)
    assert not eq430["R_cult"]["consistent_to_printed_precision"]
    assert ex["s4_1_1_step_9"]["R_total"]["recomputed"] == pytest.approx(0.7045)


def test_ch5_penalties_and_unnormalised_r_ling(spec):
    v = spec.variant("ch5_eq_5_1_s5_6_3")
    assert {(p["applies_to"], p["coefficient"], p["equation"]) for p in v["penalties"]} == {
        ("R_cult", 0.10, "5.3"), ("R_total", 0.15, "5.2")}
    statuses = {b["component"]: b["status"] for b in spec.blockers("ch5_eq_5_1_s5_6_3")}
    assert statuses["R_ling"] == "range_unnormalised" and statuses["kl_penalty"] == "underspecified"
    vals = {c: 0.0 for c in spec.leaf_components("ch5_eq_5_1_s5_6_3")}
    vals.update(CMPI=1.0, BLEU4_normalised=1.0, perplexity_normalised=0.0, embedding_coherence=1.0,
                proverb_repetition=1.0, coherence_penalty=1.0)
    res = spec.evaluate("ch5_eq_5_1_s5_6_3", vals, allow_undefined=True)
    assert res["nodes"]["R_ling"] == pytest.approx(2.0)                  # unweighted sum exceeds 1
    assert res["nodes"]["R_cult"] == pytest.approx(0.9)
    assert res["value"] == pytest.approx(0.45 * 0.9 + 0.35 * 2.0 + 0.20 * 1.0 - 0.15)


def test_same_weights_assigned_to_different_terms_are_flagged(spec):
    groups = {tuple(g["weights"]): g for g in spec.same_weights_different_terms()}
    assert (0.35, 0.3, 0.25, 0.1) in groups and (0.45, 0.35, 0.2) in groups
    uses = {u["variant"] for u in groups[(0.35, 0.3, 0.25, 0.1)]["uses"]}
    assert uses == {"ch4_eq_4_26_to_4_29", "ch5_s5_5_3_table_5_9_cultural_subweights", "appendix_a3_a11"}


def test_consistent_with_thesis_variants_transcription(spec):
    old = yaml.safe_load((ROOT / "configs/specification/thesis_variants.yaml").read_text(encoding="utf-8"))["reward_variants"]
    nodes = lambda vid: {n: {t.name: t.weight for t in ts} for n, ts in spec.nodes(vid).items()}
    ch4 = nodes("ch4_eq_4_26_to_4_29")
    assert ch4["R_total"] == {"R_cult": 0.45, "R_ling": 0.35, "R_coh": 0.20}
    assert ch4["R_cult"] == {"CMPI": 0.35, "virtue_coherence": 0.30, "rhetorical_devices": 0.25, "register": 0.10}
    assert sorted(ch4["R_ling"].values()) == sorted(old["ch4_eq_4_26_4_29"]["linguistic"].values())
    assert sorted(ch4["R_coh"].values()) == sorted(old["ch4_eq_4_26_4_29"]["coherence"].values())
    assert list(nodes("ch4_table_4_1_and_s4_1_1")["R_total"].values()) == list(old["ch4_table_4_1_and_4_1_1"]["total"].values())
    assert nodes("appendix_a3_a11")["R_total"] == old["appendix_a3_a11"]["total"]
    assert nodes("ch5_s5_1_cmpi_composition")["CMPI_ch5_composition"] == {
        "rhetorical_devices": 0.35, "dialectal_appropriateness": 0.25, "narrative_structure": 0.25, "moral_framing": 0.15}
    assert sorted(old["ch5_5_1"]["CMPI_internal"].values()) == [0.15, 0.25, 0.25, 0.35]
    pen = old["ch5_5_6_3"]["penalties"]
    v = spec.variant("ch5_eq_5_1_s5_6_3")
    assert {p["coefficient"] for p in v["penalties"]} == {pen["coherence_penalty"], pen["proverb_repetition"]}
    assert v["kl_penalty"]["coefficient"] == pen["kl"]
    assert list(nodes("appendix_a21")["cultural_vs_fluency"].values()) == old["appendix_a21"]["cultural_vs_fluency"]


def test_code_alignment_reports_defaults_without_choosing(spec):
    ca = spec.code_alignment()
    assert ca["cultural_reward_default_weights"] == [0.45, 0.35, 0.2]
    eq = ca["cultural_reward_default_weights_equal_top_level_weights_of"]
    assert eq["ch4_eq_4_26_to_4_29"] and eq["ch5_eq_5_1_s5_6_3"] and not eq["appendix_a3_a11"]
    rows = ca["ppo_variants"]["ch5_s5_6_3_table_5_9"]
    assert rows["clip_range"]["status"] == "match" and rows["value_coef"]["status"] == "match"
    assert rows["max_grad_norm"]["status"] == "code_default_differs_selectable_by_argument"
    assert rows["kl_coef"]["status"] == "caller_or_not_implemented"


def test_ppo_kwargs_require_an_explicit_variant(spec):
    out = spec.ppo_update_kwargs("ch5_s5_6_3_table_5_9")
    assert out["kwargs"] == {"clip_eps": 0.2, "value_coef": 0.5, "max_grad_norm": 0.5, "entropy_coef": 0.01}
    assert {"kl_coef", "gae_lambda", "learning_rate"} <= set(out["not_applied_by_ppo_update"])
    assert spec.ppo_update_kwargs("appendix_a3_listing_a2")["kwargs"] == {}
    with pytest.raises(KeyError):
        spec.ppo_update_kwargs("no_such_variant")


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d["variants"]["appendix_a3_a11"]["formula"]["R_total"]["sum"].__setitem__("diversity", 0.2), "not 1"),
    (lambda d: d["variants"]["ch4_table_4_1_and_s4_1_1"]["formula"]["R_total"]["sum"].__setitem__("XYZ", 0.0), "neither a node"),
    (lambda d: d["variants"]["ch4_eq_4_26_to_4_29"]["formula"]["R_cult"]["sum"].update({"R_total": 0.0}), "cycle"),
    (lambda d: d.__setitem__("selected_variant", "invented"), "not a variant"),
    (lambda d: d["components"]["CMPI"].__setitem__("status", "defined"), "status"),
])
def test_validation_catches_broken_specs(spec, mutate, message):
    data = copy.deepcopy(spec.data)
    mutate(data)
    assert any(message in p for p in RewardSpec(data).validate())


def test_cli_exit_codes(tmp_path, capsys):
    cli = _cli()
    assert cli.main([]) == 0
    assert cli.main(["--variant", "ch5_eq_5_1_s5_6_3"]) == 2
    out = capsys.readouterr().out
    assert "BLOCKED" in out and "no thesis result is reproduced" in out
    broken = yaml.safe_load(DEFAULT_SPEC.read_text(encoding="utf-8"))
    broken["variants"]["appendix_a21"]["formula"]["cultural_vs_fluency"]["sum"]["fluency"] = 0.5
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(broken, allow_unicode=True), encoding="utf-8")
    assert cli.main(["--spec", str(p)]) == 1
