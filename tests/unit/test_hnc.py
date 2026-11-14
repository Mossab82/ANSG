"""ANSG-HNC (optional research extension): plans, controller, conditions A–D, metrics.

Scenario plans are ORIGINAL research materials (research/hnc/scenarios/, unreviewed
draft). Generators here are deterministic FAKES: no model, no generated story, no
result. Negative tests mutate a valid plan in exactly one way.
"""
import copy
import glob
import sys
from pathlib import Path

import pytest

from ansg.hnc.controller import HNCError, NarrativeController, replay, schedule
from ansg.hnc.metrics import consistency_counts, structural_metrics
from ansg.hnc.model import HierarchicalPlan
from ansg.hnc.realize import HNCConfig, control_leaks, flat_prompt, realise
from ansg.hnc.validation import flatten, validate_plan
from ansg.narrative.events import Proposition, Relation, RelationKind
from ansg.narrative.validation import knowledge_item

ROOT = Path(__file__).resolve().parents[2]
SCEN = ROOT / "research" / "hnc" / "scenarios"
sys.path.insert(0, str(SCEN))
from build_scenarios import SCENARIOS, nested_chain  # noqa: E402


def load(name):
    return HierarchicalPlan.read(SCEN / f"{name}.json")


def codes(plan, **kw):
    return {i.code for i in validate_plan(plan, **kw).errors}


class FakeGen:
    """Deterministic stand-in for a language model; records every prompt."""

    def __init__(self, echo=None):
        self.prompts, self.echo = [], echo

    def __call__(self, prompt, i):
        self.prompts.append(prompt)
        return self.echo or f"مقطع {i}."


# ------------------------------------------------------------------ scenarios
def test_committed_scenarios_match_their_builder_and_validate():
    files = sorted(Path(p).stem for p in glob.glob(str(SCEN / "*.json")))
    assert files == sorted(SCENARIOS)
    for name, build in SCENARIOS.items():
        assert load(name).to_dict() == build().to_dict(), f"{name}.json is stale: rerun build_scenarios.py"
        rep = validate_plan(load(name))
        assert rep.ok, (name, rep.errors)
    assert {load(n).setting for n in SCENARIOS} == {"aqaba", "alexandria", "kuwait", "fes"}
    assert max(validate_plan(load("kuwait_3level")).depths.values()) == 3
    assert max(validate_plan(load("fes_4level_stress")).depths.values()) == 4


def test_json_roundtrip():
    p = load("kuwait_3level")
    assert HierarchicalPlan.from_dict(p.to_dict()).to_dict() == p.to_dict()
    with pytest.raises(ValueError):
        HierarchicalPlan.from_dict({"schema": "x"})


# ------------------------------------------------------------------ controller
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_schedule_replays_and_loses_no_event(name):
    plan = load(name)
    trace = replay(plan, schedule(plan))
    m = structural_metrics(plan, trace)
    assert m["event_coverage"] == 1.0 and m["lost_events"] == [] and m["duplicated_events"] == []
    assert m["narrative_level_accuracy"] == 1.0
    assert m["story_resumption_accuracy"] in (1.0, None) and m["narrative_closure_accuracy"] in (1.0, None)


def test_embedded_tale_keeps_parent_context_and_returns_to_correct_level():
    plan = load("kuwait_3level")
    c = NarrativeController(plan)
    c.start()
    for e in ("kf1", "kf2", "kf3"):
        c.emit(e)
    c.enter("kw-grand")
    for e in ("kg1", "kg2", "kg3"):
        c.emit(e)
    c.enter("kw-diver")
    assert c.path() == ["kw-frame", "kw-grand", "kw-diver"]
    for e in ("kd1", "kd2", "kd3"):
        c.emit(e)
    c.close()
    assert c.path() == ["kw-frame", "kw-grand"]                     # back one level, not to the root
    assert (c.top.anchor, c.top.pending_resume) == ("kg3", "kg4")   # parent context preserved
    with pytest.raises(HNCError, match="must resume with 'kg4'"):
        c.emit("kf4")                                               # wrong level
    c.emit("kg4")
    c.close()
    assert c.path() == ["kw-frame"] and c.top.pending_resume == "kf4"
    c.emit("kf4")
    trace = c.finish()
    assert [t["frame"] for t in trace.knowledge_transfers] == ["kw-diver", "kw-grand"]


def test_illegal_transitions(tmp_path):
    plan = load("aqaba_2level")
    c = NarrativeController(plan)
    with pytest.raises(HNCError):
        c.emit("aq1")                                  # not started
    c.start()
    with pytest.raises(HNCError, match="not the next event"):
        c.emit("aq2")                                  # out of order
    c.emit("aq1")
    with pytest.raises(HNCError, match="opens after"):
        c.enter("aq-storm")                            # anchor not reached
    c.emit("aq2")
    with pytest.raises(HNCError, match="before the embedded tale"):
        c.emit("aq3")                                  # skipping the tale
    c.enter("aq-storm")
    c.emit("st1")
    with pytest.raises(HNCError, match="not told"):
        c.close()                                      # closure with untold events (event loss)
    c.emit("st2")
    c.emit("st3")
    c.close()
    with pytest.raises(HNCError, match="already told"):
        c.enter("aq-storm")
    c.emit("aq3")
    with pytest.raises(HNCError, match="before all its events"):
        c.finish()
    with pytest.raises(HNCError, match="root frame is finished"):
        c.close()


def test_sibling_tales_at_the_same_anchor_then_resume():
    plan = load("alexandria_2level")
    sib = copy.deepcopy(plan.frame("alx-key"))
    sib.id = "alx-key2"
    for e in sib.graph.events:
        e.id = e.id + "b"
    sib.graph.relations = [Relation(r.kind, r.source + "b", r.target + "b", status=r.status)
                           for r in sib.graph.relations]
    plan.frames.append(sib)
    assert validate_plan(plan).ok
    ops = schedule(plan)
    enters = [a for o, a in ops if o == "enter"]
    assert enters == ["alx-key", "alx-key2"]
    after = ops[[i for i, (o, _) in enumerate(ops) if o == "close"][-1] + 1]
    assert after == ("emit", "al3")


# ------------------------------------------------------------------ validation: wrong cases
def test_knowledge_does_not_cross_levels_without_an_event():
    plan = load("aqaba_2level")
    plan.frame("aq-frame").graph.event("aq3").knowledge_required.append(
        knowledge_item("eid", "rescued", "fishers"))       # fact of the inner tale, never conveyed
    assert "E-KNOWLEDGE-UNSOURCED" in codes(plan) and "HNC-E-FOREIGN-ENTITY" in codes(plan)

    plan = load("alexandria_2level")                        # zeinab heard, hassan did not
    plan.frame("alx-frame").graph.event("al4").knowledge_required.append(
        knowledge_item("hassan", "hasty_accusation_wrongs", "innocents"))
    assert "E-KNOWLEDGE-UNSOURCED" in codes(plan)

    plan = load("aqaba_2level")                             # acting on the lesson before hearing it
    plan.frame("aq-frame").graph.event("aq1").knowledge_required.append(
        knowledge_item("odeh", "saved_lives", "boat"))
    assert "E-KNOWLEDGE-PREMATURE" in codes(plan)


def test_inner_tale_cannot_know_frame_events():
    plan = load("aqaba_2level")
    plan.frame("aq-storm").graph.event("st3").knowledge_required.append(
        knowledge_item("eid_young", "wants_to_sell", "odeh", "boat"))
    found = codes(plan)
    assert {"HNC-E-FOREIGN-ENTITY", "E-KNOWLEDGE-UNSOURCED"} <= found


def test_time_axes_are_separate():
    plan = load("aqaba_2level")
    plan.frame("aq-frame").graph.relations.append(Relation(RelationKind.BEFORE, "st3", "aq4"))
    assert "HNC-E-CROSS-FRAME-RELATION" in codes(plan)
    plan = load("aqaba_2level")
    plan.frame("aq-storm").time_context = "  "
    assert "HNC-E-TIME-CONTEXT" in codes(plan)


def test_embedded_tale_must_have_a_function():
    plan = load("fes_2level")
    plan.frame("fs-frame").graph.event("fs3").knowledge_required = []
    assert "HNC-E-NO-FUNCTION" in codes(plan)
    plan = load("fes_2level")
    plan.frame("fs-weaver").function = ""
    assert "HNC-E-NO-FUNCTION" in codes(plan)
    plan = load("fes_2level")
    plan.frame("fs-weaver").conveys = []
    assert "HNC-E-NO-FUNCTION" in codes(plan)


@pytest.mark.parametrize("mutate,code", [
    (lambda p: setattr(p.frame("aq-storm"), "narrator", "boat"), "HNC-E-NARRATOR-UNKNOWN"),
    (lambda p: setattr(p.frame("aq-storm"), "audience", []), "HNC-E-AUDIENCE"),
    (lambda p: setattr(p.frame("aq-storm"), "narrator_in_frame", None), "HNC-E-PERSPECTIVE"),
    (lambda p: setattr(p.frame("aq-storm"), "resumed_by", "aq4"), "HNC-E-RESUME-ORDER"),
    (lambda p: setattr(p.frame("aq-storm"), "opened_by", "zz"), "HNC-E-ANCHOR-UNKNOWN"),
    (lambda p: setattr(p.frame("aq-storm"), "parent", None), "HNC-E-ROOT"),
    (lambda p: setattr(p.frame("aq-storm"), "parent", "nowhere"), "HNC-E-PARENT-UNKNOWN"),
    (lambda p: setattr(p.frame("aq-frame"), "parent", "aq-storm"), "HNC-E-ROOT"),
    (lambda p: setattr(p.frame("aq-storm").graph.events[0], "id", "aq1"), "HNC-E-DUPLICATE-EVENT"),
])
def test_structural_errors(mutate, code):
    plan = load("aqaba_2level")
    mutate(plan)
    assert code in codes(plan)


def test_cycle_and_depth_limit():
    plan = load("kuwait_3level")
    plan.frame("kw-grand").parent = "kw-diver"
    assert "HNC-E-CYCLE" in codes(plan)
    deep = nested_chain(9)
    assert "HNC-E-DEPTH" in codes(deep)
    assert validate_plan(deep, max_depth=9).ok


def test_ablation_c_skips_only_the_knowledge_layer():
    plan = load("aqaba_2level")
    plan.frame("aq-frame").graph.event("aq1").knowledge_required.append(knowledge_item("odeh", "saved_lives", "boat"))
    assert "E-KNOWLEDGE-PREMATURE" in codes(plan)
    assert validate_plan(plan, epistemic=False).ok          # C: level tracking only
    plan.frame("aq-storm").resumed_by = "aq4"
    assert "HNC-E-RESUME-ORDER" in codes(plan, epistemic=False)


# ------------------------------------------------------------------ conditions A–D
def test_disabled_hnc_is_one_untouched_call():
    plan, gen = load("kuwait_3level"), FakeGen()
    r = realise(plan, gen, HNCConfig(), "مدينة الكويت")
    assert HNCConfig().hnc_enabled is False and r.variant == "A"
    assert gen.prompts == [flat_prompt(plan, "مدينة الكويت")] and r.text == "مقطع 0."
    d = r.to_dict()
    assert d["hnc"]["enabled"] is False and d["trace"] is None


def test_hnc_enabled_generates_per_segment_with_resumption():
    plan, gen = load("kuwait_3level"), FakeGen()
    r = realise(plan, gen, HNCConfig.for_variant("B"), "مدينة الكويت")
    d = r.to_dict()
    assert d["hnc"] == {"enabled": True, "variant": "B", "config": d["hnc"]["config"]}
    levels = [s["level"] for s in r.segments]
    assert levels == [1, 2, 3, 2, 1]                        # down to 3 levels and back up
    assert [s["events"] for s in r.segments][-2:] == [["kg4"], ["kf4"]]
    assert "عُد الآن إلى القصة" in gen.prompts[3] and "عُد الآن إلى القصة" in gen.prompts[4]
    assert structural_metrics(plan, d["trace"])["story_resumption_accuracy"] == 1.0
    assert r.text.count("مقطع") == 5 and r.checks["control_leaks"] == []


def test_prompts_never_contain_internal_identifiers():
    plan, gen = load("kuwait_3level"), FakeGen()
    realise(plan, gen, HNCConfig.for_variant("B"), "مدينة الكويت")
    flat = flat_prompt(plan, "مدينة الكويت")
    ids = [f.id for f in plan.frames] + [e.id for f in plan.frames for e in f.graph.events]
    for prompt in gen.prompts + [flat]:
        assert not [i for i in ids if i in prompt]
    leaky = realise(plan, FakeGen(echo="kw-diver kd2"), HNCConfig.for_variant("B"), "x")
    assert leaky.checks["control_leaks"] == ["kd2", "kw-diver"]


def test_variants_c_and_d():
    plan = load("aqaba_2level")
    plan.frame("aq-frame").graph.event("aq1").knowledge_required.append(knowledge_item("odeh", "saved_lives", "boat"))
    with pytest.raises(HNCError):
        realise(plan, FakeGen(), HNCConfig.for_variant("B"), "العقبة")
    assert realise(plan, FakeGen(), HNCConfig.for_variant("C"), "العقبة").variant == "C"
    with pytest.raises(ValueError, match="flattened plan"):
        realise(plan, FakeGen(), HNCConfig.for_variant("D"), "العقبة")
    ok = realise(load("aqaba_2level"), FakeGen(), HNCConfig.for_variant("D"), "العقبة")
    assert ok.variant == "D" and len(ok.segments) == 1
    assert validate_plan(load("aqaba_2level")).ok and flatten(load("aqaba_2level")).events
    with pytest.raises(ValueError):
        HNCConfig(hnc_enabled=True, level_tracking=False, epistemic_checks=False).variant


# ------------------------------------------------------------------ metrics on annotated traces
def test_metrics_detect_level_resumption_narrator_and_loss_errors():
    plan = load("aqaba_2level")
    good = [dict(op="start", frame="aq-frame", level=1),
            dict(op="emit", frame="aq-frame", event="aq1", level=1, narrator=None),
            dict(op="emit", frame="aq-frame", event="aq2", level=1, narrator=None),
            dict(op="enter", frame="aq-storm", level=2),
            dict(op="emit", frame="aq-storm", event="st1", level=2, narrator="eid"),
            dict(op="emit", frame="aq-storm", event="st2", level=2, narrator="eid"),
            dict(op="emit", frame="aq-storm", event="st3", level=2, narrator="eid"),
            dict(op="close", frame="aq-storm", level=2),
            dict(op="emit", frame="aq-frame", event="aq3", level=1, narrator=None),
            dict(op="emit", frame="aq-frame", event="aq4", level=1, narrator=None)]
    m = structural_metrics(plan, good)
    assert (m["narrative_level_accuracy"], m["story_resumption_accuracy"], m["narrative_closure_accuracy"],
            m["narrator_consistency"], m["event_coverage"]) == (1.0, 1.0, 1.0, 1.0, 1.0)
    bad = copy.deepcopy(good)
    bad[5]["level"] = 1                       # inner event realised at the frame level
    bad[6]["narrator"] = "odeh"               # wrong narrator
    del bad[8]                                # resumption event aq3 lost
    m = structural_metrics(plan, bad)
    assert m["narrative_level_accuracy"] < 1 and m["narrator_consistency"] < 1
    assert m["story_resumption_accuracy"] == 0.0 and m["lost_events"] == ["aq3"]
    unclosed = [s for s in good if s["op"] != "close"]
    assert structural_metrics(plan, unclosed)["narrative_closure_accuracy"] == 0.0


def test_consistency_counts_on_annotated_graphs():
    plan = load("aqaba_2level")
    g = copy.deepcopy(plan.frame("aq-frame").graph)
    out = consistency_counts({"aq-frame": g})
    assert out["aq-frame"]["knowledge_violations"] == 1        # lesson used, never heard (no hearing step)
    assert out["aq-frame"]["character_knowledge_consistency"] == 0.0
    g.event("aq4").preconditions.append(Proposition("broken", ("boat",)))
    assert consistency_counts({"aq-frame": g}, closed_world=True)["aq-frame"]["causal_violations"] >= 1


def test_hnc_has_no_trainable_parameters_and_no_torch():
    import subprocess
    code = ("import sys, ansg.hnc, ansg.hnc.controller, ansg.hnc.realize, ansg.hnc.metrics, ansg.hnc.integration;"
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
