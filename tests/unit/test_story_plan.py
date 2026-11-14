"""Story plans for original generation in four cultural settings (synthetic plans)."""
import copy
import json

import pytest

from ansg.evaluation import story_proxies as sp
from ansg.narrative.events import Entity, EntityKind, Event, EventGraph, Modality, Participant, Proposition, \
    Relation, RelationKind
from ansg.narrative.plan import StoryPlan, build_prompt, load_settings, validate_plan
from ansg.narrative.validation import knowledge_item


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def plan(setting="aqaba", names=("سالم", "عودة"), place="خليج العقبة"):
    a, b = names
    ents = [Entity("a", a, EntityKind.CHARACTER, "protagonist"), Entity("b", b, EntityKind.CHARACTER, "owner"),
            Entity("box", "صندوق", EntityKind.OBJECT), Entity("p", place, EntityKind.PLACE)]
    E = lambda i, trig, n, parts, gloss, **kw: Event(i, trig, None, n, parts, Modality.PLANNED, gloss=gloss, **kw)
    events = [
        E("p1", "وجد", 0, [Participant("Agent", "a"), Participant("Theme", "box"), Participant("Location", "p")],
          f"{a} يجد صندوقًا مغلقًا", effects=[Proposition("has", ("a", "box"))]),
        E("p2", "سأل", 1, [Participant("Agent", "a"), Participant("Recipient", "b")], f"{a} يسأل عن صاحب الصندوق",
          knowledge_gained=[knowledge_item("a", "owner", "box", "b", channel="told", source="b")]),
        E("p3", "أعاد", 2, [Participant("Agent", "a"), Participant("Theme", "box"), Participant("Recipient", "b")],
          f"{a} يعيد الصندوق إلى {b}", preconditions=[Proposition("has", ("a", "box"))],
          effects=[Proposition("has", ("a", "box"), True), Proposition("has", ("b", "box"))],
          knowledge_required=[knowledge_item("a", "owner", "box", "b")]),
    ]
    rels = [Relation(RelationKind.ENABLES, "p1", "p2"), Relation(RelationKind.ENABLES, "p2", "p3")]
    g = EventGraph(f"plan-{setting}", None, ents, events, rels,
                   initial_knowledge=[knowledge_item("b", "owner", "box", "b", channel="initial")])
    return StoryPlan(setting, "الأمانة", g)


def test_four_settings_are_explicit(settings):
    assert set(settings) == {"aqaba", "alexandria", "kuwait", "fes"}
    assert {s.dal_dialect for s in settings.values()} == {"Levantine", "Egyptian", "Gulf", "Maghrebi"}
    assert all(s.review_status == "unreviewed_draft" for s in settings.values())


@pytest.mark.parametrize("setting,names,place", [
    ("aqaba", ("سالم", "عودة"), "خليج العقبة"), ("alexandria", ("محمود", "زينب"), "الكورنيش"),
    ("kuwait", ("فهد", "دانة"), "سوق المباركية"), ("fes", ("إدريس", "خديجة"), "باب بوجلود")])
def test_valid_plans_build_prompts(settings, setting, names, place):
    p = plan(setting, names, place)
    rep = validate_plan(p, settings)
    assert rep.ok and not [i for i in rep.issues if i.code.startswith("W-PLAN")], rep.issues
    prompt = build_prompt(p, settings)
    assert settings[setting].label in prompt and "لا تنقل أي نص" in prompt
    assert prompt.index("يجد صندوقًا") < prompt.index("يسأل") < prompt.index("يعيد الصندوق")
    assert StoryPlan.from_dict(json.loads(json.dumps(p.to_dict(), ensure_ascii=False))).to_dict() == p.to_dict()


def test_incoherent_plans_are_rejected_before_generation(settings):
    p = plan()
    p.graph.relations = [Relation(RelationKind.BEFORE, "p3", "p2")]   # returns the box before learning the owner
    with pytest.raises(ValueError, match="E-KNOWLEDGE-PREMATURE"):
        build_prompt(p, settings)
    p = plan()
    p.graph.events[0].effects = []          # the box is never found, yet it is returned (closed world for plans)
    assert "E-PRECONDITION-NOT-ESTABLISHED" in validate_plan(p, settings).codes()
    p = plan()
    p.graph.events[0].modality = Modality.ACTUAL
    assert "E-PLAN-MODALITY" in validate_plan(p, settings).codes()
    p = plan()
    p.setting = "atlantis"
    assert "E-PLAN-SETTING-UNKNOWN" in validate_plan(p, settings).codes()
    p = plan()
    p.register["dialogue"] = "english"
    assert "E-PLAN-REGISTER" in validate_plan(p, settings).codes()


def test_cultural_mismatch_is_flagged(settings):
    p = plan("fes", ("سالم", "عودة"), "خليج العقبة")    # Aqaba names and place in a Fes story
    codes = validate_plan(p, settings).codes()
    assert {"W-PLAN-NAME-OUTSIDE-SETTING", "W-PLAN-PLACE-OUTSIDE-SETTING"} <= codes


def test_story_proxies_are_labelled_and_detect_copying(settings):
    source = "كان يا ما كان في قديم الزمان رجل يبيع الخبز في السوق كل صباح ويعود إلى بيته في المساء"
    copied = "حكاية جديدة: " + source
    original = "خرج سالم إلى خليج العقبة فوجد صندوقًا وسأل عودة: شو هذا الإشي؟ قال: هسّا بنعرف."
    assert sp.copy_overlap(copied, [source])["longest_copied_run_words"] >= 15
    assert sp.copy_overlap(original, [source])["containment"] == 0.0
    s = sp.summarise(original, sources=[source], plan_graph=plan().graph,
                     markers=settings["aqaba"].dialect_markers)
    assert all(v["metric_status"] == "proxy" for v in s.values())
    assert s["dialect_marker_rate"]["per_100_words"] > 0
    assert s["plan_surface_coverage"]["characters_named"] == "2/2"
    assert sp.distinct_n([original, original], 2)["distinct"] < sp.distinct_n([original, source], 2)["distinct"]


def test_valid_plan_has_no_textual_cue_warnings(settings):
    # plans have no source text: a missing connective is not a review item for them
    assert validate_plan(plan(), settings).issues == []


def test_unknown_setting_does_not_hide_other_plan_errors(settings):
    p = plan()
    p.setting = "atlantis"
    p.graph.events[0].modality = Modality.ACTUAL
    assert {"E-PLAN-SETTING-UNKNOWN", "E-PLAN-MODALITY"} <= validate_plan(p, settings).codes()


@pytest.mark.parametrize("words", [0, -5, True, "500"])
def test_target_length_must_be_positive(settings, words):
    p = plan()
    p.target_words = words
    with pytest.raises(ValueError, match="E-PLAN-TARGET-WORDS"):
        build_prompt(p, settings)


def test_plan_without_events_or_characters_is_rejected(settings):
    p = plan()
    p.graph.events, p.graph.relations = [], []
    assert "E-PLAN-EMPTY" in validate_plan(p, settings).codes()
    p = plan()
    p.graph.entities = [e for e in p.graph.entities if e.kind is not EntityKind.CHARACTER]
    p.graph.events, p.graph.relations, p.graph.initial_knowledge = [], [], []
    assert "E-PLAN-NO-CHARACTER" in validate_plan(p, settings).codes()
    p = plan()
    p.graph.events[1].gloss = None             # warning only: build_prompt falls back to the trigger
    assert "W-PLAN-NO-GLOSS" in validate_plan(p, settings).codes()
    assert "\n2. سأل" in build_prompt(p, settings)
