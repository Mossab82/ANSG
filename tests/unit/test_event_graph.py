"""Event Graph validation, including deliberately WRONG causal/epistemic cases.

The story text below is SYNTHETIC: written for these tests, not taken from any
edition of the Nights or any corpus. Each negative test mutates a correct graph
in exactly one way and asserts the specific error code.
"""
import copy
import json

import pytest

from ansg.data.records import AnnotationStatus
from ansg.knowledge.arabic_verbnet import ArabicVerbNet
from ansg.narrative.events import (
    Entity, EntityKind, Event, EventGraph, Evidence, Modality, Participant, Proposition, Relation, RelationKind,
    ValueAnnotation,
)
from ansg.narrative.extraction import causal_cue_candidates, verb_trigger_candidates
from ansg.narrative.validation import knowledge_item, mention_matches, text_sha256, validate_event_graph

TEXT = ("خرج الصياد سالم إلى البحر في الصباح. فوجد صندوقًا قديمًا بين الصخور. ففتح سالم الصندوق، "
        "فرأى فيه خاتمًا. ثم سأل سالم التاجر عن صاحب الخاتم، فقال التاجر إن الخاتم لحليمة. "
        "فلما عرف سالم ذلك ردّ الخاتم إلى حليمة.")


def span(sub, occurrence=1, within=None):
    """Evidence for the n-th occurrence of ``sub`` (optionally inside the span ``within``)."""
    lo, hi = (within.start, within.end) if within else (0, len(TEXT))
    i = lo - 1
    for _ in range(occurrence):
        i = TEXT.index(sub, i + 1, hi)
    return Evidence(i, i + len(sub), sub)


def P(pred, *args, neg=False):
    return Proposition(pred, args, neg)


def sentence(sub):
    return span(sub)


def build():
    s1 = sentence("خرج الصياد سالم إلى البحر في الصباح.")
    s2 = sentence("فوجد صندوقًا قديمًا بين الصخور.")
    s3 = sentence("ففتح سالم الصندوق،")
    s4 = sentence("فرأى فيه خاتمًا.")
    s5 = sentence("ثم سأل سالم التاجر عن صاحب الخاتم،")
    s6 = sentence("فقال التاجر إن الخاتم لحليمة.")
    s7 = sentence("فلما عرف سالم ذلك ردّ الخاتم إلى حليمة.")
    ents = [
        Entity("salem", "سالم", EntityKind.CHARACTER, "protagonist", ["الصياد"], [span("سالم")]),
        Entity("merchant", "التاجر", EntityKind.CHARACTER, "helper"),
        Entity("halima", "حليمة", EntityKind.CHARACTER, "owner"),
        Entity("box", "الصندوق", EntityKind.OBJECT, aliases=["صندوقًا"]),
        Entity("ring", "الخاتم", EntityKind.OBJECT, aliases=["خاتمًا"]),
        Entity("sea", "البحر", EntityKind.PLACE),
    ]
    ev = [
        Event("e1", "خرج", s1, 0, [Participant("Agent", "salem", span("سالم", within=s1)),
                                   Participant("Destination", "sea", span("البحر", within=s1))],
              effects=[P("at", "salem", "sea")]),
        Event("e2", "وجد", s2, 1, [Participant("Agent", "salem", None, "pro-drop: 3ms verb agreement"),
                                  Participant("Theme", "box", span("صندوقًا", within=s2))],
              preconditions=[P("at", "salem", "sea")], effects=[P("has", "salem", "box")]),
        Event("e3", "فتح", s3, 2, [Participant("Agent", "salem", span("سالم", within=s3)),
                                  Participant("Patient", "box", span("الصندوق", within=s3))],
              preconditions=[P("has", "salem", "box")], effects=[P("open", "box")]),
        Event("e4", "رأى", s4, 3, [Participant("Experiencer", "salem", None, "pro-drop"),
                                  Participant("Theme", "ring", span("خاتمًا", within=s4))],
              preconditions=[P("open", "box")], effects=[P("has", "salem", "ring")]),
        Event("e5", "سأل", s5, 4, [Participant("Agent", "salem", span("سالم", within=s5)),
                                  Participant("Recipient", "merchant", span("التاجر", within=s5))]),
        Event("e6", "قال", s6, 5, [Participant("Agent", "merchant", span("التاجر", within=s6)),
                                  Participant("Recipient", "salem", None, "addressee of the question")],
              knowledge_gained=[knowledge_item("salem", "owner", "ring", "halima", channel="told", source="merchant")]),
        Event("e7", "ردّ", s7, 6, [Participant("Agent", "salem", None, "subject of عرف in the same clause"),
                                  Participant("Theme", "ring", span("الخاتم", within=s7)),
                                  Participant("Recipient", "halima", span("حليمة", within=s7))],
              preconditions=[P("has", "salem", "ring")],
              effects=[P("has", "salem", "ring", neg=True), P("has", "halima", "ring")],
              knowledge_required=[knowledge_item("salem", "owner", "ring", "halima")],
              values=[ValueAnnotation("keeping_a_trust", 1, "salem", span("ردّ الخاتم إلى حليمة"))]),
    ]
    rel = [
        Relation(RelationKind.BEFORE, "e1", "e2"),
        Relation(RelationKind.ENABLES, "e2", "e3", span("ففتح")),
        Relation(RelationKind.ENABLES, "e3", "e4", span("فرأى")),
        Relation(RelationKind.CAUSES, "e5", "e6", span("فقال")),
        Relation(RelationKind.ENABLES, "e6", "e7", span("فلما")),
    ]
    return EventGraph("synthetic-test-0001", text_sha256(TEXT), ents, ev, rel,
                      initial_knowledge=[knowledge_item("merchant", "owner", "ring", "halima", channel="initial")])


def codes(graph, text=TEXT, **kw):
    return validate_event_graph(graph, text, **kw).codes()


def errors(graph, text=TEXT, **kw):
    return {i.code for i in validate_event_graph(graph, text, **kw).errors}


# ---------------------------------------------------------------- correct graph
def test_correct_graph_validates():
    rep = validate_event_graph(build(), TEXT)
    assert rep.ok, rep.errors
    assert rep.order == ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
    assert rep.final_state[("has", ("halima", "ring"))] is True
    assert rep.final_state[("has", ("salem", "ring"))] is False
    assert "owner(ring, halima)" in rep.knowledge["salem"]
    # implicit (pro-drop) participants are flagged for review, never silently accepted
    assert "W-IMPLICIT-PARTICIPANT" in rep.codes()


def test_json_roundtrip_preserves_validation():
    g = build()
    g2 = EventGraph.from_dict(json.loads(g.to_json()))
    assert g2.to_dict() == g.to_dict()
    assert validate_event_graph(g2, TEXT).to_dict() == validate_event_graph(g, TEXT).to_dict()
    with pytest.raises(ValueError):
        EventGraph.from_dict({"schema": "other"})


# ---------------------------------------------------------------- grounding
def test_graph_annotated_on_other_text():
    assert "E-TEXT-HASH" in errors(build(), TEXT + " ")


def test_evidence_must_reproduce_the_text():
    g = build()
    ev = g.event("e3").evidence
    g.event("e3").evidence = Evidence(ev.start, ev.end, "ففتح التاجر الصندوق،")
    assert "E-EVIDENCE-TEXT" in errors(g)
    g = build()
    g.event("e3").evidence = Evidence(len(TEXT) - 2, len(TEXT) + 5, "x")
    assert "E-EVIDENCE-SPAN" in errors(g)


def test_missing_evidence_and_trigger_outside_evidence():
    g = build()
    g.event("e2").evidence = None
    assert "E-EVIDENCE-MISSING" in errors(g)
    g = build()
    g.event("e4").trigger = "باع"
    assert "E-TRIGGER-NOT-IN-EVIDENCE" in errors(g)


def test_wrong_agent_mention_is_detected():
    g = build()
    e3 = g.event("e3")
    e3.participants[0] = Participant("Agent", "salem", span("الصندوق", within=e3.evidence))
    assert "E-PARTICIPANT-MISMATCH" in errors(g)


def test_participant_mention_outside_its_event():
    g = build()
    g.event("e5").participants[1] = Participant("Recipient", "merchant", span("التاجر", occurrence=2))
    assert "E-PARTICIPANT-OUTSIDE-EVENT" in errors(g)


def test_inanimate_agent_is_rejected():
    g = build()
    e3 = g.event("e3")
    e3.participants[0] = Participant("Agent", "box", span("الصندوق", within=e3.evidence))
    assert "E-AGENT-INANIMATE" in errors(g)


def test_unknown_entity_and_duplicate_ids():
    g = build()
    g.event("e5").participants.append(Participant("Theme", "ghost", None, "x"))
    g.events.append(copy.deepcopy(g.event("e1")))
    assert {"E-UNKNOWN-ENTITY", "E-DUPLICATE-ID"} <= errors(g)


# ---------------------------------------------------------------- causal / temporal
def test_causal_cycle():
    g = build()
    g.relations.append(Relation(RelationKind.CAUSES, "e7", "e5", status=AnnotationStatus.GOLD))  # e5->e6->e7->e5
    rep = validate_event_graph(g, TEXT)
    assert "E-TEMPORAL-CYCLE" in {i.code for i in rep.errors} and rep.order == []


def test_cause_stamped_after_its_effect():
    g = build()
    g.event("e5").story_time, g.event("e6").story_time = 5.0, 1.0
    assert {"E-STORY-TIME-CONFLICT", "E-TEMPORAL-CYCLE"} <= errors(g)


def test_self_relation_and_unknown_event():
    g = build()
    g.relations += [Relation(RelationKind.CAUSES, "e3", "e3"), Relation(RelationKind.BEFORE, "e3", "e99")]
    assert {"E-RELATION-SELF", "E-RELATION-UNKNOWN-EVENT"} <= errors(g)


def test_cause_and_prevent_contradiction():
    g = build()
    g.relations.append(Relation(RelationKind.PREVENTS, "e5", "e6"))
    assert {"E-RELATION-CONTRADICTION", "E-PREVENTED-EVENT-OCCURRED"} <= errors(g)


def test_prevented_event_that_happens():
    g = build()
    g.relations.append(Relation(RelationKind.PREVENTS, "e1", "e2"))
    assert "E-PREVENTED-EVENT-OCCURRED" in errors(g)
    g.event("e2").modality = Modality.HYPOTHETICAL   # a prevented *possibility* is fine
    assert "E-PREVENTED-EVENT-OCCURRED" not in errors(g)


def test_caused_event_that_did_not_happen():
    g = build()
    g.event("e6").modality = Modality.NEGATED
    assert "E-CAUSED-EVENT-DID-NOT-OCCUR" in errors(g)


def test_causal_claim_without_cue_is_flagged():
    g = build()
    g.relations.append(Relation(RelationKind.CAUSES, "e1", "e5"))
    rep = validate_event_graph(g, TEXT)
    assert rep.ok and "W-CAUSE-NO-CUE" in rep.codes()


def test_enabling_without_mechanism_is_flagged():
    g = build()
    g.relations.append(Relation(RelationKind.ENABLES, "e5", "e3", span("ثم")))
    assert "W-ENABLES-WITHOUT-MECHANISM" in codes(g)


# ---------------------------------------------------------------- state
def test_violated_precondition():
    g = build()
    g.event("e3").effects = [P("open", "box", neg=True)]   # box stays closed, yet the ring is seen inside
    assert "E-PRECONDITION-VIOLATED" in errors(g)


def test_unsupported_precondition_is_a_warning():
    g = build()
    g.event("e5").preconditions = [P("in_market", "salem")]
    rep = validate_event_graph(g, TEXT)
    assert rep.ok and "W-PRECONDITION-UNSUPPORTED" in rep.codes()


def test_effect_conflict():
    g = build()
    g.event("e4").effects += [P("has", "salem", "ring", neg=True)]
    assert "E-EFFECT-CONFLICT" in errors(g)


# ---------------------------------------------------------------- epistemic
def test_character_acts_before_learning():
    g = build()
    g.relations = [r for r in g.relations if not (r.source == "e6" and r.target == "e7")]
    g.relations.append(Relation(RelationKind.BEFORE, "e7", "e6", status=AnnotationStatus.GOLD))
    assert "E-KNOWLEDGE-PREMATURE" in errors(g)


def test_character_acts_on_knowledge_never_acquired():
    g = build()
    g.event("e6").knowledge_gained = []
    assert "E-KNOWLEDGE-UNSOURCED" in errors(g)


def test_teller_must_know_what_they_tell():
    g = build()
    g.initial_knowledge = []
    assert "E-TOLD-BY-IGNORANT" in errors(g)
    g = build()
    g.event("e6").knowledge_gained[0].source_entity = None
    assert "E-TOLD-WITHOUT-SOURCE" in errors(g)


def test_participants_observe_effects_unless_disabled():
    from ansg.narrative.validation import ValidationConfig
    g = build()
    g.event("e7").knowledge_required.append(knowledge_item("salem", "has", "salem", "ring"))
    assert validate_event_graph(g, TEXT).ok           # salem saw the ring (Experiencer of e4)
    cfg = ValidationConfig(participants_observe_effects=False)
    assert "E-KNOWLEDGE-UNSOURCED" in {i.code for i in validate_event_graph(g, TEXT, config=cfg).errors}


def test_reported_events_do_not_change_the_world():
    g = build()
    g.event("e4").modality = Modality.REPORTED   # "he said he saw a ring": not established by the narrator
    rep = validate_event_graph(g, TEXT)
    assert "E-PRECONDITION-VIOLATED" not in rep.codes()
    # ... so returning the ring now rests on an unestablished fact and is flagged for review
    assert any(i.code == "W-PRECONDITION-UNSUPPORTED" and i.ref == "e7" for i in rep.issues)


# ---------------------------------------------------------------- VerbNet
@pytest.fixture
def verbnet(fixtures):
    return ArabicVerbNet.from_directory(fixtures / "verbnet")


VN_TEXT = "كتب الولد الرسالة."


def vn_graph(agent_kind=EntityKind.CHARACTER, role="Theme"):
    ev = Evidence(0, len(VN_TEXT), VN_TEXT)
    ents = [Entity("boy", "الولد", agent_kind), Entity("letter", "الرسالة", EntityKind.OBJECT)]
    e = Event("v1", "كتب", ev, 0, [Participant("Agent", "boy", Evidence(4, 9, "الولد")),
                                   Participant(role, "letter", Evidence(10, 17, "الرسالة"))],
              verbnet_class="test-write-1")
    return EventGraph("synthetic-vn", text_sha256(VN_TEXT), ents, [e])


def test_verbnet_roles_and_restrictions(verbnet):
    assert validate_event_graph(vn_graph(), VN_TEXT, verbnet=verbnet).ok
    assert "E-ROLE-NOT-IN-CLASS" in errors(vn_graph(role="Instrument"), VN_TEXT, verbnet=verbnet)
    assert "E-SELRESTR" in errors(vn_graph(agent_kind=EntityKind.OBJECT), VN_TEXT, verbnet=verbnet)
    g = vn_graph()
    g.events[0].verbnet_class = "no-such-class"
    assert "E-VERBNET-CLASS-UNKNOWN" in errors(g, VN_TEXT, verbnet=verbnet)


# ---------------------------------------------------------------- matching and candidates
def test_mention_matching_handles_clitics_and_diacritics():
    assert mention_matches("والصيّاد", ["الصياد"])
    assert mention_matches("بالخاتم", ["الخاتم"])
    assert not mention_matches("التاجر", ["الصياد", "سالم"])


def test_candidates_are_automatic_and_grounded(verbnet):
    text = "كتب الولد الرسالة ثم سجل الموعد. وكتب أخوه كذلك."
    cands = verb_trigger_candidates(text, verbnet)
    assert [c.trigger for c in cands] == ["كتب", "سجل", "وكتب"]
    for c in cands:
        assert c.status is AnnotationStatus.AUTOMATIC and text[c.evidence.start:c.evidence.end] == c.evidence.text
        assert c.participants == []       # no participant is ever guessed
    cues = causal_cue_candidates(TEXT)
    assert [(c.cue, c.evidence.text) for c in cues] == [("فلما", "فلما")]


def test_entity_mention_and_value_polarity():
    g = build()
    g.entity("salem").mentions.append(span("التاجر"))
    g.event("e7").values[0].polarity = 0
    assert {"E-MENTION-MISMATCH", "E-VALUE-POLARITY"} <= errors(g)


def test_every_documented_error_code_is_exercised():
    """Guards docs/EVENT_GRAPH_SPECIFICATION.md §4 against drifting from the tests."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = (root / "docs" / "EVENT_GRAPH_SPECIFICATION.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"\bE-[A-Z-]+[A-Z]\b", spec))
    tests = "".join((root / "tests" / "unit" / n).read_text(encoding="utf-8")
                    for n in ("test_event_graph.py", "test_story_plan.py"))
    code = (root / "src" / "ansg" / "narrative" / "validation.py").read_text(encoding="utf-8")
    assert documented and all(c in code for c in documented)
    assert sorted(c for c in documented if c not in tests) == []
