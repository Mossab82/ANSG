"""Consistency checks for an Event Graph. Torch-free; deterministic.

Each check returns :class:`Issue` objects with a stable ``code`` (catalogue in
``docs/EVENT_GRAPH_SPECIFICATION.md`` §4). Errors mean the graph contradicts its
text or itself; warnings mean a claim is not yet supported and needs review.

1. **Grounding** — the graph pins the record text by SHA-256; every evidence span
   must reproduce the text exactly; the trigger must occur in its event's evidence;
   an explicit participant mention must lie inside the event's evidence and match
   one of the entity's surface forms (clitic- and diacritic-insensitive).
2. **Participants** — Agent/Experiencer must be animate; with an ArabicVerbNet index,
   roles must belong to the event's class (with inheritance) and respect its
   ``+/- animate`` selectional restrictions. No VerbNet data is bundled.
3. **Relations** — unknown ids, self-relations, contradictory pairs (CAUSES and
   PREVENTS), cycles in story time (CAUSES/ENABLES/MOTIVATES imply BEFORE),
   explicit ``story_time`` stamps contradicting an edge, causes of events that did
   not happen, prevented events that happened anyway.
4. **State simulation** — events are replayed in a story-time order that respects
   every temporal edge (ties: ``story_time``, then discourse order). Preconditions
   must hold (three-valued: true / false / unknown), effects may not both add and
   delete a fluent.
5. **Epistemic** — a character may act on a proposition only after learning it
   (initially, by perception, by being told by someone who knew it, by inference,
   or by revelation); participants observe the effects of events they take part in
   (configurable).
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any, Iterable

from ..data.nights import FORMULA_MATCH_FOLDING
from ..data.normalization import normalize
from .events import (TEMPORAL_KINDS, Event, EventGraph, Evidence, KnowledgeChannel, KnowledgeItem, Modality,
                     Proposition, RelationKind)

ERROR, WARNING = "error", "warning"
PROCLITICS = ("و", "ف", "ب", "ل", "ك")
ANIMATE_ROLES = ("Agent", "Experiencer")
OBSERVER_ROLES = ("Agent", "Experiencer", "Recipient", "Patient", "Beneficiary")
STATE_CHANGING = (Modality.ACTUAL, Modality.PLANNED)


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str
    message: str
    ref: str | None = None


@dataclass
class ValidationConfig:
    require_evidence: bool = True                 # False for generation plans
    participants_observe_effects: bool = True
    observer_roles: tuple[str, ...] = OBSERVER_ROLES
    value_labels: frozenset[str] | None = None    # inventory; None = not validated
    closed_world: bool = False                    # plans: a precondition the plan never establishes is an error


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    order: list[str] = field(default_factory=list)          # story-time order used for simulation
    final_state: dict[tuple[str, tuple[str, ...]], bool] = field(default_factory=dict)
    knowledge: dict[str, list[str]] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": len(self.errors), "warnings": len(self.warnings),
                "issues": [i.__dict__ for i in self.issues], "order": self.order,
                "knowledge": self.knowledge}


# ---------------------------------------------------------------------- text helpers
def fold(text: str) -> str:
    """Diacritic/hamza/tatweel-insensitive form used only for *matching*."""
    return normalize(text, FORMULA_MATCH_FOLDING).text.strip()


def _strip_clitics(word: str) -> str:
    w = word
    for _ in range(2):
        if len(w) > 3 and w[0] in PROCLITICS:
            w = w[1:]
    if len(w) > 3 and w.startswith("ال"):
        w = w[2:]
    return w


def mention_matches(mention: str, forms: Iterable[str]) -> bool:
    """True if ``mention`` names one of ``forms`` up to proclitics, article and diacritics."""
    m = " ".join(_strip_clitics(t) for t in fold(mention).split())
    for form in forms:
        f = " ".join(_strip_clitics(t) for t in fold(form).split())
        if f and (m == f):
            return True
    return False


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------- validator
class _Checker:
    def __init__(self, graph: EventGraph, text: str | None, verbnet: Any, cfg: ValidationConfig):
        self.g, self.text, self.vn, self.cfg = graph, text, verbnet, cfg
        self.rep = ValidationReport()
        self.events = {e.id: e for e in graph.events}
        self.entities = {e.id: e for e in graph.entities}

    def add(self, code: str, severity: str, message: str, ref: str | None = None) -> None:
        self.rep.issues.append(Issue(code, severity, message, ref))

    # -- 0. structure
    def structure(self) -> None:
        for kind, items in (("entity", self.g.entities), ("event", self.g.events)):
            seen: set[str] = set()
            for item in items:
                if item.id in seen:
                    self.add("E-DUPLICATE-ID", ERROR, f"duplicate {kind} id {item.id!r}", item.id)
                seen.add(item.id)
        for ev in self.g.events:
            for p in ev.participants:
                if p.entity not in self.entities:
                    self.add("E-UNKNOWN-ENTITY", ERROR, f"participant {p.role} refers to unknown entity {p.entity!r}", ev.id)
            for k in ev.knowledge_required + ev.knowledge_gained:
                if k.entity not in self.entities:
                    self.add("E-UNKNOWN-ENTITY", ERROR, f"knowledge item refers to unknown entity {k.entity!r}", ev.id)
        for k in self.g.initial_knowledge:
            if k.entity not in self.entities:
                self.add("E-UNKNOWN-ENTITY", ERROR, f"initial knowledge of unknown entity {k.entity!r}")

    # -- 1. grounding
    def _span(self, ev: Evidence, ref: str, what: str) -> bool:
        assert self.text is not None
        if not (0 <= ev.start < ev.end <= len(self.text)):
            self.add("E-EVIDENCE-SPAN", ERROR, f"{what}: span [{ev.start}, {ev.end}) outside the text", ref)
            return False
        if self.text[ev.start:ev.end] != ev.text:
            self.add("E-EVIDENCE-TEXT", ERROR, f"{what}: span text {ev.text!r} != source "
                     f"{self.text[ev.start:ev.end]!r}", ref)
            return False
        return True

    def grounding(self) -> None:
        if self.text is None:
            if self.cfg.require_evidence:
                self.add("W-NO-SOURCE-TEXT", WARNING, "no source text supplied: grounding not checked")
            return
        if self.g.text_sha256 is not None and self.g.text_sha256 != text_sha256(self.text):
            self.add("E-TEXT-HASH", ERROR, "graph was annotated on a different text (SHA-256 differs)")
        for ent in self.g.entities:
            for m in ent.mentions:
                if self._span(m, ent.id, "entity mention") and not mention_matches(m.text, ent.surface_forms()):
                    self.add("E-MENTION-MISMATCH", ERROR, f"mention {m.text!r} does not name {ent.name!r}", ent.id)
        for ev in self.g.events:
            if ev.evidence is None:
                if self.cfg.require_evidence and ev.modality is not Modality.PLANNED:
                    self.add("E-EVIDENCE-MISSING", ERROR, "event has no textual evidence", ev.id)
                continue
            if not self._span(ev.evidence, ev.id, "event evidence"):
                continue
            if fold(ev.trigger) not in fold(ev.evidence.text):
                self.add("E-TRIGGER-NOT-IN-EVIDENCE", ERROR, f"trigger {ev.trigger!r} not found in its evidence", ev.id)
            for p in ev.participants:
                ent = self.entities.get(p.entity)
                if p.evidence is None:
                    self.add("W-IMPLICIT-PARTICIPANT", WARNING, f"{p.role}={p.entity} has no explicit mention"
                             f" ({p.implicit_reason or 'no reason given'}); needs review", ev.id)
                    continue
                if not self._span(p.evidence, ev.id, f"{p.role} mention"):
                    continue
                if not (ev.evidence.start <= p.evidence.start and p.evidence.end <= ev.evidence.end):
                    self.add("E-PARTICIPANT-OUTSIDE-EVENT", ERROR,
                             f"{p.role} mention {p.evidence.text!r} lies outside the event evidence", ev.id)
                if p.implicit_reason:
                    # an anaphor (pronoun, demonstrative) grounded in the clause: resolution needs review
                    self.add("W-ANAPHORIC-PARTICIPANT", WARNING, f"{p.role} mention {p.evidence.text!r} resolved to "
                             f"{p.entity} by coreference ({p.implicit_reason}); needs review", ev.id)
                elif ent is not None and not mention_matches(p.evidence.text, ent.surface_forms()):
                    self.add("E-PARTICIPANT-MISMATCH", ERROR, f"{p.role} mention {p.evidence.text!r} does not "
                             f"name entity {ent.name!r}", ev.id)
            for v in ev.values:
                if v.evidence is not None:
                    self._span(v.evidence, ev.id, f"value {v.label!r}")
        for r in self.g.relations:
            if r.cue is not None:
                self._span(r.cue, f"{r.source}->{r.target}", f"{r.kind.value} cue")

    # -- 2. participants and VerbNet
    def participants(self) -> None:
        for ev in self.g.events:
            for p in ev.participants:
                ent = self.entities.get(p.entity)
                if ent is not None and p.role in ANIMATE_ROLES and not ent.kind.animate:
                    self.add("E-AGENT-INANIMATE", ERROR, f"{p.role} {ent.name!r} is a {ent.kind.value}", ev.id)
            if ev.verbnet_class is None or self.vn is None:
                continue
            cls = self.vn.classes.get(ev.verbnet_class)
            if cls is None:
                self.add("E-VERBNET-CLASS-UNKNOWN", ERROR, f"class {ev.verbnet_class!r} not in the VerbNet index", ev.id)
                continue
            roles = {r.type: r for r in cls.inherited_roles(self.vn)}
            members = {fold(m.verb) for m in cls.members if m.verb}
            if members and fold(ev.trigger) not in members:
                self.add("W-TRIGGER-NOT-CLASS-MEMBER", WARNING, f"trigger {ev.trigger!r} is not a listed member "
                         "form of the class (inflected forms need morphological analysis)", ev.id)
            for p in ev.participants:
                role = roles.get(p.role)
                if role is None:
                    self.add("E-ROLE-NOT-IN-CLASS", ERROR, f"role {p.role!r} not defined for class {cls.class_id}", ev.id)
                    continue
                ent = self.entities.get(p.entity)
                for r in role.restrictions:
                    if r.type == "animate" and ent is not None and (r.value == "+") != ent.kind.animate:
                        self.add("E-SELRESTR", ERROR, f"{p.role} {ent.name!r} violates selectional restriction "
                                 f"{r.value}animate of {cls.class_id}", ev.id)

    # -- 3. relations
    def relations(self) -> dict[str, set[str]]:
        succ: dict[str, set[str]] = {e: set() for e in self.events}
        pairs: dict[tuple[str, str], set[RelationKind]] = {}
        for r in self.g.relations:
            ref = f"{r.kind.value}:{r.source}->{r.target}"
            if r.source not in self.events or r.target not in self.events:
                self.add("E-RELATION-UNKNOWN-EVENT", ERROR, "relation refers to an unknown event", ref)
                continue
            if r.source == r.target:
                self.add("E-RELATION-SELF", ERROR, "an event cannot relate to itself", ref)
                continue
            pairs.setdefault((r.source, r.target), set()).add(r.kind)
            src, tgt = self.events[r.source], self.events[r.target]
            if r.kind in TEMPORAL_KINDS:
                succ[r.source].add(r.target)
                if src.story_time is not None and tgt.story_time is not None and not src.story_time < tgt.story_time:
                    self.add("E-STORY-TIME-CONFLICT", ERROR, f"story_time {src.story_time} of {src.id} is not before "
                             f"{tgt.story_time} of {tgt.id}", ref)
            if r.kind in (RelationKind.CAUSES, RelationKind.ENABLES):
                if tgt.modality is Modality.NEGATED:
                    self.add("E-CAUSED-EVENT-DID-NOT-OCCUR", ERROR, f"{tgt.id} is negated but is said to be caused", ref)
                if src.modality not in STATE_CHANGING:
                    self.add("W-CAUSE-FROM-NONACTUAL", WARNING, f"cause {src.id} is {src.modality.value}", ref)
                if r.cue is None and r.status.value not in ("gold", "silver"):
                    self.add("W-CAUSE-NO-CUE", WARNING, "causal claim without a textual cue or human annotation", ref)
                produced = {p.key() for p in src.effects} | {k.proposition.key() for k in src.knowledge_gained}
                needed = {p.key() for p in tgt.preconditions} | {k.proposition.key() for k in tgt.knowledge_required}
                if r.kind is RelationKind.ENABLES and needed and not produced & needed:
                    self.add("W-ENABLES-WITHOUT-MECHANISM", WARNING,
                             "source produces none of the target's preconditions or required knowledge", ref)
            if r.kind is RelationKind.PREVENTS and src.modality in STATE_CHANGING and tgt.modality in STATE_CHANGING:
                self.add("E-PREVENTED-EVENT-OCCURRED", ERROR, f"{tgt.id} happens although {src.id} prevents it", ref)
        for (s, t), kinds in pairs.items():
            if RelationKind.PREVENTS in kinds and kinds & {RelationKind.CAUSES, RelationKind.ENABLES}:
                self.add("E-RELATION-CONTRADICTION", ERROR, f"{s} both prevents and causes/enables {t}", f"{s}->{t}")
        # story-time stamps also order events without explicit edges
        stamped = sorted((e for e in self.g.events if e.story_time is not None), key=lambda e: e.story_time)
        for a, b in zip(stamped, stamped[1:]):
            if a.story_time < b.story_time:
                succ[a.id].add(b.id)
        return succ

    def order(self, succ: dict[str, set[str]]) -> list[str] | None:
        indeg = {e: 0 for e in succ}
        for s in succ:
            for t in succ[s]:
                indeg[t] += 1

        def key(eid: str) -> tuple:
            e = self.events[eid]
            return (e.story_time if e.story_time is not None else float("inf"), e.discourse_index, eid)

        heap = [key(e) for e, d in indeg.items() if d == 0]
        heapq.heapify(heap)
        out: list[str] = []
        while heap:
            eid = heapq.heappop(heap)[-1]
            out.append(eid)
            for t in succ[eid]:
                indeg[t] -= 1
                if indeg[t] == 0:
                    heapq.heappush(heap, key(t))
        if len(out) != len(succ):
            cyclic = sorted(e for e, d in indeg.items() if d > 0)
            self.add("E-TEMPORAL-CYCLE", ERROR, f"story-time relations form a cycle among {cyclic}")
            return None
        return out

    # -- 4./5. simulation
    def simulate(self, order: list[str]) -> None:
        state: dict[tuple[str, tuple[str, ...]], bool] = {}
        for p in self.g.initial_state:
            state[p.key()] = not p.negated
        knows: dict[str, dict[tuple[str, tuple[str, ...]], bool]] = {e: {} for e in self.entities}
        for k in self.g.initial_knowledge:
            knows.setdefault(k.entity, {})[k.proposition.key()] = not k.proposition.negated
        position = {eid: i for i, eid in enumerate(order)}
        learned_at: dict[tuple[str, tuple, bool], str] = {}
        for eid in order:
            ev = self.events[eid]
            if ev.modality not in STATE_CHANGING:
                continue
            for k in ev.knowledge_gained:
                learned_at.setdefault((k.entity, k.proposition.key(), not k.proposition.negated), ev.id)
            if self.cfg.participants_observe_effects:
                for part in ev.participants:
                    ent = self.entities.get(part.entity)
                    if part.role in self.cfg.observer_roles and ent is not None and ent.kind.animate:
                        for p in ev.effects:
                            learned_at.setdefault((ent.id, p.key(), not p.negated), ev.id)

        for eid in order:
            ev = self.events[eid]
            if ev.modality not in STATE_CHANGING:
                continue
            for p in ev.preconditions:
                value = state.get(p.key())
                if value is None and self.cfg.closed_world:
                    self.add("E-PRECONDITION-NOT-ESTABLISHED", ERROR, f"precondition {p} is not established "
                             "by the initial state or an earlier event", eid)
                elif value is None:
                    self.add("W-PRECONDITION-UNSUPPORTED", WARNING, f"precondition {p} is never established", eid)
                elif value != (not p.negated):
                    self.add("E-PRECONDITION-VIOLATED", ERROR, f"precondition {p} is false at this point", eid)
            for k in ev.knowledge_required:
                truth = not k.proposition.negated
                if knows.get(k.entity, {}).get(k.proposition.key()) != truth:
                    later = learned_at.get((k.entity, k.proposition.key(), truth))
                    if later and later != eid and position.get(later, -1) > position[eid]:
                        self.add("E-KNOWLEDGE-PREMATURE", ERROR, f"{k.entity} acts on {k.proposition} before "
                                 f"learning it in {later}", eid)
                    else:
                        self.add("E-KNOWLEDGE-UNSOURCED", ERROR, f"{k.entity} acts on {k.proposition} but never "
                                 "learns it", eid)
            for k in ev.knowledge_gained:
                if k.channel is KnowledgeChannel.TOLD:
                    if k.source_entity is None:
                        self.add("E-TOLD-WITHOUT-SOURCE", ERROR, f"{k.entity} is told {k.proposition} by nobody", eid)
                    elif knows.get(k.source_entity, {}).get(k.proposition.key()) != (not k.proposition.negated):
                        self.add("E-TOLD-BY-IGNORANT", ERROR, f"{k.source_entity} tells {k.proposition} without "
                                 "knowing it", eid)
            adds = {p.key() for p in ev.effects if not p.negated}
            dels = {p.key() for p in ev.effects if p.negated}
            for key in sorted(adds & dels):
                self.add("E-EFFECT-CONFLICT", ERROR, f"effect both adds and deletes {key}", eid)
            for p in ev.effects:
                state[p.key()] = not p.negated
            for k in ev.knowledge_gained:
                knows.setdefault(k.entity, {})[k.proposition.key()] = not k.proposition.negated
            if self.cfg.participants_observe_effects:
                for part in ev.participants:
                    ent = self.entities.get(part.entity)
                    if part.role in self.cfg.observer_roles and ent is not None and ent.kind.animate:
                        for p in ev.effects:
                            knows.setdefault(ent.id, {})[p.key()] = not p.negated
        self.rep.final_state = state
        self.rep.knowledge = {e: sorted(f"{'' if v else 'not '}{k[0]}({', '.join(k[1])})" for k, v in kn.items())
                              for e, kn in knows.items()}

    def values(self) -> None:
        for ev in self.g.events:
            for v in ev.values:
                if self.cfg.value_labels is not None and v.label not in self.cfg.value_labels:
                    self.add("W-VALUE-LABEL-UNKNOWN", WARNING, f"value label {v.label!r} not in the inventory", ev.id)
                if v.holder is not None and v.holder not in self.entities:
                    self.add("E-UNKNOWN-ENTITY", ERROR, f"value holder {v.holder!r} unknown", ev.id)
                if v.polarity not in (-1, 1):
                    self.add("E-VALUE-POLARITY", ERROR, "polarity must be +1 or -1", ev.id)

    def run(self) -> ValidationReport:
        self.structure()
        self.grounding()
        self.participants()
        self.values()
        succ = self.relations()
        order = self.order(succ)
        if order is not None:
            self.rep.order = order
            self.simulate(order)
        return self.rep


def validate_event_graph(graph: EventGraph, text: str | None = None, *, verbnet: Any = None,
                         config: ValidationConfig | None = None) -> ValidationReport:
    """Validate ``graph`` against its source ``text`` (and optionally an ``ArabicVerbNet`` index)."""
    return _Checker(graph, text, verbnet, config or ValidationConfig()).run()


def knowledge_item(entity: str, predicate: str, *args: str, channel: str = "perception",
                   source: str | None = None, negated: bool = False) -> KnowledgeItem:
    """Convenience constructor used by tests and annotation tools."""
    return KnowledgeItem(entity, Proposition(predicate, tuple(args), negated), KnowledgeChannel(channel), source)


__all__ = ["Issue", "ValidationConfig", "ValidationReport", "validate_event_graph", "fold", "mention_matches",
           "text_sha256", "knowledge_item", "Event"]
