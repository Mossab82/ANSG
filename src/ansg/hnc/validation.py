"""Validation of hierarchical plans: structure, anchors, narrative function, cross-level knowledge.

Codes (``research/hnc/METRICS.md`` §1 lists them with their tests):

Structure
  HNC-E-ROOT, HNC-E-PARENT-UNKNOWN, HNC-E-CYCLE, HNC-E-DEPTH, HNC-E-DUPLICATE-EVENT
Anchors and resumption
  HNC-E-ANCHOR-UNKNOWN      opened_by / resumed_by is not an event of the parent frame
  HNC-E-RESUME-ORDER        resumed_by is not the parent event that directly follows opened_by
Narration
  HNC-E-NARRATOR-UNKNOWN    narrator is not an animate entity of the parent frame
  HNC-E-AUDIENCE            no audience, or an audience member unknown / inanimate
  HNC-E-PERSPECTIVE         first-person frame without its narrator inside the frame
  HNC-E-TIME-CONTEXT        embedded frame without a declared time context
Function
  HNC-E-NO-FUNCTION         no stated function, nothing conveyed, or the parent story never uses
                            what the tale conveys (the embedded tale changes nothing)
Isolation
  HNC-E-FOREIGN-ENTITY      a frame's propositions name an entity that exists only in another frame
  HNC-E-CROSS-FRAME-RELATION a relation links events of different frames (time axes are separate)
Knowledge (via the Event Graph validator on each frame, with the hearing step inserted)
  E-KNOWLEDGE-PREMATURE / E-KNOWLEDGE-UNSOURCED / E-TOLD-BY-IGNORANT ... prefixed by the frame id

``epistemic=False`` skips the knowledge layer (ablation C); ``hierarchy=False`` is
handled by :func:`flatten` (ablation D).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..narrative.events import (Event, EventGraph, KnowledgeChannel, KnowledgeItem, Modality, Relation,
                                RelationKind)
from ..narrative.validation import Issue, ValidationConfig, validate_event_graph
from .model import PERSPECTIVES, HierarchicalPlan, NarrativeFrame

HEAR_PREFIX = "hnc-hear-"


@dataclass
class HNCReport:
    issues: list[Issue] = field(default_factory=list)
    frame_orders: dict[str, list[str]] = field(default_factory=dict)   # planned story order per frame
    depths: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {i.code for i in self.issues}


def _hearing_event(child: NarrativeFrame, parent: NarrativeFrame) -> Event:
    opened = parent.graph.event(child.opened_by)
    gained = [KnowledgeItem(a, p, KnowledgeChannel.TOLD, child.narrator) for a in child.audience for p in child.conveys]
    return Event(id=f"{HEAR_PREFIX}{child.id}", trigger="حكى", evidence=None,
                 discourse_index=(opened.discourse_index if opened else 0) + 0.5, participants=[],
                 modality=Modality.PLANNED, knowledge_gained=gained,
                 gloss=f"{child.narrator} يحكي الحكاية الداخلية {child.id}")


def augmented_graph(plan: HierarchicalPlan, frame: NarrativeFrame) -> EventGraph:
    """The frame's graph with one 'hearing' step per embedded tale between its anchors."""
    g = copy.deepcopy(frame.graph)
    for child in plan.children(frame.id):
        if not (g.event(child.opened_by or "") and g.event(child.resumed_by or "")):
            continue
        g.events.append(_hearing_event(child, frame))
        g.relations += [Relation(RelationKind.BEFORE, child.opened_by, f"{HEAR_PREFIX}{child.id}"),
                        Relation(RelationKind.BEFORE, f"{HEAR_PREFIX}{child.id}", child.resumed_by)]
        # the narrator knows the tale they tell (it is in their past or they heard it earlier)
        if child.narrator:
            g.initial_knowledge += [KnowledgeItem(child.narrator, p, KnowledgeChannel.INITIAL) for p in child.conveys]
    return g


def validate_plan(plan: HierarchicalPlan, *, epistemic: bool = True, max_depth: int = 8,
                  closed_world: bool = True) -> HNCReport:
    rep = HNCReport()

    def add(code: str, msg: str, ref: str | None = None, severity: str = "error") -> None:
        rep.issues.append(Issue(code, severity, msg, ref))

    ids = [f.id for f in plan.frames]
    if len(set(ids)) != len(ids):
        add("HNC-E-DUPLICATE-FRAME", "duplicate frame ids")
    roots = [f for f in plan.frames if f.parent is None]
    if len(roots) != 1:
        add("HNC-E-ROOT", f"exactly one root frame required, found {len(roots)}")
    frames = {f.id: f for f in plan.frames}
    event_owner: dict[str, str] = {}
    for f in plan.frames:
        for e in f.graph.events:
            if e.id in event_owner:
                add("HNC-E-DUPLICATE-EVENT", f"event id {e.id!r} used in {event_owner[e.id]} and {f.id}", f.id)
            event_owner[e.id] = f.id
    for f in plan.frames:
        if f.parent is not None and f.parent not in frames:
            add("HNC-E-PARENT-UNKNOWN", f"parent {f.parent!r} does not exist", f.id)
        seen, cur = {f.id}, f
        while cur.parent in frames:
            cur = frames[cur.parent]
            if cur.id in seen:
                add("HNC-E-CYCLE", "frame ancestry forms a cycle", f.id)
                break
            seen.add(cur.id)
        else:
            rep.depths[f.id] = plan.depth(f.id)
            if rep.depths[f.id] > max_depth:
                add("HNC-E-DEPTH", f"nesting depth {rep.depths[f.id]} exceeds max_depth {max_depth}", f.id)
    if rep.errors:
        return rep

    # per-frame event-graph validation (with hearing steps), which also yields story orders
    for f in plan.frames:
        graph = augmented_graph(plan, f) if epistemic else f.graph
        cfg = ValidationConfig(require_evidence=False, closed_world=closed_world)
        if not epistemic:
            graph = copy.deepcopy(graph)
            for e in graph.events:
                e.knowledge_required, e.knowledge_gained = [], []
            graph.initial_knowledge = []
        frep = validate_event_graph(graph, None, config=cfg)
        for i in frep.issues:
            if i.code == "W-NO-SOURCE-TEXT":
                continue
            rep.issues.append(Issue(i.code, i.severity, f"[{f.id}] {i.message}", f"{f.id}:{i.ref}"))
        rep.frame_orders[f.id] = [e for e in frep.order if not e.startswith(HEAR_PREFIX)]
        own = {e.id for e in f.graph.events}
        for r in f.graph.relations:
            for end in (r.source, r.target):
                if end not in own and end in event_owner:
                    add("HNC-E-CROSS-FRAME-RELATION", f"relation {r.kind.value} links {end!r} of frame "
                        f"{event_owner[end]}: frames have separate time axes", f.id)
        local = {e.id for e in f.graph.entities}
        foreign = {e.id for g in plan.frames if g.id != f.id for e in g.graph.entities} - local
        props = [p for e in f.graph.events for p in e.preconditions + e.effects] + \
                [k.proposition for e in f.graph.events for k in e.knowledge_required + e.knowledge_gained]
        for p in props:
            bad = [a for a in p.args if a in foreign]
            if bad:
                add("HNC-E-FOREIGN-ENTITY", f"{p} names {bad} from another frame", f.id)

    # anchors, narration and function of every embedded frame
    for f in plan.frames:
        if f.parent is None:
            continue
        parent = frames[f.parent]
        order = rep.frame_orders.get(parent.id, [])
        if f.opened_by not in order or f.resumed_by not in order:
            add("HNC-E-ANCHOR-UNKNOWN", "opened_by/resumed_by must be events of the parent frame", f.id)
        elif order.index(f.resumed_by) != order.index(f.opened_by) + 1:
            add("HNC-E-RESUME-ORDER", f"parent resumes with {f.resumed_by!r}, which is not the event directly "
                f"after {f.opened_by!r}", f.id)
        pents = {e.id: e for e in parent.graph.entities}
        if f.narrator not in pents or not pents[f.narrator].kind.animate:
            add("HNC-E-NARRATOR-UNKNOWN", f"narrator {f.narrator!r} is not an animate entity of {parent.id}", f.id)
        if not f.audience or any(a not in pents or not pents[a].kind.animate for a in f.audience):
            add("HNC-E-AUDIENCE", "audience must be non-empty animate entities of the parent frame", f.id)
        if f.perspective not in PERSPECTIVES:
            add("HNC-E-PERSPECTIVE", f"perspective must be one of {PERSPECTIVES}", f.id)
        elif f.perspective == "first_person" and not f.graph.entity(f.narrator_in_frame or ""):
            add("HNC-E-PERSPECTIVE", "first-person tale needs narrator_in_frame, an entity of this frame", f.id)
        if not f.time_context.strip():
            add("HNC-E-TIME-CONTEXT", "embedded frame needs a declared time context", f.id)
        conveyed = {p.key() for p in f.conveys}
        used = False
        if f.resumed_by in order:
            for eid in order[order.index(f.resumed_by):]:
                ev = parent.graph.event(eid)
                if any(k.entity in f.audience and k.proposition.key() in conveyed for k in ev.knowledge_required):
                    used = True
                    break
        if not f.function.strip() or not f.conveys or not used:
            add("HNC-E-NO-FUNCTION", "embedded tale needs a stated function and conveyed knowledge that an audience "
                "member uses at or after the resumption point", f.id)
    return rep


def flatten(plan: HierarchicalPlan) -> EventGraph:
    """All frames in ONE graph (ablation D): knowledge checks without hierarchical management.

    Event and entity ids are kept (they are unique across frames); hearing steps are inserted;
    frame boundaries, narrators and resumption are dropped.
    """
    g = EventGraph(record_id=f"flat:{plan.title}", text_sha256=None)
    for f in plan.frames:
        aug = augmented_graph(plan, f)
        g.entities += [e for e in aug.entities if e.id not in {x.id for x in g.entities}]
        g.events += aug.events
        g.relations += aug.relations
        g.initial_state += aug.initial_state
        g.initial_knowledge += aug.initial_knowledge
    return g
