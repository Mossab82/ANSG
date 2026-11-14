"""Evidence-grounded story Event Graph (data model). Torch-free.

Specification: ``docs/EVENT_GRAPH_SPECIFICATION.md``. Summary:

* **Entities** — characters, groups, supernatural beings, objects, places — with
  their narrative role and grounded mentions.
* **Events** — a trigger (the verb as written) with an evidence span, thematic-role
  participants (VerbNet role names: Agent, Patient, Theme, Recipient, Location,
  Instrument, ...), modality (actual / negated / hypothetical / reported /
  planned), preconditions and effects as propositions (STRIPS-style fluents),
  knowledge required / gained per character, and value annotations.
* **Relations** between events — BEFORE, CAUSES, ENABLES, MOTIVATES, PREVENTS —
  optionally grounded in a textual cue (e.g. a causal connective).

Every annotation carries ``status`` (``AnnotationStatus``) and ``source``
(``AnnotationSource``) from the ANHC record model, so automatic candidates are
never confused with gold annotation. Offsets are character offsets into the
record text *as stored* (``NarrativeRecord.text``), whose SHA-256 the graph pins.

Nothing in this module extracts events; see ``ansg.narrative.extraction`` for the
(explicitly baseline) candidate generator and ``ansg.narrative.validation`` for
the consistency checks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from ..data.records import AnnotationSource, AnnotationStatus

SCHEMA = "ansg.event_graph/1"


class EntityKind(str, Enum):
    CHARACTER = "character"
    GROUP = "group"
    SUPERNATURAL = "supernatural"   # jinn, ifrit, ... (animate agents in the tales)
    ANIMAL = "animal"
    OBJECT = "object"
    PLACE = "place"
    ABSTRACT = "abstract"

    @property
    def animate(self) -> bool:
        return self in (EntityKind.CHARACTER, EntityKind.GROUP, EntityKind.SUPERNATURAL, EntityKind.ANIMAL)


class Modality(str, Enum):
    ACTUAL = "actual"               # happens in the story world
    NEGATED = "negated"             # explicitly did not happen
    HYPOTHETICAL = "hypothetical"   # conditional, wish, threat, plan of a character
    REPORTED = "reported"           # told by a character; truth not established by the narrator
    PLANNED = "planned"             # an event of a generation plan (no source text)


class RelationKind(str, Enum):
    BEFORE = "BEFORE"         # story-time precedence
    CAUSES = "CAUSES"         # source brings about target (implies BEFORE)
    ENABLES = "ENABLES"       # source makes target possible (implies BEFORE)
    MOTIVATES = "MOTIVATES"   # source gives a character a reason for target (implies BEFORE)
    PREVENTS = "PREVENTS"     # source stops target from happening


TEMPORAL_KINDS = (RelationKind.BEFORE, RelationKind.CAUSES, RelationKind.ENABLES, RelationKind.MOTIVATES)


class KnowledgeChannel(str, Enum):
    INITIAL = "initial"         # known at story start (declared in the graph)
    PERCEPTION = "perception"   # saw / heard the event
    TOLD = "told"               # informed by another entity (needs ``source_entity``)
    INFERENCE = "inference"     # reasoned from evidence
    REVELATION = "revelation"   # supernatural / dream / oracle


@dataclass(frozen=True)
class Evidence:
    """A character span in the record text. ``text`` must equal ``record_text[start:end]``."""
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Proposition:
    """A fluent such as ``has(fisherman, ring)``; arguments are entity ids or literals."""
    predicate: str
    args: tuple[str, ...] = ()
    negated: bool = False

    def positive(self) -> "Proposition":
        return Proposition(self.predicate, self.args, False)

    def key(self) -> tuple[str, tuple[str, ...]]:
        return (self.predicate, self.args)

    def __str__(self) -> str:
        core = f"{self.predicate}({', '.join(self.args)})"
        return f"not {core}" if self.negated else core


@dataclass
class Entity:
    id: str
    name: str
    kind: EntityKind
    narrative_role: str | None = None   # protagonist, antagonist, helper, donor, ... (free label, see spec)
    aliases: list[str] = field(default_factory=list)
    mentions: list[Evidence] = field(default_factory=list)

    def surface_forms(self) -> list[str]:
        return [self.name, *self.aliases]


@dataclass
class Participant:
    role: str                         # VerbNet thematic role name, e.g. "Agent"
    entity: str                       # entity id
    evidence: Evidence | None = None  # explicit mention; None = implicit (e.g. pro-drop subject)
    implicit_reason: str | None = None


@dataclass
class KnowledgeItem:
    entity: str
    proposition: Proposition
    channel: KnowledgeChannel = KnowledgeChannel.PERCEPTION
    source_entity: str | None = None   # who told (channel TOLD)


@dataclass
class ValueAnnotation:
    label: str                         # e.g. hospitality, honesty, keeping a trust (inventory-defined)
    polarity: int = 1                  # +1 upheld / -1 violated
    holder: str | None = None          # entity id whose value it is
    evidence: Evidence | None = None
    status: AnnotationStatus = AnnotationStatus.AUTOMATIC
    source: AnnotationSource = AnnotationSource.AUTOMATIC


@dataclass
class Event:
    id: str
    trigger: str                        # verb as written in the text (or planned lemma)
    evidence: Evidence | None           # sentence/clause span; required unless modality is PLANNED
    discourse_index: int                # order of mention in the text / plan
    participants: list[Participant] = field(default_factory=list)
    modality: Modality = Modality.ACTUAL
    story_time: float | None = None     # optional explicit story-time stamp
    verbnet_class: str | None = None
    preconditions: list[Proposition] = field(default_factory=list)
    effects: list[Proposition] = field(default_factory=list)   # negated = deleted fluent
    knowledge_required: list[KnowledgeItem] = field(default_factory=list)
    knowledge_gained: list[KnowledgeItem] = field(default_factory=list)
    values: list[ValueAnnotation] = field(default_factory=list)
    status: AnnotationStatus = AnnotationStatus.AUTOMATIC
    source: AnnotationSource = AnnotationSource.AUTOMATIC
    gloss: str | None = None            # short paraphrase (annotator note / plan step description)

    def roles(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self.participants:
            out.setdefault(p.role, []).append(p.entity)
        return out


@dataclass
class Relation:
    kind: RelationKind
    source: str
    target: str
    cue: Evidence | None = None          # e.g. connective "فلما", "لأن", "حتى"
    status: AnnotationStatus = AnnotationStatus.AUTOMATIC
    annotation_source: AnnotationSource = AnnotationSource.AUTOMATIC


@dataclass
class EventGraph:
    record_id: str
    text_sha256: str | None              # SHA-256 of the record text; None only for plans
    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    initial_state: list[Proposition] = field(default_factory=list)
    initial_knowledge: list[KnowledgeItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ lookup
    def entity(self, entity_id: str) -> Entity | None:
        return next((e for e in self.entities if e.id == entity_id), None)

    def event(self, event_id: str) -> Event | None:
        return next((e for e in self.events if e.id == event_id), None)

    # ------------------------------------------------------------------ JSON
    def to_dict(self) -> dict[str, Any]:
        def conv(o: Any) -> Any:
            if isinstance(o, Enum):
                return o.value
            if isinstance(o, tuple):
                return [conv(x) for x in o]
            if isinstance(o, list):
                return [conv(x) for x in o]
            if isinstance(o, dict):
                return {k: conv(v) for k, v in o.items()}
            return o
        return {"schema": SCHEMA, **conv(asdict(self))}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EventGraph":
        if d.get("schema") != SCHEMA:
            raise ValueError(f"not an event graph ({SCHEMA}): {d.get('schema')!r}")

        def ev(x: Any) -> Evidence | None:
            return None if x is None else Evidence(int(x["start"]), int(x["end"]), str(x["text"]))

        def prop(x: Mapping[str, Any]) -> Proposition:
            return Proposition(x["predicate"], tuple(x.get("args") or ()), bool(x.get("negated", False)))

        def know(x: Mapping[str, Any]) -> KnowledgeItem:
            return KnowledgeItem(x["entity"], prop(x["proposition"]), KnowledgeChannel(x.get("channel", "perception")),
                                 x.get("source_entity"))

        def value(x: Mapping[str, Any]) -> ValueAnnotation:
            return ValueAnnotation(x["label"], int(x.get("polarity", 1)), x.get("holder"), ev(x.get("evidence")),
                                   AnnotationStatus(x.get("status", "automatic")),
                                   AnnotationSource(x.get("source", "automatic")))

        entities = [Entity(e["id"], e["name"], EntityKind(e["kind"]), e.get("narrative_role"),
                           list(e.get("aliases") or []), [ev(m) for m in e.get("mentions") or []])
                    for e in d.get("entities") or []]
        events = [Event(
            id=e["id"], trigger=e["trigger"], evidence=ev(e.get("evidence")),
            discourse_index=int(e["discourse_index"]),
            participants=[Participant(p["role"], p["entity"], ev(p.get("evidence")), p.get("implicit_reason"))
                          for p in e.get("participants") or []],
            modality=Modality(e.get("modality", "actual")), story_time=e.get("story_time"),
            verbnet_class=e.get("verbnet_class"),
            preconditions=[prop(x) for x in e.get("preconditions") or []],
            effects=[prop(x) for x in e.get("effects") or []],
            knowledge_required=[know(x) for x in e.get("knowledge_required") or []],
            knowledge_gained=[know(x) for x in e.get("knowledge_gained") or []],
            values=[value(x) for x in e.get("values") or []],
            status=AnnotationStatus(e.get("status", "automatic")),
            source=AnnotationSource(e.get("source", "automatic")),
            gloss=e.get("gloss"),
        ) for e in d.get("events") or []]
        relations = [Relation(RelationKind(r["kind"]), r["source"], r["target"], ev(r.get("cue")),
                              AnnotationStatus(r.get("status", "automatic")),
                              AnnotationSource(r.get("annotation_source", "automatic")))
                     for r in d.get("relations") or []]
        return cls(record_id=d["record_id"], text_sha256=d.get("text_sha256"), entities=entities, events=events,
                   relations=relations, initial_state=[prop(x) for x in d.get("initial_state") or []],
                   initial_knowledge=[know(x) for x in d.get("initial_knowledge") or []],
                   metadata=dict(d.get("metadata") or {}))

    @classmethod
    def read(cls, path: str | Path) -> "EventGraph":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
