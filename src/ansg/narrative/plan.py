"""Story plans for ORIGINAL generation in a chosen cultural setting. Torch-free.

Plan-then-realise: a :class:`StoryPlan` is an Event Graph whose events have
modality ``planned`` (no source text), plus a :class:`CulturalSetting` and a
register policy (narration / dialogue: ``msa`` or ``dialect``). The plan is
validated with the same checks as an annotated story (causality, preconditions,
character knowledge), so incoherent plans are rejected *before* any text is
generated, and narrative planning quality can be evaluated separately from
surface realisation (docs/ar/ANSG_SCIENTIFIC_REVIEW_2026-11.md §5).

``build_prompt`` turns a validated plan into an Arabic instruction for a
pretrained/instruction-tuned language model. It explicitly forbids copying
heritage text; originality is still *measured* afterwards
(``ansg.evaluation.story_proxies.copy_overlap``), never assumed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..utils.config import load_yaml_config
from .events import EventGraph, EntityKind, Modality, RelationKind
from .validation import Issue, ValidationConfig, ValidationReport, mention_matches, validate_event_graph

PLAN_SCHEMA = "ansg.story_plan/1"
REGISTERS = ("msa", "dialect")
DAL_DIALECTS = ("MSA", "Egyptian", "Levantine", "Gulf", "Maghrebi")
DEFAULT_SETTINGS = Path(__file__).resolve().parents[3] / "configs" / "generation" / "settings.yaml"

ROLE_AR = {"Agent": "الفاعل", "Patient": "المفعول به", "Theme": "الموضوع", "Recipient": "المتلقي",
           "Location": "المكان", "Instrument": "الأداة", "Experiencer": "المُدرِك", "Source": "المصدر",
           "Destination": "الوجهة", "Beneficiary": "المستفيد"}
RELATION_AR = {RelationKind.CAUSES: "يؤدي إلى", RelationKind.ENABLES: "يمكّن من",
               RelationKind.MOTIVATES: "يدفع إلى", RelationKind.PREVENTS: "يمنع"}


@dataclass(frozen=True)
class CulturalSetting:
    id: str
    label: str
    country: str
    city: str
    dal_dialect: str
    variety: str
    places: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    dialect_markers: tuple[str, ...] = ()
    default_register: Mapping[str, str] = field(default_factory=lambda: {"narration": "msa", "dialogue": "dialect"})
    review_status: str = "unreviewed_draft"

    def __post_init__(self) -> None:
        if self.dal_dialect not in DAL_DIALECTS:
            raise ValueError(f"{self.id}: dal_dialect must be one of {DAL_DIALECTS}")


def load_settings(path: str | Path = DEFAULT_SETTINGS) -> dict[str, CulturalSetting]:
    data = load_yaml_config(path)
    status = data.get("review_status", "unreviewed_draft")
    out = {}
    for sid, s in (data.get("settings") or {}).items():
        out[sid] = CulturalSetting(
            id=sid, label=s["label"], country=s["country"], city=s["city"], dal_dialect=s["dal_dialect"],
            variety=s["variety"], places=tuple(s.get("places") or ()), names=tuple(s.get("names") or ()),
            dialect_markers=tuple(s.get("dialect_markers") or ()),
            default_register=dict(s.get("default_register") or {"narration": "msa", "dialogue": "dialect"}),
            review_status=status)
    return out


@dataclass
class StoryPlan:
    setting: str
    title: str
    graph: EventGraph
    register: dict[str, str] = field(default_factory=lambda: {"narration": "msa", "dialogue": "dialect"})
    target_words: int = 500
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": PLAN_SCHEMA, "setting": self.setting, "title": self.title, "register": self.register,
                "target_words": self.target_words, "notes": self.notes, "event_graph": self.graph.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "StoryPlan":
        if d.get("schema") != PLAN_SCHEMA:
            raise ValueError(f"not a story plan ({PLAN_SCHEMA}): {d.get('schema')!r}")
        return cls(setting=d["setting"], title=d["title"], graph=EventGraph.from_dict(d["event_graph"]),
                   register=dict(d.get("register") or {"narration": "msa", "dialogue": "dialect"}),
                   target_words=int(d.get("target_words", 500)), notes=d.get("notes"))

    @classmethod
    def read(cls, path: str | Path) -> "StoryPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_plan(plan: StoryPlan, settings: Mapping[str, CulturalSetting]) -> ValidationReport:
    """Event-graph checks (without text grounding) plus setting/register checks."""
    rep = validate_event_graph(plan.graph, None, config=ValidationConfig(require_evidence=False, closed_world=True))
    rep.issues = [i for i in rep.issues if i.code != "W-NO-SOURCE-TEXT"]
    setting = settings.get(plan.setting)
    if setting is None:
        rep.issues.append(Issue("E-PLAN-SETTING-UNKNOWN", "error", f"unknown setting {plan.setting!r}"))
        return rep
    for key in ("narration", "dialogue"):
        if plan.register.get(key) not in REGISTERS:
            rep.issues.append(Issue("E-PLAN-REGISTER", "error", f"register.{key} must be one of {REGISTERS}"))
    if not plan.graph.events:
        rep.issues.append(Issue("E-PLAN-EMPTY", "error", "plan has no events"))
    for ev in plan.graph.events:
        if ev.modality is not Modality.PLANNED:
            rep.issues.append(Issue("E-PLAN-MODALITY", "error", "plan events must have modality 'planned'", ev.id))
        if not ev.gloss:
            rep.issues.append(Issue("W-PLAN-NO-GLOSS", "warning", "planned event has no description", ev.id))
    if not any(e.kind in (EntityKind.CHARACTER, EntityKind.GROUP) for e in plan.graph.entities):
        rep.issues.append(Issue("E-PLAN-NO-CHARACTER", "error", "plan has no character"))
    for ent in plan.graph.entities:
        if ent.kind is EntityKind.CHARACTER and setting.names and not mention_matches(ent.name, setting.names):
            rep.issues.append(Issue("W-PLAN-NAME-OUTSIDE-SETTING", "warning",
                                    f"character name {ent.name!r} is not in the {setting.id} name list", ent.id))
        if ent.kind is EntityKind.PLACE and setting.places and not mention_matches(ent.name, setting.places):
            rep.issues.append(Issue("W-PLAN-PLACE-OUTSIDE-SETTING", "warning",
                                    f"place {ent.name!r} is not a listed place of {setting.id}", ent.id))
    return rep


def build_prompt(plan: StoryPlan, settings: Mapping[str, CulturalSetting]) -> str:
    """Arabic generation instruction from a plan that validated without errors."""
    rep = validate_plan(plan, settings)
    if not rep.ok:
        raise ValueError("plan does not validate: " + "; ".join(f"{i.code} {i.message}" for i in rep.errors))
    s = settings[plan.setting]
    g = plan.graph
    names = {e.id: e.name for e in g.entities}
    narr = "الفصحى السردية" if plan.register["narration"] == "msa" else f"لهجة {s.label}"
    dial = f"لهجة {s.label} الطبيعية" if plan.register["dialogue"] == "dialect" else "الفصحى"
    lines = [
        f"اكتب قصة عربية أصلية بعنوان «{plan.title}» تدور أحداثها في {s.label}.",
        f"لغة السرد: {narr}. لغة الحوار: {dial}.",
        "اجعل الأسماء والأماكن والعلاقات الاجتماعية والسلوك والحوار منسجمة مع هذه البيئة دون مبالغة أو قوالب نمطية.",
        "لا تنقل أي نص من «ألف ليلة وليلة» أو من غيرها؛ استلهم البناء السردي فقط.",
        f"الطول التقريبي: {plan.target_words} كلمة.",
        "",
        "الشخصيات:",
    ]
    for e in g.entities:
        if e.kind.animate:
            role = f" — {e.narrative_role}" if e.narrative_role else ""
            lines.append(f"- {e.name}{role}")
    places = [e.name for e in g.entities if e.kind is EntityKind.PLACE]
    if places:
        lines.append("الأماكن: " + "، ".join(places))
    lines += ["", "تسلسل الأحداث (يجب الالتزام به وبعلاقاته السببية):"]
    for n, eid in enumerate(rep.order, 1):
        ev = g.event(eid)
        roles = "؛ ".join(f"{ROLE_AR.get(p.role, p.role)}: {names.get(p.entity, p.entity)}" for p in ev.participants)
        lines.append(f"{n}. {ev.gloss or ev.trigger}" + (f" ({roles})" if roles else ""))
    links = [r for r in g.relations if r.kind in RELATION_AR]
    if links:
        lines += ["", "العلاقات بين الأحداث:"]
        for r in links:
            a, b = g.event(r.source), g.event(r.target)
            lines.append(f"- «{a.gloss or a.trigger}» {RELATION_AR[r.kind]} «{b.gloss or b.trigger}»")
    secrets = [(k.entity, k) for ev in g.events for k in ev.knowledge_gained]
    if secrets:
        lines += ["", "ما تعرفه الشخصيات: لا تجعل أي شخصية تتصرف بناءً على معلومة قبل أن تعرفها في القصة."]
    return "\n".join(lines)
