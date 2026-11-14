"""Realising a hierarchical plan with a text generator: conditions A–D. Torch-free.

``generate_fn(prompt, segment_index) -> str`` is any text generator. The runtime
adapter in :mod:`ansg.hnc.integration` wraps ``ansg.generation.runtime``; tests use
deterministic fakes.

* **A** — ``hnc_enabled=false``: ONE call with :func:`flat_prompt` (the whole plan
  as nested prose instructions). No HNC code runs between the prompt and the text.
* **B** — HNC: validated plan (structure + knowledge), controller-driven
  segment-by-segment generation with level context and explicit resumption.
* **C** — HNC level tracking only: controller-driven segments, plan validated
  WITHOUT the knowledge layer.
* **D** — knowledge checks only: the flattened plan (``validation.flatten``) must
  pass the Event Graph validator; generation is ONE call with :func:`flat_prompt`.

Internal control data (frame ids, event ids, levels) is never put into prompts
in a form that names the controller, and the final text is only the
concatenation of generated segments. :func:`control_leaks` checks generated text
for internal identifiers.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Callable

from ..narrative.validation import ValidationConfig, validate_event_graph
from .controller import NarrativeController, schedule
from .model import HierarchicalPlan
from .validation import flatten, validate_plan

GenerateFn = Callable[[str, int], str]
VARIANTS = ("A", "B", "C", "D")
ROLE_AR = {"Agent": "الفاعل", "Patient": "المفعول به", "Theme": "الموضوع", "Recipient": "المتلقي",
           "Location": "المكان", "Instrument": "الأداة", "Experiencer": "المُدرِك", "Destination": "الوجهة"}


@dataclass(frozen=True)
class HNCConfig:
    hnc_enabled: bool = False
    level_tracking: bool = True
    epistemic_checks: bool = True
    max_depth: int = 8
    context_chars: int = 600          # tail of the story so far given to the next segment

    @classmethod
    def for_variant(cls, variant: str, **kw) -> "HNCConfig":
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        return {"A": cls(hnc_enabled=False, **kw), "B": cls(hnc_enabled=True, **kw),
                "C": cls(hnc_enabled=True, epistemic_checks=False, **kw),
                "D": cls(hnc_enabled=True, level_tracking=False, **kw)}[variant]

    @property
    def variant(self) -> str:
        if not self.hnc_enabled:
            return "A"
        if self.level_tracking and self.epistemic_checks:
            return "B"
        if self.level_tracking:
            return "C"
        if self.epistemic_checks:
            return "D"
        raise ValueError("hnc_enabled with neither level tracking nor epistemic checks is not a defined condition")


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _names(plan: HierarchicalPlan) -> dict[str, str]:
    return {e.id: e.name for f in plan.frames for e in f.graph.entities}


def _event_line(plan: HierarchicalPlan, frame_id: str, event_id: str) -> str:
    names = _names(plan)
    ev = plan.frame(frame_id).graph.event(event_id)
    roles = "؛ ".join(f"{ROLE_AR.get(p.role, p.role)}: {names.get(p.entity, p.entity)}" for p in ev.participants)
    return (ev.gloss or ev.trigger) + (f" ({roles})" if roles else "")


def _register(plan: HierarchicalPlan, label: str) -> str:
    narr = "الفصحى السردية" if plan.register.get("narration") == "msa" else f"لهجة {label}"
    dial = f"لهجة {label} الطبيعية" if plan.register.get("dialogue") == "dialect" else "الفصحى"
    return f"لغة السرد: {narr}. لغة الحوار: {dial}."


def flat_prompt(plan: HierarchicalPlan, setting_label: str) -> str:
    """One instruction describing the whole nested plan (input of conditions A and D)."""
    order = validate_plan(plan, epistemic=False).frame_orders
    names = _names(plan)
    lines = [f"اكتب قصة عربية أصلية بعنوان «{plan.title}» تدور في {setting_label}.", _register(plan, setting_label),
             "لا تنقل أي نص من «ألف ليلة وليلة» أو من غيرها.", f"الطول التقريبي: {plan.target_words} كلمة.", ""]

    def describe(frame_id: str, indent: str) -> None:
        f = plan.frame(frame_id)
        for eid in order.get(frame_id, []):
            lines.append(f"{indent}- {_event_line(plan, frame_id, eid)}")
            for c in plan.children(frame_id):
                if c.opened_by == eid:
                    lines.append(f"{indent}  وهنا يحكي {names.get(c.narrator, c.narrator)} حكاية "
                                 f"({c.function}؛ زمنها: {c.time_context}) لـ"
                                 f"{'، '.join(names.get(a, a) for a in c.audience)}:")
                    describe(c.id, indent + "    ")
                    lines.append(f"{indent}  ثم تعود القصة إلى حيث توقفت.")
        _ = f

    describe(plan.root().id, "")
    return "\n".join(lines)


def segment_prompt(plan: HierarchicalPlan, setting_label: str, ctl: NarrativeController, frame_id: str,
                   events: list[str], resumed_from: str | None, story_so_far: str, context_chars: int) -> str:
    """Instruction for one segment: where we are in the nesting, who narrates, what happens."""
    names = _names(plan)
    f = plan.frame(frame_id)
    path = [plan.frame(x) for x in ctl.path()]
    lines = [f"تكملة لقصة عربية أصلية بعنوان «{plan.title}» في {setting_label}. {_register(plan, setting_label)}",
             "اكتب التكملة فقط، دون عناوين أو شرح أو إشارة إلى هذه التعليمات."]
    if len(path) > 1:
        chain = " ← ".join(f"{names.get(x.narrator, x.narrator)} يحكي لـ{'، '.join(names.get(a, a) for a in x.audience)}"
                           for x in path[1:])
        lines.append(f"نحن داخل حكاية محكية: {chain}. "
                     f"الراوي هنا: {names.get(f.narrator, f.narrator)}، بصيغة "
                     f"{'المتكلم' if f.perspective == 'first_person' else 'الغائب'}، وزمن هذه الحكاية: {f.time_context}.")
        lines.append("لا تجعل شخصيات هذه الحكاية تعرف ما يجري في القصة التي تُحكى فيها.")
    if resumed_from:
        c = plan.frame(resumed_from)
        lines.append(f"انتهت الحكاية التي رواها {names.get(c.narrator, c.narrator)}، وعُد الآن إلى القصة التي "
                     f"كانت تُروى فيها من حيث توقفت، وأظهر أثر الحكاية على سامعيها ({c.function}).")
    if story_so_far:
        lines.append("آخر ما كُتب: «" + story_so_far[-context_chars:] + "»")
    lines.append("الأحداث المطلوبة الآن بالترتيب:")
    lines += [f"- {_event_line(plan, frame_id, e)}" for e in events]
    upcoming = [c for c in plan.children(frame_id) if events and c.opened_by == events[-1]]
    if upcoming:
        c = upcoming[0]
        lines.append(f"اختم هذا الجزء بتمهيد طبيعي لأن يبدأ {names.get(c.narrator, c.narrator)} حكايته "
                     f"({c.function}).")
    return "\n".join(lines)


def control_leaks(plan: HierarchicalPlan, text: str) -> list[str]:
    """Internal identifiers (frame/event/entity ids) that appear verbatim in generated text."""
    ids = {f.id for f in plan.frames} | {e.id for f in plan.frames for e in f.graph.events} | {"hnc-hear-", "HNC"}
    return sorted(i for i in ids if len(i) >= 3 and i in text)


@dataclass
class Realisation:
    variant: str
    config: HNCConfig
    text: str
    segments: list[dict] = field(default_factory=list)
    trace: dict | None = None
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"hnc": {"enabled": self.config.hnc_enabled, "variant": self.variant, "config": asdict(self.config)},
                "text": self.text, "text_sha256": _sha(self.text), "segments": self.segments,
                "trace": self.trace, "checks": self.checks}


def realise(plan: HierarchicalPlan, generate_fn: GenerateFn, config: HNCConfig, setting_label: str) -> Realisation:
    variant = config.variant
    if variant in ("A", "D"):
        checks: dict = {}
        if variant == "D":
            rep = validate_event_graph(flatten(plan), None, config=ValidationConfig(require_evidence=False,
                                                                                    closed_world=True))
            checks = {"flattened_plan_errors": [f"{i.code} {i.message}" for i in rep.errors]}
            if rep.errors:
                raise ValueError("flattened plan does not validate: " + "; ".join(checks["flattened_plan_errors"][:3]))
        prompt = flat_prompt(plan, setting_label)
        text = generate_fn(prompt, 0)
        checks["control_leaks"] = control_leaks(plan, text)
        return Realisation(variant, config, text,
                           [{"index": 0, "frame": None, "level": None, "events": [], "prompt_sha256": _sha(prompt),
                             "text_sha256": _sha(text)}], None, checks)

    ctl = NarrativeController(plan, epistemic=config.epistemic_checks, max_depth=config.max_depth)
    ops = schedule(plan, epistemic=config.epistemic_checks)
    pieces: list[str] = []
    segments: list[dict] = []
    buffer: list[str] = []
    resumed_from: str | None = None
    last_closed: str | None = None

    def flush() -> None:
        nonlocal buffer, resumed_from
        if not buffer:
            return
        frame_id = ctl.top.frame
        prompt = segment_prompt(plan, setting_label, ctl, frame_id, buffer, resumed_from, "".join(pieces),
                                config.context_chars)
        out = generate_fn(prompt, len(segments))
        segments.append({"index": len(segments), "frame": frame_id, "level": len(ctl.stack), "events": list(buffer),
                         "narrator": plan.frame(frame_id).narrator, "prompt_sha256": _sha(prompt),
                         "text_sha256": _sha(out)})
        pieces.append(out)
        buffer, resumed_from = [], None

    for op, arg in ops:
        if op == "start":
            ctl.start()
        elif op == "emit":
            ctl.emit(arg)
            buffer.append(arg)
            if last_closed:
                resumed_from, last_closed = last_closed, None
        elif op == "enter":
            flush()
            last_closed = None
            ctl.enter(arg)
        elif op == "close":
            flush()
            last_closed = ctl.top.frame
            ctl.close()
        elif op == "finish":
            flush()
            trace = ctl.finish()
    text = "\n\n".join(p.strip() for p in pieces if p.strip())
    return Realisation(variant, config, text, segments, trace.to_dict(),
                       {"control_leaks": control_leaks(plan, text),
                        "plan_validation": {"errors": 0, "epistemic": config.epistemic_checks}})
