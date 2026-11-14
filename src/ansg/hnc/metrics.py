"""Deterministic structural metrics for nested narratives. Torch-free.

They are computed on a **realised trace**: the sequence of (operation, frame, event,
level, narrator) observed in a generated story. For a controlled run (variants B/C)
the controller's own trace is perfect *by construction*, so the scientifically
meaningful input is a trace **annotated from the generated text** by people who
did not see the system label (``research/hnc/EXPERIMENT_PROTOCOL.md`` §5). The same
functions apply to every condition A–D.

These are operational definitions written for this project
(``research/hnc/METRICS.md``); they are not established metrics from the literature.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from ..narrative.events import EventGraph
from ..narrative.validation import ValidationConfig, validate_event_graph
from .model import HierarchicalPlan

KNOWLEDGE_CODES = ("E-KNOWLEDGE-PREMATURE", "E-KNOWLEDGE-UNSOURCED", "E-TOLD-BY-IGNORANT", "E-TOLD-WITHOUT-SOURCE")
CAUSAL_CODES = ("E-TEMPORAL-CYCLE", "E-STORY-TIME-CONFLICT", "E-PRECONDITION-VIOLATED",
                "E-PRECONDITION-NOT-ESTABLISHED", "E-PREVENTED-EVENT-OCCURRED", "E-CAUSED-EVENT-DID-NOT-OCCUR",
                "E-RELATION-CONTRADICTION", "E-EFFECT-CONFLICT")


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 6) if den else None


def _steps(trace: Iterable[Mapping[str, Any]] | Any) -> list[dict]:
    if hasattr(trace, "to_dict"):
        steps = trace.to_dict()["steps"]
    elif isinstance(trace, Mapping):
        steps = trace["steps"]          # a serialised Trace
    else:
        steps = list(trace)
    return [dict(s) for s in steps]


def structural_metrics(plan: HierarchicalPlan, trace) -> dict:
    """Level accuracy, resumption, closure, narrator consistency and event coverage of a trace."""
    steps = _steps(trace)
    owner = {e.id: f.id for f in plan.frames for e in f.graph.events}
    narrator = {f.id: f.narrator for f in plan.frames}
    planned = Counter(owner)
    emits = [s for s in steps if s.get("op") == "emit"]
    realised = Counter(s.get("event") for s in emits)

    level_hits = sum(1 for s in emits if s.get("event") in owner
                     and s.get("level") == plan.depth(owner[s["event"]]))
    narrated = [s for s in emits if "narrator" in s and s.get("event") in owner]
    narrator_hits = sum(1 for s in narrated if s["narrator"] == narrator[owner[s["event"]]])

    # closure: LIFO enter/close with all events of the frame realised before its close
    stack: list[str] = []
    closed_ok, embedded = 0, [f for f in plan.frames if f.parent is not None]
    seen_events: set[str] = set()
    resumption_hits, resumption_total = 0, 0
    for i, s in enumerate(steps):
        op = s.get("op")
        if op == "enter":
            stack.append(s.get("frame"))
        elif op == "emit":
            seen_events.add(s.get("event"))
        elif op == "close":
            fid = s.get("frame")
            f = plan.frame(fid)
            if stack and stack[-1] == fid and f is not None and \
                    all(e.id in seen_events for e in f.graph.events):
                closed_ok += 1
            if stack and stack[-1] == fid:
                stack.pop()
            if f is not None and f.parent is not None:
                nxt = next((t for t in steps[i + 1:] if t.get("op") in ("emit", "enter")), None)
                resumption_total += 1
                siblings_left = any(t.get("op") == "enter" and plan.frame(t.get("frame")) is not None
                                    and plan.frame(t.get("frame")).opened_by == f.opened_by
                                    for t in ([nxt] if nxt else []))
                if siblings_left or (nxt and nxt.get("op") == "emit" and nxt.get("event") == f.resumed_by):
                    resumption_hits += 1
    lost = sorted(e for e in planned if realised[e] == 0)
    duplicated = sorted(e for e, n in realised.items() if n > 1 and e in planned)
    unplanned = sorted(e for e in realised if e not in planned)
    return {
        "narrative_level_accuracy": _ratio(level_hits, len(emits)),
        "story_resumption_accuracy": _ratio(resumption_hits, resumption_total),
        "narrative_closure_accuracy": _ratio(closed_ok, len(embedded)),
        "narrator_consistency": _ratio(narrator_hits, len(narrated)),
        "event_coverage": _ratio(sum(1 for e in planned if realised[e] == 1), len(planned)),
        "lost_events": lost, "duplicated_events": duplicated, "unplanned_events": unplanned,
        "metric_status": "deterministic (operational definition, research/hnc/METRICS.md)",
    }


def consistency_counts(graphs: Mapping[str, EventGraph], *, closed_world: bool = False) -> dict:
    """Knowledge and causal violations in Event Graphs annotated from generated text, per frame."""
    out = {}
    for frame_id, g in graphs.items():
        rep = validate_event_graph(g, None, config=ValidationConfig(require_evidence=False, closed_world=closed_world))
        codes = Counter(i.code for i in rep.errors)
        required = sum(len(e.knowledge_required) for e in g.events)
        k = sum(codes[c] for c in KNOWLEDGE_CODES)
        out[frame_id] = {"knowledge_violations": k, "knowledge_requirements": required,
                         "character_knowledge_consistency": None if not required else round(1 - k / required, 6),
                         "causal_violations": sum(codes[c] for c in CAUSAL_CODES),
                         "events": len(g.events)}
    return out
