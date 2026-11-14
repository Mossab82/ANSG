"""HNC data model: a hierarchical plan of narrative frames. Torch-free.

A :class:`HierarchicalPlan` is a tree of :class:`NarrativeFrame`s. Each frame has
its own planned Event Graph (``ansg.narrative``), hence its **own entities and its
own story-time axis**; frames meet only through explicit anchors:

* ``opened_by`` — the parent-frame event after which the embedded tale is told;
* ``resumed_by`` — the parent-frame event with which the parent tale resumes,
  which must be the next parent event after ``opened_by``;
* ``narrator`` / ``audience`` — parent-frame characters who tell / hear the tale;
* ``conveys`` — the propositions the audience learns at closure (the point of
  telling the tale). The parent story must *use* at least one of them, otherwise
  the embedded tale has no narrative function.

These are ANSG-HNC research extensions; they are not part of the dissertation's
ANSG architecture (see ``research/hnc/ARCHITECTURE.md``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..narrative.events import EventGraph, Proposition

HPLAN_SCHEMA = "ansg.hnc_plan/1"
PERSPECTIVES = ("first_person", "third_person")


@dataclass
class NarrativeFrame:
    id: str
    graph: EventGraph
    parent: str | None = None
    narrator: str | None = None           # entity id in the parent frame (None: external narrator of the root)
    audience: list[str] = field(default_factory=list)
    perspective: str = "third_person"
    narrator_in_frame: str | None = None  # the narrator's entity inside this frame (first person)
    time_context: str = ""                # e.g. "زمن الحكاية الإطارية", "شباب الراوي"
    function: str = ""                    # why the tale is told (warn, persuade, ransom, explain, ...)
    opened_by: str | None = None
    resumed_by: str | None = None
    conveys: list[Proposition] = field(default_factory=list)
    setting_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "parent": self.parent, "narrator": self.narrator, "audience": list(self.audience),
                "perspective": self.perspective, "narrator_in_frame": self.narrator_in_frame,
                "time_context": self.time_context, "function": self.function, "opened_by": self.opened_by,
                "resumed_by": self.resumed_by, "setting_note": self.setting_note,
                "conveys": [{"predicate": p.predicate, "args": list(p.args), "negated": p.negated}
                            for p in self.conveys],
                "graph": self.graph.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NarrativeFrame":
        return cls(id=d["id"], graph=EventGraph.from_dict(d["graph"]), parent=d.get("parent"),
                   narrator=d.get("narrator"), audience=list(d.get("audience") or []),
                   perspective=d.get("perspective", "third_person"), narrator_in_frame=d.get("narrator_in_frame"),
                   time_context=d.get("time_context", ""), function=d.get("function", ""),
                   opened_by=d.get("opened_by"), resumed_by=d.get("resumed_by"),
                   setting_note=d.get("setting_note", ""),
                   conveys=[Proposition(c["predicate"], tuple(c.get("args") or ()), bool(c.get("negated", False)))
                            for c in d.get("conveys") or []])


@dataclass
class HierarchicalPlan:
    setting: str
    title: str
    frames: list[NarrativeFrame]
    register: dict[str, str] = field(default_factory=lambda: {"narration": "msa", "dialogue": "dialect"})
    target_words: int = 700
    notes: str = ""

    def frame(self, frame_id: str) -> NarrativeFrame | None:
        return next((f for f in self.frames if f.id == frame_id), None)

    def root(self) -> NarrativeFrame | None:
        roots = [f for f in self.frames if f.parent is None]
        return roots[0] if len(roots) == 1 else None

    def children(self, frame_id: str) -> list[NarrativeFrame]:
        return [f for f in self.frames if f.parent == frame_id]

    def depth(self, frame_id: str) -> int:
        d, cur, seen = 1, self.frame(frame_id), set()
        while cur is not None and cur.parent is not None and cur.id not in seen:
            seen.add(cur.id)
            cur = self.frame(cur.parent)
            d += 1
        return d

    def to_dict(self) -> dict[str, Any]:
        return {"schema": HPLAN_SCHEMA, "setting": self.setting, "title": self.title, "register": self.register,
                "target_words": self.target_words, "notes": self.notes,
                "frames": [f.to_dict() for f in self.frames]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HierarchicalPlan":
        if d.get("schema") != HPLAN_SCHEMA:
            raise ValueError(f"not a hierarchical plan ({HPLAN_SCHEMA}): {d.get('schema')!r}")
        return cls(setting=d["setting"], title=d["title"], frames=[NarrativeFrame.from_dict(f) for f in d["frames"]],
                   register=dict(d.get("register") or {"narration": "msa", "dialogue": "dialect"}),
                   target_words=int(d.get("target_words", 700)), notes=d.get("notes", ""))

    @classmethod
    def read(cls, path: str | Path) -> "HierarchicalPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
