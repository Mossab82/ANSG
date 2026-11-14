"""Narrative stack machine for nested tales. Torch-free; no learned parameters.

The controller *enforces* a legal traversal of a validated :class:`HierarchicalPlan`:

* ``emit(e)`` — the next planned event of the frame on top of the stack;
* ``enter(f)`` — push an embedded frame, only right after its ``opened_by`` event;
* ``close()`` — pop a frame only when all its events were emitted; its audience
  learns ``conveys``; the parent must then resume with ``resumed_by`` (or open the
  next sibling tale anchored at the same event);
* ``finish()`` — every frame entered and closed, every event emitted exactly once.

Any other transition raises :class:`HNCError`. The resulting :class:`Trace` is the
object the deterministic metrics in :mod:`ansg.hnc.metrics` are computed on.

This is *programmatic control* of narrative levels. It is not evidence that a
language model has learned nested narration (``research/hnc/ARCHITECTURE.md`` §5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .model import HierarchicalPlan
from .validation import HNCReport, validate_plan


class HNCError(RuntimeError):
    """An illegal narrative transition or an invalid plan."""


@dataclass(frozen=True)
class TraceStep:
    op: str                 # start | emit | enter | close | finish
    frame: str
    level: int
    event: str | None = None


@dataclass
class Trace:
    steps: list[TraceStep] = field(default_factory=list)
    knowledge_transfers: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"steps": [asdict(s) for s in self.steps], "knowledge_transfers": self.knowledge_transfers}

    def emitted(self) -> list[tuple[str, str]]:
        return [(s.frame, s.event) for s in self.steps if s.op == "emit"]


@dataclass
class _Context:
    frame: str
    level: int
    order: list[str]
    position: int = 0
    pending_resume: str | None = None      # parent event that must come next after a closure
    anchor: str | None = None              # last emitted event (children open after it)

    @property
    def done(self) -> bool:
        return self.position >= len(self.order)


class NarrativeController:
    def __init__(self, plan: HierarchicalPlan, *, epistemic: bool = True, max_depth: int = 8,
                 report: HNCReport | None = None):
        self.plan = plan
        self.report = report or validate_plan(plan, epistemic=epistemic, max_depth=max_depth)
        if not self.report.ok:
            raise HNCError("plan does not validate: " +
                           "; ".join(f"{i.code} {i.message}" for i in self.report.errors[:5]))
        self.stack: list[_Context] = []
        self.trace = Trace()
        self.entered: set[str] = set()
        self.closed: set[str] = set()
        self.knowledge: dict[str, set] = {}

    # -- helpers
    @property
    def top(self) -> _Context:
        if not self.stack:
            raise HNCError("no open frame")
        return self.stack[-1]

    def _step(self, op: str, frame: str, event: str | None = None) -> None:
        self.trace.steps.append(TraceStep(op, frame, self.plan.depth(frame), event))

    def path(self) -> list[str]:
        return [c.frame for c in self.stack]

    # -- transitions
    def start(self) -> None:
        if self.stack or self.trace.steps:
            raise HNCError("already started")
        root = self.plan.root()
        self.stack.append(_Context(root.id, 1, list(self.report.frame_orders[root.id])))
        self.entered.add(root.id)
        self._step("start", root.id)

    def expected_children(self) -> list[str]:
        ctx = self.top
        return [f.id for f in self.plan.children(ctx.frame)
                if f.opened_by == ctx.anchor and f.id not in self.entered]

    def emit(self, event_id: str) -> None:
        ctx = self.top
        if self.expected_children():
            raise HNCError(f"{event_id!r} emitted before the embedded tale(s) {self.expected_children()} were told")
        if ctx.pending_resume is not None and event_id != ctx.pending_resume:
            raise HNCError(f"frame {ctx.frame} must resume with {ctx.pending_resume!r}, not {event_id!r}")
        if ctx.done or ctx.order[ctx.position] != event_id:
            want = None if ctx.done else ctx.order[ctx.position]
            raise HNCError(f"{event_id!r} is not the next event of frame {ctx.frame} (expected {want!r})")
        ctx.position += 1
        ctx.pending_resume = None
        ctx.anchor = event_id
        self._step("emit", ctx.frame, event_id)

    def enter(self, frame_id: str) -> None:
        f = self.plan.frame(frame_id)
        if f is None:
            raise HNCError(f"unknown frame {frame_id!r}")
        if frame_id in self.entered:
            raise HNCError(f"frame {frame_id} was already told")
        if f.parent != self.top.frame:
            raise HNCError(f"frame {frame_id} is embedded in {f.parent}, not in the open frame {self.top.frame}")
        if self.top.anchor != f.opened_by:
            raise HNCError(f"frame {frame_id} opens after {f.opened_by!r}; last event is {self.top.anchor!r}")
        self.entered.add(frame_id)
        self.stack.append(_Context(frame_id, len(self.stack) + 1, list(self.report.frame_orders[frame_id])))
        self._step("enter", frame_id)

    def close(self) -> None:
        ctx = self.top
        f = self.plan.frame(ctx.frame)
        if f.parent is None:
            raise HNCError("the root frame is finished, not closed")
        if not ctx.done:
            raise HNCError(f"frame {ctx.frame} closed with {len(ctx.order) - ctx.position} event(s) not told")
        open_children = [c.id for c in self.plan.children(ctx.frame) if c.id not in self.closed]
        if open_children:
            raise HNCError(f"frame {ctx.frame} closed before its embedded tales {open_children}")
        self.stack.pop()
        self.closed.add(ctx.frame)
        for a in f.audience:
            self.knowledge.setdefault(a, set()).update(p.key() for p in f.conveys)
        self.trace.knowledge_transfers.append({"frame": f.id, "narrator": f.narrator, "audience": list(f.audience),
                                               "conveys": [str(p) for p in f.conveys]})
        self._step("close", f.id)
        parent = self.top
        if not self.expected_children():
            parent.pending_resume = f.resumed_by

    def finish(self) -> Trace:
        if len(self.stack) != 1:
            raise HNCError(f"open frames remain: {self.path()}")
        ctx = self.top
        if not ctx.done or self.expected_children():
            raise HNCError("root frame finished before all its events and tales were told")
        missing = [f.id for f in self.plan.frames if f.id not in self.closed and f.parent is not None]
        if missing:
            raise HNCError(f"embedded tales never told: {missing}")
        self.stack.pop()
        self._step("finish", ctx.frame)
        return self.trace


def schedule(plan: HierarchicalPlan, *, epistemic: bool = True) -> list[tuple[str, str | None]]:
    """The canonical legal operation sequence ``[(op, arg)]`` for a valid plan (depth-first)."""
    ctl = NarrativeController(plan, epistemic=epistemic)
    ops: list[tuple[str, str | None]] = [("start", None)]
    ctl.start()

    def run_frame() -> None:
        while True:
            for child in sorted(ctl.expected_children(), key=lambda c: [f.id for f in plan.frames].index(c)):
                ops.append(("enter", child))
                ctl.enter(child)
                run_frame()
                ops.append(("close", None))
                ctl.close()
            if ctl.top.done:
                return
            nxt = ctl.top.order[ctl.top.position]
            ops.append(("emit", nxt))
            ctl.emit(nxt)

    run_frame()
    ops.append(("finish", None))
    ctl.finish()
    return ops


def replay(plan: HierarchicalPlan, ops: list[tuple[str, str | None]], *, epistemic: bool = True) -> Trace:
    """Apply an operation sequence; raises :class:`HNCError` on the first illegal transition."""
    ctl = NarrativeController(plan, epistemic=epistemic)
    for op, arg in ops:
        if op == "start":
            ctl.start()
        elif op == "emit":
            ctl.emit(arg)
        elif op == "enter":
            ctl.enter(arg)
        elif op == "close":
            ctl.close()
        elif op == "finish":
            return ctl.finish()
        else:
            raise HNCError(f"unknown operation {op!r}")
    raise HNCError("operation sequence ended without finish")
