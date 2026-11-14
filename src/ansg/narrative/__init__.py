"""Narrative semantics: evidence-grounded Event Graph, consistency checks, story plans.

Torch-free. ``events`` (data model), ``validation`` (grounding, causal, state and
epistemic checks), ``extraction`` (annotation-candidate baseline only), ``plan``
(plans for original generation in a cultural setting). Specification:
``docs/EVENT_GRAPH_SPECIFICATION.md``.
"""

from .events import EventGraph
from .validation import ValidationReport, validate_event_graph

__all__ = ["EventGraph", "ValidationReport", "validate_event_graph"]
