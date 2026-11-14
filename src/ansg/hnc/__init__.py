"""Hierarchical Narrative Controller (HNC) — OPTIONAL research extension of ANSG.

ANSG-HNC is not part of the dissertation's ANSG architecture and its results are
never reported as dissertation results. It is torch-free, has no trainable
parameters and does not modify CSAM, DAL, the backbone or any ANSG equation.
Disabled by default (``hnc_enabled = false``, ``configs/hnc/default.yaml``).

Modules: ``model`` (hierarchical plans), ``validation`` (structure, anchors,
function, cross-level knowledge), ``controller`` (narrative stack machine),
``realize`` (conditions A–D), ``metrics`` (deterministic structural metrics),
``integration`` (adapter to ``ansg.generation.runtime``). Research documentation:
``research/hnc/``.
"""

from .model import HierarchicalPlan, NarrativeFrame
from .realize import HNCConfig

__all__ = ["HierarchicalPlan", "NarrativeFrame", "HNCConfig"]
