"""Linguistic and cultural knowledge resources: ArabicVerbNet, ArabicNameNet, CKG.

Backward compatibility (Issue #11): the former flat module ``ansg/knowledge.py``
of the model branch now lives in :mod:`ansg.knowledge.lexical_baseline`. Its
names are re-exported lazily.

Name hazard (audit I-CKG-1): ``ansg.knowledge.CulturalKnowledgeGraph`` resolves to
the model branch's *lexical baseline* graph (integer motif ids, keyword
overlap), because existing model code imports it from this path. The data
layer's typed graph with CST-Onto support is
:class:`ansg.knowledge.ckg.CulturalKnowledgeGraph`. Accessing the ambiguous name
here emits a ``DeprecationWarning``; import from the explicit module instead.
"""

import warnings as _warnings
from importlib import import_module as _import_module

_LAZY = {
    "Motif": "lexical_baseline",
    "CulturalKnowledgeGraph": "lexical_baseline",
}
_AMBIGUOUS = {"CulturalKnowledgeGraph"}


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        if name in _AMBIGUOUS:
            _warnings.warn(
                "ansg.knowledge.CulturalKnowledgeGraph is the lexical baseline from the model branch; "
                "import ansg.knowledge.lexical_baseline.CulturalKnowledgeGraph or "
                "ansg.knowledge.ckg.CulturalKnowledgeGraph explicitly",
                DeprecationWarning, stacklevel=2,
            )
            # Not cached in globals(), so every access through the ambiguous path warns.
            return getattr(_import_module(f".{_LAZY[name]}", __name__), name)
        value = getattr(_import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
