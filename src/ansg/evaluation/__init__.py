"""Evaluation. CRAFT metrics are specified but NOT implemented (no formulas in the
dissertation, see ``docs/CRAFT_EVALUATION_SPECIFICATION.md``).

Backward compatibility (Issue #11): the former flat module ``ansg/evaluation.py``
of the model branch now lives in :mod:`ansg.evaluation.basic_metrics`; its names
are re-exported lazily. ``weighted_cmpi`` is a **surrogate** weighted aggregate
(Ch. 5 §5.1 weights), not the dissertation's CMPI — see its docstring.
"""

from importlib import import_module as _import_module

_LAZY = {
    "bleu4": "basic_metrics",
    "weighted_cmpi": "basic_metrics",
    "summarize": "basic_metrics",
}


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        value = getattr(_import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
