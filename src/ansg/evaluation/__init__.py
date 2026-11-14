"""Evaluation. CRAFT metrics are specified but not computable from text.

CMPI's equation is defined in the dissertation (Ch.5 §5.1 p.131; §8.7.4 p.223) and
implemented in :mod:`ansg.evaluation.cmpi` as the ANSG reference; how its four
dimensions are scored from a text is not specified, so CRAFT still refuses to
score samples (``docs/CRAFT_EVALUATION_SPECIFICATION.md``).

Backward compatibility (Issue #11): the former flat module ``ansg/evaluation.py``
of the model branch now lives in :mod:`ansg.evaluation.basic_metrics`; its names
are re-exported lazily. ``weighted_cmpi`` applies the dissertation's CMPI equation
to caller-supplied dimension scores — see its docstring.
"""

from importlib import import_module as _import_module

_LAZY = {
    "bleu4": "basic_metrics",
    "weighted_cmpi": "basic_metrics",
    "ansg_cmpi": "cmpi",
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
