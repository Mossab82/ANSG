"""Corpus ingestion, normalisation, ANHC records and dataset splitting.

Backward compatibility (Issue #11): the former flat module ``ansg/data.py`` of
the model branch now lives in :mod:`ansg.data.training_dataset`. Its public
names are re-exported here **lazily**, so ``import ansg.data`` never imports
PyTorch; torch is imported only when one of these names is first accessed.
"""

from importlib import import_module as _import_module

_LAZY = {
    "ByteTokenizer": "training_dataset",
    "NarrativeDataset": "training_dataset",
    "collate": "training_dataset",
}


def __getattr__(name):  # PEP 562
    if name in _LAZY:
        value = getattr(_import_module(f".{_LAZY[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
