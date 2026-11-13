"""Arabic story generation runtime and capability gate (Issue #15).

Torch-free modules: ``labels``, ``artifacts``, ``readiness``, ``cli`` (imports the
runtime lazily), ``stub`` (imports torch/transformers only when called).
Model layer (``model`` + ``pretrained`` extras): ``runtime``, ``cultural``.

See ``docs/GENERATION_READINESS.md``. Nothing here reproduces a dissertation result.
"""

from .labels import GenerationMode, ReadinessLevel

__all__ = ["GenerationMode", "ReadinessLevel"]
