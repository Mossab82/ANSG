"""OPTIONAL, EXPERIMENTAL dialect identification interface. Torch-free; nothing loads by default.

What the thesis used, and why it is not here
--------------------------------------------
§4.2.4 (p. 94) and §5.4.3 (p. 140) describe a character-trigram CNN (three convolutional
layers, kernel sizes 3/4/5, max-pooling) reaching 94.3% accuracy on MADAR. Neither the
model, its weights nor its training split are available
(docs/reproduction/DISCREPANCY_REPORT.md, D-ART-1), so the thesis classifier is NOT
reproduced. The DIA layer of the thesis's ANHC cannot be regenerated with this module.

What this module offers
-----------------------
A narrow interface for *supplementary* experiments with an external classifier, for
example CAMeL Tools' MADAR-trained dialect identification (``camel_tools.dialectid``).
Rules:

* Explicit opt-in: ``experimental=True`` is required; nothing is imported or downloaded
  otherwise.
* Every prediction is labelled ``status = "experimental_supplementary"`` and
  ``reproduces_thesis = False``. It never replaces the thesis's DIA annotation or the
  repository's proxy metrics silently.
* MADAR terms: the corpus licence (CMU/NYUAD) allows internal research and evaluation only
  and forbids redistribution and modification. Models trained on MADAR, and their data
  packages, carry their own terms. Check them (``data/sources/source_registry.json``, id
  ``madar``) before use. No MADAR file is, or may be, committed to this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

STATUS = "experimental_supplementary"


class DialectBackend(Protocol):
    name: str
    version: str

    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:  # (label, score)
        ...


@dataclass(frozen=True)
class DialectPrediction:
    text: str
    label: str
    score: float
    backend: str
    backend_version: str
    status: str = STATUS
    reproduces_thesis: bool = False


class ExperimentalDialectIdentifier:
    """Wraps a backend; refuses to run without an explicit experimental opt-in."""

    def __init__(self, backend: DialectBackend, *, experimental: bool = False):
        if not experimental:
            raise PermissionError("dialect identification is an optional experimental interface: pass "
                                  "experimental=True and record the backend's licence review")
        self.backend = backend

    def predict(self, sentences: Iterable[str]) -> list[DialectPrediction]:
        texts = [s for s in sentences]
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("sentences must be non-empty strings")
        out = self.backend.predict(texts)
        if len(out) != len(texts):
            raise RuntimeError("backend returned a different number of predictions")
        return [DialectPrediction(t, label, float(score), self.backend.name, self.backend.version)
                for t, (label, score) in zip(texts, out)]


class CamelToolsBackend:
    """CAMeL Tools ``DialectIdentifier`` (MADAR-trained). Imported only when constructed.

    Requires ``pip install camel-tools`` and its data package (``camel_data``), whose terms
    must be reviewed first. Not exercised in CI.
    """

    name = "camel_tools.dialectid"

    def __init__(self, loader: Callable[[], object] | None = None):
        if loader is None:
            from camel_tools.dialectid import DialectIdentifier  # noqa: PLC0415 - optional dependency
            import camel_tools  # noqa: PLC0415
            self._model = DialectIdentifier.pretrained()
            self.version = getattr(camel_tools, "__version__", "unknown")
        else:
            self._model = loader()
            self.version = "injected"

    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        preds = self._model.predict(sentences)
        return [(p.top, float(p.scores.get(p.top, 0.0))) for p in preds]
