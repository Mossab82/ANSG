"""Inference adapter for the optional QCRI tweet-domain DIA baseline.

The classifier predicts one of Egyptian, Levantine, Gulf, or Maghrebi. It has no
MSA class and must not be used to force a dialect or to claim story quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SUPPORTED_LABELS = ("Egyptian", "Levantine", "Gulf", "Maghrebi")


@dataclass(frozen=True)
class DialectPrediction:
    label: str
    decision_scores: dict[str, float]
    score_type: str = "raw_linear_svc_decision_margin"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QCRIDialectClassifier:
    """Load and run a QCRI-trained char-TF-IDF + LinearSVC artifact."""

    def __init__(self, artifact: dict[str, Any]):
        required = {"vectorizer", "classifier", "labels"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError("model artifact must contain vectorizer, classifier, and labels")
        labels = tuple(str(label) for label in artifact["labels"])
        if labels != SUPPORTED_LABELS:
            raise ValueError(f"expected QCRI labels in order {SUPPORTED_LABELS}, got {labels}")
        classes = tuple(str(label) for label in getattr(artifact["classifier"], "classes_", ()))
        if set(classes) != set(SUPPORTED_LABELS):
            raise ValueError("classifier classes do not match the four supported QCRI dialects")
        self.vectorizer = artifact["vectorizer"]
        self.classifier = artifact["classifier"]
        self.labels = labels
        self.classes = classes

    @classmethod
    def from_file(cls, model_path: str | Path) -> "QCRIDialectClassifier":
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("install joblib to load the optional QCRI model") from exc
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(joblib.load(path))

    def predict(self, text: str) -> DialectPrediction:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("story text must be a non-empty string")
        features = self.vectorizer.transform([text])
        predicted = str(self.classifier.predict(features)[0])
        if predicted not in self.labels:
            raise ValueError(f"classifier returned unsupported label {predicted!r}")
        margins = self.classifier.decision_function(features)
        if getattr(margins, "ndim", 1) != 2 or margins.shape[0] != 1:
            raise ValueError("expected one row of one-vs-rest decision scores")
        scores = {label: float(margins[0, idx]) for idx, label in enumerate(self.classes)}
        return DialectPrediction(label=predicted, decision_scores=scores)
