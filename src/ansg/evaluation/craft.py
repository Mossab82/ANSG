"""CRAFT evaluation framework — interfaces and specification registry only.

**No CRAFT metric is implemented in this module.** The mathematical
definitions, normalisation rules and weights must be transcribed from the
dissertation (Chapter 6) into ``configs/evaluation/craft.yaml`` and
``docs/CRAFT_EVALUATION_SPECIFICATION.md`` before any implementation is
written. Until then:

* every :class:`MetricSpec` carries ``status`` ``"unverified"`` and
  ``thesis_reference=None``;
* :class:`UnimplementedMetric` raises :class:`MetricNotSpecifiedError` when
  called, so no pipeline can silently report a CRAFT number;
* :func:`aggregate` refuses to combine scores unless every weight is
  explicitly configured *and* marked as verified against the thesis.

The four dimension names below are those listed in the project brief; their
exact names, number, metrics and weights in Chapter 6 are unverified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class CraftDimension(str, Enum):
    CULTURAL_AUTHENTICITY = "cultural_authenticity"
    DIALECTAL_FIDELITY = "dialectal_fidelity"
    LINGUISTIC_QUALITY = "linguistic_quality"
    NARRATIVE_COHERENCE = "narrative_coherence"


class SpecStatus(str, Enum):
    UNVERIFIED = "unverified"  # not yet checked against the dissertation
    INCOMPLETE = "incomplete"  # thesis definition found but under-specified
    SPECIFIED = "specified"  # complete definition transcribed with section reference
    IMPLEMENTED = "implemented"  # implementation + tests against the spec


class MetricNotSpecifiedError(NotImplementedError):
    pass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    dimension: CraftDimension
    status: SpecStatus = SpecStatus.UNVERIFIED
    thesis_reference: str | None = None
    equation: str | None = None  # LaTeX as written in the thesis
    range: tuple[float, float] | None = None
    normalization: str | None = None
    higher_is_better: bool | None = None
    requires: tuple[str, ...] = ()  # e.g. ("reference_texts", "ckg", "human_ratings")
    notes: str | None = None


@dataclass
class EvaluationSample:
    """One generated story with optional conditioning and references."""

    sample_id: str
    generated_text: str
    prompt: str | None = None
    reference_texts: Sequence[str] = ()
    target_dialect: str | None = None
    target_motifs: Sequence[str] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    metric: str
    value: float
    per_sample: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CraftMetric(Protocol):
    spec: MetricSpec

    def __call__(self, samples: Sequence[EvaluationSample]) -> MetricResult: ...


@dataclass
class UnimplementedMetric:
    """Registry entry for a metric that is not yet specified/implemented."""

    spec: MetricSpec

    def __call__(self, samples: Sequence[EvaluationSample]) -> MetricResult:
        raise MetricNotSpecifiedError(
            f"CRAFT metric '{self.spec.name}' ({self.spec.dimension.value}) has status "
            f"'{self.spec.status.value}'. Transcribe its definition from the dissertation "
            "(Chapter 6) before implementing it."
        )


@dataclass(frozen=True)
class HumanEvaluationProtocol:
    """Declarative description of the human-evaluation protocol (Chapter 6).

    All fields default to ``None`` = not yet transcribed.
    """

    n_raters: int | None = None
    rater_qualifications: str | None = None
    rating_scale: tuple[int, int] | None = None
    dimensions: tuple[CraftDimension, ...] | None = None
    n_items: int | None = None
    agreement_statistic: str | None = None  # e.g. "Krippendorff alpha" — unverified
    blinding: str | None = None
    thesis_reference: str | None = None


class CraftRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, CraftMetric] = {}

    def register(self, metric: CraftMetric) -> None:
        if metric.spec.name in self._metrics:
            raise ValueError(f"metric {metric.spec.name!r} already registered")
        self._metrics[metric.spec.name] = metric

    def get(self, name: str) -> CraftMetric:
        return self._metrics[name]

    def specs(self) -> list[MetricSpec]:
        return [m.spec for m in self._metrics.values()]

    def status_report(self) -> dict[str, dict[str, Any]]:
        return {s.name: {"dimension": s.dimension.value, "status": s.status.value,
                         "thesis_reference": s.thesis_reference} for s in self.specs()}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CraftRegistry":
        """Build a registry of :class:`UnimplementedMetric` entries from YAML."""
        reg = cls()
        for m in config.get("metrics") or []:
            rng = m.get("range")
            spec = MetricSpec(
                name=m["name"], dimension=CraftDimension(m["dimension"]),
                status=SpecStatus(m.get("status", "unverified")),
                thesis_reference=m.get("thesis_reference"), equation=m.get("equation"),
                range=tuple(rng) if rng else None, normalization=m.get("normalization"),
                higher_is_better=m.get("higher_is_better"), requires=tuple(m.get("requires") or ()),
                notes=m.get("notes"),
            )
            reg.register(UnimplementedMetric(spec))
        return reg


@dataclass(frozen=True)
class AggregationWeights:
    weights: Mapping[str, float]
    verified: bool = False
    thesis_reference: str | None = None


def aggregate(scores: Mapping[str, float], weights: AggregationWeights | None) -> float:
    """Weighted combination of dimension/metric scores.

    Refuses to run unless weights are supplied, verified against the thesis and
    cover exactly the provided scores. Whether CRAFT aggregates linearly at all
    is itself unverified; this function exists so that the check is enforced in
    one place.
    """
    if weights is None or not weights.verified or not weights.thesis_reference:
        raise MetricNotSpecifiedError(
            "CRAFT aggregation weights are not verified against the dissertation; refusing to aggregate."
        )
    if set(weights.weights) != set(scores):
        raise ValueError(f"weights {sorted(weights.weights)} do not match scores {sorted(scores)}")
    return sum(weights.weights[k] * scores[k] for k in scores)
