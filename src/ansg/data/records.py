"""ANHC annotated narrative record structure.

Implements the record layers named in the project brief for the ANHC corpus
(dissertation Chapter 3): **DIA** (dialectal), **MOTIF** (cultural motif),
**MORAL** (moral framework), source provenance / copyright, and split
membership.

What this module does *not* do
------------------------------
* It does not define the label inventories. Dialect codes, motif identifiers
  and moral-framework categories must come from the dissertation's annotation
  guidelines or supplied gold files and are configured in YAML
  (``configs/data/anhc.yaml``). Until then validation of labels is skipped and
  reported, never filled in.
* It never creates annotations. Records ingested from raw text carry the
  status ``unannotated`` for every layer.

The schema identifier ``ansg.anhc-record/0.1`` is defined by this
reimplementation; it is not claimed to be the historical ANHC file format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "ansg.anhc-record/0.1"
LAYERS = ("dia", "motif", "moral")


class AnnotationStatus(str, Enum):
    """Status of an annotation layer for one record."""

    UNANNOTATED = "unannotated"  # nobody has annotated this layer
    AUTOMATIC = "automatic"  # produced by a model/heuristic (never gold)
    SILVER = "silver"  # human, not adjudicated
    GOLD = "gold"  # human, adjudicated per guidelines


class AnnotationSource(str, Enum):
    HUMAN_ADJUDICATED = "human_adjudicated"
    HUMAN_SINGLE = "human_single"
    AUTOMATIC = "automatic"
    IMPORTED = "imported"  # taken from a supplied file whose provenance is recorded


class CopyrightStatus(str, Enum):
    PUBLIC_DOMAIN = "public_domain"
    LICENSED = "licensed"
    RESTRICTED = "restricted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Span:
    """Half-open character span ``[start, end)`` into ``NarrativeRecord.text``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span ({self.start}, {self.end})")


@dataclass
class SourceProvenance:
    source_id: str
    title: str
    copyright_status: CopyrightStatus = CopyrightStatus.UNKNOWN
    redistributable: bool = False
    license: str | None = None
    edition: str | None = None
    editor: str | None = None
    publisher: str | None = None
    year: str | None = None
    url: str | None = None
    file_path: str | None = None
    file_sha256: str | None = None
    notes: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceProvenance":
        d = dict(data)
        d["copyright_status"] = CopyrightStatus(d.get("copyright_status", "unknown"))
        return cls(**d)


@dataclass
class _AnnotationBase:
    span: Span | None  # None = document-level annotation
    annotator_id: str | None
    source: AnnotationSource
    confidence: float | None = None
    guideline_version: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


@dataclass
class DialectAnnotation(_AnnotationBase):
    """DIA layer: dialect / language-variety label for a span or the document."""

    label: str = ""


@dataclass
class MotifAnnotation(_AnnotationBase):
    """MOTIF layer: reference to a cultural motif identifier in the CKG."""

    motif_id: str = ""


@dataclass
class MoralAnnotation(_AnnotationBase):
    """MORAL layer: moral-framework category (and optional value) for a span."""

    framework: str = ""
    value: str | None = None


@dataclass
class NarrativeMetadata:
    collection: str | None = None
    title: str | None = None
    story_index: int | None = None
    night_numbers: list[int] = field(default_factory=list)
    parent_story_id: str | None = None
    nesting_level: int | None = None
    source_span: Span | None = None  # span in the normalised *source file*
    formulae: list[dict[str, Any]] = field(default_factory=list)  # detected rhetorical formulae
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeRecord:
    record_id: str
    text: str
    source: SourceProvenance
    narrative: NarrativeMetadata = field(default_factory=NarrativeMetadata)
    dia: list[DialectAnnotation] = field(default_factory=list)
    motif: list[MotifAnnotation] = field(default_factory=list)
    moral: list[MoralAnnotation] = field(default_factory=list)
    annotation_status: dict[str, AnnotationStatus] = field(
        default_factory=lambda: {layer: AnnotationStatus.UNANNOTATED for layer in LAYERS}
    )
    split: str | None = None
    normalization_config_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ io
    def to_dict(self, include_text: bool = True) -> dict[str, Any]:
        def conv(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, dict):
                return {k: conv(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [conv(v) for v in obj]
            return obj

        d = conv(asdict(self))
        if not include_text:
            d["text"] = None
            d["text_withheld_reason"] = "source not redistributable"
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NarrativeRecord":
        def span(x: Any) -> Span | None:
            return None if x is None else Span(int(x["start"]), int(x["end"]))

        def ann_kwargs(a: Mapping[str, Any]) -> dict[str, Any]:
            k = dict(a)
            k["span"] = span(k.get("span"))
            k["source"] = AnnotationSource(k["source"])
            return k

        nm = dict(d.get("narrative") or {})
        nm["source_span"] = span(nm.get("source_span"))
        status = {
            layer: AnnotationStatus(v)
            for layer, v in (d.get("annotation_status") or {}).items()
        }
        for layer in LAYERS:
            status.setdefault(layer, AnnotationStatus.UNANNOTATED)
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {d.get('schema_version')}")
        if d.get("text") is None:
            raise ValueError(f"record {d.get('record_id')} has no text (withheld export?)")
        return cls(
            record_id=d["record_id"],
            text=d["text"],
            source=SourceProvenance.from_mapping(d["source"]),
            narrative=NarrativeMetadata(**nm),
            dia=[DialectAnnotation(**ann_kwargs(a)) for a in d.get("dia", [])],
            motif=[MotifAnnotation(**ann_kwargs(a)) for a in d.get("motif", [])],
            moral=[MoralAnnotation(**ann_kwargs(a)) for a in d.get("moral", [])],
            annotation_status=status,
            split=d.get("split"),
            normalization_config_sha256=d.get("normalization_config_sha256"),
        )


@dataclass
class LabelInventory:
    """Allowed labels per layer. ``None`` means 'not supplied' (validation skipped)."""

    dia: frozenset[str] | None = None
    motif: frozenset[str] | None = None
    moral: frozenset[str] | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Iterable[str] | None] | None) -> "LabelInventory":
        data = data or {}
        return cls(**{
            k: (frozenset(data[k]) if data.get(k) is not None else None)
            for k in LAYERS if k in data
        })


@dataclass
class ValidationIssue:
    record_id: str
    severity: str  # "error" | "warning"
    message: str


def validate_record(rec: NarrativeRecord, labels: LabelInventory | None = None) -> list[ValidationIssue]:
    """Structural and policy checks. Returns issues; never mutates the record."""
    labels = labels or LabelInventory()
    issues: list[ValidationIssue] = []

    def err(msg: str) -> None:
        issues.append(ValidationIssue(rec.record_id, "error", msg))

    def warn(msg: str) -> None:
        issues.append(ValidationIssue(rec.record_id, "warning", msg))

    if not rec.record_id:
        err("empty record_id")
    if not rec.text.strip():
        err("empty text")
    if rec.source.copyright_status == CopyrightStatus.UNKNOWN:
        warn("copyright_status is unknown")

    layer_items: dict[str, list[_AnnotationBase]] = {
        "dia": list(rec.dia), "motif": list(rec.motif), "moral": list(rec.moral)
    }
    for layer, items in layer_items.items():
        status = rec.annotation_status.get(layer, AnnotationStatus.UNANNOTATED)
        if status == AnnotationStatus.UNANNOTATED and items:
            err(f"{layer}: annotations present but status is 'unannotated'")
        for i, a in enumerate(items):
            where = f"{layer}[{i}]"
            if a.span is not None and a.span.end > len(rec.text):
                err(f"{where}: span {a.span} exceeds text length {len(rec.text)}")
            if a.confidence is not None and not 0.0 <= a.confidence <= 1.0:
                err(f"{where}: confidence {a.confidence} outside [0, 1]")
            if status == AnnotationStatus.GOLD:
                if a.source not in (AnnotationSource.HUMAN_ADJUDICATED, AnnotationSource.IMPORTED):
                    err(f"{where}: gold layer contains non-adjudicated annotation ({a.source.value})")
                if not a.annotator_id and a.source != AnnotationSource.IMPORTED:
                    err(f"{where}: gold annotation without annotator_id")
            if a.source == AnnotationSource.AUTOMATIC and status in (
                AnnotationStatus.GOLD, AnnotationStatus.SILVER
            ):
                err(f"{where}: automatic annotation in a {status.value} layer")
        inventory = getattr(labels, layer)
        if inventory is None:
            if items:
                warn(f"{layer}: no label inventory configured; labels not validated")
            continue
        for i, a in enumerate(items):
            label = (
                a.label if isinstance(a, DialectAnnotation)
                else a.motif_id if isinstance(a, MotifAnnotation)
                else a.framework  # type: ignore[attr-defined]
            )
            if label not in inventory:
                err(f"{layer}[{i}]: label '{label}' not in configured inventory")
    return issues
