"""ANHC record -> model training example adapter (Issue #11; plan contracts K4-K6).

Torch-free. Produces rows that :class:`ansg.data.training_dataset.NarrativeDataset`
reads, without inventing labels:

* **Schema ids** come from a CKG activator and a :class:`SchemaVocabulary` with
  ``PAD = 0`` and CKG nodes numbered ``1..N``. A record with no activated schema
  gets an empty list (all positions masked) — never a real cultural node.
* **Dialect** comes only from exactly one *document-level* DIA annotation. Missing,
  span-level-only or conflicting DIA annotations give ``dialect_id = None``
  (written as JSON ``null``; the model's DAL then uses its learned soft gate).
  It is never defaulted to MSA.
* Activated schemas are **model inputs derived by an activator**, not MOTIF
  annotations; they are never written back to the record.
* Records whose text was withheld for copyright reasons are skipped and counted.

Thesis references: DIA labels §3.5.1; DAL dialects §4.4.1; schema activation
Eq. 4.20 / App. A.4; schema inventory Table 4.2 (898) vs A.3 (1,847) — C-CKG-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..knowledge.ckg import CulturalKnowledgeGraph, LexicalTriggerEvidence, SchemaActivator
from ..utils.provenance import RunManifest, sha256_file
from .records import AnnotationStatus, NarrativeRecord

# --------------------------------------------------------------------- dialects
# (thesis label §3.5.1, DAL code = ansg.config.DIALECTS entry, integer id = position)
DIALECT_TABLE: tuple[tuple[str, str, int], ...] = (
    ("MSA", "msa", 0),
    ("Egyptian", "egyptian", 1),
    ("Levantine", "levantine", 2),
    ("Gulf", "gulf", 3),
    ("Maghrebi", "maghrebi", 4),
)
UNKNOWN_DIALECT_ID = -1  # sentinel used by the model's collate/DAL for "no label"
_BY_ALIAS = {alias: (code, i) for label, code, i in DIALECT_TABLE for alias in (label, code)}


def dialect_code(label: str) -> tuple[str, int]:
    """Map a thesis DIA label (``"MSA"``) or DAL code (``"msa"``) to ``(code, id)``.

    Unknown labels raise ``ValueError``; they are never guessed (Iraqi and
    "Mixed" from Table 8.6 / A.17 are deliberately not in the table, C-ANHC-5).
    """
    try:
        return _BY_ALIAS[label]
    except KeyError:
        raise ValueError(f"unknown dialect label {label!r}; known: {[l for l, _, _ in DIALECT_TABLE]}") from None


def document_dialect(record: NarrativeRecord) -> tuple[str | None, str]:
    """Return ``(dal_code | None, reason)`` from the record's DIA layer."""
    if record.annotation_status.get("dia", AnnotationStatus.UNANNOTATED) == AnnotationStatus.UNANNOTATED:
        return None, "dia_unannotated"
    doc_labels = {a.label for a in record.dia if a.span is None}
    if not doc_labels:
        return None, "no_document_level_dia" if not record.dia else "span_level_dia_only"
    if len(doc_labels) > 1:
        return None, "conflicting_document_level_dia"
    return dialect_code(doc_labels.pop())[0], "document_level_dia"


# ---------------------------------------------------------------- schema vocab
@dataclass(frozen=True)
class SchemaVocabulary:
    """CKG node id <-> model schema id, with ``PAD = 0`` reserved.

    ``source_sha256`` ties the ids to the exact CKG file; a checkpoint must store
    it so ids are never reinterpreted against another graph.
    """

    node_to_id: Mapping[str, int]
    source_sha256: str | None = None
    kinds: tuple[str, ...] | None = None
    PAD: int = 0

    def __post_init__(self) -> None:
        ids = sorted(self.node_to_id.values())
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError("schema ids must be exactly 1..N (0 is reserved for PAD)")

    @classmethod
    def from_graph(cls, graph: CulturalKnowledgeGraph, kinds: Sequence[str] | None = None,
                   source_path: str | Path | None = None) -> "SchemaVocabulary":
        index = graph.index(kinds)  # 0-based, sorted by node id
        return cls(node_to_id={n: i + 1 for n, i in index.items()},
                   source_sha256=sha256_file(source_path) if source_path else None,
                   kinds=tuple(kinds) if kinds is not None else None)

    @property
    def schema_count(self) -> int:
        """Value for ``ModelConfig.schema_count`` (includes PAD)."""
        return len(self.node_to_id) + 1

    def encode(self, node_ids: Iterable[str]) -> list[int]:
        out = []
        for n in node_ids:
            if n not in self.node_to_id:
                raise KeyError(f"CKG node {n!r} is not in this schema vocabulary")
            out.append(self.node_to_id[n])
        return out

    def decode(self, ids: Iterable[int]) -> list[str]:
        inv = {v: k for k, v in self.node_to_id.items()}
        return [inv[i] for i in ids if i != self.PAD]

    def to_dict(self) -> dict[str, Any]:
        return {"pad_id": self.PAD, "node_to_id": dict(sorted(self.node_to_id.items(), key=lambda kv: kv[1])),
                "source_sha256": self.source_sha256, "kinds": list(self.kinds) if self.kinds else None}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SchemaVocabulary":
        if d.get("pad_id", 0) != 0:
            raise ValueError("only PAD = 0 is supported")
        return cls(node_to_id=dict(d["node_to_id"]), source_sha256=d.get("source_sha256"),
                   kinds=tuple(d["kinds"]) if d.get("kinds") else None)


# ------------------------------------------------------------- training example
@dataclass(frozen=True)
class TrainingExample:
    record_id: str
    text: str
    split: str | None
    dialect: str | None
    dialect_id: int | None
    dialect_reason: str
    dialect_label_status: str
    schema_node_ids: tuple[str, ...]
    schema_ids: tuple[int, ...]
    schema_weights: tuple[float, ...]
    activation_method: str
    source_file_sha256: str | None
    vocabulary_sha256: str | None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_core_row(self) -> dict[str, Any]:
        """Row for ``NarrativeDataset``: ``text``, ``schema_ids``, ``dialect_id`` (or null),
        plus provenance fields that the dataset ignores."""
        row = asdict(self)
        row["schema_ids"] = list(self.schema_ids)
        row["schema_node_ids"] = list(self.schema_node_ids)
        row["schema_weights"] = list(self.schema_weights)
        return row


def build_example(record: NarrativeRecord, graph: CulturalKnowledgeGraph, vocab: SchemaVocabulary,
                  activator: SchemaActivator | None = None, top_k: int | None = None) -> TrainingExample:
    activator = activator or LexicalTriggerEvidence()
    scores = activator.activate(record.text, graph).scores
    ranked = sorted((n for n in scores if n in vocab.node_to_id), key=lambda n: (-scores[n], n))
    if top_k is not None:
        ranked = ranked[:top_k]
    code, reason = document_dialect(record)
    return TrainingExample(
        record_id=record.record_id, text=record.text, split=record.split,
        dialect=code, dialect_id=None if code is None else dialect_code(code)[1], dialect_reason=reason,
        dialect_label_status=record.annotation_status.get("dia", AnnotationStatus.UNANNOTATED).value,
        schema_node_ids=tuple(ranked), schema_ids=tuple(vocab.encode(ranked)),
        schema_weights=tuple(float(scores[n]) for n in ranked),
        activation_method=activator.method, source_file_sha256=record.source.file_sha256,
        vocabulary_sha256=vocab.source_sha256,
    )


def build_examples_from_jsonl(path: str | Path, graph: CulturalKnowledgeGraph, vocab: SchemaVocabulary,
                              activator: SchemaActivator | None = None, top_k: int | None = None
                              ) -> tuple[list[TrainingExample], dict[str, int]]:
    """Read an ANHC split JSONL (as written by ``ansg-ingest``); skip withheld text."""
    examples: list[TrainingExample] = []
    stats = {"records": 0, "skipped_withheld_text": 0, "with_dialect": 0, "with_schemas": 0}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            stats["records"] += 1
            if raw.get("text") is None:
                stats["skipped_withheld_text"] += 1
                continue
            ex = build_example(NarrativeRecord.from_dict(raw), graph, vocab, activator, top_k)
            stats["with_dialect"] += ex.dialect_id is not None
            stats["with_schemas"] += bool(ex.schema_ids)
            examples.append(ex)
    return examples, stats


def write_examples(examples: Iterable[TrainingExample], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_core_row(), ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build model training examples from ANHC split files (Issue #11)")
    ap.add_argument("--records-dir", required=True, help="directory with {train,validation,test}.jsonl from ansg-ingest")
    ap.add_argument("--ckg", required=True, help="CKG JSON (format: ansg.knowledge.ckg)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--kinds", nargs="*", default=None, help="CKG node kinds to index as schemas (default: all)")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args(argv)
    records_dir, out = Path(args.records_dir), Path(args.output)
    graph = CulturalKnowledgeGraph.from_json(args.ckg)
    vocab = SchemaVocabulary.from_graph(graph, args.kinds, source_path=args.ckg)
    activator = LexicalTriggerEvidence()
    manifest = RunManifest(run_type="training_examples", config={**vars(args), "activator": activator.method})
    manifest.add_inputs([args.ckg])
    manifest.warnings.append("activation uses LexicalTriggerEvidence, an inspection baseline, "
                             "not the dissertation's schema-activation function (Eq. 4.20 needs embeddings)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema_vocabulary.json").write_text(json.dumps(vocab.to_dict(), ensure_ascii=False, indent=1) + "\n",
                                                encoding="utf-8")
    manifest.add_output(out / "schema_vocabulary.json")
    found = False
    for split in ("train", "validation", "test"):
        src = records_dir / f"{split}.jsonl"
        if not src.is_file():
            continue
        found = True
        manifest.add_inputs([src])
        examples, stats = build_examples_from_jsonl(src, graph, vocab, activator, args.top_k)
        dest = out / f"{split}.examples.jsonl"
        write_examples(examples, dest)
        manifest.add_output(dest)
        manifest.stats[split] = stats
    if not found:
        print(f"error: no split files in {records_dir}", file=sys.stderr)
        return 2
    manifest.stats["schema_count"] = vocab.schema_count
    manifest.finish()
    manifest.write(out / "manifest.json")
    print(json.dumps({"schema_count": vocab.schema_count, **manifest.stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
