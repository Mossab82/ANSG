"""ANHC corpus I/O: JSONL serialisation, export policy, annotation import.

* :func:`write_jsonl` withholds the text of non-redistributable sources unless
  explicitly allowed, so derived datasets can be shared without leaking
  copyrighted narratives.
* :func:`import_annotations` merges *supplied* annotation files (e.g. the
  original gold DIA/MOTIF/MORAL annotations, once provided) into records. It
  requires the caller to state the status of the imported layer; it never
  upgrades automatic annotations to gold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Iterator

from .records import (
    LAYERS,
    AnnotationSource,
    AnnotationStatus,
    DialectAnnotation,
    MoralAnnotation,
    MotifAnnotation,
    NarrativeRecord,
    Span,
)


def write_jsonl(
    records: Iterable[NarrativeRecord],
    path: str | os.PathLike[str],
    allow_restricted_text: bool = False,
) -> int:
    """Write records as UTF-8 JSONL in the given order. Returns the count written."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            include = allow_restricted_text or rec.source.redistributable
            fh.write(json.dumps(rec.to_dict(include_text=include), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[NarrativeRecord]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                yield NarrativeRecord.from_dict(json.loads(line))
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


_FACTORIES = {"dia": DialectAnnotation, "motif": MotifAnnotation, "moral": MoralAnnotation}


def import_annotations(
    records: list[NarrativeRecord],
    annotation_path: str | os.PathLike[str],
    layer: str,
    status: AnnotationStatus,
    replace: bool = True,
) -> dict[str, int]:
    """Merge a JSONL annotation file into ``records`` for one ``layer``.

    Each line: ``{"record_id": ..., "span": {"start":..,"end":..} | null,
    "annotator_id": ..., "source": "human_adjudicated"|..., <layer fields>}``
    where layer fields are ``label`` (dia), ``motif_id`` (motif) or
    ``framework``/``value`` (moral).

    Returns counts: ``{"imported": n, "unknown_record": m}``. Records listed in
    the file get ``annotation_status[layer] = status``; others are untouched.
    """
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer!r}")
    if status == AnnotationStatus.UNANNOTATED:
        raise ValueError("cannot import annotations with status 'unannotated'")
    by_id = {r.record_id: r for r in records}
    touched: set[str] = set()
    counts = {"imported": 0, "unknown_record": 0}
    factory = _FACTORIES[layer]
    with open(annotation_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid = row.pop("record_id")
            rec = by_id.get(rid)
            if rec is None:
                counts["unknown_record"] += 1
                continue
            sp = row.pop("span", None)
            row["span"] = None if sp is None else Span(int(sp["start"]), int(sp["end"]))
            row["source"] = AnnotationSource(row.get("source", "imported"))
            row.setdefault("annotator_id", None)
            if status == AnnotationStatus.GOLD and row["source"] == AnnotationSource.AUTOMATIC:
                raise ValueError(f"{annotation_path}:{lineno}: automatic annotation cannot be gold")
            ann = factory(**row)
            target = getattr(rec, layer)
            if replace and rid not in touched:
                target.clear()
            touched.add(rid)
            target.append(ann)
            rec.annotation_status[layer] = status
            counts["imported"] += 1
    return counts
