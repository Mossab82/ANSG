#!/usr/bin/env python3
"""Audit story text with the optional QCRI tweet-domain dialect classifier.

Input JSONL rows require story_id, source_corpus, and text. Add target_dialect
only when it is an independently assigned reference label. The report omits
story text and never treats the classifier's raw margins as probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ansg.dialects.qcri_classifier import QCRIDialectClassifier, SUPPORTED_LABELS
from ansg.utils.provenance import sha256_file

ALIASES = {
    "egyptian": "Egyptian",
    "levantine": "Levantine",
    "shami": "Levantine",
    "gulf": "Gulf",
    "maghrebi": "Maghrebi",
    "msa": "MSA",
}


def normalize_target(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().casefold()
    if key not in ALIASES:
        raise ValueError(f"unsupported target_dialect {value!r}")
    return ALIASES[key]


def load_story_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        for field in ("story_id", "source_corpus", "text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
        if "qcri" in row["source_corpus"].casefold():
            raise ValueError(f"{path}:{line_number}: story source must be distinct from QCRI")
        if row["story_id"] in seen:
            raise ValueError(f"{path}:{line_number}: duplicate story_id {row['story_id']!r}")
        seen.add(row["story_id"])
        row["_target"] = normalize_target(row.get("target_dialect"))
        rows.append(row)
    if not rows:
        raise ValueError("input contains no story records")
    return rows


def summarize(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        (row["_target"], result["prediction"])
        for row, result in zip(rows, predictions)
        if row["_target"] in SUPPORTED_LABELS
    ]
    counts = Counter(target for target, _ in scored)
    correct = sum(target == pred for target, pred in scored)
    f1_by_label = {}
    for label in SUPPORTED_LABELS:
        tp = sum(target == label and pred == label for target, pred in scored)
        fp = sum(target != label and pred == label for target, pred in scored)
        fn = sum(target == label and pred != label for target, pred in scored)
        denom = 2 * tp + fp + fn
        f1_by_label[label] = (2 * tp / denom) if denom else 0.0
    macro_f1 = (sum(f1_by_label.values()) / len(SUPPORTED_LABELS)) if scored else None
    return {
        "scored_story_count": len(scored),
        "accuracy": correct / len(scored) if scored else None,
        "macro_f1": macro_f1,
        "per_label_f1": f1_by_label if scored else None,
        "target_counts": {label: counts.get(label, 0) for label in SUPPORTED_LABELS},
        "unsupported_target_count": sum(row["_target"] == "MSA" for row in rows),
        "unlabeled_story_count": sum(row["_target"] is None for row in rows),
        "interpretation": (
            "Exploratory cross-domain story audit only. The QCRI classifier was trained "
            "on tweets; these scores are not thesis results or proof of dialect quality."
        ),
    }


def run(model_path: Path, stories_path: Path) -> dict[str, Any]:
    rows = load_story_rows(stories_path)
    classifier = QCRIDialectClassifier.from_file(model_path)
    predictions = []
    for row in rows:
        result = classifier.predict(row["text"])
        predictions.append({
            "story_id": row["story_id"],
            "source_corpus": row["source_corpus"],
            "target_dialect": row["_target"],
            **result.to_dict(),
            "matches_reference": (
                result.label == row["_target"] if row["_target"] in SUPPORTED_LABELS else None
            ),
        })
    return {
        "schema": "ansg.qcri_story_dialect_audit/1",
        "model_sha256": sha256_file(model_path),
        "model_training_domain": "QCRI Arabic POS tweets",
        "supported_labels": list(SUPPORTED_LABELS),
        "metrics": summarize(rows, predictions),
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="QCRI qcri_dia_baseline.joblib")
    parser.add_argument("--stories", required=True, type=Path, help="independent story JSONL")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report path")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = run(args.model, args.stories)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
