#!/usr/bin/env python3
"""Audit ANHC training admission without emitting or copying narrative text."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(paths: list[Path], decisions_path: Path) -> dict[str, Any]:
    decision_file = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions = decision_file.get("decisions", {})
    sources: dict[str, dict[str, Any]] = {}
    annotations = Counter()
    split_status = Counter()
    source_rows: dict[str, Counter] = defaultdict(Counter)
    input_summaries = []

    for path in paths:
        rows = read_jsonl(path)
        eligible = 0
        groups: set[str] = set()
        for row in rows:
            source = row.get("source") or {}
            source_id = str(source.get("source_id") or "UNKNOWN_SOURCE")
            decision = decisions.get(source_id) or {}
            approved = (
                decision.get("training_allowed") is True
                and bool(decision.get("basis"))
                and bool(decision.get("evidence_url"))
                and bool(decision.get("reviewed_on"))
                and bool(decision.get("reviewer"))
            )
            eligible += int(approved and isinstance(row.get("text"), str) and bool(row["text"].strip()))
            src_count = source_rows[source_id]
            src_count["records"] += 1
            src_count[f"copyright_status:{source.get('copyright_status', 'missing')}"] += 1
            src_count[f"training_allowed:{str(decision.get('training_allowed', 'undecided')).lower()}"] += 1
            src_count[f"license:{source.get('license') or 'missing'}"] += 1
            statuses = row.get("annotation_status") or {}
            for layer in ("dia", "motif", "moral"):
                annotations[f"{layer}:{statuses.get(layer, 'missing')}"] += 1
            split_status[str(row.get("split") or "unassigned")] += 1
            extra = (row.get("narrative") or {}).get("extra") or {}
            group = extra.get("candidate_story_family_id") or (row.get("narrative") or {}).get("parent_story_id") or row.get("record_id")
            groups.add(str(group))
        input_summaries.append({
            "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "records": len(rows),
            "source_count": len({str((r.get("source") or {}).get("source_id") or "UNKNOWN_SOURCE") for r in rows}),
            "provisionally_training_eligible": eligible,
            "candidate_groups": len(groups),
        })

    source_summary = []
    for source_id, counts in sorted(source_rows.items()):
        decision = decisions.get(source_id) or {}
        source_summary.append({
            "source_id": source_id,
            "records": counts["records"],
            "copyright_status_counts": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("copyright_status:")},
            "license_claims": {k.split(":", 1)[1]: v for k, v in counts.items() if k.startswith("license:")},
            "training_decision": "approved_with_evidence" if (
                decision.get("training_allowed") is True and decision.get("basis") and decision.get("evidence_url") and decision.get("reviewed_on") and decision.get("reviewer")
            ) else "pending_or_incomplete",
        })

    total = sum(item["records"] for item in input_summaries)
    summary = {
        "schema": "ansg.anhc-training-readiness-audit/1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_source": str(decisions_path.relative_to(ROOT) if decisions_path.is_relative_to(ROOT) else decisions_path),
        "inputs": input_summaries,
        "total_input_rows_including_overlaps": total,
        "unique_sources": len(source_rows),
        "training_eligible_rows": sum(i["provisionally_training_eligible"] for i in input_summaries),
        "training_ready": False,
        "split_counts_including_overlaps": dict(sorted(split_status.items())),
        "annotation_status_counts_including_overlaps": dict(sorted(annotations.items())),
        "sources": source_summary,
        "blocking_gates": [
            "No source has a complete, source-specific training permission decision with evidence.",
            "Current story-boundary and parallel-source links remain candidates requiring review.",
            "DIA, MOTIF, and MORAL layers have no human-adjudicated gold annotations in current staging records.",
            "No leakage-reviewed train/validation/test assignments are present.",
            "Tokenizer-specific token counts and selected training tokenizer are not finalized.",
        ],
        "warning": "This audit reports eligibility only. It does not adjudicate copyright, create annotations, assign splits, or copy text into a training export.",
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=ROOT / "data/corpora/anhc/staging_records.jsonl")
    parser.add_argument("--merged", type=Path, default=ROOT / "data/corpora/anhc/quarantine/merged_narrative_candidates.jsonl")
    parser.add_argument("--decisions", type=Path, default=ROOT / "data/corpora/anhc/training_admission_decisions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/corpora/anhc/training_readiness_audit.json")
    args = parser.parse_args()
    # Prefer the merged superset; fall back to the tracked heritage seed when
    # private/quarantined candidates have not been materialized locally.
    inputs = [args.merged] if args.merged.is_file() else ([args.staging] if args.staging.is_file() else [])
    if not inputs:
        parser.error("neither merged candidates nor staging records were found")
    result = audit(inputs, args.decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {k: result[k] for k in ("inputs", "unique_sources", "training_eligible_rows", "training_ready", "blocking_gates")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
