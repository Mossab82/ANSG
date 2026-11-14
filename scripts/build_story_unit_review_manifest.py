#!/usr/bin/env python3
"""Build provisional, source-linked review units for classical narrative sources."""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_record(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.loads(f.readline())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alf-records", type=Path, required=True)
    ap.add_argument("--alf-candidates", type=Path, required=True)
    ap.add_argument("--kalila-jsonl", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    candidates = defaultdict(list)
    for path in sorted(args.alf_candidates.glob("hindawi-*_boundaries*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                candidates[row["record_id"]].append(row)

    output_path = args.output / "story_units_provisional.jsonl"
    rows = []
    report = {
        "schema": "ansg.story_unit_review_manifest/1",
        "boundary_policy": "retain source offsets; heading-delimited spans are provisional and never treated as gold story boundaries",
        "nested_story_policy": "do not infer parent-child links; keep every unit from the same work-cycle in one split when splitting is later assigned",
        "sources": {},
        "counts": Counter(),
        "unsegmented_prefixes": []
    }

    for path in sorted(args.alf_records.glob("hindawi-*.jsonl.gz")):
        record = read_record(path)
        record_id = record["record_id"]
        marks = sorted(candidates.get(record_id, []), key=lambda x: x["char_start"])
        if not marks:
            continue
        digest = sha256_text(record["text"])
        if digest != record["source_text_sha256"]:
            raise ValueError(f"Source-text digest mismatch: {record_id}")
        for mark in marks:
            if mark["source_text_sha256"] != digest or record["text"][mark["char_start"]:mark["char_end"]] != mark["heading"]:
                raise ValueError(f"Candidate offset mismatch: {mark['candidate_id']}")
        if marks[0]["char_start"] > 0:
            report["unsegmented_prefixes"].append({
                "record_id": record_id,
                "char_start": 0,
                "char_end": marks[0]["char_start"],
                "reason": "text_before_first_heading_candidate_not_assigned_to_a_story"
            })
        report["sources"][record_id] = {
            "cycle_id": "alf_layla_work_cycle",
            "candidate_markers": len(marks),
            "source_text_sha256": digest,
            "edition_title": record["source"]["title"],
            "translation": bool(record["source"].get("translation")),
                "source_license": record["source"].get("license"),
                "source_attribution": record["source"].get("attribution"),
            "split_lock": "all volumes and candidate tales of Alf Layla wa-Layla stay together pending a deliberate cycle-level split"
        }
        for i, mark in enumerate(marks):
            end = marks[i + 1]["char_start"] if i + 1 < len(marks) else len(record["text"])
            if end <= mark["char_end"]:
                raise ValueError(f"Invalid candidate order/extent: {mark['candidate_id']}")
            rows.append({
                "schema": "ansg.story_unit_candidate/1",
                "unit_id": mark["candidate_id"],
                "source_record_id": record_id,
                "source_cycle_id": "alf_layla_work_cycle",
                "source_group_split_lock": "alf_layla_work_cycle",
                "unit_type": "heading_delimited_provisional_span",
                "title_candidate": mark["heading"],
                "char_start": mark["char_start"],
                "char_end": end,
                "boundary_start_evidence": "standalone heading beginning with story term",
                "boundary_end_evidence": "next detected heading candidate" if i + 1 < len(marks) else "end of source volume",
                "source_text_sha256": digest,
                "status": "provisional_needs_human_review",
                "parent_story_id": None,
                "split": "unassigned",
                "review_required": ["confirm_start_boundary", "confirm_end_boundary", "identify_parent_or_frame_story", "confirm_work_cycle"]
            })
            report["counts"]["alf_layla_provisional_spans"] += 1

    kalila_count = 0
    for record in (json.loads(line) for line in args.kalila_jsonl.open(encoding="utf-8") if line.strip()):
        text = record["text"]
        source = record["source"]
        digest = sha256_text(text)
        if digest != record["source_text_sha256"]:
            raise ValueError(f"Kalila source-text digest mismatch: {record['record_id']}")
        rows.append({
            "schema": "ansg.story_unit_candidate/1",
            "unit_id": record["record_id"],
            "source_record_id": record["record_id"],
            "source_cycle_id": "kalila_1816_frame_cycle",
            "source_group_split_lock": "kalila_1816_frame_cycle",
            "unit_type": "source_chapter_container",
            "title_candidate": record["narrative"].get("chapter_title"),
            "char_start": 0,
            "char_end": len(text),
            "boundary_start_evidence": "source edition chapter page",
            "boundary_end_evidence": "source chapter end marker",
            "source_page_title": source.get("page_title"),
            "source_page_url": source.get("page_url"),
            "source_revision_id": source.get("page_revision_id"),
            "source_license": source.get("license"),
            "source_attribution": source.get("attribution"),
            "source_text_sha256": digest,
            "status": "chapter_container_nested_tales_unsegmented",
            "parent_story_id": None,
            "split": "unassigned",
            "review_required": ["segment_nested_tales_if_supported_by_text", "identify_parent_or_frame_story", "confirm_cycle_membership"]
        })
        kalila_count += 1
    report["counts"]["kalila_chapter_containers"] = kalila_count
    report["counts"]["total_provisional_units"] = len(rows)
    report["counts"] = dict(report["counts"])
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output / "story_unit_review_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
