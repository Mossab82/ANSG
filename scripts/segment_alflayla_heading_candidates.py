#!/usr/bin/env python3
"""Find reviewable story-title boundary candidates in licensed Alf Layla records."""
import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

HEADING = re.compile(r"^(?:حكاية|حكايات|قصة|قصّة)(?:\s|ٌ|ً|ُ|َ|ِ|ّ|$)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True, help="directory with per-book .jsonl.gz records")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("hindawi-*_boundaries*.jsonl"):
        stale.unlink()
    report = {"schema": "ansg.boundary_candidate_report/1", "source_work": "Alf Layla wa-Layla", "method": "standalone line starts with حكاية/حكايات/قصة/قصّة", "status": "provisional_review_required", "records": {}}
    total = 0
    for path in sorted(args.records.glob("hindawi-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.loads(stream.readline())
        if "ألف ليلة وليلة" not in (record.get("source", {}).get("title") or ""):
            continue
        if record["source"].get("translation"):
            continue
        text = record["text"]
        offset = 0
        line_number = 0
        rows = []
        for line in text.splitlines(keepends=True):
            line_number += 1
            content = line.rstrip("\r\n")
            heading = content.strip()
            if heading and HEADING.match(heading):
                start = offset + len(content) - len(content.lstrip())
                end = start + len(heading)
                rows.append({
                    "schema": "ansg.story_boundary_candidate/1",
                    "candidate_id": f"{record['record_id']}-b{len(rows)+1:03d}",
                    "record_id": record["record_id"],
                    "char_start": start,
                    "char_end": end,
                    "line_number": line_number,
                    "heading": heading,
                    "evidence": "standalone_title_line_prefix",
                    "status": "provisional_needs_human_review",
                    "story_id": None,
                    "parent_story_id": None,
                    "source_text_sha256": record["source_text_sha256"]
                })
            offset += len(line)
        if not rows:
            continue
        files = []
        for part, start in enumerate(range(0, len(rows), 20), 1):
            out = args.output / f"{record['record_id']}_boundaries_part{part:02d}.jsonl"
            out.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows[start:start+20]), encoding="utf-8")
            files.append(out.name)
        report["records"][record["record_id"]] = {"title": record["source"]["title"], "candidate_boundaries": len(rows), "sha256": record["source_text_sha256"], "files": files}
        total += len(rows)
    report["total_candidate_boundaries"] = total
    (args.output / "boundary_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(report["records"]), "total_candidate_boundaries": total, "by_record": {k:v["candidate_boundaries"] for k,v in report["records"].items()}}, ensure_ascii=False))

if __name__ == "__main__":
    main()
