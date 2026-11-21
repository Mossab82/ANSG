#!/usr/bin/env python3
"""Extract user-supplied Kalila wa Dimna chapter candidates to local quarantine.

The output intentionally stays out of git until the edition and text rights are
resolved. Table-of-contents pages are candidate boundaries, not human-adjudicated
story segmentation; private-use glyphs are preserved and counted for review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import unicodedata
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    default_pdf = root.parent / "upload" / "كليلة ودمنة_31354_Foulabook.com_.pdf"
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", type=Path, default=default_pdf)
    parser.add_argument(
        "--index", type=Path,
        default=root / "data/corpora/anhc/source_intake.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data/corpora/anhc/quarantine/kalila_pdf_candidates.jsonl",
    )
    parser.add_argument(
        "--report", type=Path,
        default=root / "data/corpora/anhc/quarantine/kalila_pdf_report.json",
    )
    args = parser.parse_args()

    intake = json.loads(args.index.read_text(encoding="utf-8"))
    source = next(
        row for row in intake["sources"]
        if row.get("work_title_observed") == "كليلة ودمنة"
    )
    actual_sha = sha256(args.pdf)
    if actual_sha != source["sha256"]:
        raise SystemExit("PDF hash differs from the recorded user-supplied source")

    extracted = subprocess.run(
        ["pdftotext", "-layout", str(args.pdf), "-"],
        check=True, capture_output=True, text=True,
    ).stdout
    pages = extracted.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    chapters = source["chapter_index"]
    rows = []
    warnings = []
    for idx, chapter in enumerate(chapters, start=1):
        start = int(chapter["printed_page_start"])
        end = chapter["printed_page_end"]
        end = int(end) if end is not None else len(pages)
        if start < 1 or end < start or end > len(pages):
            warnings.append({"index": idx, "reason": "page_range_outside_extracted_pdf", "start": start, "end": end})
            continue
        text = "\n".join(pages[start - 1:end]).strip()
        if not text:
            warnings.append({"index": idx, "reason": "empty_text_range", "start": start, "end": end})
            continue
        private_use = sum("\ue000" <= ch <= "\uf8ff" for ch in text)
        normalized_text = "".join(
            ch for ch in unicodedata.normalize("NFKC", text)
            if unicodedata.category(ch) != "Cf"
        )
        row = {
            "record_id": f"anhc-kalila-pdf-{idx:03d}",
            "parent_work_id": "user-pdf-kalila-wa-dimna",
            "title_candidate": chapter["title"],
            "unit_kind_candidate": "front_matter" if idx <= 4 else "fable_chapter_or_story_cycle",
            "source": {
                "filename": args.pdf.name,
                "sha256": actual_sha,
                "pdf_page_start": start,
                "pdf_page_end": end,
                "printed_page_start": start,
                "printed_page_end": end,
                "edition": None,
                "copyright_status": "unknown_review_required",
                "license": None,
                "redistributable": False,
                "training_admission": False,
                "local_quarantine_only": True,
            },
            "text": text,
            "whitespace_word_count": len(normalized_text.split()),
            "arabic_character_count_after_nfkc": sum("\u0600" <= ch <= "\u06ff" for ch in normalized_text),
            "private_use_glyph_count": private_use,
            "segmentation_status": "toc_page_range_candidate_not_human_reviewed",
            "nested_story_status": "unreviewed",
            "annotation_status": {"dia": "unannotated", "motif": "unannotated", "moral": "unannotated"},
            "split": None,
        }
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema": "ansg.anhc-local-quarantine-intake/0.1",
        "source_filename": args.pdf.name,
        "source_sha256": actual_sha,
        "pdf_pages": len(pages) - (1 if pages and not pages[-1].strip() else 0),
        "toc_candidates": len(chapters),
        "records_written": len(rows),
        "front_matter_candidates": sum(r["unit_kind_candidate"] == "front_matter" for r in rows),
        "fable_chapter_candidates": sum(r["unit_kind_candidate"] == "fable_chapter_or_story_cycle" for r in rows),
        "whitespace_word_count": sum(r["whitespace_word_count"] for r in rows),
        "arabic_character_count_after_nfkc": sum(r["arabic_character_count_after_nfkc"] for r in rows),
        "private_use_glyph_count": sum(r["private_use_glyph_count"] for r in rows),
        "warnings": warnings,
        "training_admission": False,
        "redistributable": False,
        "output_is_gitignored": True,
        "policy": "Local extraction for structural review only; no rights clearance, annotations, or independent tale segmentation is implied.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
