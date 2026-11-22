#!/usr/bin/env python3
"""Build a metadata-only intake report for user-supplied ANHC reference PDFs.

The script records file hashes, extractability, and printed chapter starts. It
does not export copyrighted PDF text into the repository or clear reuse rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path


ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return "".join(ch for ch in value if unicodedata.category(ch) != "Cf")


def inspect_pdf(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    text = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True, capture_output=True, text=True).stdout
    text = normalized(text)
    fields = {}
    for line in info.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fields[key.strip()] = val.strip()
    arabic_chars = sum("\u0600" <= ch <= "\u06ff" for ch in text)
    private_use_chars = sum("\ue000" <= ch <= "\uf8ff" for ch in text)
    record = {
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "file_size_bytes": len(raw),
        "pdf_pages": int(fields.get("Pages", "0")),
        "pdf_title_metadata": fields.get("Title"),
        "pdf_author_metadata": fields.get("Author"),
        "pdf_creation_date": fields.get("CreationDate"),
        "extracted_word_count_whitespace": len(text.split()),
        "extracted_arabic_characters": arabic_chars,
        "private_use_glyph_count": private_use_chars,
        "text_extractability": "searchable_arabic_text" if arabic_chars > 10000 else "arabic_ocr_required",
        "rights_status": "unknown_review_required",
        "training_admission": False,
        "redistributable": False,
    }
    return record, text


def kalila_contents(text: str) -> list[dict]:
    pages = text.split("\f")
    contents = next((page for page in reversed(pages) if "الفهرس" in page), "")
    entries = []
    for line in contents.splitlines():
        line = " ".join(line.split())
        match = re.match(r"^(.*?)\s+([0-9٠-٩]{1,3})\s*[.…·]{3,}\s*$", line)
        if not match:
            continue
        title = match.group(1).strip()
        printed_page = int(match.group(2).translate(ARABIC_DIGITS))
        entries.append({"title": title, "printed_page_start": printed_page})
    for i, item in enumerate(entries):
        item["printed_page_end"] = (entries[i + 1]["printed_page_start"] - 1) if i + 1 < len(entries) else None
        item["unit_status"] = "contents_page_candidate_boundary_review_required"
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("nights_pdf", type=Path)
    parser.add_argument("kalila_pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nights, _ = inspect_pdf(args.nights_pdf)
    kalila, kalila_text = inspect_pdf(args.kalila_pdf)
    chapters = kalila_contents(kalila_text)
    report = {
        "schema": "ansg.anhc-source-intake/0.1",
        "dataset_id": "ANHC-reconstruction",
        "status": "source_intake_only_not_rights_cleared",
        "sources": [
            {
                **nights,
                "work_title_observed_on_cover": "ألف ليلة وليلة",
                "bibliographic_edition_identified": False,
                "observations": [
                    "The PDF is primarily page images; pdftotext yielded no usable Arabic text.",
                    "A W. R. Taylor Collection label is visible in the scanned front matter; it is not an edition statement.",
                    "Arabic OCR and bibliographic identification are required before record ingestion.",
                ],
            },
            {
                **kalila,
                "work_title_observed": "كليلة ودمنة",
                "author_attribution_observed": "ابن المقفع",
                "bibliographic_edition_identified": False,
                "chapter_index": chapters,
                "observations": [
                    "Searchable Arabic text was extracted, with private-use glyph substitutions that need correction or review.",
                    "The PDF includes an al-mostafa.com conversion credit; publisher/editor/year are not established by the supplied file metadata.",
                    "Contents-page chapter entries are structural candidates, not human-adjudicated story boundaries.",
                ],
            },
        ],
        "policy": "User-supplied files are source candidates. Their presence does not establish copyright permission for redistribution or training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "nights_pages": nights["pdf_pages"],
        "nights_arabic_chars": nights["extracted_arabic_characters"],
        "kalila_pages": kalila["pdf_pages"],
        "kalila_words": kalila["extracted_word_count_whitespace"],
        "kalila_private_use_glyphs": kalila["private_use_glyph_count"],
        "kalila_toc_entries": len(chapters),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
