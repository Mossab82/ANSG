#!/usr/bin/env python3
"""Materialize narrative-candidate books from the CC-BY Arabic E-Book Corpus."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_URL = "https://open.doris.gu.snd.se/downloads/public/2024-145/1/data/corpus-txt.zip"
METADATA_URL = "https://open.doris.gu.snd.se/downloads/public/2024-145/1/documentation/metadata.tsv"
DATASET_DOI = "https://doi.org/10.5878/7rbh-gy93"
ARTICLE_DOI = "https://doi.org/10.1016/j.dib.2025.111456"
CATEGORIES = {
    "narrative_core": {"novels", "detective.fiction", "science.fiction", "children.stories", "plays", "literature"},
    "narrative_adjacent": {"history", "biographies", "travel.literature"},
}
UA = "ANSGResearch/0.4 (reproducible Arabic narrative-source intake; https://github.com/Mossab82/ANSG)"


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)


def period(year_text: str) -> str:
    try:
        year = int(year_text[:4])
    except (TypeError, ValueError):
        return "unknown"
    if year < 1800:
        return "classical_pre_1800"
    if year <= 2000:
        return "modern_1800_2000"
    return "contemporary_post_2000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path("arabic-corpus-txt.zip"))
    parser.add_argument("--metadata", type=Path, default=Path("metadata.tsv"))
    parser.add_argument("--output", type=Path, default=Path("anhc_direct_build/arabic_ebooks"))
    args = parser.parse_args()
    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    if not args.zip_path.exists():
        download(DATA_URL, args.zip_path)
    if not args.metadata.exists():
        download(METADATA_URL, args.metadata)
    args.output.mkdir(parents=True, exist_ok=True)
    records_dir = args.output / "records"
    records_dir.mkdir(exist_ok=True)

    with args.metadata.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    selected = []
    for row in rows:
        tags = set((row.get("category") or "").split(","))
        scopes = [name for name, allowed in CATEGORIES.items() if tags & allowed]
        if scopes:
            row["_scopes"] = scopes
            selected.append(row)

    by_scope = Counter()
    by_scope_words = Counter()
    by_category = Counter()
    by_category_words = Counter()
    by_period = Counter()
    by_period_words = Counter()
    by_translation = Counter()
    by_translation_words = Counter()
    total_words_metadata = 0
    total_words_observed = 0
    records, missing = [], []
    with zipfile.ZipFile(args.zip_path) as archive:
        names = set(archive.namelist())
        for row in selected:
            book_id = row["booknr"]
            member = f"corpus-txt/{book_id}.txt"
            if member not in names:
                missing.append(book_id)
                continue
            raw = archive.read(member)
            text = raw.decode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()
            tags = sorted(set((row.get("category") or "").split(",")))
            scope = "narrative_core" if "narrative_core" in row["_scopes"] else "narrative_adjacent"
            year = row.get("origpubdate") or ""
            era = period(year)
            record = {
                "schema": "ansg.narrative_candidate/1",
                "record_id": f"hindawi-{book_id}",
                "text": text,
                "source": {
                    "dataset": "The Arabic E-Book Corpus", "dataset_version": 1,
                    "dataset_doi": DATASET_DOI, "article_doi": ARTICLE_DOI,
                    "dataset_author": "Andreas Hallberg; University of Gothenburg",
                    "original_source": f"https://www.hindawi.org/books/{book_id}/",
                    "book_id": book_id, "title": row.get("title"), "author": row.get("author"),
                    "publication_date": row.get("pubdate"), "original_publication_year": year or None,
                    "original_title": row.get("origtitle") or None,
                    "original_author": row.get("origauthor") or None,
                    "original_language": row.get("origlang") or None,
                    "translation": row.get("translation") == "TRUE",
                    "license": "CC BY 4.0 (dataset; the dataset README states included works/translations are public domain or CC BY 4.0)",
                    "attribution": "Hallberg, A. (2024). The Arabic E-Book Corpus (Version 1). University of Gothenburg. DOI: 10.5878/7rbh-gy93; original book metadata retained.",
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "source_file_sha256": sha
                },
                "narrative": {
                    "candidate_scope": scope, "publisher_categories": tags,
                    "period": era, "period_basis": "original_publication_year_from_dataset_metadata",
                    "language": "ar", "dialect": "unannotated; edition-specific",
                    "unit": "whole_book_candidate; needs story-boundary review",
                    "admission_status": "candidate_not_yet_story_segmented_or_annotated"
                },
                "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
                "split": "unassigned",
                "counts": {"metadata_word_count": int(row.get("wc") or 0), "observed_whitespace_words": len(text.split()), "utf8_bytes": len(raw)},
                "source_text_sha256": sha
            }
            rel = f"records/{record['record_id']}.jsonl.gz"
            payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            (args.output / rel).write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
            records.append({k: v for k, v in record.items() if k != "text"} | {"text_file": rel})
            total_words_metadata += int(row.get("wc") or 0)
            total_words_observed += len(text.split())
            observed_words = len(text.split())
            by_category.update(tags)
            for category in tags:
                by_category_words[category] += observed_words
            by_period[era] += 1
            by_period_words[era] += observed_words
            translation_key = "translated" if row.get("translation") == "TRUE" else "Arabic_original"
            by_translation[translation_key] += 1
            by_translation_words[translation_key] += observed_words
            for item_scope in row["_scopes"]:
                by_scope[item_scope] += 1
                by_scope_words[item_scope] += observed_words

    manifest = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    (args.output / "manifest.jsonl").write_text(manifest, encoding="utf-8")
    alf_books = [row for row in records if "ألف ليلة وليلة" in (row.get("source", {}).get("title") or "")]
    if alf_books:
        report_alf = {
            "work": "Alf Layla wa-Layla",
            "books": [{"record_id": row["record_id"], "title": row["source"]["title"],
                       "book_id": row["source"]["book_id"], "translation": row["source"]["translation"],
                       "original_language": row["source"]["original_language"],
                       "observed_whitespace_words": row["counts"]["observed_whitespace_words"],
                       "source_text_sha256": row["source_text_sha256"]} for row in alf_books],
            "arabic_original_volume_count": sum(not row["source"]["translation"] for row in alf_books),
            "arabic_original_observed_whitespace_words": sum(row["counts"]["observed_whitespace_words"] for row in alf_books if not row["source"]["translation"]),
            "all_book_observed_whitespace_words": sum(row["counts"]["observed_whitespace_words"] for row in alf_books),
            "note": "Whole-book records; run segment_alflayla_heading_candidates.py for provisional title-line markers."
        }
    else:
        report_alf = None
    report = {
        "schema": "ansg.source_ingestion_report/1", "built_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "The Arabic E-Book Corpus", "version": 1, "dataset_doi": DATASET_DOI,
        "article_doi": ARTICLE_DOI, "license": "CC BY 4.0",
        "archive_sha256": hashlib.sha256(args.zip_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "archive_books": len(rows), "selected_candidates": len(selected), "materialized_records": len(records),
        "missing_archive_files": missing, "scope_record_counts_nonexclusive": dict(by_scope),
        "scope_whitespace_word_counts_nonexclusive": dict(by_scope_words),
        "publisher_category_counts_nonexclusive": dict(by_category),
        "category_whitespace_word_counts_nonexclusive": dict(by_category_words),
        "period_record_counts": dict(by_period), "period_whitespace_word_counts": dict(by_period_words),
        "translation_counts": dict(by_translation), "translation_whitespace_word_counts": dict(by_translation_words),
        "featured_narrative_work": report_alf,
        "metadata_reported_words": total_words_metadata, "observed_whitespace_words": total_words_observed,
        "token_count": None, "tokenizer": None,
        "limitations": ["publisher genres are candidate filters, not verified story-level segmentation", "books remain unsegmented, unannotated, and unassigned to splits", "this source is not a dialect corpus", "do not equate word counts or the source catalogue token figure with the thesis tokenizer count"]
    }
    (args.output / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
