#!/usr/bin/env python3
"""Convert the locally downloaded TinyStories Algerian Darija shards to quarantine.

The source corpus labels itself MIT but declares the English source as
roneneldan/TinyStories (CDLA-Sharing-1.0). Records remain quarantined and are
not licensed or quality-cleared for ANHC training by this conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/corpora/anhc/quarantine/external/tinystories-algerian-darija"
DEFAULT_OUTPUT = ROOT / "data/corpora/anhc/quarantine/tinystories_darija_candidates.jsonl"
DEFAULT_MANIFEST = ROOT / "data/corpora/anhc/quarantine/tinystories_darija_intake_manifest.json"
SOURCE_REPO = "touati-kamel/TinyStories-Algerian-Darija"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_REPO}"
UPSTREAM_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
SOURCE_REVISION = "82760dc9de372ff56a531766194f91f78e270520"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    shards = sorted((args.source_dir / "data").glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No Parquet shards found under {args.source_dir / 'data'}")

    rows: list[dict[str, Any]] = []
    text_hashes: Counter[str] = Counter()
    char_lengths: list[int] = []
    word_total = 0
    empty_count = 0
    short_count = 0
    arabic_ratio_low_count = 0
    ids: set[int] = set()
    source_hashes = {p.name: sha256(p) for p in shards}

    for shard in shards:
        table = pq.read_table(shard)
        needed = {"id", "story_algerian_darija"}
        if not needed.issubset(table.column_names):
            raise ValueError(f"Unexpected source schema in {shard.name}: {table.column_names}")
        for record in table.to_pylist():
            raw_text = record.get("story_algerian_darija") or ""
            text = norm(raw_text)
            ext_id = int(record["id"])
            ids.add(ext_id)
            words = text.split()
            ratio = arabic_ratio(text)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            text_hashes[digest] += 1
            word_total += len(words)
            char_lengths.append(len(text))
            empty_count += not bool(text)
            short_count += len(words) < 20
            arabic_ratio_low_count += ratio < 0.65
            record_id = f"tinystories-dz-{ext_id:06d}"
            rows.append({
                "schema_version": "ansg.anhc-record/0.1",
                "record_id": record_id,
                "text": text,
                "source": {
                    "source_id": "tinystories_algerian_darija_synthetic",
                    "title": "TinyStories Algerian Darija",
    "url": f"{SOURCE_URL}/tree/{SOURCE_REVISION}",
                    "license": "MIT (as declared by dataset card; upstream TinyStories is declared CDLA-Sharing-1.0)",
                    "copyright_status": "rights_review_required_derivative_source_terms_not_adjudicated",
                    "redistributable": False,
                    "file_path": str(shard.relative_to(ROOT)),
                    "file_sha256": source_hashes[shard.name],
                    "edition": None,
                    "editor": None,
                    "publisher": None,
                    "year": None,
                    "notes": "Synthetic Algerian Darija adaptation of TinyStories according to source card; local quarantine only; no human quality review or training clearance.",
                },
                "narrative": {
                    "collection": "External synthetic dialect story candidates",
                    "title": f"Synthetic story {ext_id}",
                    "story_index": ext_id,
                    "parent_story_id": None,
                    "nesting_level": None,
                    "night_numbers": [],
                    "formulae": [],
                    "source_span": None,
                    "extra": {
                        "source_record_id": str(ext_id),
                        "source_dataset_id": SOURCE_REPO,
                        "upstream_dataset_id": "roneneldan/TinyStories",
                        "origin_type": "synthetic_translation_and_cultural_adaptation",
                        "source_dialect_label": "Algerian Darija (unverified source label)",
                        "source_theme": record.get("theme"),
                        "source_age_group": record.get("age_group"),
                        "source_moral_lesson": record.get("moral_lesson_darija"),
                        "source_english_word_count": record.get("english_word_count"),
                        "source_darija_word_count": record.get("darija_word_count"),
                        "human_review_required": True,
                        "quality_gate_status": "not_reviewed",
                        "rights_gate_status": "blocked_pending_upstream_and_derivative_terms_review",
                        "text_sha256": digest,
                    },
                },
                "dia": [],
                "moral": [],
                "motif": [],
                "annotation_status": {
                    "dia": "source_label_only_not_gold",
                    "moral": "source_metadata_only_not_gold",
                    "motif": "unannotated",
                },
                "split": None,
                "normalization_config_sha256": None,
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "schema": "ansg.anhc-external-intake-manifest/0.1",
        "dataset_id": "ANHC-TinyStories-Darija-candidates",
        "status": "downloaded_local_quarantine_not_training_cleared",
        "source_dataset": SOURCE_REPO,
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "upstream_dataset": "roneneldan/TinyStories",
        "upstream_url": UPSTREAM_URL,
        "source_license_claim": "MIT",
        "upstream_license_claim": "CDLA-Sharing-1.0",
        "source_card_claims_synthetic": True,
        "records": len(rows),
        "unique_source_ids": len(ids),
        "duplicate_source_id_count": len(rows) - len(ids),
        "duplicate_text_record_count": sum(n - 1 for n in text_hashes.values() if n > 1),
        "unique_text_count": len(text_hashes),
        "candidate_whitespace_word_count": word_total,
        "candidate_char_count": sum(char_lengths),
        "empty_text_count": empty_count,
        "under_20_word_count": short_count,
        "arabic_letter_ratio_below_0_65_count": arabic_ratio_low_count,
        "human_review_status": "not_reviewed",
        "rights_review_status": "blocked_pending_upstream_and_derivative_terms_review",
        "redistributable": False,
        "token_count_status": "not_computed; tokenizer-specific count required",
        "split_status": "unassigned",
        "source_shards": [
            {"path": str((args.source_dir / "data" / name).relative_to(ROOT)), "sha256": digest}
            for name, digest in sorted(source_hashes.items())
        ],
        "output_local_quarantine": str(args.output.relative_to(ROOT)),
        "limitations": [
            "This is a synthetic children-story cohort in one source-labeled dialect only.",
            "The source license label does not resolve obligations inherited from the declared TinyStories upstream source.",
            "The dataset card's authenticity and quality claims are not treated as human validation or ANHC gold labels.",
            "This intake does not establish the thesis corpus size, dialect breadth, annotation volume, or training readiness.",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in (
        "records", "unique_source_ids", "duplicate_source_id_count", "duplicate_text_record_count",
        "candidate_whitespace_word_count", "empty_text_count", "under_20_word_count",
        "arabic_letter_ratio_below_0_65_count", "rights_review_status", "output_local_quarantine",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
