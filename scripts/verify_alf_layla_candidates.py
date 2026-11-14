#!/usr/bin/env python3
"""Verify the committed Alf Layla candidate shards against their integrity manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


MANIFEST_REL = Path("data/corpora/anhc_direct_build/alf_layla/story_candidates_manifest.json")
EXPECTED_DOI = "10.5878/7rbh-gy93"
EXPECTED_ANNOTATIONS = {"DIA": None, "MOTIF": None, "MORAL": None}


def git_blob_sha1(data: bytes) -> str:
    """Return the Git object id for a blob containing exactly data."""
    header = b"blob " + str(len(data)).encode("ascii") + bytes([0])
    return hashlib.sha1(header + data).hexdigest()


def fail(errors: list[str]) -> int:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1


def verify(root: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = root / MANIFEST_REL
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read manifest {manifest_path}: {exc}"]

    if manifest.get("schema") != "ansg.alf_layla_story_candidate_integrity_manifest/1":
        errors.append("unexpected manifest schema")
    if manifest.get("dataset_doi") != EXPECTED_DOI:
        errors.append(f"expected dataset DOI {EXPECTED_DOI}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        return errors + ["manifest has no file entries"]

    all_ids: set[str] = set()
    all_sources: set[str] = set()
    all_groups: set[str] = set()
    records = words = pieces = 0

    for entry in files:
        rel = Path(str(entry.get("path", "")))
        if rel.is_absolute() or ".." in rel.parts:
            errors.append(f"unsafe shard path: {rel}")
            continue
        path = root / rel
        try:
            raw = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read {rel}: {exc}")
            continue

        if len(raw) != entry.get("bytes"):
            errors.append(f"{rel}: byte length {len(raw)} != manifest {entry.get('bytes')}")
        actual_blob = git_blob_sha1(raw)
        if actual_blob != entry.get("git_blob_sha1"):
            errors.append(f"{rel}: Git blob SHA-1 mismatch")

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            errors.append(f"{rel}: not UTF-8: {exc}")
            continue

        shard_records = 0
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line:
                continue
            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{rel}:{line_no}: invalid JSON: {exc}")
                continue
            shard_records += 1
            records += 1

            cid = row.get("candidate_id")
            if not isinstance(cid, str) or not cid:
                errors.append(f"{rel}:{line_no}: missing candidate_id")
            elif cid in all_ids:
                errors.append(f"{rel}:{line_no}: duplicate candidate_id {cid}")
            else:
                all_ids.add(cid)

            source = row.get("source") or {}
            rid = source.get("record_id")
            if isinstance(rid, str):
                all_sources.add(rid)
            if source.get("dataset_doi") != EXPECTED_DOI:
                errors.append(f"{rel}:{line_no}: source DOI mismatch")
            if "CC BY 4.0" not in str(source.get("license", "")):
                errors.append(f"{rel}:{line_no}: source license is not recorded as CC BY 4.0")
            if not source.get("attribution"):
                errors.append(f"{rel}:{line_no}: missing attribution")
            if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("source_text_sha256", ""))):
                errors.append(f"{rel}:{line_no}: invalid source SHA-256")

            span = row.get("span") or {}
            start, end = span.get("start_char"), span.get("end_char")
            payload = row.get("text")
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(payload, str):
                errors.append(f"{rel}:{line_no}: invalid span offsets or text")
            elif start < 0 or end <= start or len(payload) != end - start:
                errors.append(f"{rel}:{line_no}: span offsets do not match text length")

            group = row.get("work_group")
            if isinstance(group, str):
                all_groups.add(group)
            count = row.get("counts") or {}
            row_words, row_pieces = count.get("whitespace_words"), count.get("direct_sentencepiece_tokens")
            if not isinstance(row_words, int) or row_words < 0:
                errors.append(f"{rel}:{line_no}: invalid whitespace word count")
            else:
                words += row_words
                if isinstance(payload, str) and len(payload.split()) != row_words:
                    errors.append(f"{rel}:{line_no}: whitespace word count mismatch")
            if not isinstance(row_pieces, int) or row_pieces < 0:
                errors.append(f"{rel}:{line_no}: invalid SentencePiece count")
            else:
                pieces += row_pieces
            if row.get("annotations") != EXPECTED_ANNOTATIONS:
                errors.append(f"{rel}:{line_no}: annotation fields must remain null")

        if shard_records != entry.get("records"):
            errors.append(f"{rel}: {shard_records} records != manifest {entry.get('records')}")

    expected_counts = manifest.get("counts") or {}
    if records != manifest.get("record_count"):
        errors.append(f"total records {records} != manifest {manifest.get('record_count')}")
    if len(all_sources) != manifest.get("source_records"):
        errors.append(f"source records {len(all_sources)} != manifest {manifest.get('source_records')}")
    if words != expected_counts.get("whitespace_words"):
        errors.append(f"whitespace words {words} != manifest {expected_counts.get('whitespace_words')}")
    if pieces != expected_counts.get("direct_sentencepiece_tokens"):
        errors.append(f"SentencePiece pieces {pieces} != manifest {expected_counts.get('direct_sentencepiece_tokens')}")
    if len(all_groups) != 1:
        errors.append(f"expected one locked work group; found {len(all_groups)}")
    if manifest.get("annotation_coverage") != {
        "DIA": 0, "MOTIF": 0, "MORAL": 0, "triple_covered_tokens": 0
    }:
        errors.append("manifest annotation coverage is not all zero")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="repository root (defaults to the parent of scripts/)")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors = verify(root)
    if errors:
        return fail(errors)
    manifest = json.loads((root / MANIFEST_REL).read_text(encoding="utf-8"))
    print(
        f"Verified {manifest['record_count']} Alf Layla candidate records, "
        f"{len(manifest['files'])} shards, and "
        f"{manifest['counts']['direct_sentencepiece_tokens']} direct SentencePiece pieces."
    )
    print("This verifies file identity and metadata consistency; it does not validate story boundaries or annotations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
