#!/usr/bin/env python3
"""Audit the archive structure and provenance of an ANSG story seed pack.

This verifies metadata and file integrity only. It does not establish legal
permission, historical ANHC equivalence, story boundaries, or gold annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile

MANIFEST_SCHEMA = "ansg.story_seed_pack/1"
DATASETS = {
    "data/alf_layla_story_candidates.jsonl": ("alf_layla_candidate_spans", "candidate_id", "text"),
    "data/kalila_1816_source_chapters.jsonl": ("kalila_1816_source_chapters", "record_id", "text"),
    "data/kalila_nested_story_openings_review.jsonl": ("kalila_nested_opening_cues", "marker_id", "cue"),
}
ANNOTATION_KEYS = ("DIA", "MOTIF", "MORAL")
MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def audit_archive(archive_path: str) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    character_counts: dict[str, int] = {}
    provenance_rows = 0
    unannotated_rows = 0
    chapter_ids: set[str] = set()

    try:
        with ZipFile(archive_path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                errors.append("archive contains duplicate member names")
            if any(not _safe_member(name) for name in names):
                errors.append("archive contains an unsafe member path")
            if sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
                errors.append("archive exceeds the 100 MiB uncompressed safety limit")
            if errors:
                return _report(errors, warnings, counts, character_counts, provenance_rows, unannotated_rows)

            manifests = [name for name in names if PurePosixPath(name).name == "manifest.json"]
            if len(manifests) != 1:
                errors.append(f"expected one manifest.json, found {len(manifests)}")
                return _report(errors, warnings, counts, character_counts, provenance_rows, unannotated_rows)
            manifest_name = manifests[0]
            root = str(PurePosixPath(manifest_name).parent)
            root = "" if root == "." else root + "/"
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
            if manifest.get("schema") != MANIFEST_SCHEMA:
                errors.append(f"unsupported manifest schema: {manifest.get('schema')!r}")

            listed = {}
            for entry in manifest.get("files", []):
                rel = entry.get("path", "")
                if not rel or not _safe_member(rel) or rel.startswith("/"):
                    errors.append(f"unsafe manifest path: {rel!r}")
                    continue
                listed[rel] = entry
                member_name = root + rel
                if member_name not in names:
                    errors.append(f"manifest file is missing from archive: {rel}")
                    continue
                data = archive.read(member_name)
                if len(data) != entry.get("bytes"):
                    errors.append(f"byte count mismatch: {rel}")
                if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                    errors.append(f"SHA-256 mismatch: {rel}")

            expected_members = {root + rel for rel in listed}
            actual_payload = {name for name in names if name != manifest_name}
            if actual_payload != expected_members:
                extra = sorted(actual_payload - expected_members)
                missing = sorted(expected_members - actual_payload)
                if extra:
                    errors.append(f"unlisted archive files: {extra}")
                if missing:
                    errors.append(f"listed archive files are absent: {missing}")

            manifest_counts = manifest.get("counts", {})
            for rel, (count_key, id_key, row_kind) in DATASETS.items():
                member_name = root + rel
                if member_name not in names:
                    errors.append(f"required dataset is missing: {rel}")
                    continue
                rows = []
                for line_number, line in enumerate(archive.read(member_name).decode("utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        errors.append(f"{rel}:{line_number}: invalid JSON: {exc.msg}")
                        continue
                    if not isinstance(row, dict):
                        errors.append(f"{rel}:{line_number}: JSON row must be an object")
                        continue
                    rows.append(row)
                    if not row.get(id_key):
                        errors.append(f"{rel}:{line_number}: missing {id_key}")
                    source = row.get("source") or {}
                    if row_kind == "cue":
                        parent_id = row.get("source_record_id")
                        if not parent_id:
                            errors.append(f"{rel}:{line_number}: missing source_record_id")
                        elif parent_id not in chapter_ids:
                            errors.append(f"{rel}:{line_number}: unknown source chapter {parent_id!r}")
                    elif source.get("license") and source.get("attribution"):
                        provenance_rows += 1
                    else:
                        errors.append(f"{rel}:{line_number}: missing source license or attribution")
                    annotations = row.get("annotations") or {}
                    if all(annotations.get(key) is None for key in ANNOTATION_KEYS):
                        unannotated_rows += 1
                    else:
                        errors.append(f"{rel}:{line_number}: unexpected DIA/MOTIF/MORAL annotation")
                    if row_kind == "text" and (
                        not isinstance(row.get("text"), str) or not row["text"].strip()
                    ):
                        errors.append(f"{rel}:{line_number}: missing text")
                if len({row.get(id_key) for row in rows if row.get(id_key)}) != len(rows):
                    errors.append(f"{rel}: duplicate record identifiers")
                if row_kind == "text" and id_key == "record_id":
                    chapter_ids = {row[id_key] for row in rows if row.get(id_key)}
                counts[count_key] = len(rows)
                character_counts[count_key] = sum(len(row.get("text", "")) for row in rows
                                                   if isinstance(row.get("text"), str))
                if manifest_counts.get(count_key) != len(rows):
                    errors.append(
                        f"{rel}: manifest count {manifest_counts.get(count_key)!r} "
                        f"does not match {len(rows)} rows"
                    )

            if manifest_counts.get("human_adjudicated_annotations") != 0:
                errors.append("manifest must report zero human-adjudicated annotations")
            if sum(character_counts.values()) == 0:
                errors.append("no usable text was found")

    except (OSError, BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read story seed pack: {exc}")

    if unannotated_rows != sum(counts.values()):
        warnings.append("some records are annotated or lack a complete annotation map")
    return _report(errors, warnings, counts, character_counts, provenance_rows, unannotated_rows)


def _report(errors, warnings, counts, character_counts, provenance_rows, unannotated_rows):
    return {
        "schema": "ansg.story_seed_pack_audit/1",
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "record_counts": counts,
        "text_character_counts": character_counts,
        "records_with_source_license_and_attribution": provenance_rows,
        "records_with_unannotated_DIA_MOTIF_MORAL": unannotated_rows,
        "claims": {
            "is_historical_anhc": False,
            "has_gold_story_segmentation": False,
            "license_permission_verified_by_this_audit": False,
            "training_started": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to ANSG_story_seed_pack_v1.zip")
    parser.add_argument("--json", action="store_true", help="print the complete audit as JSON")
    args = parser.parse_args(argv)
    result = audit_archive(args.archive)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"story seed pack audit: {result['status']}")
        print(f"records: {sum(result['record_counts'].values())}")
        print(f"text characters: {sum(result['text_character_counts'].values())}")
        print(f"source-attributed records: {result['records_with_source_license_and_attribution']}")
        for error in result["errors"]:
            print(f"error: {error}", file=sys.stderr)
        for warning in result["warnings"]:
            print(f"warning: {warning}", file=sys.stderr)
        print("This checks archive integrity/provenance fields only; it does not authorize training.")
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
