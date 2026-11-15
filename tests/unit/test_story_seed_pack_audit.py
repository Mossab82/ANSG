"""Tests for story seed pack integrity checks using tiny temporary archives."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.audit_story_seed_pack import audit_archive


def _archive(path: Path, *, tamper=False, bad_parent=False, bad_annotations=False):
    root = "seedpack"
    rows = {
        "data/alf_layla_story_candidates.jsonl": [{
            "candidate_id": "alf-1", "text": "حكاية", "status": "candidate",
            "source": {"license": "CC BY 4.0", "attribution": "Corpus source"},
            "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
        }],
        "data/kalila_1816_source_chapters.jsonl": [{
            "record_id": "kalila-1", "text": "باب الأسد والثور",
            "source": {"license": "CC BY-SA 4.0", "attribution": "Edition source"},
            "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
        }],
        "data/kalila_nested_story_openings_review.jsonl": [{
            "marker_id": "cue-1", "source_record_id": "missing" if bad_parent else "kalila-1",
            "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
        }],
    }
    if bad_annotations:
        rows["data/alf_layla_story_candidates.jsonl"][0].pop("annotations")

    payloads = {"README.md": b"test pack\n"}
    for name, records in rows.items():
        payloads[name] = ("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
                          .encode("utf-8"))
    manifest = {
        "schema": "ansg.story_seed_pack/1",
        "counts": {
            "alf_layla_candidate_spans": 1,
            "kalila_1816_source_chapters": 1,
            "kalila_nested_opening_cues": 1,
            "human_adjudicated_annotations": 0,
        },
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        ],
    }
    with ZipFile(path, "w") as archive:
        for name, data in payloads.items():
            if tamper and name == "README.md":
                data += b"changed"
            archive.writestr(f"{root}/{name}", data)
        archive.writestr(
            f"{root}/manifest.json",
            json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
        )


class StorySeedPackAuditTests(unittest.TestCase):
    def test_accepts_a_valid_attributed_pack_without_inventing_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.zip"
            _archive(path)
            result = audit_archive(str(path))

        self.assertEqual(result["status"], "pass")
        self.assertEqual(sum(result["record_counts"].values()), 3)
        self.assertEqual(result["records_with_source_license_and_attribution"], 2)
        self.assertEqual(result["records_with_unannotated_DIA_MOTIF_MORAL"], 3)
        self.assertFalse(result["claims"]["is_historical_anhc"])
        self.assertFalse(result["claims"]["license_permission_verified_by_this_audit"])

    def test_rejects_a_payload_that_does_not_match_manifest_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.zip"
            _archive(path, tamper=True)
            result = audit_archive(str(path))

        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("SHA-256 mismatch" in error for error in result["errors"]))

    def test_rejects_missing_explicit_annotation_placeholders(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.zip"
            _archive(path, bad_annotations=True)
            result = audit_archive(str(path))

        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("explicit DIA/MOTIF/MORAL" in error for error in result["errors"]))

    def test_rejects_a_nested_cue_without_a_known_source_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed.zip"
            _archive(path, bad_parent=True)
            result = audit_archive(str(path))

        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("unknown source chapter" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
