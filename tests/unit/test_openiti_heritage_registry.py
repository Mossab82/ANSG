"""OpenITI heritage registry: metadata only, everything unresolved stays pending."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "data/sources/openiti/openiti_heritage_registry.json"
EVIDENCE = ROOT / "data/sources/openiti/openiti_metadata_evidence.json"

pytestmark = pytest.mark.skipif(not EVIDENCE.exists(), reason="OpenITI metadata evidence not collected yet")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reg():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_is_current():
    builder = _load("build_openiti_heritage_registry", "scripts/openiti/build_openiti_heritage_registry.py")
    assert builder.main(["--check"]) == 0


def test_registry_never_claims_anhc_or_ingested_text(reg):
    assert reg["status"] == "pending"
    assert reg["is_anhc"] is False and reg["reproduces_thesis_data"] is False and reg["text_committed"] is False
    assert reg["token_comparison"]["ratio"] < 1
    assert "not a claim" in reg["token_comparison"]["units_note"]


def test_every_work_is_pending_with_rights_unresolved(reg):
    assert {w["work_key"] for w in reg["works"]} == {"maqamat_al_hariri", "kitab_al_aghani", "al_jahiz_works"}
    for w in reg["works"]:
        assert w["admission"]["status"] == "pending" and w["admission"]["text_ingested"] is False
        r = w["rights"]
        for key in ("underlying_edition_and_digital_source_rights", "permitted_use_for_ansg_training",
                    "redistribution_in_this_repository"):
            assert r[key] == "pending"
        assert w["openiti_books"], f"no OpenITI book found for {w['work_key']}"


def test_versions_have_identifiers_and_fingerprints_or_pending(reg):
    for w in reg["works"]:
        for b in w["openiti_books"]:
            for v in b["versions"]:
                assert v["version_uri"].startswith(b["book_uri"] + ".")
                if v["fingerprint_status"] == "collected":
                    assert len(v["sha256"]) == 64 and v["bytes"] > 0 and v["repository_commit"]
                else:
                    assert v["fingerprint_status"] == "pending"
                assert v["edition_identity"] in {"catalogued", "pending"}


def test_direct_build_overlap_refers_to_real_records(reg):
    split_ids = {json.loads(l)["record_id"] for l in
                 open(ROOT / "data/corpora/anhc_direct_build/release_v1/split_assignments.jsonl", encoding="utf-8")}
    for w in reg["works"]:
        for b in w["openiti_books"]:
            for o in b["direct_build_work_overlap"]:
                assert o["record_id"] in split_ids
        for s in w["direct_build_author_scan"]:
            assert s["status"] in {"mapped", "not_target", "pending"}
    s = reg["direct_build_overlap_summary"]
    assert set(s["record_ids"]) <= split_ids
    assert s["direct_build_records_also_in_openiti_as_hindawi_versions"] == len(s["record_ids"])


def test_no_text_payload_in_committed_openiti_files():
    files = sorted((ROOT / "data/sources/openiti").iterdir())
    assert {f.suffix for f in files} <= {".json", ".md"}

    def strings(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)
        elif isinstance(o, str):
            yield o

    for f in files:
        if f.suffix == ".json":
            longest = max((len(s) for s in strings(json.loads(f.read_text(encoding="utf-8")))), default=0)
            assert longest < 20000, f"{f.name} holds a string of {longest} characters; text must not be committed"


def test_overlap_records_point_to_existing_review_queues(reg):
    """Each overlapping direct-build record is referenced (not copied) in a PR #28 review queue."""
    builder = _load("build_openiti_heritage_registry", "scripts/openiti/build_openiti_heritage_registry.py")
    idx = builder.review_queue_index()
    seen = 0
    for w in reg["works"]:
        for b in w["openiti_books"]:
            for o in b["direct_build_work_overlap"]:
                seen += 1
                assert o["direct_build_review_queues"], o["record_id"]
                assert o["direct_build_review_queues"] == idx[o["record_id"]]
    assert seen == 6
