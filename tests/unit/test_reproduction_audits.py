"""Reproduction audit tooling: registry, rights gate, pilot preflight, corpus/split audits, reported values.

These tests run on the REAL committed classical corpus (data/corpora/classical_arabic) only to
check plumbing and audit logic; they make no quality claim. Synthetic rows are labelled.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def registry():
    return json.loads((ROOT / "data/sources/source_registry.json").read_text("utf-8"))


# ------------------------------------------------------------------ source registry
def test_registry_is_generated_and_complete(registry, tmp_path, monkeypatch):
    builder = load("data/sources/build_source_registry.py", "bsr")
    monkeypatch.setattr(builder, "HERE", tmp_path)
    builder.main()
    assert json.loads((tmp_path / "source_registry.json").read_text("utf-8")) == registry, \
        "source_registry.json is stale: rerun data/sources/build_source_registry.py"
    ids = [r["id"] for r in registry["resources"]]
    assert len(ids) == len(set(ids))
    for r in registry["resources"]:
        assert r["category"] in registry["categories"] and r["permission_status"] in registry["permission_statuses"]
        assert set(r["allowed_uses"]) <= set(registry["uses"])
    dialect = json.loads((ROOT / "data/sources/dialect_registry.json").read_text("utf-8"))
    mapped = {r["dialect_registry_id"] for r in registry["resources"] if r["dialect_registry_id"]}
    assert {d["id"] for d in dialect["datasets"]} <= mapped, "every dialect_registry entry needs a registry row"


def test_madar_is_thesis_traceable_but_never_committed(registry):
    madar = next(r for r in registry["resources"] if r["id"] == "madar")
    assert madar["category"] == "thesis_original" and madar["redistribution_allowed"] is False
    assert madar["permission_status"] == "requires_license_agreement" and "model_training" not in madar["allowed_uses"]
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    assert not [f for f in tracked if "madar" in f.lower() and not f.endswith((".md", ".py", ".json", ".yaml"))]


def test_thesis_original_resources_are_not_marked_available(registry):
    for r in registry["resources"]:
        if r["category"] == "thesis_original" and r["id"] != "madar":
            assert r["permission_status"] == "unavailable", r["id"]


# ------------------------------------------------------------------ pilot preflight and rights gate
def test_preflight_runs_the_real_data_path():
    pre = load("scripts/preflight_heritage_pilot.py", "pre")
    report = pre.run(require_rights=False)
    assert report["status"] == "data_path_ok"
    assert all(n > 0 for n in report["chunks_with_whitespace_stand_in"].values())
    assert report["rights_gate"]["passed"] is False            # owner review pending
    assert pre.run(require_rights=True)["status"] == "blocked_by_rights_gate"


def test_rights_gate_passes_only_with_verified_training_permission(tmp_path, registry):
    pre = load("scripts/preflight_heritage_pilot.py", "pre2")
    reg = json.loads(json.dumps(registry))
    res = next(r for r in reg["resources"] if r["id"] == pre.CORPUS_RESOURCE)
    for status, uses, expected in (("verified", ["model_training"], True), ("verified", ["evaluation"], False),
                                   ("pending_owner_review", ["model_training"], False)):
        res["permission_status"], res["allowed_uses"] = status, uses
        p = tmp_path / "reg.json"
        p.write_text(json.dumps(reg), encoding="utf-8")
        assert pre.rights_gate(p)["passed"] is expected


def test_old_night_keyed_chunking_failure_is_fixed():
    """Run 43561944084: KeyError 'night'. SYNTHETIC rows with only the corpus's real keys."""
    pilot = load("kaggle/heritage/ansg_heritage_pilot.py", "pilot")
    tok = lambda t, add_special_tokens=False: {"input_ids": list(range(len(t.split())))}  # noqa: E731
    rows = [{"id": "synthetic-a", "text": " ".join(["كلمة"] * 600)}]
    chunks = pilot.chunks(rows, tok, max_length=256)
    assert [c["group"] for c in chunks] == ["synthetic-a", "synthetic-a"]
    leak = {"train": rows, "validation": rows, "test": rows}
    with pytest.raises(ValueError, match="leakage"):
        pilot.prepare_data(leak, tok, max_length=256)


# ------------------------------------------------------------------ split and corpus audits
def test_pilot_split_is_the_derived_split_and_leak_free():
    dps = load("scripts/reproduction/derive_pilot_split.py", "dps")
    d = dps.derive()
    assert dps.check(d) == []
    merges = {tuple(m["pages"]) for m in d["group_merges"]}
    assert ("kalila_dimna-007", "kalila_dimna-019") in merges and ("kalila_dimna-010", "kalila_dimna-017") in merges
    for a, b in merges:
        assert {s for s, ids in d["split"].items() if a in ids} == {s for s, ids in d["split"].items() if b in ids}


@pytest.mark.parametrize("script", ["scripts/reproduction/audit_classical_corpus.py",
                                    "scripts/reproduction/audit_reported_values.py",
                                    "scripts/reproduction/build_story_records.py"])
def test_committed_audit_outputs_are_current(script):
    r = subprocess.run([sys.executable, str(ROOT / script), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_corpus_audit_findings():
    audit = json.loads((ROOT / "evidence/corpus/classical_corpus_audit.json").read_text("utf-8"))
    assert audit["verdict"]["equivalent_to_anhc"] is False
    assert audit["pilot_split_v1_leaked_pages"] == ["kalila_dimna-019"] and audit["pilot_split_leaked_pages"] == []
    assert audit["ratios"]["corpus_words_over_anhc_tokens_pct"] < 1
    assert audit["ali_baba_mentions"] == 0


def test_story_records_never_claim_anhc_or_dialect_labels():
    recs = json.loads((ROOT / "data/corpora/classical_arabic/story_records.json").read_text("utf-8"))["records"]
    assert recs and all(r["is_anhc"] is False and r["dialect"] is None and r["dialect_status"] == "unannotated"
                        for r in recs)
    assert all(r["nesting"]["evidence"] in (None, "heading_hierarchy") for r in recs)   # cues never accepted


def test_reported_values_audit_flags_known_inconsistencies():
    audit = json.loads((ROOT / "evidence/reproduction/reported_values_audit.json").read_text("utf-8"))
    status = {r["id"]: r["status"] for r in audit["results"]}
    assert status["GAP-CLOSURE-EQ6.1"] == "CONSISTENT"
    assert status["A20-NARRATIVES-VS-S5.4.4"] == "INCONSISTENT"
    assert status["MADAR-VARIETIES"] == "INCONSISTENT"


# ------------------------------------------------------------------ experimental dialect ID
def test_dialect_identifier_requires_opt_in_and_labels_output():
    from ansg.evaluation.dialect_id import CamelToolsBackend, ExperimentalDialectIdentifier

    class Pred:   # SYNTHETIC stand-in for a classifier output
        def __init__(self, top):
            self.top, self.scores = top, {top: 0.9}

    class Model:
        def predict(self, s):
            return [Pred("CAI") for _ in s]

    backend = CamelToolsBackend(loader=Model)
    with pytest.raises(PermissionError):
        ExperimentalDialectIdentifier(backend)
    out = ExperimentalDialectIdentifier(backend, experimental=True).predict(["ازيك عامل ايه"])
    assert out[0].label == "CAI" and out[0].status == "experimental_supplementary" and not out[0].reproduces_thesis
