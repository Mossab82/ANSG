"""Generation labels, artefact manifests and the readiness gate (Issue #15). Torch-free.

Backbone directories here are FAKE: JSON config/tokenizer files plus placeholder
weight bytes. The readiness gate never loads weights, so this exercises its logic
without PyTorch or any download. Nothing here is a real checkpoint.
"""
import hashlib
import json

import pytest

from ansg.data.training_examples import SchemaVocabulary
from ansg.generation import labels as L
from ansg.generation.artifacts import (
    BACKBONE_PROVENANCE_FILE, cultural_compatibility_problems, cultural_manifest_path, digest_of, file_inventory,
    inspect_backbone, read_backbone_provenance, tokenizer_json_size, validate_cultural_manifest,
)
from ansg.generation.readiness import assess, main as readiness_main
from ansg.knowledge.ckg import CulturalKnowledgeGraph
from ansg.utils.provenance import sha256_file

H = "a" * 64


# ----------------------------------------------------------------- helpers
def fake_backbone(root, *, synthetic=False, digest=True, provenance=True, vocab=10, tok_vocab=8):
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2",
                                                  "vocab_size": vocab, "n_embd": 12, "n_positions": 64}))
    (root / "model.safetensors").write_bytes(b"FAKE WEIGHTS - test placeholder")
    (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {f"t{i}": i for i in range(tok_vocab)}},
                                                     "added_tokens": []}))
    if provenance:
        prov = {"name": "fake-test-backbone", "source": "unit test", "license": "test only",
                "synthetic": synthetic, "language": "ar"}
        if digest:
            prov["directory_digest"] = digest_of(file_inventory(root))
        (root / BACKBONE_PROVENANCE_FILE).write_text(json.dumps(prov))
    return root


def cultural_manifest(ckpt, backbone_info, vocab_sha, *, status="trained", steps=5, data_kind="synthetic",
                      data=None):
    return {
        "format": "ansg-cultural-modules", "format_version": 2, "checkpoint_sha256": sha256_file(ckpt),
        "architecture": {"schema_count": 4, "schema_dim": 8, "cultural_heads": 12, "lora_rank": 2, "lora_alpha": 4.0},
        "backbone": {"directory_digest": backbone_info["directory_digest"]},
        "tokenizer": {"digest": backbone_info["tokenizer_digest"]},
        "schema_vocabulary": {"sha256": vocab_sha},
        "training": {"status": status, "optimizer_steps": steps, "data_kind": data_kind,
                     "data_sha256": data if data is not None else {"examples": H}},
    }


@pytest.fixture
def schema_setup(tmp_path, fixtures):
    """Synthetic CKG -> schema vocabulary -> a fake examples manifest that lists them."""
    ckg = fixtures / "ckg" / "synthetic_ckg.json"
    vocab = SchemaVocabulary.from_graph(CulturalKnowledgeGraph.from_json(ckg), source_path=ckg)
    ex = tmp_path / "examples"
    ex.mkdir()
    (ex / "schema_vocabulary.json").write_text(json.dumps(vocab.to_dict()))
    (ex / "train.examples.jsonl").write_text('{"text": "نص تركيبي", "schema_ids": [], "dialect_id": null}\n',
                                            encoding="utf-8")
    (ex / "manifest.json").write_text(json.dumps({
        "inputs": {str(ckg): sha256_file(ckg)},
        "outputs": {str(ex / n): sha256_file(ex / n) for n in ("schema_vocabulary.json", "train.examples.jsonl")}}))
    return {"ckg": ckg, "dir": ex, "vocab": ex / "schema_vocabulary.json",
            "data": {"examples": sha256_file(ex / "train.examples.jsonl"),
                     "schema_vocabulary": sha256_file(ex / "schema_vocabulary.json")}}


def write_cultural(tmp_path, backbone, vocab_path, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "cultural.pt"
    ckpt.write_bytes(b"FAKE CULTURAL TENSORS - test placeholder")
    manifest = cultural_manifest(ckpt, inspect_backbone(backbone), sha256_file(vocab_path), **kw)
    cultural_manifest_path(ckpt).write_text(json.dumps(manifest))
    return ckpt


# ----------------------------------------------------------------- labels
def test_mode_is_derived_from_evidence_only():
    trained = {"status": "trained", "optimizer_steps": 3, "data_sha256": {"x": H}, "data_kind": "real"}
    assert L.classify_mode(backbone_synthetic=True, cultural_training=trained, cultural_loaded=True) \
        is L.GenerationMode.SYNTHETIC_STUB
    assert L.classify_mode(backbone_synthetic=False, cultural_training=None, cultural_loaded=False) \
        is L.GenerationMode.BACKBONE_ONLY
    assert L.classify_mode(backbone_synthetic=False, cultural_training=None, cultural_loaded=True) \
        is L.GenerationMode.ANSG_UNTRAINED
    assert L.classify_mode(backbone_synthetic=False, cultural_training={**trained, "data_kind": "synthetic"},
                           cultural_loaded=True) is L.GenerationMode.ANSG_TRAINED_SYNTHETIC_ONLY
    assert L.classify_mode(backbone_synthetic=False, cultural_training=trained, cultural_loaded=True) \
        is L.GenerationMode.ANSG_TRAINED
    for bad in ({**trained, "optimizer_steps": 0}, {**trained, "optimizer_steps": True},
                {**trained, "data_sha256": {}}, {**trained, "status": "untrained"}):
        assert L.classify_mode(backbone_synthetic=False, cultural_training=bad, cultural_loaded=True) \
            is L.GenerationMode.ANSG_UNTRAINED


@pytest.mark.parametrize("mode", list(L.GenerationMode))
def test_no_label_claims_a_dissertation_result(mode):
    block = L.label_block(mode, dialect_requested=True)
    assert block["reproduces_dissertation"] is False and block["dissertation_result"] is False
    assert block["dialect_adaptation_claimed"] is (mode is L.GenerationMode.ANSG_TRAINED)
    assert block["cultural_conditioning_claimed"] is (mode is L.GenerationMode.ANSG_TRAINED)
    if mode is not L.GenerationMode.ANSG_TRAINED:
        assert block["warnings"], "dialect requested without trained DAL must warn"


def test_inflated_labels_are_rejected():
    untrained = L.label_block(L.GenerationMode.ANSG_UNTRAINED)
    L.validate_label(untrained, backbone_synthetic=False, cultural_loaded=True, cultural_training=None)
    promoted = {**untrained, "mode": "ansg_trained"}
    with pytest.raises(ValueError, match="not supported by the evidence"):
        L.validate_label(promoted, backbone_synthetic=False, cultural_loaded=True, cultural_training=None)
    with pytest.raises(ValueError, match="dissertation"):
        L.validate_label({**untrained, "reproduces_dissertation": True}, backbone_synthetic=False,
                         cultural_loaded=True, cultural_training=None)
    with pytest.raises(ValueError, match="cultural conditioning"):
        L.validate_label({**untrained, "cultural_conditioning_claimed": True}, backbone_synthetic=False,
                         cultural_loaded=True, cultural_training=None)


# ----------------------------------------------------------------- artefacts
def test_directory_digest_is_order_independent_and_change_sensitive(tmp_path):
    root = fake_backbone(tmp_path / "bb")
    inv = file_inventory(root)
    assert BACKBONE_PROVENANCE_FILE not in inv  # sidecar records the digest, so it is excluded
    assert digest_of(dict(reversed(list(inv.items())))) == digest_of(inv)
    (root / ".cache").mkdir()
    (root / ".cache" / "x").write_text("ignored")
    assert digest_of(file_inventory(root)) == digest_of(inv)
    (root / "model.safetensors").write_bytes(b"tampered")
    assert digest_of(file_inventory(root)) != digest_of(inv)


def test_backbone_provenance_validation(tmp_path):
    root = fake_backbone(tmp_path / "bb")
    assert read_backbone_provenance(root)["synthetic"] is False
    (root / BACKBONE_PROVENANCE_FILE).write_text(json.dumps({"name": "x", "source": "y", "license": ""}))
    with pytest.raises(ValueError, match="license"):
        read_backbone_provenance(root)
    (root / BACKBONE_PROVENANCE_FILE).write_text(json.dumps({"name": "x", "source": "y", "license": "z",
                                                             "synthetic": "no"}))
    with pytest.raises(ValueError, match="synthetic"):
        read_backbone_provenance(root)
    assert read_backbone_provenance(tmp_path) is None


def test_tokenizer_size_and_inspection(tmp_path):
    root = fake_backbone(tmp_path / "bb", tok_vocab=8)
    assert tokenizer_json_size(root) == 8
    info = inspect_backbone(root)
    assert info["causal_lm"] and not info["problems"] and info["tokenizer_files"] == ["tokenizer.json"]
    assert not inspect_backbone(tmp_path / "missing")["exists"]
    (root / "config.json").unlink()
    assert "config.json missing" in inspect_backbone(root)["problems"]


def test_cultural_manifest_validation(tmp_path):
    root = fake_backbone(tmp_path / "bb")
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x")
    info = inspect_backbone(root)
    good = cultural_manifest(ckpt, info, H)
    validate_cultural_manifest(good)
    for patch, msg in [({"status": "trained", "optimizer_steps": 0}, "optimizer_steps > 0"),
                       ({"status": "untrained", "optimizer_steps": 2}, "cannot record optimizer steps"),
                       ({"data_sha256": {}}, "training-data hashes"),
                       ({"data_kind": "unknown"}, "data_kind"),
                       ({"data_sha256": {"x": "not-a-hash"}}, "SHA-256")]:
        bad = json.loads(json.dumps(good))
        bad["training"].update(patch)
        with pytest.raises(ValueError, match=msg):
            validate_cultural_manifest(bad)
    assert cultural_compatibility_problems(good, checkpoint_sha256=sha256_file(ckpt),
                                           backbone_digest=info["directory_digest"],
                                           tokenizer_digest=info["tokenizer_digest"], schema_vocabulary_sha256=H) == []
    problems = cultural_compatibility_problems(good, checkpoint_sha256="b" * 64, backbone_digest="c" * 64,
                                               tokenizer_digest="d" * 64, schema_vocabulary_sha256="e" * 64)
    assert len(problems) == 4


# ----------------------------------------------------------------- readiness levels
def _level(**kw):
    return assess(**kw)["level"]


def test_nothing_supplied_is_smoke_only():
    report = assess()
    assert report["level"] == "smoke_only" and not report["generation_exercised"]
    assert {c["status"] for c in report["checks"]} == {"MISSING"}


def test_backbone_levels(tmp_path):
    good = fake_backbone(tmp_path / "good")
    assert _level(backbone=str(good)) == "backbone_only"
    assert _level(backbone=str(fake_backbone(tmp_path / "synthetic", synthetic=True))) == "smoke_only"
    assert _level(backbone=str(fake_backbone(tmp_path / "undeclared", provenance=False))) == "smoke_only"
    nodigest = fake_backbone(tmp_path / "nodigest", digest=False)
    assert _level(backbone=str(nodigest)) == "smoke_only"
    assert _level(backbone=str(nodigest), backbone_sha256=inspect_backbone(nodigest)["directory_digest"]) \
        == "backbone_only"
    assert _level(backbone=str(nodigest), backbone_sha256="f" * 64) == "smoke_only"
    misaligned = fake_backbone(tmp_path / "misaligned", vocab=4, tok_vocab=8)
    report = assess(backbone=str(misaligned))
    assert report["level"] == "smoke_only" and "tokenizer_embedding_alignment" in report["failed_checks"]


def test_cultural_levels(tmp_path, schema_setup):
    bb = fake_backbone(tmp_path / "bb")
    common = dict(backbone=str(bb), schema_vocabulary=str(schema_setup["vocab"]), ckg=str(schema_setup["ckg"]),
                  examples_dir=str(schema_setup["dir"]))
    untrained = write_cultural(tmp_path / "u", bb, schema_setup["vocab"], status="untrained", steps=0, data={})
    assert _level(cultural_checkpoint=str(untrained), **common) == "ansg_untrained"
    synth = write_cultural(tmp_path / "s", bb, schema_setup["vocab"], data=schema_setup["data"])
    assert _level(cultural_checkpoint=str(synth), **common) == "ansg_trained_synthetic_only"
    real = write_cultural(tmp_path / "r", bb, schema_setup["vocab"], data_kind="real", data=schema_setup["data"])
    assert _level(cultural_checkpoint=str(real), **common) == "trained_ansg"
    # Untraceable training data, or missing CKG evidence, does not count as trained ANSG.
    untraceable = write_cultural(tmp_path / "x", bb, schema_setup["vocab"], data_kind="real")
    report = assess(cultural_checkpoint=str(untraceable), **common)
    assert report["level"] == "backbone_only" and "cultural_training_data_traceable" in report["failed_checks"]
    assert _level(cultural_checkpoint=str(real), backbone=str(bb), schema_vocabulary=str(schema_setup["vocab"])) \
        == "backbone_only"


def test_cultural_checkpoint_rejections(tmp_path, schema_setup):
    bb = fake_backbone(tmp_path / "bb")
    other = fake_backbone(tmp_path / "other", vocab=12)
    common = dict(schema_vocabulary=str(schema_setup["vocab"]))
    ckpt = write_cultural(tmp_path / "c", bb, schema_setup["vocab"])
    assert _level(backbone=str(other), cultural_checkpoint=str(ckpt), **common) == "backbone_only"
    ckpt.write_bytes(b"tampered")
    report = assess(backbone=str(bb), cultural_checkpoint=str(ckpt), **common)
    assert "cultural_checkpoint" in report["failed_checks"]
    cultural_manifest_path(ckpt).unlink()
    report = assess(backbone=str(bb), cultural_checkpoint=str(ckpt), **common)
    assert any("manifest not found" in c["detail"] for c in report["checks"])


def test_generation_records_are_validated(tmp_path):
    bb = fake_backbone(tmp_path / "bb")
    digest = inspect_backbone(bb)["directory_digest"]

    def record(name, label, synthetic=False, cultural=None, tokens=3, bb_digest=digest):
        p = tmp_path / name
        p.write_text(json.dumps({"schema": L.RECORD_SCHEMA, "label": label, "cultural_modules": cultural,
                                 "backbone": {"synthetic": synthetic, "directory_digest": bb_digest},
                                 "samples": [{"new_tokens": tokens}]}))
        return str(p)

    ok = record("ok.json", L.label_block(L.GenerationMode.BACKBONE_ONLY))
    assert assess(backbone=str(bb), generation_records=[ok])["generation_exercised"] is True
    stub = record("stub.json", L.label_block(L.GenerationMode.SYNTHETIC_STUB), synthetic=True)
    assert assess(backbone=str(bb), generation_records=[stub])["generation_exercised"] is False
    inflated = record("inflated.json", {**L.label_block(L.GenerationMode.BACKBONE_ONLY), "mode": "ansg_trained"})
    report = assess(backbone=str(bb), generation_records=[inflated])
    assert "generation_exercised" in report["failed_checks"]
    other = record("other.json", L.label_block(L.GenerationMode.BACKBONE_ONLY), bb_digest="0" * 64)
    assert "different backbone" in json.dumps(assess(backbone=str(bb), generation_records=[other])["checks"])
    empty = record("empty.json", L.label_block(L.GenerationMode.BACKBONE_ONLY), tokens=0)
    assert not assess(backbone=str(bb), generation_records=[empty])["generation_exercised"]


def test_readiness_cli_require(tmp_path, capsys):
    assert readiness_main([]) == 0
    assert "smoke_only" in capsys.readouterr().out
    assert readiness_main(["--require", "backbone_only"]) == 1
    capsys.readouterr()
    bb = fake_backbone(tmp_path / "bb")
    assert readiness_main(["--backbone", str(bb), "--require", "backbone_only", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["level"] == "backbone_only"
    assert data["facts"]["backbone_directory_digest"] == hashlib.sha256(
        "".join(f"{k}\t{v}\n" for k, v in sorted(file_inventory(bb).items())).encode()).hexdigest()
