"""Synthetic, offline tests for independently auditable experiment records."""
import hashlib
import json

import pytest

from ansg.reproducibility import (
    assert_disjoint_splits, compare_measurement, create_manifest, save_manifest, sha256_file,
)


def test_sha256_streamed(tmp_path):
    p = tmp_path / "narratives.txt"
    p.write_bytes("قصة عربية".encode("utf-8"))
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_splits_accept_disjoint_ids():
    assert assert_disjoint_splits({"train": ["t1", "t2"], "validation": ["v1"], "test": ["s1"]}) == {
        "train": 2, "validation": 1, "test": 1
    }


@pytest.mark.parametrize("splits", [
    {"train": ["a"], "test": ["a"]},
    {"train": ["a", "a"]},
    {"training": ["a"]},
    {"train": [""]},
    {},
])
def test_split_validation_rejects_leaks_and_bad_ids(splits):
    with pytest.raises(ValueError):
        assert_disjoint_splits(splits)


def test_manifest_contains_real_file_hash_and_is_create_only(tmp_path):
    p = tmp_path / "synthetic.jsonl"
    p.write_text('{"text":"قصة"}\n', encoding="utf-8")
    manifest = create_manifest(
        run_id="smoke-1", git_sha="a" * 40, configuration={"size": 16},
        seed=42, data_files={"smoke": p}, timestamp="2026-11-13T00:00:00+00:00",
    )
    assert manifest["kind"] == "synthetic"
    assert manifest["data_sha256"]["smoke"] == sha256_file(p)
    assert manifest["checkpoint_sha256"] is None
    target = tmp_path / "manifest.json"
    save_manifest(manifest, target)
    assert json.loads(target.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        save_manifest(manifest, target)


def test_manifest_rejects_invalid_provenance(tmp_path):
    p = tmp_path / "data"
    p.write_text("synthetic")
    args = dict(run_id="x", git_sha="a" * 40, configuration={}, seed=1, data_files={"x": p})
    with pytest.raises(ValueError):
        create_manifest(**{**args, "git_sha": "invalid"})
    with pytest.raises(ValueError):
        create_manifest(**{**args, "kind": "not_run"})
    with pytest.raises(ValueError):
        create_manifest(**{**args, "configuration": {"bad": float("nan")}})


def test_comparison_distinguishes_claim_and_real_run():
    result = compare_measurement(
        metric="BLEU-4", thesis_value=29.8, measured_value=28.3, unit="points",
        thesis_reference="Dissertation, Chapter 6", kind="real", source_run_id="run-123",
    )
    assert result["difference"] == pytest.approx(-1.5)
    assert result["thesis_claim"] == 29.8
    assert result["new_measurement"] == 28.3
    assert result["measurement_kind"] == "real"


def test_not_run_never_has_a_measured_number():
    result = compare_measurement(
        metric="CMPI", thesis_value=.82, measured_value=None, unit="score",
        thesis_reference="Dissertation, Chapter 6", kind="not_run",
    )
    assert result["difference"] is None
    with pytest.raises(ValueError):
        compare_measurement(
            metric="CMPI", thesis_value=.82, measured_value=.82, unit="score",
            thesis_reference="Chapter 6", kind="not_run",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_invalid_numbers_rejected(value):
    with pytest.raises(ValueError):
        compare_measurement(
            metric="M", thesis_value=1.0, measured_value=value, unit="score",
            thesis_reference="chapter", kind="synthetic", source_run_id="smoke",
        )
