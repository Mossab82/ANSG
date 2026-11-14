"""Tests for scripts/validate_ansg_direct_profile.py (configuration audit only).

Mutated profiles are written to tmp_path; the committed profile is only read.
"""
import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs" / "training" / "ansg_direct_profile.yaml"
_spec = importlib.util.spec_from_file_location("validate_ansg_direct_profile", ROOT / "scripts" / "validate_ansg_direct_profile.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)


def _base():
    return yaml.safe_load(PROFILE.read_text(encoding="utf-8"))


def _audit(tmp_path, profile):
    path = tmp_path / "ansg_direct_profile.yaml"
    path.write_text(yaml.safe_dump(profile, allow_unicode=True), encoding="utf-8")
    report = V.audit_profile(path)
    return report, {c["check"]: c["ok"] for c in report["checks"]}


def _stage(profile, sid):
    return next(s for s in profile["training"]["stages"] if s["id"] == sid)


def test_committed_profile_passes_audit_but_training_is_blocked(capsys):
    assert V.main(["--profile", str(PROFILE)]) == 2
    out = capsys.readouterr().out
    assert "Configuration audit: PASSED" in out
    assert "Training: BLOCKED" in out
    assert "corpus-staging conflict is unresolved" in out
    assert "no training runner consumes this profile" in out
    assert "No training was run and no thesis result was reproduced" in out
    lowered = out.lower()
    assert "training passed" not in lowered and "reproduced successfully" not in lowered


def test_committed_profile_report_contents():
    report = V.audit_profile(PROFILE)
    assert report["configuration_audit"] == "passed"
    assert report["training"] == "blocked"
    assert report["training_runners_found"] == []
    assert V.exit_code(report) == 2


@pytest.mark.parametrize("epochs, problem", [
    ([[1, 30], [32, 65], [66, 100]], "gap"),
    ([[1, 30], [30, 65], [66, 100]], "overlap"),
    ([[2, 30], [31, 65], [66, 100]], "starts at epoch 2"),
    ([[1, 30], [31, 65], [66, 99]], "ends at epoch 99"),
])
def test_stage_coverage_errors_fail_the_audit(tmp_path, epochs, problem):
    p = _base()
    for stage, ep in zip(p["training"]["stages"], epochs):
        stage["epochs"] = ep
    report, checks = _audit(tmp_path, p)
    assert checks["stages_ordered_contiguous_1_100"] is False
    detail = next(c["detail"] for c in report["checks"] if c["check"] == "stages_ordered_contiguous_1_100")
    assert problem in detail
    assert V.exit_code(report) == 1


def test_reordered_stages_fail(tmp_path):
    p = _base()
    p["training"]["stages"].reverse()
    _, checks = _audit(tmp_path, p)
    assert checks["stage_ids"] is False and checks["stages_ordered_contiguous_1_100"] is False


def test_reward_weights_must_sum_to_one(tmp_path):
    p = _base()
    _stage(p, "culturally_grounded_rl")["reward"]["equation"] = "R_total = 0.45 * R_cult + 0.35 * R_ling + 0.25 * R_coh"
    report, checks = _audit(tmp_path, p)
    assert checks["reward_weights_sum_to_1"] is False
    assert checks["reward_weights_match_code_default"] is False
    assert V.exit_code(report) == 1


def test_csam_head_counts_and_width(tmp_path):
    p = _base()
    p["model"]["csam"]["heads_by_family"]["dialect"] = 3
    _, checks = _audit(tmp_path, p)
    assert checks["csam_heads_sum_to_12"] is False
    p = _base()
    p["model"]["csam"]["implied_internal_width"] = 1280
    _, checks = _audit(tmp_path, p)
    assert checks["csam_head_dim_x_heads_equals_width"] is False


@pytest.mark.parametrize("mutate", [
    lambda p: p.__setitem__("status", "historical_run_config"),
    lambda p: p.__setitem__("historical_reproduction", True),
    lambda p: p["corpus_claims"]["current_direct_build"].__setitem__("historical_reproduction", True),
])
def test_profile_must_stay_marked_inferred(tmp_path, mutate):
    p = _base()
    mutate(p)
    report, _ = _audit(tmp_path, p)
    assert report["configuration_audit"] == "failed"
    assert V.exit_code(report) == 1


def test_unresolved_conflict_cannot_be_marked_ready(tmp_path):
    p = _base()
    p["training_ready"] = True
    p["training_start_permitted"] = True
    report, checks = _audit(tmp_path, p)
    assert checks["unresolved_conflict_keeps_training_not_ready"] is False
    assert any("corpus-staging conflict is unresolved" in r for r in report["blocking_reasons"])
    assert V.exit_code(report) == 1


def test_resolving_conflict_alone_does_not_unblock_training(tmp_path):
    p = copy.deepcopy(_base())
    p["data_use_conflict"]["status"] = "resolved"
    report, _ = _audit(tmp_path, p)
    assert report["configuration_audit"] == "passed"
    assert not any("corpus-staging conflict" in r for r in report["blocking_reasons"])
    assert report["training"] == "blocked"  # artefacts, runner and ready flags still block
    assert V.exit_code(report) == 2
