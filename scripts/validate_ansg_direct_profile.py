#!/usr/bin/env python
"""Configuration audit for configs/training/ansg_direct_profile.yaml.

This is a *configuration audit* only. It checks that the thesis-guided
reconstruction profile is internally consistent and still labelled as inferred,
and it lists why training may not start. It does not train, download data,
contact Kaggle or a GPU, or reproduce any thesis result.

Exit status:
  0  configuration audit passed and nothing blocks training (not expected today)
  1  configuration audit failed (the profile is inconsistent or mislabelled)
  2  configuration audit passed, but training is BLOCKED (reasons are printed)

    python scripts/validate_ansg_direct_profile.py [--profile PATH] [--json]
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs" / "training" / "ansg_direct_profile.yaml"
EXPECTED_EPOCHS = (1, 100)
EXPECTED_STAGE_IDS = ["supervised_pretraining", "adversarial_refinement", "culturally_grounded_rl"]
EXPECTED_REWARD_TERMS = {"R_cult", "R_ling", "R_coh"}
EXPECTED_CULTURAL_HEADS = 12
INFERRED_STATUS = "inferred_from_thesis_text"
UNRESOLVED = "unresolved_in_thesis"

# Profile flags that record a missing artefact or an unspecified definition.
# Each one that is false (or missing) is a reason training cannot start.
MISSING_ARTIFACT_FLAGS = [
    ("corpus_claims.historical.verified_artifacts_available", "historical corpus artefacts (70.4M / 23M tokens) are not available"),
    ("model.backbone.checkpoint_available", "no backbone checkpoint is available to verify the parameter count"),
    ("model.tokenizer.historical_tokenizer_artifact_available", "the historical root-aware 32K SentencePiece tokenizer is not available"),
    ("model.csam.rhetoric_bias_formula_fully_specified", "the CSAM rhetoric-bias formula is not fully specified"),
    ("model.csam.auxiliary_losses_fully_specified", "the CSAM auxiliary losses are not fully specified"),
    ("model.dal.gpt2_qv_projection_mapping_verified", "the DAL LoRA mapping onto GPT-2's fused QKV projection is not verified"),
    ("training.stages[culturally_grounded_rl].reward.CMPI_equation_recovered", "the CMPI equation (R_cult) has not been recovered"),
    ("training.stages[culturally_grounded_rl].reward.normalization_bounds_defined", "reward normalisation bounds are not defined"),
]

# Python sources that may consume the profile. A runner must reference the
# profile path; this validator and tests do not count as runners.
RUNNER_SEARCH_DIRS = ("scripts", "src", "kaggle")


def _get(data, dotted):
    """Resolve 'a.b.stages[id].c' against nested dicts / lists of {'id': ...}."""
    cur = data
    for part in dotted.split("."):
        m = re.fullmatch(r"(\w+)\[(\w+)\]", part)
        key, item_id = (m.group(1), m.group(2)) if m else (part, None)
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
        if item_id is not None:
            cur = next((s for s in cur or [] if isinstance(s, dict) and s.get("id") == item_id), None)
            if cur is None:
                return None
    return cur


class Audit:
    def __init__(self):
        self.checks = []  # {"check", "ok", "detail"}

    def check(self, name, ok, detail):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})

    @property
    def passed(self):
        return all(c["ok"] for c in self.checks)


def check_stages(profile, audit):
    stages = _get(profile, "training.stages")
    if not isinstance(stages, list) or not stages:
        audit.check("stages_present", False, "training.stages is missing or empty")
        return
    ids = [s.get("id") for s in stages]
    audit.check("stage_ids", ids == EXPECTED_STAGE_IDS,
                f"stage order {ids}; expected {EXPECTED_STAGE_IDS} (Chapter 5 three-stage schedule)")
    spans = []
    for s in stages:
        ep = s.get("epochs")
        valid = (isinstance(ep, list) and len(ep) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in ep)
                 and ep[0] <= ep[1])
        if not valid:
            audit.check("stage_epoch_ranges", False, f"stage {s.get('id')!r} has invalid epochs {ep!r}")
            return
        spans.append((s.get("id"), ep[0], ep[1]))
    problems = []
    if spans[0][1] != EXPECTED_EPOCHS[0]:
        problems.append(f"first stage starts at epoch {spans[0][1]}, not {EXPECTED_EPOCHS[0]}")
    if spans[-1][2] != EXPECTED_EPOCHS[1]:
        problems.append(f"last stage ends at epoch {spans[-1][2]}, not {EXPECTED_EPOCHS[1]}")
    for (a, _, a_end), (b, b_start, _) in zip(spans, spans[1:]):
        if b_start != a_end + 1:
            kind = "overlap" if b_start <= a_end else "gap"
            problems.append(f"{kind} between {a} (ends {a_end}) and {b} (starts {b_start})")
    audit.check("stages_ordered_contiguous_1_100", not problems,
                "; ".join(problems) or " -> ".join(f"{i} {s}-{e}" for i, s, e in spans))


def check_reward(profile, audit):
    eq = _get(profile, "training.stages[culturally_grounded_rl].reward.equation")
    if not isinstance(eq, str):
        audit.check("reward_equation_present", False, "RL stage reward.equation is missing")
        return
    terms = re.findall(r"([0-9]*\.?[0-9]+)\s*\*\s*(R_\w+)", eq)
    names = [t[1] for t in terms]
    weights = {n: float(w) for w, n in terms}
    ok_terms = sorted(names) == sorted(EXPECTED_REWARD_TERMS) and len(names) == 3
    audit.check("reward_terms", ok_terms, f"terms {names} parsed from {eq!r}; expected {sorted(EXPECTED_REWARD_TERMS)}")
    total = sum(weights.values())
    audit.check("reward_weights_sum_to_1", ok_terms and math.isclose(total, 1.0, abs_tol=1e-9),
                f"{' + '.join(f'{weights[n]:g}' for n in names)} = {total:g}")
    code_weights = _code_reward_weights()
    if code_weights is not None and ok_terms:
        profile_weights = (weights["R_cult"], weights["R_ling"], weights["R_coh"])
        audit.check("reward_weights_match_code_default",
                    all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(profile_weights, code_weights)),
                    f"profile (cult, ling, coh) = {profile_weights}; ansg.training.cultural_reward default = {code_weights}")


def _code_reward_weights():
    """Default ``weights`` of ansg.training.cultural_reward, read with ast (no torch import)."""
    src = ROOT / "src" / "ansg" / "training.py"
    if not src.is_file():
        return None
    for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef) and node.name == "cultural_reward":
            names = [a.arg for a in node.args.args]
            defaults = dict(zip(names[len(names) - len(node.args.defaults):], node.args.defaults))
            if "weights" in defaults:
                return tuple(float(v) for v in ast.literal_eval(defaults["weights"]))
    return None


def check_csam(profile, audit):
    csam = _get(profile, "model.csam") or {}
    heads = csam.get("cultural_heads")
    fam = csam.get("heads_by_family") or {}
    fam_sum = sum(v for v in fam.values() if isinstance(v, int))
    audit.check("csam_heads_sum_to_12", heads == EXPECTED_CULTURAL_HEADS and fam_sum == EXPECTED_CULTURAL_HEADS,
                f"heads_by_family {fam} sums to {fam_sum}; cultural_heads = {heads}; expected {EXPECTED_CULTURAL_HEADS}")
    hd, width = csam.get("head_dimension"), csam.get("implied_internal_width")
    product = hd * heads if isinstance(hd, int) and isinstance(heads, int) else None
    audit.check("csam_head_dim_x_heads_equals_width", product is not None and product == width,
                f"head_dimension {hd} x cultural_heads {heads} = {product}; implied_internal_width = {width}")
    backbone = csam.get("backbone_width_in_appendix")
    if isinstance(backbone, int) and isinstance(width, int) and backbone != width:
        flag = csam.get(f"projection_{backbone}_to_{width}_is_reconstruction_choice")
        audit.check("csam_width_projection_declared", flag is True,
                    f"backbone width {backbone} != CSAM width {width}; projection must be declared a reconstruction choice (flag = {flag!r})")


def check_labelling(profile, audit):
    status = profile.get("status")
    audit.check("profile_marked_inferred", status == INFERRED_STATUS, f"status = {status!r}; expected {INFERRED_STATUS!r}")
    for path in ("historical_reproduction", "corpus_claims.current_direct_build.historical_reproduction",
                 "corpus_claims.current_direct_build.tokenizer_is_historical_root_aware_tokenizer",
                 "model.tokenizer.direct_build_tokenizer_is_equivalent"):
        val = _get(profile, path)
        audit.check(f"not_historical:{path}", val is False, f"{path} = {val!r}; must be false")


def check_conflict_gate(profile, audit, blocking):
    conflict = _get(profile, "data_use_conflict.status")
    ready = profile.get("training_ready")
    permitted = profile.get("training_start_permitted")
    stage1_mapping = _get(profile, "training.stages[supervised_pretraining].data_mapping")
    if conflict != "resolved":
        blocking.append(
            "Chapter 3 vs Chapter 5 corpus-staging conflict is unresolved "
            f"(data_use_conflict.status = {conflict!r}; stage 1 data_mapping = {stage1_mapping!r}). "
            "Chapter 3 pretrains on 70.4M then fine-tunes on 23M; Chapter 5 §5.1.3 uses both in stage 1 (epochs 1-30).")
        # An unresolved conflict must not coexist with a ready/permitted flag.
        audit.check("unresolved_conflict_keeps_training_not_ready", ready is False and permitted is False,
                    f"training_ready = {ready!r}, training_start_permitted = {permitted!r}; both must be false while the conflict is unresolved")
    else:
        audit.check("unresolved_conflict_keeps_training_not_ready", True, "data_use_conflict.status = 'resolved'")
    if ready is not True:
        blocking.append(f"profile declares training_ready = {ready!r}")
    if permitted is not True:
        blocking.append(f"profile declares training_start_permitted = {permitted!r}")


def find_runners(profile_path):
    """Python files outside tests, including repository-root entry points, that reference the profile."""
    name = profile_path.name
    me = Path(__file__).resolve()
    candidates = set(ROOT.glob("*.py"))
    for directory in RUNNER_SEARCH_DIRS:
        base = ROOT / directory
        if base.is_dir():
            candidates.update(base.rglob("*.py"))
    hits = []
    for py in sorted(candidates):
        if py.resolve() == me or "__pycache__" in py.parts:
            continue
        try:
            if name in py.read_text(encoding="utf-8", errors="ignore"):
                hits.append(str(py.relative_to(ROOT)))
        except OSError:
            continue
    return hits


def existing_training_entry_points():
    """Known training entry points, whether or not they consume this profile."""
    candidates = ("train.py", "train_pipeline.py", "scripts/train_cultural_modules.py")
    return [path for path in candidates if (ROOT / path).is_file()]


def audit_profile(profile_path):
    profile_path = Path(profile_path)
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ValueError(f"{profile_path} does not contain a YAML mapping")
    audit, blocking = Audit(), []
    check_stages(profile, audit)
    check_reward(profile, audit)
    check_csam(profile, audit)
    check_labelling(profile, audit)
    check_conflict_gate(profile, audit, blocking)

    runners = find_runners(profile_path)
    entry_points = existing_training_entry_points()
    if not runners:
        locations = "repository root and " + ", ".join(d + "/" for d in RUNNER_SEARCH_DIRS)
        reason = (f"no training runner consumes this profile: no Python file in {locations} "
                  f"references {profile_path.name}")
        if entry_points:
            reason += "; existing entry points that do not consume this profile: " + ", ".join(entry_points)
        blocking.append(reason)
    for path, reason in MISSING_ARTIFACT_FLAGS:
        if _get(profile, path) is not True:
            blocking.append(f"{reason} ({path} = {_get(profile, path)!r})")
    for item in _get(profile, "training_gate.required_before_any_run") or []:
        blocking.append(f"training_gate requirement not evidenced: {item}")

    return {
        "profile": str(profile_path),
        "profile_id": profile.get("profile_id"),
        "configuration_audit": "passed" if audit.passed else "failed",
        "checks": audit.checks,
        "training": "blocked" if blocking else "not_blocked_by_profile",
        "blocking_reasons": blocking,
        "training_runners_found": runners,
        "existing_training_entry_points_not_using_profile": entry_points,
        "notice": ("Configuration audit only. No training was run and no thesis result was reproduced; "
                   "this profile is inferred from the thesis text, not a recovered historical configuration."),
    }


def exit_code(report):
    if report["configuration_audit"] != "passed":
        return 1
    return 2 if report["blocking_reasons"] else 0


def render(report):
    lines = [f"Profile: {report['profile']} ({report['profile_id']})", ""]
    for c in report["checks"]:
        lines.append(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    n_ok = sum(c["ok"] for c in report["checks"])
    lines += ["", f"Configuration audit: {report['configuration_audit'].upper()} ({n_ok}/{len(report['checks'])} checks)"]
    if report["blocking_reasons"]:
        lines.append(f"Training: BLOCKED ({len(report['blocking_reasons'])} reasons)")
        lines += [f"  {i}. {r}" for i, r in enumerate(report["blocking_reasons"], 1)]
    else:
        lines.append("Training: not blocked by the profile itself (artefacts must still be verified by a runner)")
    lines += ["", report["notice"]]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)
    try:
        report = audit_profile(args.profile)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render(report))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
