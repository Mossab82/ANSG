#!/usr/bin/env python3
"""Report whether an explicitly configured ANSG reconstruction may start.

This is a preflight only. It never starts training and never marks a run as a
historical reproduction. Full thesis-stage orchestration is not implemented.
"""
import argparse
import hashlib
import json
import os
import re
import sys

import yaml
from training_authorization import validate_real_data_authorization


ALLOWED_DECISIONS = {
    "corpus_stage_mapping": {"chapter3_two_stage", "chapter5_three_stage"},
    "backbone_and_checkpoint": {"ch4_gpt2_medium", "ch5_774m_base", "appendix_aragpt2_large"},
    "tokenizer": {"standard_backbone_bpe", "sentencepiece_root_aware_32k"},
    "csam_variant": {"eq_4_9_to_4_13", "section_4_3_6", "appendix_a25"},
    "schema_activation_variant": {"eq_4_20_table_4_2", "appendix_a4"},
    "dal_projection_mapping": {"generic_per_layer", "qv_projection", "gpt2_fused_c_attn"},
    "training_schedule": {"table_4_1", "section_4_1_6", "chapter5_three_stage", "appendix_a26"},
    "reward_variant": {"eq_4_26_to_4_29", "table_4_1", "section_5_6_3", "appendix_a3_a11"},
    "cmpi_definition": {"owner_supplied_original", "new_operational_definition"},
}
UNRESOLVED = {"", "pending", "unresolved", "unknown", "choose", "tbd"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_plan(plan_path, profile_path):
    blockers = []
    try:
        with open(profile_path, encoding="utf-8") as source:
            profile = yaml.safe_load(source)
    except (OSError, yaml.YAMLError) as exc:
        return {"status": "invalid", "blockers": [f"Cannot read profile: {exc}"],
                "historical_reproduction": False}

    try:
        with open(plan_path, encoding="utf-8") as source:
            plan = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "blockers": [f"Cannot read run plan: {exc}"],
                "historical_reproduction": False}

    if not isinstance(profile, dict):
        return {"status": "invalid", "blockers": ["Profile root must be a YAML object."],
                "historical_reproduction": False}
    if not isinstance(plan, dict):
        return {"status": "invalid", "blockers": ["Run plan root must be a JSON object."],
                "historical_reproduction": False}
    if plan.get("schema") != "ansg.training_run_plan/1":
        blockers.append("Run plan schema must be ansg.training_run_plan/1.")
    if plan.get("run_kind") != "reconstruction":
        blockers.append("Only run_kind=reconstruction is accepted; this tool cannot authorize historical reproduction.")
    if plan.get("historical_reproduction") is not False:
        blockers.append("historical_reproduction must be false.")

    decisions = plan.get("decisions")
    if not isinstance(decisions, dict):
        blockers.append("decisions must be an object containing explicit thesis choices.")
        decisions = {}
    for name, allowed in ALLOWED_DECISIONS.items():
        value = decisions.get(name)
        if not isinstance(value, str) or value.strip().lower() in UNRESOLVED:
            blockers.append(f"Unresolved decision: {name}.")
        elif value not in allowed:
            blockers.append(f"Unsupported value for {name}: {value!r}; choose one of {sorted(allowed)}.")

    artifacts = plan.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        blockers.append("No artifact files are listed.")
        artifacts = []
    roles = set()
    for item in artifacts:
        if not isinstance(item, dict):
            blockers.append("Each artifact entry must be an object.")
            continue
        role, path, expected = item.get("role"), item.get("path"), item.get("sha256")
        if not role or role in roles:
            blockers.append(f"Artifact role is missing or duplicated: {role!r}.")
            continue
        roles.add(role)
        if not path or not os.path.isfile(path):
            blockers.append(f"Artifact file missing for role {role}: {path!r}.")
            continue
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            blockers.append(f"Artifact SHA-256 missing or malformed for role {role}.")
        else:
            try:
                actual_hash = sha256_file(path)
            except OSError as exc:
                blockers.append(f"Cannot read artifact for role {role}: {exc}.")
            else:
                if actual_hash != expected.lower():
                    blockers.append(f"Artifact SHA-256 mismatch for role {role}: {path}.")
        if not item.get("license"):
            blockers.append(f"License is not recorded for role {role}.")
        if item.get("training_allowed") is not True:
            blockers.append(f"Training permission is not approved for role {role}.")

    data_kind = plan.get("data_kind")
    if data_kind not in {"synthetic", "real"}:
        blockers.append("data_kind must explicitly be synthetic or real.")
    elif data_kind == "real":
        by_role = {item.get("role"): item for item in artifacts if isinstance(item, dict)}
        examples = by_role.get("licensed_training_examples", {}).get("path")
        vocabulary = by_role.get("schema_vocabulary", {}).get("path")
        manifest_path = plan.get("authorization_manifest")
        manifest_hash = plan.get("authorization_manifest_sha256")
        if not manifest_path or not os.path.isfile(manifest_path):
            blockers.append("Real-data training requires an existing authorization_manifest.")
        if not isinstance(manifest_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_hash):
            blockers.append("Real-data training requires authorization_manifest_sha256.")
        elif manifest_path and os.path.isfile(manifest_path):
            try:
                actual_manifest_hash = sha256_file(manifest_path)
                if actual_manifest_hash != manifest_hash.lower():
                    blockers.append("Authorization manifest SHA-256 does not match the run plan.")
                if examples and vocabulary:
                    validate_real_data_authorization(manifest_path, examples, vocabulary)
            except (OSError, ValueError) as exc:
                blockers.append(f"Real-data authorization check failed: {exc}.")

    # A successful preflight is not sufficient to start the thesis's three-stage run.
    # Current code has isolated update primitives and a stage-1 cultural fine-tuner only.
    runner_scope = plan.get("runner_scope")
    supported = runner_scope == "cultural_modules_sft"
    if not supported:
        blockers.append(
            "No full thesis-stage runner is implemented; available training entry point "
            "covers cultural_modules_sft only."
        )
    elif runner_scope == "cultural_modules_sft":
        required_roles = {"licensed_training_examples", "backbone_checkpoint", "schema_vocabulary"}
        missing_roles = sorted(required_roles - roles)
        for role in missing_roles:
            blockers.append(f"Required artifact role is missing for cultural_modules_sft: {role}.")
    if profile.get("training_ready") is not True:
        blockers.append("The operational profile keeps training_ready=false.")
    if profile.get("training_start_permitted") is not True:
        blockers.append(
            "The checked-in operational profile keeps training_start_permitted=false; "
            "do not edit it to bypass the thesis and artifact gates."
        )

    return {
        "status": "blocked" if blockers else "preflight_ready_for_cultural_modules_sft",
        "run_kind": plan.get("run_kind"),
        "runner_scope": runner_scope,
        "data_kind": data_kind,
        "decisions_checked": sorted(ALLOWED_DECISIONS),
        "artifact_roles_checked": sorted(roles),
        "blockers": blockers,
        "historical_reproduction": False,
        "starts_training": False,
        "note": "Preflight only; readiness does not assert scientific reproduction or authorize absent rights.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True, help="JSON run plan; see the checked-in template")
    parser.add_argument("--profile", default="configs/training/ansg_direct_profile.yaml")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args(argv)
    report = check_plan(args.plan, args.profile)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Training preflight: {report['status']}")
        for blocker in report["blockers"]:
            print(f"- BLOCKED: {blocker}")
        print("Historical reproduction: false")
        print("Training started: false")
    return 0 if report["status"] == "preflight_ready_for_cultural_modules_sft" else 2


if __name__ == "__main__":
    sys.exit(main())
