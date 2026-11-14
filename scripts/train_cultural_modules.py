#!/usr/bin/env python
"""Train ANSG cultural modules (CSAM + DAL) on a frozen, hash-verified pretrained backbone.

Stage-1 supervised objective only (next-token cross-entropy); see
``ansg.generation.cultural``. Writes ``--output`` (tensors) plus
``--output.manifest.json`` with backbone/tokenizer/schema-vocabulary digests,
training-data hashes, the declared data kind and the measured loss per step.

    python scripts/train_cultural_modules.py --backbone DIR --backbone-sha256 DIGEST \\
        --examples data/processed/examples/train.examples.jsonl \\
        --schema-vocabulary data/processed/examples/schema_vocabulary.json \\
        --data-kind real --authorization-manifest data/private/training_authorization.json \\
        --steps 2000 --output checkpoints/cultural.pt

``--data-kind`` is a declaration by the person running the training and is recorded
verbatim. Synthetic examples (for example ``tests/fixtures``) must be declared
``synthetic``; such checkpoints are labelled ``ansg_trained_synthetic_only``.
"""
import argparse
import hashlib
import json
import os
import sys


from training_authorization import sha256_file, validate_real_data_authorization


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--backbone-sha256")
    ap.add_argument("--allow-unverified-backbone", action="store_true")
    ap.add_argument("--examples", required=True, help="*.examples.jsonl from ansg-build-examples")
    ap.add_argument("--schema-vocabulary", required=True)
    ap.add_argument("--data-kind", required=True, choices=["synthetic", "real"])
    ap.add_argument("--authorization-manifest", help="Required for real data; records explicit rights approval and hashes")
    ap.add_argument("--run-plan", help="Required for real data; run plan that passed the preflight gate")
    ap.add_argument("--profile", default="configs/training/ansg_direct_profile.yaml")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if args.steps <= 0 or args.batch_size <= 0 or args.max_length <= 1 or args.lr <= 0:
        ap.error("--steps, --batch-size and --lr must be positive; --max-length must exceed 1")
    authorization_hash = None
    run_plan_hash = None
    profile_hash = None
    preflight_status = None
    if args.data_kind == "real":
        if not args.run_plan:
            print("error: --run-plan is required for real-data training", file=sys.stderr)
            return 2
        from check_training_readiness import check_plan
        try:
            run_plan_hash = sha256_file(args.run_plan)
            profile_hash = sha256_file(args.profile)
        except OSError as exc:
            print(f"error: cannot hash run plan or profile: {exc}", file=sys.stderr)
            return 2
        report = check_plan(args.run_plan, args.profile)
        preflight_status = report.get("status")
        if report.get("status") != "preflight_ready_for_cultural_modules_sft":
            print("error: real-data training preflight is blocked", file=sys.stderr)
            for blocker in report.get("blockers", []):
                print(f"- {blocker}", file=sys.stderr)
            return 2
        try:
            with open(args.run_plan, encoding="utf-8") as source:
                plan = json.load(source)
            if plan.get("data_kind") != "real":
                raise ValueError("run plan data_kind must be real for real-data training")
            plan_artifacts = {item.get("role"): item for item in plan.get("artifacts", [])
                              if isinstance(item, dict)}
            for role, cli_path in (("licensed_training_examples", args.examples),
                                   ("schema_vocabulary", args.schema_vocabulary),
                                   ("backbone_checkpoint", args.backbone)):
                item = plan_artifacts.get(role)
                if not item or os.path.realpath(item.get("path", "")) != os.path.realpath(cli_path):
                    raise ValueError(f"run-plan artifact path does not match CLI input: {role}")
            backbone_artifact = plan_artifacts["backbone_checkpoint"]
            if backbone_artifact.get("kind") != "directory":
                raise ValueError("backbone_checkpoint in the run plan must be a directory artifact")
            if not args.backbone_sha256 or args.backbone_sha256.lower() != backbone_artifact.get("sha256", "").lower():
                raise ValueError("--backbone-sha256 must match the pinned run-plan directory digest")
            manifest_path = plan.get("authorization_manifest")
            if (not manifest_path or os.path.realpath(manifest_path) != os.path.realpath(args.authorization_manifest)):
                raise ValueError("authorization manifest path does not match the run plan")
            manifest_hash = plan.get("authorization_manifest_sha256", "")
            if not manifest_hash or sha256_file(args.authorization_manifest) != manifest_hash.lower():
                raise ValueError("authorization manifest SHA-256 does not match the run plan")
            authorization_hash = validate_real_data_authorization(
                args.authorization_manifest, args.examples, args.schema_vocabulary
            )
            if authorization_hash != manifest_hash.lower():
                raise ValueError("authorization manifest changed during preflight")
            if sha256_file(args.run_plan) != run_plan_hash or sha256_file(args.profile) != profile_hash:
                raise ValueError("run plan or operational profile changed during preflight")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    from ansg.generation.cultural import load_example_rows, save_cultural_modules, train_cultural_modules
    from ansg.generation.runtime import GenerationError, build_cultural, load_backbone, resolve_device
    import torch
    try:
        device = resolve_device(args.device)
        backbone = load_backbone(args.backbone, expected_digest=args.backbone_sha256,
                                 allow_unverified=args.allow_unverified_backbone, device=device, dtype=torch.float32)
        cultural = build_cultural(backbone, schema_vocabulary=args.schema_vocabulary, checkpoint=None,
                                  init_seed=args.seed)
        rows = load_example_rows(args.examples, cultural.schema_vocabulary.schema_count)
        record = train_cultural_modules(cultural, backbone, rows, steps=args.steps, lr=args.lr,
                                        batch_size=args.batch_size, max_length=args.max_length, seed=args.seed)
        record["backbone_hash_verified"] = backbone.hash_verified
        if authorization_hash is not None:
            record["authorization_manifest_sha256"] = authorization_hash
            record["run_plan_sha256"] = run_plan_hash
            record["operational_profile_sha256"] = profile_hash
            record["preflight_status"] = preflight_status
            record["training_plan_decisions"] = plan["decisions"]
        manifest = save_cultural_modules(cultural, backbone, args.output, training=record, data_kind=args.data_kind,
                                         data_files={"examples": args.examples,
                                                     "schema_vocabulary": args.schema_vocabulary})
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {"output": args.output, "data_kind": args.data_kind, "steps": args.steps,
               "first_loss": record["loss_per_step"][0], "last_loss": record["loss_per_step"][-1],
               "checkpoint_sha256": manifest["checkpoint_sha256"]}
    if run_plan_hash is not None:
        summary["run_plan_sha256"] = run_plan_hash
        summary["authorization_manifest_sha256"] = authorization_hash
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
