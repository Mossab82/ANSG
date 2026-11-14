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
import datetime as dt
import hashlib
import json
import os
import sys


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_real_data_authorization(manifest_path, examples_path, vocabulary_path):
    """Require explicit approval and exact hashes before a run can be labelled real."""
    if not manifest_path:
        raise ValueError("--authorization-manifest is required when --data-kind=real")
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read authorization manifest: {exc}") from exc

    if manifest.get("schema") != "ansg.training_authorization/1":
        raise ValueError("authorization manifest schema must be ansg.training_authorization/1")
    if manifest.get("review_status") != "approved" or manifest.get("training_authorized") is not True:
        raise ValueError("real-data training requires explicit approved status and training_authorized=true")
    if not manifest.get("approver") or not manifest.get("approved_at"):
        raise ValueError("authorization manifest must record approver and approved_at")
    try:
        dt.datetime.fromisoformat(manifest["approved_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("approved_at must be an ISO-8601 timestamp") from exc

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("authorization manifest must list source works")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source authorization entry must be an object")
        source_path = source.get("path")
        source_hash = source.get("sha256", "")
        if (source.get("training_allowed") is not True or not source.get("license")
                or len(source_hash) != 64 or any(ch not in "0123456789abcdef" for ch in source_hash.lower())
                or not source_path or not os.path.isfile(source_path)):
            raise ValueError("each source must record an existing file path, license, SHA-256, and training_allowed=true")
        if _sha256(source_path) != source_hash.lower():
            raise ValueError(f"source SHA-256 does not match: {source_path}")

    derived = manifest.get("derived_files")
    if not isinstance(derived, list):
        raise ValueError("authorization manifest must list derived_files")
    by_role = {item.get("role"): item for item in derived if isinstance(item, dict)}
    for role, path in (("examples", examples_path), ("schema_vocabulary", vocabulary_path)):
        item = by_role.get(role)
        if not item or not item.get("path") or not item.get("sha256"):
            raise ValueError(f"authorization manifest is missing the {role} derived file")
        expected = os.path.realpath(item["path"])
        actual = os.path.realpath(path)
        if expected != actual:
            raise ValueError(f"authorization manifest {role} path does not match CLI input")
        if _sha256(actual) != item["sha256"]:
            raise ValueError(f"authorization manifest {role} SHA-256 does not match file")
    return _sha256(manifest_path)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--backbone-sha256")
    ap.add_argument("--allow-unverified-backbone", action="store_true")
    ap.add_argument("--examples", required=True, help="*.examples.jsonl from ansg-build-examples")
    ap.add_argument("--schema-vocabulary", required=True)
    ap.add_argument("--data-kind", required=True, choices=["synthetic", "real"])
    ap.add_argument("--authorization-manifest", help="Required for real data; records explicit rights approval and hashes")
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
    if args.data_kind == "real":
        try:
            authorization_hash = validate_real_data_authorization(
                args.authorization_manifest, args.examples, args.schema_vocabulary
            )
        except ValueError as exc:
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
        manifest = save_cultural_modules(cultural, backbone, args.output, training=record, data_kind=args.data_kind,
                                         data_files={"examples": args.examples,
                                                     "schema_vocabulary": args.schema_vocabulary})
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": args.output, "data_kind": args.data_kind, "steps": args.steps,
                      "first_loss": record["loss_per_step"][0], "last_loss": record["loss_per_step"][-1],
                      "checkpoint_sha256": manifest["checkpoint_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
