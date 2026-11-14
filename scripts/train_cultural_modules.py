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
import json
import sys


from training_authorization import validate_real_data_authorization


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
