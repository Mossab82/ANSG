#!/usr/bin/env python
"""Offline smoke test of the Arabic generation path on a SYNTHETIC stub backbone (Issue #15).

    stub backbone (local, random) -> ansg-generate (backbone path, twice: same seed = same tokens)
    -> ansg-generate --untrained-cultural-modules -> train cultural modules (synthetic, 2 steps)
    -> ansg-generate --cultural-checkpoint -> readiness gate (must stay smoke_only)

Nothing is downloaded. Every generated text is meaningless; the run proves only that the
CLI contract, hashing, labelling and provenance work end to end. Not a dissertation result.

Usage:  python scripts/generation_smoke.py --output runs/generation_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
PROMPT = "كان يا ما كان في قديم الزمان"


def run(out: Path) -> dict:
    from ansg.data.anhc import write_jsonl
    from ansg.data.nights import ingest_file
    from ansg.data.records import CopyrightStatus, SourceProvenance
    from ansg.data.training_examples import main as build_examples
    from ansg.generation.cli import main as generate
    from ansg.generation.readiness import assess
    from ansg.generation.stub import make_stub_backbone

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    def check(name: str, ok: bool, **detail) -> None:
        steps.append({"step": name, "ok": bool(ok), **detail})
        if not ok:
            raise AssertionError(f"generation smoke step failed: {name}: {detail}")

    stub = out / "stub_backbone"
    prov = make_stub_backbone(stub, seed=0)
    check("stub_backbone", prov["synthetic"] is True, directory_digest=prov["directory_digest"])

    base = ["--backbone", str(stub), "--prompt", PROMPT, "--device", "cpu", "--max-new-tokens", "16"]
    records = []
    for name in ("backbone_a", "backbone_b"):
        assert generate(base + ["--seed", "7", "--output", str(out / f"{name}.json")]) == 0
        records.append(json.loads((out / f"{name}.json").read_text(encoding="utf-8")))
    a, b = records
    check("backbone_generation_reproducible",
          a["samples"][0]["token_ids_sha256"] == b["samples"][0]["token_ids_sha256"]
          and a["label"]["mode"] == "synthetic_stub" and a["backbone"]["hash_verified"],
          mode=a["label"]["mode"], token_ids_sha256=a["samples"][0]["token_ids_sha256"])

    # synthetic examples for the cultural modules
    src = SourceProvenance(source_id="synthetic", title="synthetic Nights fixture",
                           copyright_status=CopyrightStatus.PUBLIC_DOMAIN, redistributable=True)
    recs, _ = ingest_file(FIXTURES / "nights" / "synthetic_nights.txt", src)
    write_jsonl([r for r in recs if r.narrative.title], out / "records" / "train.jsonl")
    ckg = FIXTURES / "ckg" / "synthetic_ckg.json"
    assert build_examples(["--records-dir", str(out / "records"), "--ckg", str(ckg),
                           "--output", str(out / "examples")]) == 0
    vocab = out / "examples" / "schema_vocabulary.json"

    cult = ["--schema-vocabulary", str(vocab), "--schema-node", "motif:test_magic_ring"]
    assert generate(base + ["--untrained-cultural-modules", *cult, "--dialect", "Levantine",
                            "--output", str(out / "untrained.json")]) == 0
    u = json.loads((out / "untrained.json").read_text(encoding="utf-8"))
    check("untrained_cultural_modules", u["cultural_modules"]["init_seed"] == 42
          and u["label"]["dialect_adaptation_claimed"] is False, mode=u["label"]["mode"])

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_cultural_modules import main as train
    ckpt = out / "cultural" / "cultural.pt"
    assert train(["--backbone", str(stub), "--examples", str(out / "examples" / "train.examples.jsonl"),
                  "--schema-vocabulary", str(vocab), "--data-kind", "synthetic", "--steps", "2",
                  "--batch-size", "2", "--max-length", "64", "--device", "cpu", "--output", str(ckpt)]) == 0
    manifest = json.loads(Path(str(ckpt) + ".manifest.json").read_text(encoding="utf-8"))
    check("train_cultural_modules_synthetic", manifest["training"]["data_kind"] == "synthetic",
          loss_per_step=manifest["training"]["loss_per_step"], checkpoint_sha256=manifest["checkpoint_sha256"])

    assert generate(base + ["--cultural-checkpoint", str(ckpt), *cult, "--output", str(out / "cultural.json")]) == 0
    c = json.loads((out / "cultural.json").read_text(encoding="utf-8"))
    check("generation_with_cultural_checkpoint",
          c["cultural_modules"]["checkpoint"]["sha256"] == manifest["checkpoint_sha256"], mode=c["label"]["mode"])

    report = assess(backbone=str(stub), cultural_checkpoint=str(ckpt), schema_vocabulary=str(vocab), ckg=str(ckg),
                    examples_dir=str(out / "examples"),
                    generation_records=[str(out / f) for f in ("backbone_a.json", "untrained.json", "cultural.json")])
    (out / "readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check("readiness_stays_smoke_only", report["level"] == "smoke_only" and not report["generation_exercised"],
          level=report["level"], failed_checks=report["failed_checks"])

    summary = {"synthetic": True,
               "notice": "SYNTHETIC stub backbone and synthetic examples; generated text is meaningless. "
                         "Verifies the generation contract only; not a dissertation result.",
               "steps": steps}
    (out / "generation_smoke_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="runs/generation_smoke")
    args = ap.parse_args(argv)
    report = run(Path(args.output))
    print("SYNTHETIC GENERATION SMOKE TEST — not a dissertation result")
    for s in report["steps"]:
        print(f"  {'PASS' if s['ok'] else 'FAIL'}  {s['step']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
