"""Generation capability gate (Issue #15, Deliverable 1). Torch-free; reads files only.

Reports which of the following the artefacts *on disk* support, and why:

=============================== =================================================
``smoke_only``                  only tiny untrained / synthetic checkpoints
``backbone_only``               verified pretrained Arabic causal LM, licence and
                                source declared, no ANSG cultural modules
``ansg_untrained``              cultural-module checkpoint present but untrained
``ansg_trained_synthetic_only`` cultural modules trained on synthetic data only
``trained_ansg``                cultural modules trained on declared real data;
                                backbone, tokenizer, schema vocabulary and CKG
                                hashes consistent
=============================== =================================================

Freshly initialised (untrained) CSAM/DAL can always be attached at run time with
``ansg-generate --untrained-cultural-modules``; that never raises the level.

Usage::

    python scripts/check_generation_readiness.py                       # nothing supplied
    python scripts/check_generation_readiness.py --backbone DIR --json
    python scripts/check_generation_readiness.py --backbone DIR --cultural-checkpoint C.pt \\
        --schema-vocabulary V.json --ckg ckg.json --examples-dir data/processed/examples \\
        --generation-record runs/gen.json --require backbone_only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..utils.provenance import sha256_file
from . import labels as L
from .labels import RECORD_SCHEMA
from .artifacts import (cultural_compatibility_problems, expected_backbone_digest, inspect_backbone, read_json,
                        read_cultural_manifest)

PASS, FAIL, MISSING, NOT_CHECKED = "PASS", "FAIL", "MISSING", "NOT_CHECKED"


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.facts: dict[str, Any] = {}

    def add(self, name: str, status: str, detail: str) -> str:
        self.checks.append({"check": name, "status": status, "detail": detail})
        return status

    def status(self, name: str) -> str | None:
        return next((c["status"] for c in self.checks if c["check"] == name), None)


def _backbone_checks(rep: Report, path: str | None, cli_digest: str | None) -> tuple[bool, dict | None]:
    """Returns (counts as a verified real pretrained backbone, inspection info)."""
    if not path:
        rep.add("backbone_present", MISSING, "no --backbone supplied; no pretrained Arabic causal LM is "
                "bundled with the repository")
        return False, None
    info = inspect_backbone(path)
    if info["problems"]:
        rep.add("backbone_present", FAIL, "; ".join(info["problems"]))
        return False, info
    rep.add("backbone_present", PASS, f"{len(info['weight_files'])} weight file(s), {len(info['tokenizer_files'])} "
            f"tokenizer file(s); directory digest {info['directory_digest']}")
    rep.facts["backbone_directory_digest"] = info["directory_digest"]
    rep.facts["tokenizer_digest"] = info["tokenizer_digest"]
    ok = True
    if info["causal_lm"]:
        rep.add("backbone_is_causal_lm", PASS, f"architectures {info['config']['architectures']}")
    else:
        ok = rep.add("backbone_is_causal_lm", FAIL, f"architectures {info['config']['architectures']}") == PASS
    try:
        expected, source = expected_backbone_digest(info, cli_digest)
    except ValueError as exc:
        expected, source = None, str(exc)
    if expected is None:
        ok = rep.add("backbone_hash_verified", FAIL, f"no expected digest ({source}); computed "
                     f"{info['directory_digest']}") == PASS and ok
    elif expected != info["directory_digest"]:
        ok = rep.add("backbone_hash_verified", FAIL, f"expected {expected} ({source}) != computed "
                     f"{info['directory_digest']}") == PASS and ok
    else:
        rep.add("backbone_hash_verified", PASS, f"matches digest from {source}")
    prov = info.get("provenance")
    if prov is None:
        ok = rep.add("backbone_licence_and_source_declared", FAIL,
                     "ansg_backbone.json missing: licence and source unknown") == PASS and ok
    else:
        rep.add("backbone_licence_and_source_declared", PASS,
                f"{prov['name']} | source: {prov['source']} | licence: {prov['license']}")
        if prov["synthetic"]:
            ok = rep.add("backbone_is_real_pretrained", FAIL, "declared synthetic (test stub)") == PASS and ok
        else:
            rep.add("backbone_is_real_pretrained", PASS, "declared non-synthetic by ansg_backbone.json "
                    "(declaration recorded, not independently verifiable)")
        lang = prov.get("language")
        langs = [lang] if isinstance(lang, str) else list(lang or [])
        if not langs:
            rep.add("backbone_language", NOT_CHECKED, "ansg_backbone.json declares no 'language'")
        elif "ar" in langs:
            rep.add("backbone_language", PASS, f"declared language(s) {langs}")
        else:
            ok = rep.add("backbone_language", FAIL, f"declared language(s) {langs} do not include Arabic ('ar')") == PASS and ok
    vocab_size, tok_size = info["config"].get("vocab_size"), info.get("tokenizer_size")
    if tok_size is None or vocab_size is None:
        rep.add("tokenizer_embedding_alignment", NOT_CHECKED, "needs tokenizer.json and config.vocab_size "
                "(ansg-generate verifies alignment at load time)")
    elif tok_size <= vocab_size:
        rep.add("tokenizer_embedding_alignment", PASS, f"tokenizer {tok_size} <= embedding rows {vocab_size}")
    else:
        ok = rep.add("tokenizer_embedding_alignment", FAIL,
                     f"tokenizer {tok_size} > embedding rows {vocab_size}") == PASS and ok
    return ok, info


def _schema_checks(rep: Report, vocab_path: str | None, ckg_path: str | None, examples_dir: str | None) -> dict:
    out: dict[str, Any] = {}
    if not vocab_path:
        rep.add("schema_vocabulary_matches_ckg", MISSING, "no --schema-vocabulary supplied")
        return out
    p = Path(vocab_path)
    if not p.is_file():
        rep.add("schema_vocabulary_matches_ckg", FAIL, f"{p} not found")
        return out
    out["sha256"] = sha256_file(p)
    vocab = read_json(p)
    out["source_sha256"] = vocab.get("source_sha256")
    rep.facts["schema_vocabulary_sha256"] = out["sha256"]
    if not ckg_path:
        rep.add("schema_vocabulary_matches_ckg", MISSING, "no --ckg supplied")
    elif not Path(ckg_path).is_file():
        rep.add("schema_vocabulary_matches_ckg", FAIL, f"{ckg_path} not found")
    else:
        ckg_sha = sha256_file(ckg_path)
        out["ckg_sha256"] = ckg_sha
        rep.add("schema_vocabulary_matches_ckg", PASS if ckg_sha == vocab.get("source_sha256") else FAIL,
                f"CKG sha256 {ckg_sha} vs vocabulary source_sha256 {vocab.get('source_sha256')}")
    if examples_dir:
        mpath = Path(examples_dir) / "manifest.json"
        if not mpath.is_file():
            rep.add("examples_manifest_consistent", FAIL, f"{mpath} not found")
        else:
            manifest = read_json(mpath)
            outputs = set((manifest.get("outputs") or {}).values())
            inputs = set((manifest.get("inputs") or {}).values())
            out["example_output_hashes"] = outputs
            problems = []
            if out["sha256"] not in outputs:
                problems.append("schema vocabulary is not an output of this examples run")
            if "ckg_sha256" in out and out["ckg_sha256"] not in inputs:
                problems.append("CKG is not an input of this examples run")
            rep.add("examples_manifest_consistent", FAIL if problems else PASS,
                    "; ".join(problems) or "vocabulary and CKG hashes match the examples manifest")
    return out


def _cultural_checks(rep: Report, ckpt: str | None, info: dict | None, schema: dict) -> dict | None:
    """Returns the validated manifest if the checkpoint is present, intact and compatible."""
    if not ckpt:
        rep.add("cultural_checkpoint", MISSING, "no --cultural-checkpoint supplied; no trained CSAM/DAL "
                "weights exist in the repository")
        return None
    if not Path(ckpt).is_file():
        rep.add("cultural_checkpoint", FAIL, f"{ckpt} not found")
        return None
    try:
        manifest = read_cultural_manifest(ckpt)
    except (FileNotFoundError, ValueError) as exc:
        rep.add("cultural_checkpoint", FAIL, str(exc))
        return None
    problems = cultural_compatibility_problems(
        manifest, checkpoint_sha256=sha256_file(ckpt),
        backbone_digest=info.get("directory_digest") if info else None,
        tokenizer_digest=info.get("tokenizer_digest") if info else None,
        schema_vocabulary_sha256=schema.get("sha256"))
    if info is None or "directory_digest" not in info:
        problems.append("no verified backbone to check compatibility against")
    if "sha256" not in schema:
        problems.append("no schema vocabulary supplied to check against")
    if problems:
        rep.add("cultural_checkpoint", FAIL, "; ".join(problems))
        return None
    t = manifest["training"]
    rep.add("cultural_checkpoint", PASS, f"hash and backbone/tokenizer/schema-vocabulary compatibility verified; "
            f"status={t['status']}, steps={t['optimizer_steps']}, data_kind={t.get('data_kind')}")
    outputs = schema.get("example_output_hashes")
    if outputs is None:
        rep.add("cultural_training_data_traceable", NOT_CHECKED, "supply --examples-dir to trace training data")
    elif t["status"] == "trained":
        missing = sorted(k for k, v in t["data_sha256"].items() if v not in outputs)
        rep.add("cultural_training_data_traceable", FAIL if missing else PASS,
                f"not produced by the examples run: {missing}" if missing else
                "every training-data hash is an output of the examples manifest")
    return manifest


def _record_checks(rep: Report, records: list[str], info: dict | None) -> None:
    if not records:
        rep.add("generation_exercised", MISSING, "no --generation-record supplied: generation not shown")
        return
    modes, problems = [], []
    for path in records:
        try:
            rec = read_json(path)
            if rec.get("schema") != RECORD_SCHEMA:
                raise ValueError(f"schema is not {RECORD_SCHEMA}")
            cult = rec.get("cultural_modules")
            training = ((cult or {}).get("manifest") or {}).get("training")
            L.validate_label(rec["label"], backbone_synthetic=bool(rec["backbone"]["synthetic"]),
                             cultural_loaded=cult is not None, cultural_training=training)
            if info and info.get("directory_digest") and rec["backbone"]["directory_digest"] != info["directory_digest"]:
                raise ValueError("generated with a different backbone")
            if not rec["samples"] or not all(s["new_tokens"] > 0 for s in rec["samples"]):
                raise ValueError("no generated tokens")
            modes.append(rec["label"]["mode"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{path}: {exc}")
    real = [m for m in modes if m != L.GenerationMode.SYNTHETIC_STUB.value]
    if problems:
        rep.add("generation_exercised", FAIL, "; ".join(problems))
    elif real:
        rep.add("generation_exercised", PASS, f"verified records with modes {sorted(set(real))}")
    else:
        rep.add("generation_exercised", FAIL, "only synthetic-stub generation has been exercised")
    rep.facts["generation_modes_exercised"] = sorted(set(modes))


def assess(*, backbone: str | None = None, backbone_sha256: str | None = None, cultural_checkpoint: str | None = None,
           schema_vocabulary: str | None = None, ckg: str | None = None, examples_dir: str | None = None,
           generation_records: list[str] | None = None) -> dict:
    rep = Report()
    backbone_ok, info = _backbone_checks(rep, backbone, backbone_sha256)
    schema = _schema_checks(rep, schema_vocabulary, ckg, examples_dir)
    manifest = _cultural_checks(rep, cultural_checkpoint, info, schema)
    _record_checks(rep, list(generation_records or []), info)

    reasons = []
    if not backbone_ok:
        level = L.ReadinessLevel.SMOKE_ONLY
        reasons.append("no verified, non-synthetic pretrained backbone with declared licence and source; only the "
                       "in-repository byte-level research model and synthetic smoke checkpoints are usable")
    elif manifest is None:
        level = L.ReadinessLevel.BACKBONE_ONLY
        reasons.append("no compatible cultural-module checkpoint")
    else:
        mode = L.classify_mode(backbone_synthetic=False, cultural_training=manifest["training"], cultural_loaded=True)
        level = L.MODE_TO_LEVEL[mode]
        if level is L.ReadinessLevel.TRAINED_ANSG:
            weak = [n for n in ("schema_vocabulary_matches_ckg", "examples_manifest_consistent",
                                "cultural_training_data_traceable") if rep.status(n) != PASS]
            if weak:
                level = L.ReadinessLevel.BACKBONE_ONLY
                reasons.append(f"trained cultural checkpoint present but data provenance unverified ({', '.join(weak)}); "
                               "it is not counted until CKG, examples manifest and training-data hashes are consistent")
    failed = [c["check"] for c in rep.checks if c["status"] == FAIL]
    return {
        "level": level.value,
        "description": L.LEVEL_DESCRIPTIONS[level],
        "reasons": reasons,
        "generation_exercised": rep.status("generation_exercised") == PASS,
        "failed_checks": failed,
        "checks": rep.checks,
        "facts": {k: (sorted(v) if isinstance(v, set) else v) for k, v in rep.facts.items()},
        "notice": "Readiness describes available artefacts only; it is not an evaluation and reproduces no "
                  "dissertation result. Declarations in ansg_backbone.json and data_kind are recorded, not verified.",
    }


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="ANSG generation capability gate (Issue #15)")
    ap.add_argument("--backbone")
    ap.add_argument("--backbone-sha256")
    ap.add_argument("--cultural-checkpoint")
    ap.add_argument("--schema-vocabulary")
    ap.add_argument("--ckg")
    ap.add_argument("--examples-dir", help="output directory of ansg-build-examples (contains manifest.json)")
    ap.add_argument("--generation-record", action="append", default=[])
    ap.add_argument("--require", choices=[lvl.value for lvl in L.ReadinessLevel],
                    help="exit 1 unless at least this level is reached")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = assess(backbone=args.backbone, backbone_sha256=args.backbone_sha256,
                    cultural_checkpoint=args.cultural_checkpoint, schema_vocabulary=args.schema_vocabulary,
                    ckg=args.ckg, examples_dir=args.examples_dir, generation_records=args.generation_record)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"ANSG generation readiness: {report['level']}\n  {report['description']}")
        for r in report["reasons"]:
            print(f"  reason: {r}")
        for c in report["checks"]:
            print(f"  {c['status']:<11} {c['check']}: {c['detail']}")
        print(f"  generation exercised (non-synthetic): {'yes' if report['generation_exercised'] else 'no'}")
        print(report["notice"])
    if args.require and L.ReadinessLevel(report["level"]).rank < L.ReadinessLevel(args.require).rank:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
