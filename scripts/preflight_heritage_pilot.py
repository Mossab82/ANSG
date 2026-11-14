#!/usr/bin/env python3
"""CPU preflight for the Kaggle heritage pilot: run its real data path before any GPU launch.

Run 43561944084 passed the old preflight (py_compile + corpus validator) and then failed
on Kaggle with ``KeyError: 'night'`` inside ``chunks()``. This preflight imports the
pilot module itself and executes, on CPU and on the committed corpus:

    load_verified_corpus(local)  ->  prepare_data(splits, tokenizer)

with a deterministic whitespace stand-in tokenizer (the real Qwen tokenizer needs a
model download; chunk counts here are therefore NOT the GPU run's counts).

``--require-rights`` additionally enforces the rights gate of data/sources/source_registry.json:
the corpus must be recorded as verified for model training. It is the gate the GPU
workflow uses; it fails until the owner records that review.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "kaggle" / "heritage" / "ansg_heritage_pilot.py"
CORPUS = ROOT / "data" / "corpora" / "classical_arabic" / "stories.jsonl"
REGISTRY = ROOT / "data" / "sources" / "source_registry.json"
CORPUS_RESOURCE = "wikisource_classical_arabic_pages"


class WhitespaceTokenizer:
    """Deterministic CPU stand-in: one id per whitespace token (stable hash)."""

    def __call__(self, text, add_special_tokens=False):
        import zlib
        return {"input_ids": [zlib.crc32(w.encode("utf-8")) % 50000 for w in text.split()]}


def load_pilot():
    spec = importlib.util.spec_from_file_location("ansg_heritage_pilot", PILOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rights_gate(registry_path: Path = REGISTRY) -> dict:
    reg = json.loads(registry_path.read_text(encoding="utf-8"))
    res = next((r for r in reg["resources"] if r["id"] == CORPUS_RESOURCE), None)
    if res is None:
        return {"passed": False, "reason": f"{CORPUS_RESOURCE} missing from the source registry"}
    ok = res.get("permission_status") == "verified" and "model_training" in res.get("allowed_uses", [])
    return {"passed": ok, "permission_status": res.get("permission_status"),
            "allowed_uses": res.get("allowed_uses"),
            "reason": None if ok else "corpus rights not recorded as verified for model training "
                                      "(owner review pending); GPU training is blocked"}


def run(require_rights: bool = False) -> dict:
    pilot = load_pilot()
    splits = pilot.load_verified_corpus(local_path=CORPUS, write_manifest=False)
    data = pilot.prepare_data(splits, WhitespaceTokenizer())
    report = {
        "status": "data_path_ok",
        "pilot": str(PILOT.relative_to(ROOT)),
        "corpus_commit_pinned_in_pilot": pilot.CORPUS_COMMIT,
        "split_records": {k: sorted(r["id"] for r in v) for k, v in splits.items()},
        "chunks_with_whitespace_stand_in": {k: len(v) for k, v in data.items()},
        "groups_per_split": {k: len({c["group"] for c in v}) for k, v in data.items()},
        "note": "chunk counts use a whitespace stand-in tokenizer, not the model tokenizer",
    }
    report["rights_gate"] = rights_gate()
    if require_rights and not report["rights_gate"]["passed"]:
        report["status"] = "blocked_by_rights_gate"
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--require-rights", action="store_true")
    args = ap.parse_args(argv)
    try:
        report = run(args.require_rights)
    except Exception as exc:  # noqa: BLE001 - any failure blocks the GPU launch
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "data_path_ok" else 2


if __name__ == "__main__":
    sys.exit(main())
