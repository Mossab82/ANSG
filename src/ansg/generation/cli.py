"""``ansg-generate``: Arabic story generation from an explicitly supplied local checkpoint.

Importable without PyTorch: the runtime is imported only after argument parsing,
so a data-only install reports the missing extra instead of crashing.

Exit status: 0 success, 2 invalid request or failed verification, 3 missing extra.

Examples::

    ansg-generate --backbone /models/aragpt2-base --prompt "كان يا ما كان" --output runs/gen.json
    ansg-generate --backbone DIR --cultural-checkpoint runs/cultural.pt \\
        --schema-vocabulary data/processed/examples/schema_vocabulary.json \\
        --schema-node schema:hospitality --dialect Egyptian --prompt-file prompt.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ansg-generate", description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("prompt")
    g = src.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Arabic prompt text")
    g.add_argument("--prompt-file", help="UTF-8 file containing the prompt")
    m = ap.add_argument_group("model (local files only; nothing is downloaded)")
    m.add_argument("--backbone", required=True, help="local Hugging Face-format causal-LM directory")
    m.add_argument("--backbone-sha256", help="expected directory digest (see check_generation_readiness.py)")
    m.add_argument("--allow-unverified-backbone", action="store_true",
                   help="run without an expected digest; the record is marked hash_verified=false")
    m.add_argument("--cultural-checkpoint", help="trained CSAM/DAL checkpoint with <file>.manifest.json")
    m.add_argument("--untrained-cultural-modules", action="store_true",
                   help="attach freshly initialised (UNTRAINED) CSAM/DAL; labelled as such")
    m.add_argument("--schema-vocabulary", help="schema_vocabulary.json from ansg-build-examples")
    m.add_argument("--schema-node", action="append", default=[], help="CKG node id to activate (repeatable)")
    m.add_argument("--dialect", help="MSA, Egyptian, Levantine, Gulf or Maghrebi (cultural modules only)")
    r = ap.add_argument_group("runtime and sampling")
    r.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N or mps (default: auto)")
    r.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    r.add_argument("--deterministic", action="store_true", help="torch deterministic algorithms (GPU repeatability)")
    r.add_argument("--seed", type=int, default=42, help="default 42 (Ch. 6 §6.1.2)")
    r.add_argument("--max-new-tokens", type=int, default=512, help="default 512 (Ch. 6 §6.1.2)")
    r.add_argument("--temperature", type=float, default=0.7, help="0 = greedy; default 0.7 (Ch. 6 §6.1.2)")
    r.add_argument("--top-p", type=float, default=0.95, help="default 0.95 (Ch. 6 §6.1.2; C-EVAL-7)")
    r.add_argument("--top-k", type=int, default=None)
    r.add_argument("--repetition-penalty", type=float, default=1.0)
    r.add_argument("--num-samples", type=int, default=1, help="sample i uses seed + i")
    r.add_argument("--no-stop-at-eos", action="store_true")
    o = ap.add_argument_group("output")
    o.add_argument("--output", help="write the JSON generation record here (never overwritten)")
    o.add_argument("--json", action="store_true", help="print the full JSON record to stdout")
    return ap


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _utf8_streams()
    args = build_parser().parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")         # belt and braces: no hub access
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.output and Path(args.output).exists():
        print(f"error: {args.output} exists; generation records are never overwritten", file=sys.stderr)
        return 2
    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"error: cannot read prompt file: {exc}", file=sys.stderr)
            return 2
    else:
        prompt = args.prompt
    try:
        from .runtime import GenerationError, GenerationRequest, SamplingConfig, run_generation
    except ImportError as exc:
        print(f"error: {exc}. Install the model extras: pip install -e '.[model,pretrained]'", file=sys.stderr)
        return 3
    try:
        sampling = SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                  top_p=args.top_p, top_k=args.top_k, repetition_penalty=args.repetition_penalty,
                                  seed=args.seed, num_samples=args.num_samples, stop_at_eos=not args.no_stop_at_eos)
        record = run_generation(GenerationRequest(
            prompt=prompt, backbone=args.backbone, backbone_sha256=args.backbone_sha256,
            allow_unverified_backbone=args.allow_unverified_backbone, device=args.device, dtype=args.dtype,
            deterministic=args.deterministic, sampling=sampling, cultural_checkpoint=args.cultural_checkpoint,
            untrained_cultural_modules=args.untrained_cultural_modules, schema_vocabulary=args.schema_vocabulary,
            schema_nodes=tuple(args.schema_node), dialect=args.dialect))
    except ImportError as exc:
        print(f"error: {exc}. Install the model extras: pip install -e '.[model,pretrained]'", file=sys.stderr)
        return 3
    except GenerationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    label = record["label"]
    print(f"[{label['mode']}] {label['banner']}", file=sys.stderr)
    for w in label["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    payload = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"record: {dest}", file=sys.stderr)
    if args.json:
        sys.stdout.write(payload)
    else:
        for s in record["samples"]:
            print(f"--- sample {s['index']} (seed {s['seed']}, {s['new_tokens']} new tokens, {s['finish_reason']})")
            print(s["text"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
