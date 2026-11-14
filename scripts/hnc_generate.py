#!/usr/bin/env python
"""Run one HNC experiment condition (A–D) for a hierarchical plan with a local backbone.

ANSG-HNC is an OPTIONAL research extension (research/hnc/). Its runs are never
dissertation results. Condition A (hnc_enabled=false) uses the original generation
path unchanged. Nothing is downloaded; the backbone must be a verified local
directory (see docs/GENERATION_READINESS.md).

    python scripts/hnc_generate.py --hplan research/hnc/scenarios/aqaba_2level.json \\
        --variant B --register r1 --backbone /models/x --backbone-sha256 <digest> \\
        --chat-template --output runs/hnc/aqaba_B_r1_seed42.json
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REGISTERS = {"r1": {"narration": "msa", "dialogue": "dialect"}, "r2": {"narration": "dialect", "dialogue": "dialect"}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hplan", required=True)
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--register", choices=sorted(REGISTERS), help="override the plan's register (R1/R2 study)")
    ap.add_argument("--settings", help="settings YAML (default configs/generation/settings.yaml)")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--backbone-sha256")
    ap.add_argument("--allow-unverified-backbone", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=256, help="per segment (B/C) or for the whole story (A/D)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--system-prompt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    if Path(args.output).exists():
        print(f"error: {args.output} exists; run records are never overwritten", file=sys.stderr)
        return 2

    from ansg.generation.runtime import GenerationError, GenerationRequest, SamplingConfig
    from ansg.hnc.integration import runtime_generate_fn
    from ansg.hnc.model import HierarchicalPlan
    from ansg.hnc.realize import HNCConfig, realise
    from ansg.hnc.validation import validate_plan
    from ansg.narrative.plan import DEFAULT_SETTINGS, load_settings

    plan = HierarchicalPlan.read(args.hplan)
    if args.register:
        plan.register = dict(REGISTERS[args.register])
    settings = load_settings(args.settings or DEFAULT_SETTINGS)
    if plan.setting not in settings:
        print(f"error: unknown setting {plan.setting!r}", file=sys.stderr)
        return 2
    config = HNCConfig.for_variant(args.variant)
    req = GenerationRequest(prompt="-", backbone=args.backbone, backbone_sha256=args.backbone_sha256,
                            allow_unverified_backbone=args.allow_unverified_backbone, device=args.device,
                            dtype=args.dtype, chat_template=args.chat_template, system_prompt=args.system_prompt,
                            sampling=SamplingConfig(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                                    top_p=args.top_p, seed=args.seed))
    records: list[dict] = []
    try:
        result = realise(plan, runtime_generate_fn(req, records), config, settings[plan.setting].label)
    except (GenerationError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = validate_plan(plan, epistemic=config.epistemic_checks)
    out = {"schema": "ansg.hnc_run/1",
           "notice": "ANSG-HNC research extension run. Not ANSG-Original and not a dissertation result.",
           "plan": {"path": args.hplan, "sha256": plan.sha256(), "setting": plan.setting, "register": plan.register,
                    "setting_review_status": settings[plan.setting].review_status,
                    "validation_warnings": sorted(i.code for i in report.issues if i.severity == "warning")},
           **result.to_dict(),
           "generation_records": records}
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"[HNC {'enabled' if config.hnc_enabled else 'disabled'} | variant {config.variant}] "
          f"{len(result.segments)} segment(s) -> {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
