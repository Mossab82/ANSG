#!/usr/bin/env python
"""Audit the dissertation's reward / RL variants (configs/specification/reward_rl_variants.yaml).

Specification audit only: it validates the transcription, recomputes the printed
worked examples, lists weight sets reused for different terms, compares the
defaults in ansg.training with the PPO variants, and reports which components
block each variant. It computes no reward from text, defines no CMPI, selects no
variant and trains nothing.

Exit status:
  0  specification valid (without --variant)
  1  specification invalid
  2  with --variant: the variant cannot be computed (undefined components)

    python scripts/audit_reward_rl_spec.py [--spec PATH] [--variant ID] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ansg.reward_spec import DEFAULT_SPEC, load_reward_spec  # noqa: E402


def build_report(spec_path, variant=None):
    spec = load_reward_spec(spec_path)
    problems = spec.validate()
    report = {
        "spec": str(spec_path),
        "problems": problems,
        "selected_variant": spec.selected_variant,
        "selected_ppo_variant": spec.data.get("selected_ppo_variant"),
        "variants": {vid: {"kind": v["kind"], "reference": v["reference"],
                           "blockers": spec.blockers(vid)} for vid, v in spec.variants.items()} if not problems else {},
        "worked_examples": spec.worked_examples() if not problems else [],
        "same_weights_different_terms": spec.same_weights_different_terms() if not problems else [],
        "code_alignment": spec.code_alignment() if not problems else {},
        "notice": ("Specification audit only. No reward was computed from text, no CMPI is defined, "
                   "no variant is selected and no thesis result is reproduced."),
    }
    if variant is not None and not problems:
        spec.variant(variant)
        report["requested_variant"] = {"id": variant, "blockers": spec.blockers(variant),
                                       "computable": not spec.blockers(variant)}
    return report


def exit_code(report):
    if report["problems"]:
        return 1
    rv = report.get("requested_variant")
    return 2 if rv is not None and not rv["computable"] else 0


def render(r):
    out = [f"Spec: {r['spec']}", f"Selected variant: {r['selected_variant']} (none selected = author decision pending)"]
    if r["problems"]:
        out += ["", "Specification problems:"] + [f"  - {p}" for p in r["problems"]]
        return "\n".join(out)
    out += ["", "Variants and blocking components:"]
    for vid, v in r["variants"].items():
        ref = v["reference"]
        blk = ", ".join(f"{b['component']} ({b['status']})" for b in v["blockers"]) or "none"
        out.append(f"  {vid} [{v['kind']}] {ref['section']} pp.{ref['pages']}: blocked by {blk}")
    out += ["", "Worked examples (printed vs recomputed from printed inputs):"]
    for ex in r["worked_examples"]:
        out.append(f"  {ex['id']} ({ex['reference']})")
        for name, row in ex["rows"].items():
            flag = "ok" if row["consistent_to_printed_precision"] else "INCONSISTENT"
            out.append(f"    {name}: printed {row['printed']} recomputed {row['recomputed']} diff {row['difference']:+} [{flag}]")
    out += ["", "Same weight vector used for different terms:"]
    for c in r["same_weights_different_terms"]:
        out.append(f"  {c['weights']}: " + "; ".join(f"{u['variant']}.{u['node']}={u['terms_by_weight']}" for u in c["uses"]))
    ca = r["code_alignment"]
    out += ["", f"Code: ansg.training.cultural_reward default weights {ca['cultural_reward_default_weights']}"]
    for vid, eq in ca["cultural_reward_default_weights_equal_top_level_weights_of"].items():
        out.append(f"  equal to top-level weights of {vid}: {eq}")
    out.append(f"Code: ansg.training.ppo_update defaults {ca['ppo_update_defaults']}")
    for pid, rows in ca["ppo_variants"].items():
        out.append(f"  {pid}: " + ", ".join(f"{k}={v['thesis']} ({v['status']})" for k, v in rows.items()))
    if "requested_variant" in r:
        rv = r["requested_variant"]
        out += ["", f"Requested variant {rv['id']}: " + ("computable" if rv["computable"] else "BLOCKED")]
        out += [f"  - {b['component']}: {b['status']} — {b['reference']}" for b in rv["blockers"]]
    out += ["", r["notice"]]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--variant")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    try:
        report = build_report(args.spec, args.variant)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=1) if args.json else render(report))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
