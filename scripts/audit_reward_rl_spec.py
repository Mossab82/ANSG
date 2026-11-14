#!/usr/bin/env python
"""Audit the dissertation's reward / RL variants (configs/specification/reward_rl_variants.yaml).

Specification audit only: it validates the transcription, recomputes the printed
worked examples, lists weight sets reused for different terms, compares the
defaults in ansg.training with the PPO variants, reports the dissertation's CMPI
definition and the separate CULTURA / CST-Onto / MM-GTA metrics, and reports
which inputs block each variant. It scores no text, selects no variant and
trains nothing.

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

from ansg.evaluation.cmpi import (  # noqa: E402
    ANSG_CMPI_REFERENCE, ANSG_CMPI_WEIGHTS, cultura_printed_upper_bound, dimension_scorer_report, load_cmpi_variants,
    validate_cmpi_variants,
)
from ansg.reward_spec import DEFAULT_SPEC, load_reward_spec  # noqa: E402


def build_report(spec_path, variant=None):
    spec = load_reward_spec(spec_path)
    problems = spec.validate()
    cmpi_variants = load_cmpi_variants()
    problems += [f"cmpi_variants: {p}" for p in validate_cmpi_variants(cmpi_variants)]
    report = {
        "spec": str(spec_path),
        "problems": problems,
        "selected_variant": spec.selected_variant,
        "selected_ppo_variant": spec.data.get("selected_ppo_variant"),
        "cmpi_definition": {"weights": ANSG_CMPI_WEIGHTS, "reference": ANSG_CMPI_REFERENCE,
                            "equation": cmpi_variants["ansg_thesis"]["equation"],
                            "dimension_scorers": cmpi_variants["ansg_thesis"]["dimension_scorers"],
                            "open_issues": cmpi_variants["ansg_thesis"]["open_issues"]},
        "other_cmpi_metrics": {k: {"status": cmpi_variants[k]["status"], "publication": cmpi_variants[k]["publication"],
                                   "name_in_paper": cmpi_variants[k]["name_in_paper"],
                                   "formula": cmpi_variants[k]["formula"],
                                   "formula_source": cmpi_variants[k]["formula_source"],
                                   "results_reported": cmpi_variants[k].get("results_reported"),
                                   "conflicts": cmpi_variants[k].get("printed_formula_conflicts", [])
                                   + cmpi_variants[k]["conflicts_with_dissertation"]}
                               for k in ("cultura", "cst_onto", "mm_gta", "ai_cinema_cpm")},
        "cmpi_dimension_scorers": dimension_scorer_report(cmpi_variants),
        "cultura_conflict_computation": cmpi_variants["cultura"].get("conflict_computation"),
        "cultura_author_decision_options": cmpi_variants["cultura"].get("author_decision_options"),
        "cultura_printed_formula_upper_bound": cultura_printed_upper_bound(
            [c["weight"] for c in cmpi_variants["cultura"]["criteria"]]),
        "cultura_eq_4_1_apparent_conflicts": cmpi_variants["cultura"]["printed_equation_in_thesis"]["apparent_conflicts"],
        "variants": {vid: {"kind": v["kind"], "reference": v["reference"],
                           "blockers": spec.blockers(vid)} for vid, v in spec.variants.items()} if not problems else {},
        "worked_examples": spec.worked_examples() if not problems else [],
        "same_weights_different_terms": spec.same_weights_different_terms() if not problems else [],
        "code_alignment": spec.code_alignment() if not problems else {},
        "notice": ("Specification audit only. No reward or CMPI was computed from text (CMPI's dimension "
                   "scorers are unspecified), no variant is selected and no thesis result is reproduced."),
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
    c = r["cmpi_definition"]
    out += ["", f"CMPI (dissertation definition, ANSG reference; {c['reference']}):", f"  {c['equation']}"]
    ds = r["cmpi_dimension_scorers"]
    out.append(f"  computable from text: {ds['cmpi_computable_from_text']}; "
               f"from caller-supplied dimension scores: {ds['cmpi_computable_from_supplied_dimension_scores']}")
    for name, d in ds["dimensions"].items():
        out.append(f"    {name}: scorer specified = {d['scorer_specified']}; missing: {d['missing']}")
    out += ["  open issues:"] + [f"    - {i}" for i in c["open_issues"]]
    out += ["", "Different metrics sharing the acronym (never mixed with ANSG CMPI):"]
    for k, v in r["other_cmpi_metrics"].items():
        out.append(f"  {k}: '{v['name_in_paper']}' — {v['publication']}")
        out.append(f"    formula: {v['formula']}  [source: {v['formula_source']['kind']}, pages {v['formula_source']['pages']}]")
        out += [f"    conflict: {c}" for c in v["conflicts"]]
    out.append(f"  CULTURA printed-formula maximum (all judge scores 10, K=5): {r['cultura_printed_formula_upper_bound']:g}")
    cc = r.get("cultura_conflict_computation") or {}
    if cc:
        out.append(f"    implied by 0.82: {cc.get('implied_by_0_82')}")
    out.append("    options for the author (none chosen):")
    out += [f"      - {o['option']} -> {o['consequence']}" for o in r.get("cultura_author_decision_options") or []]
    out += ["  CULTURA Eq. 4.1 (p.87, schema gating, not CMPI) apparent conflicts:"]
    out += [f"    - {x}" for x in r["cultura_eq_4_1_apparent_conflicts"]]
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
