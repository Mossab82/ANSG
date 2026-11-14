#!/usr/bin/env python3
"""Recompute derived thesis values from the thesis's own reported inputs (arithmetic consistency only).

This is NOT a reproduction of any experiment: it checks whether the reported numbers agree
with each other. Inputs: configs/reproduction/thesis_reported_values.yaml (transcribed values
with section/page). Output: evidence/reproduction/reported_values_audit.json and a Markdown
table on stdout.

    python scripts/reproduction/audit_reported_values.py            # write + print
    python scripts/reproduction/audit_reported_values.py --check    # fail if the committed JSON is stale
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SRC = ROOT / "configs" / "reproduction" / "thesis_reported_values.yaml"
OUT = ROOT / "evidence" / "reproduction" / "reported_values_audit.json"


def evaluate(check: dict) -> dict:
    value = eval(check["formula"], {"__builtins__": {}, "sqrt": math.sqrt}, dict(check["inputs"]))  # noqa: S307
    diff = value - check["reported"]
    status = "CONSISTENT" if abs(diff) <= check["tolerance"] + 1e-12 else "INCONSISTENT"
    return {"id": check["id"], "ref": check["ref"], "claim": check["claim"], "reported": check["reported"],
            "recomputed_from_reported_inputs": round(value, 6), "difference": round(diff, 6),
            "tolerance": check["tolerance"], "status": status, "note": check.get("note")}


def build() -> dict:
    from ansg.utils.config import load_yaml_config
    from ansg.utils.provenance import sha256_file
    checks = load_yaml_config(SRC)["checks"]
    ids = [c["id"] for c in checks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate check ids")
    results = [evaluate(c) for c in checks]
    return {"schema": "ansg.reported_values_audit/1",
            "source": str(SRC.relative_to(ROOT)), "source_sha256": sha256_file(SRC),
            "nature": "arithmetic consistency of values REPORTED in the thesis; no experiment is reproduced",
            "summary": {"checks": len(results),
                        "consistent": sum(r["status"] == "CONSISTENT" for r in results),
                        "inconsistent": sum(r["status"] == "INCONSISTENT" for r in results)},
            "results": results}


def markdown(audit: dict) -> str:
    rows = ["| ID | Thesis reference | Reported | Recomputed from reported inputs | Δ | Status |",
            "| --- | --- | --- | --- | --- | --- |"]
    for r in audit["results"]:
        rows.append(f"| {r['id']} | {r['ref']} | {r['reported']} | {r['recomputed_from_reported_inputs']} | "
                    f"{r['difference']} | {r['status']} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    audit = build()
    payload = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != payload:
            print(f"{OUT} is stale: rerun scripts/reproduction/audit_reported_values.py", file=sys.stderr)
            return 1
        print(json.dumps(audit["summary"]))
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="utf-8")
    print(markdown(audit))
    print(json.dumps(audit["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
