#!/usr/bin/env python
"""Load the supplied knowledge resources named in the configs and print statistics.

Usage (from the repository root):
    python scripts/inspect_resources.py
Missing resources are reported, not substituted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ansg.knowledge.arabic_namenet import ArabicNameNet, TableMapping
from ansg.knowledge.arabic_verbnet import ArabicVerbNet
from ansg.knowledge.ckg import CulturalKnowledgeGraph, load_declared_variants, load_ontology_rdf
from ansg.utils.config import load_yaml_config


def main() -> int:
    res = load_yaml_config("configs/knowledge/resources.yaml")
    ckg_cfg = load_yaml_config("configs/knowledge/ckg.yaml")
    report: dict[str, object] = {}

    vn = res["arabic_verbnet"]
    try:
        report["arabic_verbnet"] = ArabicVerbNet.from_directory(
            vn["directory"], vn.get("pattern", "*.xml"),
            verb_attributes=tuple(vn.get("verb_attributes") or ("transname", "name"))).stats()
    except FileNotFoundError as exc:
        report["arabic_verbnet"] = f"NOT SUPPLIED: {exc}"

    nn = res["arabic_namenet"]
    if not (nn["entities"].get("columns") or {}).get("name"):
        report["arabic_namenet"] = "NOT CONFIGURED: column mapping is empty (format unverified)"
    else:
        try:
            rel = TableMapping.from_mapping(nn["relations"]) if nn.get("relations") else None
            report["arabic_namenet"] = ArabicNameNet.load(TableMapping.from_mapping(nn["entities"]), rel).stats()
        except FileNotFoundError as exc:
            report["arabic_namenet"] = f"NOT SUPPLIED: {exc}"

    data = ckg_cfg["data"]
    if Path(data["graph_json"]).is_file():
        g = CulturalKnowledgeGraph.from_json(data["graph_json"])
        if data.get("cst_onto") and Path(data["cst_onto"]).is_file():
            g.attach_ontology(load_ontology_rdf(data["cst_onto"], data.get("cst_onto_format")))
        variants = load_declared_variants(ckg_cfg.get("declared_variants"))
        report["ckg"] = {
            "problems": g.validate(),
            # One report per thesis statement; they disagree with each other (C-CKG-1/2).
            "counts_vs_declared": {vid: g.count_report(d) for vid, d in variants.items()},
        }
    else:
        report["ckg"] = f"NOT SUPPLIED: {data['graph_json']}"

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
