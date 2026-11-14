#!/usr/bin/env python
"""Audit the CKG / CST-Onto resources: presence, provenance, structure and counts.

Reads configs/knowledge/ckg.yaml. It:
* checks whether the configured CKG graph and CST-Onto files exist, with SHA-256;
* scans the repository for other ontology/graph files, labelling known synthetic
  fixtures and demos as such;
* reports the cited sources' status from evidence/ckg/ckg_source_evidence.json;
* checks the internal arithmetic of the counts the thesis states;
* if files are supplied (or passed with --graph / --ontology): validates
  structure, checks provenance metadata, and compares supplied counts with every
  declared variant. Declared counts are never turned into data.

Exit status:
  0  resources present, structurally valid and not marked synthetic/demo
     (identity with the dissertation's CKG still requires owner review)
  1  a supplied file is invalid
  2  BLOCKED: resources missing, or only synthetic/demo data supplied

    python scripts/audit_ckg_resources.py [--graph PATH] [--ontology PATH] [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ansg.knowledge.ckg import (  # noqa: E402
    CulturalKnowledgeGraph, compare_inventory, load_declared_variants, load_ontology_rdf, ontology_inventory,
)

CONFIG = ROOT / "configs" / "knowledge" / "ckg.yaml"
EVIDENCE = ROOT / "evidence" / "ckg" / "ckg_source_evidence.json"
ONTOLOGY_SUFFIXES = (".owl", ".ttl", ".rdf", ".nt", ".n3", ".jsonld", ".owx", ".ofn", ".graphml", ".gexf")
KNOWN_NON_THESIS = {
    "tests/fixtures/ckg/synthetic_ckg.json": "synthetic test fixture",
    "tests/fixtures/ckg/synthetic_onto.ttl": "synthetic test fixture",
    "tests/fixtures/ckg/synthetic_onto_inventory.ttl": "synthetic test fixture",
    "examples/ckg_demo.json": "3-node demo for the lexical baseline (examples/README.md)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "present": False}
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "present": True,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def scan_repository() -> list[dict]:
    try:
        files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        files = [str(p.relative_to(ROOT)) for p in ROOT.rglob("*") if p.is_file()]
    hits = []
    for f in files:
        low = f.lower()
        if low.endswith(ONTOLOGY_SUFFIXES) or "ckg" in Path(low).name or "cst_onto" in low or "cst-onto" in low:
            if low.endswith((".py", ".md", ".yml", ".yaml")):
                continue
            if f.startswith("evidence/ckg/"):
                cls = "metadata evidence about the cited sources (not a graph or ontology)"
            else:
                cls = KNOWN_NON_THESIS.get(f, "unclassified: review required")
            hits.append({"path": f, "classification": cls})
    return hits


def cited_sources_status() -> dict:
    if not EVIDENCE.exists():
        return {"status": "not_collected", "evidence": str(EVIDENCE.relative_to(ROOT))}
    e = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    crossref = e.get("cst_onto_paper_crossref") or {}
    rec = crossref.get("record") or {}
    return {
        "evidence": str(EVIDENCE.relative_to(ROOT)), "collected_at_utc": e.get("collected_at_utc"), "ci_run": e.get("ci_run"),
        "ckg_zenodo_8847378": {"api": e["ckg_zenodo_record"]["api"].get("status"),
                               "doi": e["ckg_zenodo_record"]["doi_resolution"].get("status")},
        "ckg_github_repo": e["ckg_github_repo"].get("status"),
        "ckg_github_owner": e["ckg_github_owner"].get("status"),
        "cst_onto_paper": {"crossref_status": crossref.get("status"), "title": rec.get("title"),
                           "container": rec.get("container"), "published": rec.get("published"),
                           "license_urls": rec.get("license"), "links": rec.get("links"),
                           "relations": rec.get("relation"),
                           "landing_page_status": (e.get("cst_onto_paper_landing_page") or {}).get("status")},
        "zenodo_search_hits": {q: len(r.get("hits") or []) for q, r in (e.get("zenodo_search") or {}).items()},
        "github_search_hits": {q: [i["full_name"] for i in r.get("items") or []] for q, r in (e.get("github_search") or {}).items()},
        "owner_repositories": _owner_repo_summary(e.get("owner_repositories") or {}),
    }


def _owner_repo_summary(o: dict) -> dict:
    """Ontology-like files in the owner's public repositories, excluding this repository's own code/fixtures."""
    repos = o.get("repositories") or []
    found = []
    for r in repos:
        for f in r.get("ontology_like_files") or []:
            path = f.get("path") or ""
            if r.get("full_name", "").lower() == "mossab82/ansg":
                continue
            found.append({"repository": r["full_name"], "path": path, "size": f.get("size"),
                          "is_rdf_owl": path.lower().endswith((".owl", ".ttl", ".rdf", ".nt", ".n3", ".jsonld", ".owx", ".ofn"))})
    return {"status": o.get("status"), "repositories_scanned": len(repos),
            "repositories": [{"name": r.get("full_name"), "license": r.get("license"), "tree_status": r.get("tree_status")} for r in repos],
            "ontology_like_files_outside_ansg": found,
            "rdf_owl_files_found": [f for f in found if f["is_rdf_owl"]]}


def declared_arithmetic(declared: dict) -> list[dict]:
    """Internal consistency of counts stated in the thesis (no data involved)."""
    checks = []
    ch3 = declared.get("cst_onto_ch3")
    if ch3:
        k = ch3.node_counts_by_kind
        checks.append({"check": "§3.1.4 classes = archetypes + specialized concepts",
                       "stated": ch3.ontology_class_count, "computed": k.get("archetype", 0) + k.get("specialized_concept", 0)})
    a16 = declared.get("cst_onto_table_a16")
    if a16:
        o = a16.other_counts
        rows = sum(a16.node_counts_by_kind.values()) + sum(o.get(x, 0) for x in
                   ("object_properties", "datatype_properties", "swrl_rules", "annotation_instances"))
        checks.append({"check": "Table A.16 rows sum = stated total entities", "stated": o.get("total_entities_stated"), "computed": rows})
    t48 = declared.get("ckg_ch4_table_4_8")
    if t48:
        o = t48.other_counts
        seg = sum(v for k, v in o.items() if k.startswith("segments_"))
        checks.append({"check": "Table 4.8 per-source segments sum = 12,341", "stated": o.get("grounding_segments"), "computed": seg})
        checks.append({"check": "Table 4.2 tags = 847 specialized concepts + 51 relational patterns",
                       "stated": t48.node_counts_by_kind.get("tag"),
                       "computed": o.get("tags_specialized_concepts", 0) + o.get("tags_cross_cutting_relational_patterns", 0)})
    for c in checks:
        c["consistent"] = c["stated"] == c["computed"]
    return checks


def audit(graph_path: Path, onto_path: Path, onto_fmt: str | None) -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    declared = load_declared_variants(cfg.get("declared_variants"))
    files = {"graph_json": file_entry(graph_path), "cst_onto": file_entry(onto_path)}
    report = {
        "config": str(CONFIG.relative_to(ROOT)),
        "files": files,
        "repository_scan": scan_repository(),
        "cited_sources": cited_sources_status(),
        "declared_variants": {vid: {"source_reference": d.source_reference,
                                    "node_counts_by_kind": dict(d.node_counts_by_kind),
                                    "ontology_class_count": d.ontology_class_count,
                                    "other_counts": dict(d.other_counts)} for vid, d in declared.items()},
        "declared_arithmetic": declared_arithmetic(declared),
        "rights": cfg.get("rights") or {},
        "problems": [], "blocking": [],
    }
    graph = None
    if files["graph_json"]["present"]:
        try:
            graph = CulturalKnowledgeGraph.from_json(graph_path)
        except (ValueError, KeyError, TypeError) as exc:
            report["problems"].append(f"graph JSON could not be loaded: {exc}")
    if files["cst_onto"]["present"]:
        try:
            inv = ontology_inventory(onto_path, onto_fmt)
            report["ontology_inventory"] = inv
            report["ontology_vs_declared"] = {vid: compare_inventory(inv, d) for vid, d in declared.items()}
            report["problems"] += [f"ontology subclass cycle: {' -> '.join(c)}" for c in inv["subclass_cycles"]]
            if not inv["ontology_iris"] or not inv["version_info"]:
                report["blocking"].append("supplied ontology has no owl:Ontology IRI and version (owl:versionIRI/versionInfo); "
                                          "its identity cannot be recorded")
            if graph is not None:
                graph.attach_ontology(load_ontology_rdf(onto_path, onto_fmt))
        except ImportError as exc:
            report["blocking"].append(str(exc))
        except Exception as exc:  # rdflib parse errors have many types
            report["problems"].append(f"ontology could not be parsed: {type(exc).__name__}: {exc}")
    if graph is not None:
        report["graph_validation"] = graph.validate()
        report["problems"] += report["graph_validation"]
        report["graph_provenance"] = graph.provenance()
        report["graph_vs_declared"] = {vid: graph.count_report(d) for vid, d in declared.items()}
        prov = report["graph_provenance"]
        if prov["marked_synthetic_or_demo"]:
            report["blocking"].append("supplied graph is marked synthetic/test/demo in its metadata; it cannot stand in for the CKG")
        if prov["missing_fields"]:
            report["blocking"].append(f"supplied graph metadata lacks provenance fields {prov['missing_fields']}")
    for key, entry in files.items():
        if not entry["present"]:
            report["blocking"].append(f"{key} not present at {entry['path']}")
    cs = report["cited_sources"]
    # The cited locations matter only while no local copy is supplied.
    if cs.get("status") != "not_collected" and not all(e["present"] for e in files.values()):
        report["blocking"].append(
            f"cited CKG locations unavailable: Zenodo 8847378 API {cs['ckg_zenodo_8847378']['api']} / DOI "
            f"{cs['ckg_zenodo_8847378']['doi']}; github.com/mossab-ansg/ckg-arabic {cs['ckg_github_repo']}")
    report["problems"] = list(dict.fromkeys(report["problems"]))
    pending_rights = sorted(f"{res}.{k}" for res, fields in report["rights"].items() if isinstance(fields, dict)
                            for k, v in fields.items() if v == "pending")
    if pending_rights:
        report["blocking"].append("rights/use/admission not decided (pending): " + ", ".join(pending_rights))
    report["status"] = "invalid" if report["problems"] else ("blocked" if report["blocking"] else "loaded_pending_identity_review")
    report["notice"] = ("Resource audit only. Counts stated in the thesis are declarations, not data; "
                        "no CKG or CST-Onto content was created or inferred.")
    return report


def exit_code(report: dict) -> int:
    return {"invalid": 1, "blocked": 2}.get(report["status"], 0)


def render(r: dict) -> str:
    out = [f"Config: {r['config']}", "", "Configured files:"]
    for k, e in r["files"].items():
        out.append(f"  {k}: {e['path']} -> " + (f"present, sha256 {e['sha256'][:16]}…, {e['bytes']} bytes" if e["present"] else "MISSING"))
    out += ["", "Ontology/graph files tracked in the repository:"]
    out += [f"  {h['path']}: {h['classification']}" for h in r["repository_scan"]] or ["  none"]
    cs = r["cited_sources"]
    out += ["", "Cited sources (CI evidence):"]
    if cs.get("status") == "not_collected":
        out.append(f"  not collected ({cs['evidence']})")
    else:
        out.append(f"  run {cs['ci_run'].get('GITHUB_RUN_ID')} at {cs['collected_at_utc']}")
        out.append(f"  CKG Zenodo 8847378: API {cs['ckg_zenodo_8847378']['api']}, DOI {cs['ckg_zenodo_8847378']['doi']}")
        out.append(f"  github.com/mossab-ansg/ckg-arabic: {cs['ckg_github_repo']} (owner {cs['ckg_github_owner']})")
        p = cs["cst_onto_paper"]
        out.append(f"  CST-Onto paper: Crossref {p['crossref_status']} '{p['title']}' ({p['container']}, {p['published']}); "
                   f"licence {p['license_urls'] or 'none listed'}; links {p['links']}; landing page {p['landing_page_status']}")
        out.append(f"  searches: Zenodo {cs['zenodo_search_hits']}; GitHub {cs['github_search_hits']}")
        o = cs.get("owner_repositories") or {}
        if o.get("repositories_scanned"):
            listed = ", ".join("%s [%s]" % (r["name"], r["license"]) for r in o["repositories"])
            out.append(f"  owner's public repositories scanned: {o['repositories_scanned']} ({listed})")
            out.append(f"  RDF/OWL files found: {len(o['rdf_owl_files_found'])}; other ontology-like paths: "
                       + (", ".join(f"{f['repository']}:{f['path']}" for f in o["ontology_like_files_outside_ansg"] if not f["is_rdf_owl"]) or "none"))
    out += ["", "Rights and admission (configs/knowledge/ckg.yaml):"]
    for res, fields in (r.get("rights") or {}).items():
        if res == "stated_in_sources":
            out += [f"  stated: {x['resource']}: {x['statement']} ({x['source']}; artifact found: {x['artifact_found']})" for x in fields]
            continue
        out.append(f"  {res}: " + ", ".join(f"{k}={v}" for k, v in fields.items()))
    out += ["", "Arithmetic of counts stated in the thesis:"]
    for c in r["declared_arithmetic"]:
        out.append(f"  [{'ok' if c['consistent'] else 'INCONSISTENT'}] {c['check']}: stated {c['stated']}, computed {c['computed']}")
    if "graph_provenance" in r:
        out += ["", f"Supplied graph provenance: {r['graph_provenance']}"]
    if "ontology_inventory" in r:
        inv = {k: v for k, v in r["ontology_inventory"].items() if k != "subclass_cycles"}
        out += ["", f"Supplied ontology inventory: {inv}"]
    if r["problems"]:
        out += ["", "Problems:"] + [f"  - {p}" for p in r["problems"]]
    out += ["", f"Status: {r['status'].upper()}"]
    out += [f"  {i}. {b}" for i, b in enumerate(r["blocking"], 1)]
    out += ["", r["notice"]]
    return "\n".join(out)


def main(argv=None):
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph", default=str(ROOT / cfg["data"]["graph_json"]))
    ap.add_argument("--ontology", default=str(ROOT / cfg["data"]["cst_onto"]))
    ap.add_argument("--ontology-format", default=cfg["data"].get("cst_onto_format"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = audit(Path(args.graph).resolve(), Path(args.ontology).resolve(), args.ontology_format)
    print(json.dumps(report, ensure_ascii=False, indent=1) if args.json else render(report))
    return exit_code(report)


if __name__ == "__main__":
    sys.exit(main())
