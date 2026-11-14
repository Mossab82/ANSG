"""CKG / CST-Onto audit: blocks without real resources and never turns stated counts into data."""
import importlib.util
import math
from pathlib import Path

import pytest

from ansg.knowledge.ckg import (
    CulturalKnowledgeGraph, DeclaredStructure, OntologyClass, compare_inventory, subclass_cycles,
)

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "ckg"


def _cli():
    s = importlib.util.spec_from_file_location("audit_ckg_resources", ROOT / "scripts" / "audit_ckg_resources.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_default_audit_is_blocked_and_reports_missing_resources(capsys):
    cli = _cli()
    assert cli.main([]) == 2
    out = capsys.readouterr().out
    assert "graph_json not present" in out and "cst_onto not present" in out
    assert "Counts stated in the thesis are declarations, not data" in out


def test_declared_counts_are_never_reported_as_present():
    cli = _cli()
    r = cli.audit(ROOT / "data/private/ckg/ckg.json", ROOT / "data/private/ckg/cst_onto.ttl", "turtle")
    assert r["status"] == "blocked"
    assert "graph_vs_declared" not in r and "ontology_inventory" not in r
    assert not any(e["present"] for e in r["files"].values())


def test_declared_arithmetic_flags_table_a16():
    cli = _cli()
    r = cli.audit(ROOT / "missing.json", ROOT / "missing.ttl", "turtle")
    checks = {c["check"]: c for c in r["declared_arithmetic"]}
    a16 = next(c for k, c in checks.items() if k.startswith("Table A.16"))
    assert (a16["stated"], a16["computed"], a16["consistent"]) == (1847, 2047, False)
    for k, c in checks.items():
        if not k.startswith("Table A.16"):
            assert c["consistent"], k


def test_repository_scan_classifies_every_ontology_like_file():
    cli = _cli()
    for hit in cli.scan_repository():
        assert not hit["classification"].startswith("unclassified"), hit


def test_cited_sources_come_from_recorded_evidence():
    cli = _cli()
    cs = cli.cited_sources_status()
    if cs.get("status") == "not_collected":
        pytest.skip("CKG source evidence not collected")
    assert cs["ckg_zenodo_8847378"]["api"] == 404 and cs["ckg_github_repo"] == 404


def test_synthetic_graph_cannot_unblock(tmp_path):
    cli = _cli()
    r = cli.audit(FIX / "synthetic_ckg.json", tmp_path / "absent.ttl", "turtle")
    assert r["graph_provenance"]["marked_synthetic_or_demo"] is True
    assert r["status"] == "blocked"
    assert any("synthetic" in b for b in r["blocking"])


def test_graph_provenance_requires_fields_and_non_synthetic_metadata():
    g = CulturalKnowledgeGraph.from_dict({"metadata": {"name": "x", "version": "1", "source": "owner export", "license": "CC BY-NC-SA 4.0"},
                                          "nodes": [{"id": "a", "kind": "archetype"}]})
    assert g.provenance()["eligible_for_thesis_identity_review"] is True
    g2 = CulturalKnowledgeGraph.from_dict({"metadata": {"name": "x"}, "nodes": []})
    assert g2.provenance()["missing_fields"] == ["version", "source", "license"]


def test_validate_detects_duplicate_edges_nonfinite_weights_and_cycles():
    g = CulturalKnowledgeGraph.from_dict({
        "ontology_classes": [{"iri": "u:A", "parents": ["u:B"]}, {"iri": "u:B", "parents": ["u:A"]}],
        "nodes": [{"id": "a", "kind": "k"}, {"id": "b", "kind": "k"}],
        "edges": [{"source": "a", "relation": "r", "target": "b"}, {"source": "a", "relation": "r", "target": "b"},
                  {"source": "b", "relation": "r", "target": "a", "weight": math.inf}],
    })
    problems = g.validate()
    assert any(p.startswith("duplicate edge") for p in problems)
    assert any("non-finite weight" in p for p in problems)
    assert any(p.startswith("ontology subclass cycle") for p in problems)
    assert subclass_cycles([OntologyClass("u:A", None, ("u:B",)), OntologyClass("u:B")]) == []


def test_ontology_inventory_and_comparison():
    pytest.importorskip("rdflib")
    from ansg.knowledge.ckg import ontology_inventory
    inv = ontology_inventory(FIX / "synthetic_onto_inventory.ttl", "turtle")
    assert (inv["owl_classes"], inv["object_properties"], inv["datatype_properties"], inv["swrl_rules"],
            inv["named_individuals"]) == (4, 2, 1, 1, 1)
    assert inv["ontology_iris"] == ["urn:test:inventory"] and inv["subclass_cycles"]
    declared = DeclaredStructure(ontology_class_count=934, other_counts={"object_properties": 143, "swrl_rules": 234,
                                                                          "annotation_instances": 647})
    cmp = compare_inventory(inv, declared)
    assert cmp["rows"]["ontology_class_count"] == {"declared": 934, "found": 4, "status": "mismatch"}
    assert cmp["rows"]["datatype_properties"]["status"] == "undeclared"
    assert cmp["not_comparable_declared_counts"] == ["annotation_instances"]


def test_audit_reports_invalid_ontology_structure():
    pytest.importorskip("rdflib")
    cli = _cli()
    r = cli.audit(FIX / "synthetic_ckg.json", FIX / "synthetic_onto_inventory.ttl", "turtle")
    assert r["status"] == "invalid" and cli.exit_code(r) == 1
    assert sum(p.startswith("ontology subclass cycle") for p in r["problems"]) == 1


def test_rights_use_and_admission_stay_pending_and_block():
    cli = _cli()
    r = cli.audit(ROOT / "missing.json", ROOT / "missing.ttl", "turtle")
    assert set(r["rights"]) == {"ckg", "cst_onto"}
    for fields in r["rights"].values():
        assert fields == {"licence": "pending", "permitted_use": "pending", "redistribution": "pending", "admission": "pending"}
    assert any(b.startswith("rights/use/admission not decided") for b in r["blocking"])
