import pytest

from ansg.knowledge.ckg import (
    ActivationScores, CulturalKnowledgeGraph, DeclaredStructure, LexicalTriggerEvidence,
    SchemaActivator, load_ontology_rdf,
)


@pytest.fixture
def g(fixtures):
    return CulturalKnowledgeGraph.from_json(fixtures / "ckg" / "synthetic_ckg.json")


def test_valid_graph(g):
    assert g.validate() == []
    assert g.ontology_ancestors("urn:test:onto#Transformation") == [
        "urn:test:onto#Motif", "urn:test:onto#CulturalSchema"]


def test_declared_vs_supplied_counts(g):
    rep = g.count_report(DeclaredStructure(node_counts_by_kind={"motif": 2, "schema": 5}))
    assert rep["nodes_by_kind"]["motif"] == {"declared": 2, "supplied": 2, "status": "match"}
    assert rep["nodes_by_kind"]["schema"]["status"] == "mismatch"
    assert rep["ontology_classes"]["status"] == "undeclared"
    # declarations never add nodes
    assert len(g.nodes) == 3


def test_declared_relation_types_enforced(g):
    problems = g.validate(DeclaredStructure(relation_types=("instance_of",)))
    assert problems == ["edge relation 'related_to' not in declared relation types"]


def test_dangling_edge_detected():
    g = CulturalKnowledgeGraph.from_dict({"nodes": [{"id": "a", "kind": "motif"}],
                                          "edges": [{"source": "a", "relation": "r", "target": "b"}]})
    assert g.validate() == ["edge target 'b' is not a node"]


def test_index_and_adjacency_stable(g):
    idx = g.index()
    assert list(idx) == sorted(idx)
    adj = g.adjacency(idx)
    assert adj == sorted(adj)
    assert (idx["motif:test_magic_ring"], idx["schema:test_hospitality"], 0.5) in adj
    assert g.adjacency(g.index(kinds=["motif"])) == [(1, 0, 1.0)]


def test_lexical_trigger_evidence_is_labelled(g):
    act = LexicalTriggerEvidence()
    assert isinstance(act, SchemaActivator)
    res = act.activate("فوجد خاتمًا في بطن سمكة", g)
    assert res.method == "lexical_trigger_count"
    assert res.scores == {"motif:test_magic_ring": 1.0}
    vec = res.as_vector(g.index())
    assert sum(vec) == 1.0 and len(vec) == 3


def test_rdf_ontology_loader(fixtures, g):
    pytest.importorskip("rdflib")
    classes = load_ontology_rdf(fixtures / "ckg" / "synthetic_onto.ttl", fmt="turtle")
    by = {c.iri: c for c in classes}
    assert by["urn:test:onto#CulturalSchema"].label == "مخطط"
    assert by["urn:test:onto#Motif"].parents == ("urn:test:onto#CulturalSchema",)
