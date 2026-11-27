import pytest

from ansg.generation.seed_prompt import REGIONS, load_seed
from ansg.knowledge.ckg import CulturalKnowledgeGraph
from ansg.knowledge.seed_ckg import VALUE_TO_VIRTUE, VIRTUES, build_seed_ckg


@pytest.fixture(scope="module")
def graph():
    return CulturalKnowledgeGraph.from_dict(build_seed_ckg())


def test_every_seed_value_is_mapped_to_a_virtue():
    for region in REGIONS:
        for value in load_seed(region)["values"]:
            assert VALUE_TO_VIRTUE.get(value["concept_ar"]) in VIRTUES, value["concept_ar"]


def test_graph_is_valid_and_typed(graph):
    assert graph.validate() == []
    kinds = {n.kind for n in graph.nodes.values()}
    assert {"virtue", "schema", "region", "proverb", "marker"} <= kinds
    assert sum(n.kind == "virtue" for n in graph.nodes.values()) == 9


def test_shared_values_are_one_node(graph):
    generosity = [n for n in graph.nodes.values() if n.kind == "schema" and n.labels["ar"] == "الكرم"]
    assert len(generosity) == 1
    regions = [e.source for e in graph.edges if e.target == generosity[0].node_id and e.relation == "values"]
    assert len(regions) == 5


def test_graph_embeddings_follow_the_recipe_and_cluster(graph):
    pytest.importorskip("gensim")
    from ansg.knowledge.graph_embeddings import WalkSettings, train_node_embeddings
    import numpy as np

    ids, vecs = train_node_embeddings(graph, WalkSettings(walks=4000, epochs=5, dim=32))
    assert vecs.shape == (len(graph.nodes), 32)
    idx = {nid: i for i, nid in enumerate(ids)}
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    courage = idx["virtue:courage"]
    near = unit[idx["schema:الشجاعة"]] @ unit[courage]
    far = unit[idx["region:maghrebi"]] @ unit[courage]
    assert near > far
