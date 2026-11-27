"""OWL2Vec*-style node embeddings for the Cultural Knowledge Graph (dissertation §4.3.2).

The dissertation initialises schema embeddings with OWL2Vec* (Chen et al. 2021)
on CST-Onto: 500,000 random walks of length 10 over the ontology triples,
then Word2Vec skip-gram with 300 dimensions, window 5, 15 negative samples and
25 epochs. This module follows that recipe on any ``CulturalKnowledgeGraph``:

* **structural document**: random walks that alternate entity and relation
  tokens (``node, relation, node, ...``) over edges in both directions
  (inverse relations are prefixed ``inv:``);
* **lexical document**: one sentence per node, the node token followed by its
  normalised Arabic label words, as in OWL2Vec*'s lexical corpus.

It is a re-implementation of the recipe, not the OWL2Vec* package; it reads no
OWL reasoning output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .ckg import CulturalKnowledgeGraph


@dataclass(frozen=True)
class WalkSettings:
    walks: int = 500_000      # §4.3.2
    walk_length: int = 10     # entities per walk, §4.3.2
    dim: int = 300            # §4.3.2
    window: int = 5
    negative: int = 15
    epochs: int = 25
    seed: int = 42
    workers: int = 1          # 1 keeps results deterministic; more workers are faster but not bit-reproducible


def _normalise(text: str) -> list[str]:
    from ..evaluation.seed_adherence import tokens
    return tokens(text)


def structural_walks(graph: CulturalKnowledgeGraph, settings: WalkSettings) -> list[list[str]]:
    out_edges: dict[str, list[tuple[str, str]]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        out_edges[e.source].append((e.relation, e.target))
        out_edges[e.target].append(("inv:" + e.relation, e.source))
    rng = random.Random(settings.seed)
    starts = sorted(n for n, nbrs in out_edges.items() if nbrs)
    walks = []
    for _ in range(settings.walks):
        cur = rng.choice(starts)
        walk = [cur]
        for _ in range(settings.walk_length - 1):
            nbrs = out_edges[cur]
            if not nbrs:
                break
            rel, cur = rng.choice(nbrs)
            walk += [rel, cur]
        walks.append(walk)
    return walks


def lexical_sentences(graph: CulturalKnowledgeGraph) -> list[list[str]]:
    return [[nid] + _normalise(n.labels.get("ar", "")) for nid, n in sorted(graph.nodes.items())]


def train_node_embeddings(graph: CulturalKnowledgeGraph, settings: WalkSettings = WalkSettings()):
    """Return (node_ids, matrix [N, dim]) in ``graph.index()`` order."""
    from gensim.models import Word2Vec

    corpus = structural_walks(graph, settings) + lexical_sentences(graph)
    model = Word2Vec(
        sentences=corpus, vector_size=settings.dim, window=settings.window, negative=settings.negative,
        sg=1, min_count=1, epochs=settings.epochs, seed=settings.seed, workers=settings.workers,
    )
    index = graph.index()
    ids = sorted(index, key=index.get)
    return ids, np.stack([model.wv[nid] for nid in ids]).astype(np.float32)


def load_schema_embeddings(npz_path, graph: CulturalKnowledgeGraph, kinds=("schema", "virtue")):
    """Rows of a saved embedding file for the CSAM schemas (value concepts and virtue classes).

    Returns (schema_ids, matrix [S, dim]); pass the matrix to ``CulturalSchemaBias``.
    """
    data = np.load(npz_path, allow_pickle=False)
    row = {nid: i for i, nid in enumerate(data["ids"].tolist())}
    ids = sorted(nid for nid, n in graph.nodes.items() if n.kind in kinds)
    missing = [nid for nid in ids if nid not in row]
    if missing:
        raise KeyError(f"{len(missing)} schema nodes have no embedding, e.g. {missing[0]!r}; rebuild the embeddings")
    return ids, data["vectors"][[row[nid] for nid in ids]]
