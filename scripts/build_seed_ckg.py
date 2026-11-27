"""Build the seed CKG and its OWL2Vec*-style embeddings (dissertation §4.3.2 settings).

Usage: PYTHONPATH=src python scripts/build_seed_ckg.py [--walks N]
Writes data/ckg/seed_ckg.json and data/ckg/seed_ckg_embeddings.npz.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ansg.knowledge.ckg import CulturalKnowledgeGraph
from ansg.knowledge.graph_embeddings import WalkSettings, train_node_embeddings
from ansg.knowledge.seed_ckg import build_seed_ckg

OUT = Path(__file__).resolve().parents[1] / "data" / "ckg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", type=int, default=WalkSettings.walks)
    ap.add_argument("--workers", type=int, default=WalkSettings.workers)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_seed_ckg()
    (OUT / "seed_ckg.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    graph = CulturalKnowledgeGraph.from_dict(data)
    settings = WalkSettings(walks=args.walks, workers=args.workers)
    start = time.time()
    ids, vecs = train_node_embeddings(graph, settings)
    np.savez_compressed(OUT / "seed_ckg_embeddings.npz", ids=np.array(ids), vectors=vecs,
                        settings=json.dumps(asdict(settings)))
    print(f"{len(ids)} nodes, dim {vecs.shape[1]}, {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
