"""Cultural Knowledge Graph (CKG) interface for CSAM, with CST-Onto integration.

Scope of this module (Phase 1)
------------------------------
* A typed, validated in-memory graph of **cultural schemas / motifs**, their
  **relations**, and their mapping to **CST-Onto** ontology classes.
* A stable integer indexing (:meth:`CulturalKnowledgeGraph.index`) and
  adjacency export that CSAM (Phase 2) will consume to build schema
  embeddings and attention masks — without importing PyTorch here.
* A distinction, enforced in code, between **declared** structure (counts and
  relation types stated in configuration, e.g. transcribed from the
  dissertation) and **supplied** structure (nodes/edges actually loaded from
  data files). :meth:`count_report` compares the two and never fills gaps.
* The :class:`SchemaActivator` protocol for schema activation scores. The
  dissertation's activation function is **not** implemented here because its
  definition has not been transcribed into this repository; see
  ``docs/IMPLEMENTATION_GAPS.md``. :class:`LexicalTriggerEvidence` is a
  transparent data-inspection utility, not the thesis activation function.

Input format (JSON, defined by this reimplementation)::

    {
      "metadata": {"name": "...", "version": "...", "source": "...", "license": "..."},
      "ontology_classes": [{"iri": "...", "label": "...", "parents": ["..."]}],
      "nodes": [{"id": "...", "kind": "motif|schema|...", "labels": {"ar": "...", "en": "..."},
                 "ontology_class": "<iri>", "triggers": ["..."], "attributes": {...}}],
      "edges": [{"source": "...", "relation": "...", "target": "...", "weight": 1.0}]
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

from ..data.normalization import NormalizationConfig, normalize, simple_tokenize

_TRIGGER_FOLD = NormalizationConfig(strip_diacritics=True, normalize_alef=True,
                                    normalize_alef_maqsura=True, normalize_teh_marbuta=True)


@dataclass(frozen=True)
class OntologyClass:
    iri: str
    label: str | None = None
    parents: tuple[str, ...] = ()


@dataclass
class CKGNode:
    node_id: str
    kind: str
    labels: dict[str, str] = field(default_factory=dict)
    ontology_class: str | None = None
    triggers: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CKGEdge:
    source: str
    relation: str
    target: str
    weight: float | None = None


@dataclass(frozen=True)
class DeclaredStructure:
    """Structure *stated* in configuration (e.g. transcribed from the thesis).

    ``None`` means "not declared". Values here are never treated as data.
    """

    node_counts_by_kind: Mapping[str, int | None] = field(default_factory=dict)
    ontology_class_count: int | None = None
    relation_types: tuple[str, ...] | None = None
    node_kinds: tuple[str, ...] | None = None
    source_reference: str | None = None  # e.g. "Dissertation §3.1.4, Table A.16"
    # Counts stated in the thesis that the in-memory graph cannot check directly
    # (e.g. SWRL rules, datatype properties); carried into reports verbatim.
    other_counts: Mapping[str, int | None] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None) -> "DeclaredStructure":
        d = dict(d or {})
        for k in ("relation_types", "node_kinds"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        d["node_counts_by_kind"] = dict(d.get("node_counts_by_kind") or {})
        d["other_counts"] = dict(d.get("other_counts") or {})
        return cls(**d)


@dataclass
class ActivationScores:
    """Schema activation output consumed by CSAM.

    ``scores`` maps node ids to real values; ``method`` names the function
    that produced them (must identify a thesis equation once implemented).
    """

    scores: dict[str, float]
    method: str
    evidence: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)

    def as_vector(self, index: Mapping[str, int]) -> list[float]:
        vec = [0.0] * len(index)
        for nid, s in self.scores.items():
            vec[index[nid]] = s
        return vec


@runtime_checkable
class SchemaActivator(Protocol):
    """Computes activation scores over CKG nodes for an input context."""

    method: str

    def activate(self, text: str, graph: "CulturalKnowledgeGraph") -> ActivationScores: ...


class CKGValidationError(ValueError):
    pass


@dataclass
class CulturalKnowledgeGraph:
    nodes: dict[str, CKGNode]
    edges: list[CKGEdge]
    ontology: dict[str, OntologyClass] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ load
    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "CulturalKnowledgeGraph":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CulturalKnowledgeGraph":
        nodes: dict[str, CKGNode] = {}
        for n in data.get("nodes", []):
            node = CKGNode(node_id=str(n["id"]), kind=str(n["kind"]), labels=dict(n.get("labels") or {}),
                           ontology_class=n.get("ontology_class"), triggers=list(n.get("triggers") or []),
                           attributes=dict(n.get("attributes") or {}))
            if node.node_id in nodes:
                raise CKGValidationError(f"duplicate node id {node.node_id!r}")
            nodes[node.node_id] = node
        edges = [CKGEdge(str(e["source"]), str(e["relation"]), str(e["target"]),
                         None if e.get("weight") is None else float(e["weight"]))
                 for e in data.get("edges", [])]
        ontology = {c["iri"]: OntologyClass(c["iri"], c.get("label"), tuple(c.get("parents") or ()))
                    for c in data.get("ontology_classes", [])}
        return cls(nodes=nodes, edges=edges, ontology=ontology, metadata=dict(data.get("metadata") or {}))

    def attach_ontology(self, classes: Iterable[OntologyClass]) -> None:
        for c in classes:
            self.ontology[c.iri] = c

    # -------------------------------------------------------------- validate
    def validate(self, declared: DeclaredStructure | None = None) -> list[str]:
        """Return a list of integrity problems (empty list = valid)."""
        problems: list[str] = []
        for e in self.edges:
            if e.source not in self.nodes:
                problems.append(f"edge source {e.source!r} is not a node")
            if e.target not in self.nodes:
                problems.append(f"edge target {e.target!r} is not a node")
        if self.ontology:
            for n in self.nodes.values():
                if n.ontology_class and n.ontology_class not in self.ontology:
                    problems.append(f"node {n.node_id!r}: ontology class {n.ontology_class!r} not loaded")
            for c in self.ontology.values():
                for p in c.parents:
                    if p not in self.ontology:
                        problems.append(f"ontology class {c.iri!r}: parent {p!r} not loaded")
        if declared is not None:
            if declared.relation_types is not None:
                for e in self.edges:
                    if e.relation not in declared.relation_types:
                        problems.append(f"edge relation {e.relation!r} not in declared relation types")
            if declared.node_kinds is not None:
                for n in self.nodes.values():
                    if n.kind not in declared.node_kinds:
                        problems.append(f"node {n.node_id!r}: kind {n.kind!r} not declared")
        return problems

    def count_report(self, declared: DeclaredStructure) -> dict[str, Any]:
        """Compare supplied counts against declared counts. Never infers missing data."""
        actual_kinds: dict[str, int] = {}
        for n in self.nodes.values():
            actual_kinds[n.kind] = actual_kinds.get(n.kind, 0) + 1
        rows = {}
        for kind in sorted(set(actual_kinds) | set(declared.node_counts_by_kind)):
            d = declared.node_counts_by_kind.get(kind)
            a = actual_kinds.get(kind, 0)
            rows[kind] = {"declared": d, "supplied": a,
                          "status": "undeclared" if d is None else ("match" if d == a else "mismatch")}
        onto_decl = declared.ontology_class_count
        return {
            "source_reference": declared.source_reference,
            "other_declared_counts": dict(declared.other_counts),
            "nodes_by_kind": rows,
            "ontology_classes": {
                "declared": onto_decl, "supplied": len(self.ontology),
                "status": "undeclared" if onto_decl is None else (
                    "match" if onto_decl == len(self.ontology) else "mismatch"),
            },
            "edges_supplied": len(self.edges),
        }

    # ------------------------------------------------------------ CSAM view
    def index(self, kinds: Sequence[str] | None = None) -> dict[str, int]:
        """Stable node-id -> row index (sorted by id) for embedding tables."""
        ids = sorted(n.node_id for n in self.nodes.values() if kinds is None or n.kind in kinds)
        return {nid: i for i, nid in enumerate(ids)}

    def adjacency(
        self, index: Mapping[str, int], relations: Iterable[str] | None = None,
        default_weight: float = 1.0,
    ) -> list[tuple[int, int, float]]:
        """Sparse ``(row, col, weight)`` triples over the indexed nodes."""
        rel = set(relations) if relations is not None else None
        out = []
        for e in self.edges:
            if (rel is None or e.relation in rel) and e.source in index and e.target in index:
                out.append((index[e.source], index[e.target],
                            e.weight if e.weight is not None else default_weight))
        return sorted(out)

    def neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        return [e.target for e in self.edges
                if e.source == node_id and (relation is None or e.relation == relation)]

    def ontology_ancestors(self, iri: str) -> list[str]:
        seen: list[str] = []
        stack = list(self.ontology[iri].parents) if iri in self.ontology else []
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.append(p)
                stack.extend(self.ontology[p].parents if p in self.ontology else ())
        return seen


class LexicalTriggerEvidence:
    """Find supplied lexical triggers of CKG nodes in text.

    **Not the dissertation's schema activation function.** It reports which
    configured trigger phrases occur (folded match on whitespace tokens) and
    returns their *counts* as ``scores`` with ``method='lexical_trigger_count'``,
    for corpus inspection and for testing the CSAM data path. Nodes without
    supplied triggers can never be activated by it.
    """

    method = "lexical_trigger_count"

    def activate(self, text: str, graph: CulturalKnowledgeGraph) -> ActivationScores:
        toks = simple_tokenize(text)
        folded = [normalize(t, _TRIGGER_FOLD).text for _, _, t in toks]
        scores: dict[str, float] = {}
        evidence: dict[str, list[tuple[int, int, str]]] = {}
        for node in graph.nodes.values():
            for trig in node.triggers:
                tt = [normalize(t, _TRIGGER_FOLD).text for _, _, t in simple_tokenize(trig)]
                if not tt:
                    continue
                for i in range(len(folded) - len(tt) + 1):
                    if folded[i:i + len(tt)] == tt:
                        s, e = toks[i][0], toks[i + len(tt) - 1][1]
                        evidence.setdefault(node.node_id, []).append((s, e, text[s:e]))
                        scores[node.node_id] = scores.get(node.node_id, 0.0) + 1.0
        return ActivationScores(scores=scores, method=self.method, evidence=evidence)


class CosineThresholdActivator:
    """Schema activation by cosine similarity — dissertation Eq. (4.20) / App. A.4.

    ``w_s = cos(e_context, c_s)``; schema ``s`` is active if ``w_s`` passes the
    threshold. Two thesis variants exist (SPECIFICATION_CONFLICTS C-CSAM-5):

    * Ch. 4 Eq. (4.20) / Table 4.2: ``w_s > τ`` with τ = 0.5, 768-d embeddings
      (Eq. 4.8), "3–5 typical" active schemas;
    * App. A.4 Table A.5: ``cos ≥ 0.75`` on 256-d embeddings, at most 50 active.

    Embeddings are **inputs**: the thesis derives schema embeddings from
    OWL2Vec* over CST-Onto (Eq. 4.8) and does not specify how the context
    embedding is computed ("derived from the narrative segment"), so both the
    schema vectors and ``context_encoder`` must be supplied.
    """

    def __init__(
        self,
        schema_embeddings: Mapping[str, Sequence[float]],
        context_encoder: Any,
        threshold: float,
        inclusive: bool,
        max_active: int | None = None,
        method: str = "cosine_threshold",
    ) -> None:
        dims = {len(v) for v in schema_embeddings.values()}
        if len(dims) > 1:
            raise ValueError(f"schema embeddings have mixed dimensions {sorted(dims)}")
        self.schema_embeddings = {k: list(v) for k, v in schema_embeddings.items()}
        self.context_encoder = context_encoder
        self.threshold = threshold
        self.inclusive = inclusive
        self.max_active = max_active
        self.method = method

    @staticmethod
    def _cos(a: Sequence[float], b: Sequence[float]) -> float:
        na = sum(x * x for x in a) ** 0.5
        nb = sum(x * x for x in b) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    def activate(self, text: str, graph: "CulturalKnowledgeGraph") -> ActivationScores:
        ctx = list(self.context_encoder(text))
        scores: dict[str, float] = {}
        for nid in graph.nodes:
            vec = self.schema_embeddings.get(nid)
            if vec is None:
                continue
            if len(vec) != len(ctx):
                raise ValueError(f"context dim {len(ctx)} != schema dim {len(vec)}")
            sim = self._cos(ctx, vec)
            if sim > self.threshold or (self.inclusive and sim == self.threshold):
                scores[nid] = sim
        if self.max_active is not None and len(scores) > self.max_active:
            keep = sorted(scores, key=lambda k: (-scores[k], k))[: self.max_active]
            scores = {k: scores[k] for k in keep}
        return ActivationScores(scores=scores, method=self.method)


def load_declared_variants(items: Iterable[Mapping[str, Any]] | None) -> dict[str, DeclaredStructure]:
    """Parse ``declared_variants`` from ``configs/knowledge/ckg.yaml``.

    The dissertation states incompatible counts in different places; each
    statement is kept as its own declaration keyed by ``id``."""
    out: dict[str, DeclaredStructure] = {}
    for item in items or []:
        d = dict(item)
        vid = d.pop("id")
        d.pop("notes", None)
        if vid in out:
            raise ValueError(f"duplicate declared variant {vid!r}")
        out[vid] = DeclaredStructure.from_mapping(d)
    return out


def load_ontology_rdf(path: str | os.PathLike[str], fmt: str | None = None) -> list[OntologyClass]:
    """Load OWL/RDFS classes (``owl:Class``/``rdfs:Class``) from an RDF file.

    Requires the optional ``rdflib`` dependency (``pip install .[ontology]``).
    Labels prefer Arabic, then English, then any language.
    """
    try:
        import rdflib  # type: ignore[import-not-found]
        from rdflib.namespace import OWL, RDF, RDFS  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("load_ontology_rdf requires rdflib: pip install '.[ontology]'") from exc
    g = rdflib.Graph()
    g.parse(str(Path(path)), format=fmt)
    iris = {s for s in g.subjects(RDF.type, OWL.Class)} | {s for s in g.subjects(RDF.type, RDFS.Class)}
    iris = {s for s in iris if isinstance(s, rdflib.URIRef)}
    out = []
    for s in sorted(iris, key=str):
        labels = {getattr(o, "language", None): str(o) for o in g.objects(s, RDFS.label)}
        label = labels.get("ar") or labels.get("en") or next(iter(labels.values()), None)
        parents = tuple(sorted(str(p) for p in g.objects(s, RDFS.subClassOf) if isinstance(p, rdflib.URIRef)))
        out.append(OntologyClass(str(s), label, parents))
    return out
