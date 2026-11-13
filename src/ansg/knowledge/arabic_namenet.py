"""Adapter for the ArabicNameNet entity resource.

Format status: **unverified.** Neither the legacy code nor any file in this
repository contains or reads ArabicNameNet. The legacy scripts recognised
named entities with polyglot's Arabic NER (``ANER``) and by probing Arabic
Wikipedia page titles (``Wiki_NER``) — neither is ArabicNameNet, and this
adapter does not substitute them.

Because the actual file layout is not available, the adapter is driven by an
explicit column mapping supplied in YAML (``configs/knowledge/resources.yaml``)
for delimited (CSV/TSV) or JSON-Lines exports:

* an **entity table**: id, name, entity type, optional variant/alias field
  and any number of attribute columns;
* an optional **relation table**: source id, relation type, target id.

Once the real resource is supplied, its mapping should be fixed in the config
and a fixture derived from (a licensed excerpt of) it added to the tests.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..data.normalization import NormalizationConfig, normalize, simple_tokenize

# Matching uses an explicit, lossy folding that is *separate* from corpus normalisation.
MATCH_FOLDING = NormalizationConfig(
    strip_diacritics=True, strip_dagger_alef=True, normalize_alef=True,
    normalize_alef_maqsura=True, normalize_teh_marbuta=True,
)


def fold_name(name: str) -> str:
    return normalize(name, MATCH_FOLDING).text


@dataclass(frozen=True)
class TableMapping:
    path: str
    format: str = "csv"  # csv | tsv | jsonl
    encoding: str = "utf-8"
    columns: Mapping[str, str] = field(default_factory=dict)
    attribute_columns: tuple[str, ...] = ()
    variant_separator: str | None = None

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> "TableMapping":
        d = dict(d)
        d["attribute_columns"] = tuple(d.get("attribute_columns") or ())
        return cls(**d)


@dataclass
class NameEntity:
    entity_id: str
    name: str
    entity_type: str | None
    variants: list[str] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NameRelation:
    source_id: str
    relation: str
    target_id: str


@dataclass(frozen=True)
class NameMatch:
    start: int
    end: int
    surface: str
    entity_ids: tuple[str, ...]


def _read_rows(m: TableMapping, base_dir: Path) -> Iterator[dict[str, Any]]:
    p = Path(m.path)
    p = p if p.is_absolute() else base_dir / p
    if not p.is_file():
        raise FileNotFoundError(f"ArabicNameNet file not found: {p}")
    with open(p, encoding=m.encoding, newline="") as fh:
        if m.format == "jsonl":
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        elif m.format in ("csv", "tsv"):
            yield from csv.DictReader(fh, delimiter="," if m.format == "csv" else "\t")
        else:
            raise ValueError(f"unsupported format {m.format!r}")


def _col(row: Mapping[str, Any], m: TableMapping, key: str, required: bool) -> Any:
    col = m.columns.get(key)
    if col is None:
        if required:
            raise ValueError(f"column mapping for '{key}' is required")
        return None
    if col not in row:
        raise KeyError(f"column '{col}' (for '{key}') not in source row keys {sorted(row)}")
    return row[col]


@dataclass
class ArabicNameNet:
    entities: dict[str, NameEntity]
    relations: list[NameRelation]
    _index: dict[str, set[str]] = field(default_factory=dict, repr=False)
    _max_tokens: int = field(default=1, repr=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self._index.clear()
        self._max_tokens = 1
        for e in self.entities.values():
            for n in [e.name, *e.variants]:
                key = " ".join(t for _, _, t in simple_tokenize(fold_name(n)))
                if key:
                    self._index.setdefault(key, set()).add(e.entity_id)
                    self._max_tokens = max(self._max_tokens, key.count(" ") + 1)

    @classmethod
    def load(
        cls,
        entity_table: TableMapping,
        relation_table: TableMapping | None = None,
        base_dir: str | os.PathLike[str] = ".",
    ) -> "ArabicNameNet":
        base = Path(base_dir)
        entities: dict[str, NameEntity] = {}
        for i, row in enumerate(_read_rows(entity_table, base)):
            eid = _col(row, entity_table, "id", required=False)
            eid = str(eid) if eid not in (None, "") else f"row{i + 1}"
            name = _col(row, entity_table, "name", required=True)
            if not name:
                raise ValueError(f"entity {eid}: empty name")
            variants_raw = _col(row, entity_table, "variants", required=False)
            if isinstance(variants_raw, str) and entity_table.variant_separator:
                variants = [v.strip() for v in variants_raw.split(entity_table.variant_separator) if v.strip()]
            elif isinstance(variants_raw, list):
                variants = [str(v) for v in variants_raw]
            else:
                variants = []
            attrs = {c: row.get(c) for c in entity_table.attribute_columns}
            if eid in entities:
                raise ValueError(f"duplicate entity id {eid!r}")
            entities[eid] = NameEntity(eid, str(name), _col(row, entity_table, "type", required=False) or None,
                                       variants, attrs)
        relations: list[NameRelation] = []
        if relation_table is not None:
            for row in _read_rows(relation_table, base):
                relations.append(NameRelation(
                    str(_col(row, relation_table, "source", True)),
                    str(_col(row, relation_table, "relation", True)),
                    str(_col(row, relation_table, "target", True)),
                ))
        return cls(entities=entities, relations=relations)

    # ---------------------------------------------------------------- queries
    def lookup(self, name: str) -> list[NameEntity]:
        key = " ".join(t for _, _, t in simple_tokenize(fold_name(name)))
        return [self.entities[i] for i in sorted(self._index.get(key, ()))]

    def related(self, entity_id: str, relation: str | None = None) -> list[NameRelation]:
        return [r for r in self.relations
                if r.source_id == entity_id and (relation is None or r.relation == relation)]

    def dangling_relations(self) -> list[NameRelation]:
        return [r for r in self.relations
                if r.source_id not in self.entities or r.target_id not in self.entities]

    def find_in_text(self, text: str) -> list[NameMatch]:
        """Greedy longest-match gazetteer lookup on folded tokens.

        Offsets refer to ``text`` as given. This is a lexical lookup, not NER:
        it does not handle clitics (e.g. "وشهريار") or disambiguation.
        """
        tokens = simple_tokenize(text)
        folded = [fold_name(t) for _, _, t in tokens]
        out: list[NameMatch] = []
        i = 0
        while i < len(tokens):
            hit = None
            for n in range(min(self._max_tokens, len(tokens) - i), 0, -1):
                key = " ".join(folded[i:i + n])
                if key in self._index:
                    hit = (n, key)
                    break
            if hit:
                n, key = hit
                s, e = tokens[i][0], tokens[i + n - 1][1]
                out.append(NameMatch(s, e, text[s:e], tuple(sorted(self._index[key]))))
                i += n
            else:
                i += 1
        return out

    def stats(self) -> dict[str, Any]:
        types: dict[str, int] = {}
        for e in self.entities.values():
            types[e.entity_type or "<none>"] = types.get(e.entity_type or "<none>", 0) + 1
        return {"entities": len(self.entities), "relations": len(self.relations),
                "entity_types": dict(sorted(types.items())),
                "dangling_relations": len(self.dangling_relations())}
