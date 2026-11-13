"""Deterministic, leakage-safe dataset splitting.

Three strategies:

``predefined``
    Use a supplied ``record_id -> split`` (or ``group -> split``) mapping. This
    is the only way to reproduce the dissertation's original partition, and
    should be used as soon as that partition file is available.
``hash_exact``
    Groups are ordered by ``sha256(seed:group_key)`` and cut at the configured
    ratios. Exact proportions at group level, independent of input order.
``hash_bucket``
    Each group is assigned by its own hash value only. Adding new groups never
    moves existing ones, at the cost of approximate proportions.

All strategies split at *group* level (default group = story) so that sentences
or segments of one tale never appear in more than one split.

The split ratios are configuration values. The dissertation's ratios have not
been verified in this repository (see ``docs/EXPERIMENTAL_SPECIFICATIONS.md``).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from .records import NarrativeRecord

Strategy = Literal["predefined", "hash_exact", "hash_bucket"]


@dataclass(frozen=True)
class SplitConfig:
    strategy: Strategy = "hash_exact"
    ratios: Mapping[str, float] = field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    seed: int = 0
    group_by: Literal["record_id", "parent_story_id", "title", "split_cluster"] = "record_id"
    predefined: Mapping[str, str] | None = None  # key -> split name
    predefined_key: Literal["record_id", "group"] = "record_id"

    def __post_init__(self) -> None:
        if self.strategy != "predefined":
            total = sum(self.ratios.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(f"split ratios must sum to 1, got {total}")
            if any(v < 0 for v in self.ratios.values()):
                raise ValueError("split ratios must be non-negative")

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any] | None) -> "SplitConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown split options: {sorted(unknown)}")
        return cls(**d)


def _group_key(rec: NarrativeRecord, by: str) -> str:
    if by == "record_id":
        return rec.record_id
    if by == "parent_story_id":
        return rec.narrative.parent_story_id or rec.record_id
    if by == "title":
        return rec.narrative.title or rec.record_id
    if by == "split_cluster":
        cid = rec.narrative.extra.get("split_cluster")
        if cid is None:
            raise KeyError(f"{rec.record_id}: no split_cluster; run cluster_records_dbscan first")
        return str(cid)
    raise ValueError(f"unknown group_by {by!r}")


def _hash_unit(seed: int, key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _largest_remainder(n: int, ratios: Mapping[str, float]) -> dict[str, int]:
    raw = {k: n * v for k, v in ratios.items()}
    counts = {k: int(v) for k, v in raw.items()}
    rest = n - sum(counts.values())
    for k in sorted(ratios, key=lambda k: (-(raw[k] - counts[k]), list(ratios).index(k)))[:rest]:
        counts[k] += 1
    return counts


def assign_splits(
    records: Sequence[NarrativeRecord],
    config: SplitConfig,
    group_fn: Callable[[NarrativeRecord], str] | None = None,
) -> dict[str, str]:
    """Return ``record_id -> split``; also sets ``record.split`` in place."""
    key_of = group_fn or (lambda r: _group_key(r, config.group_by))
    groups: dict[str, list[NarrativeRecord]] = {}
    for r in records:
        groups.setdefault(key_of(r), []).append(r)

    group_split: dict[str, str] = {}
    if config.strategy == "predefined":
        if not config.predefined:
            raise ValueError("predefined strategy requires a 'predefined' mapping")
        for g, members in groups.items():
            if config.predefined_key == "group":
                if g not in config.predefined:
                    raise KeyError(f"group {g!r} missing from predefined split")
                group_split[g] = config.predefined[g]
            else:
                names = {config.predefined.get(m.record_id) for m in members}
                if None in names:
                    missing = [m.record_id for m in members if m.record_id not in config.predefined]
                    raise KeyError(f"records missing from predefined split: {missing[:5]}")
                if len(names) > 1:
                    raise ValueError(f"group {g!r} spans several predefined splits: {sorted(names)}")
                group_split[g] = names.pop()  # type: ignore[assignment]
    elif config.strategy == "hash_exact":
        ordered = sorted(groups, key=lambda g: (_hash_unit(config.seed, g), g))
        counts = _largest_remainder(len(ordered), config.ratios)
        i = 0
        for name in config.ratios:
            for g in ordered[i:i + counts[name]]:
                group_split[g] = name
            i += counts[name]
    elif config.strategy == "hash_bucket":
        names = list(config.ratios)
        cum: list[tuple[float, str]] = []
        acc = 0.0
        for name in names:
            acc += config.ratios[name]
            cum.append((acc, name))
        for g in groups:
            u = _hash_unit(config.seed, g)
            group_split[g] = next((n for c, n in cum if u < c), names[-1])
    else:
        raise ValueError(f"unknown strategy {config.strategy!r}")

    out: dict[str, str] = {}
    for g, members in groups.items():
        for m in members:
            m.split = group_split[g]
            out[m.record_id] = group_split[g]
    return out


def check_no_leakage(
    records: Sequence[NarrativeRecord], group_fn: Callable[[NarrativeRecord], str]
) -> list[str]:
    """Return group keys that occur in more than one split."""
    seen: dict[str, set[str | None]] = {}
    for r in records:
        seen.setdefault(group_fn(r), set()).add(r.split)
    return sorted(g for g, s in seen.items() if len(s) > 1)


def cluster_records_dbscan(
    records: Sequence[NarrativeRecord],
    eps: float = 0.3,
    min_samples: int = 2,
    metric: str = "cosine",
    tfidf_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Narrative-level clustering for leakage control — dissertation Ch. 5 §5.4.4.

    "Compute TF-IDF representations for all narratives; cluster using DBSCAN
    (ε = 0.3, min_samples = 2); assign entire clusters to a single split."
    Motivated by multi-witness texts (Bulaq / Calcutta II / Mahdi, §3.2.4).

    Not specified by the thesis and therefore explicit arguments here:
    the distance ``metric`` (cosine is assumed because ε = 0.3 is only
    meaningful on a bounded distance), and the TF-IDF settings (tokeniser,
    n-grams, weighting) — scikit-learn defaults unless ``tfidf_kwargs`` is given.
    DBSCAN noise points (label −1) each form their own singleton group.

    Writes ``narrative.extra["split_cluster"]`` and returns ``record_id -> cluster``.
    Requires scikit-learn (``pip install '.[splits]'``).
    """
    try:
        from sklearn.cluster import DBSCAN  # type: ignore[import-not-found]
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("cluster_records_dbscan requires scikit-learn: pip install '.[splits]'") from exc
    if not records:
        return {}
    texts = [r.text for r in records]
    x = TfidfVectorizer(**dict(tfidf_kwargs or {})).fit_transform(texts)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(x).labels_
    out: dict[str, str] = {}
    for rec, lab in zip(records, labels):
        cid = f"noise:{rec.record_id}" if lab == -1 else f"dbscan:{int(lab)}"
        rec.narrative.extra["split_cluster"] = cid
        out[rec.record_id] = cid
    return out


def split_distribution(
    records: Sequence[NarrativeRecord], key_fn: Callable[[NarrativeRecord], str | None]
) -> dict[str, dict[str, float]]:
    """Per-split proportion of each value of ``key_fn`` (e.g. dialect, period).

    Ch. 5 §5.4.4 states stratification *targets* (dialect 35/25/20/15/5; cultural
    density 30/50/20; period 35/30/35) but not how they are combined with cluster
    assignment, so stratification is reported for comparison, not enforced."""
    by_split: dict[str, Counter[str]] = {}
    for r in records:
        by_split.setdefault(r.split or "<none>", Counter())[str(key_fn(r))] += 1
    return {sp: {k: v / sum(c.values()) for k, v in sorted(c.items())} for sp, c in sorted(by_split.items())}
