"""Train/test leakage audits for frame-tale corpora such as *One Thousand and One Nights*. Torch-free.

Why the Nights need special care
--------------------------------
* **Nights are not stories.** A night ends wherever Shahrazad stops, usually in
  the middle of a tale, and one tale spans many nights. A split *by night number*
  therefore puts the beginning of a tale in training and its continuation in the
  test set: test loss then measures in-story continuation (same characters, same
  plot), not generalisation to unseen narratives.
* **Embedded tales.** Tales are told inside tales (frame narratives). Splitting
  embedded tales independently leaks the frame's characters and plot.
* **Formulae.** Every night opens and closes with near-identical frame formulae
  (بلغني أيها الملك السعيد / وأدرك شهرزاد الصباح فسكتت عن الكلام المباح), which
  inflate n-gram overlap and lower held-out loss without any narrative learning.

Tools
-----
* :func:`night_split_crossings` — given tale → nights and night → split, list the
  tales whose nights fall in more than one split.
* :func:`records_night_split_audit` — the same, from ANHC ``NarrativeRecord``s
  (``narrative.night_numbers``) for a proposed night-level split.
* :func:`story_cycle_groups` — group records by the root of their
  ``parent_story_id`` chain so that frame and embedded tales stay together.
* :func:`ngram_overlap` — contamination of test texts by training n-grams, with
  optional masking of formulae (regexes), reported per test document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .normalization import simple_tokenize
from .records import NarrativeRecord


def night_split_crossings(tale_nights: Mapping[str, Iterable[int]],
                          night_split: Mapping[int, str]) -> dict[str, dict[str, list[int]]]:
    """Tales whose nights are assigned to more than one split: ``{tale: {split: nights}}``.

    Nights absent from ``night_split`` are reported under the split ``"unassigned"``.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for tale, nights in tale_nights.items():
        by_split: dict[str, list[int]] = {}
        for n in sorted(set(nights)):
            by_split.setdefault(night_split.get(n, "unassigned"), []).append(n)
        if len(by_split) > 1:
            out[tale] = by_split
    return out


def records_night_split_audit(records: Sequence[NarrativeRecord], night_split: Mapping[int, str]) -> dict:
    """Audit a night-level split against tale segmentation from ingestion."""
    tale_nights = {(r.narrative.title or r.record_id): r.narrative.night_numbers for r in records}
    crossings = night_split_crossings(tale_nights, night_split)
    return {"tales": len(tale_nights), "tales_crossing_splits": len(crossings), "crossings": crossings,
            "verdict": "LEAKAGE: tales span several splits" if crossings else "no tale spans several splits"}


def story_cycle_groups(records: Sequence[NarrativeRecord]) -> dict[str, str]:
    """``record_id -> root story id`` following ``parent_story_id`` (cycle-safe)."""
    parent = {r.record_id: r.narrative.parent_story_id for r in records}
    roots: dict[str, str] = {}
    for rid in parent:
        seen = [rid]
        cur = rid
        while parent.get(cur) and parent[cur] in parent and parent[cur] not in seen:
            cur = parent[cur]
            seen.append(cur)
        if parent.get(cur) in seen:
            raise ValueError(f"parent_story_id cycle involving {seen}")
        roots[rid] = cur
    return roots


@dataclass(frozen=True)
class OverlapReport:
    n: int
    per_document: list[dict]
    corpus_containment: float      # share of all test n-grams that occur in training
    masked_patterns: int

    def to_dict(self) -> dict:
        return {"n": self.n, "corpus_containment": self.corpus_containment,
                "masked_patterns": self.masked_patterns, "per_document": self.per_document}


def _tokens(text: str, masks: Sequence[re.Pattern]) -> list[str]:
    for m in masks:
        text = m.sub(" ", text)
    return [t for _, _, t in simple_tokenize(text)]


def _grams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_overlap(train_texts: Iterable[str], test_texts: Mapping[str, str], n: int = 8,
                  mask_patterns: Iterable[str] = ()) -> OverlapReport:
    """Share of each test document's word n-grams that also occur in the training texts."""
    if n < 1:
        raise ValueError("n must be >= 1")
    masks = [re.compile(p) for p in mask_patterns]
    train: set[tuple[str, ...]] = set()
    for t in train_texts:
        train.update(_grams(_tokens(t, masks), n))
    per_doc, hit_total, gram_total = [], 0, 0
    for doc_id, text in test_texts.items():
        grams = _grams(_tokens(text, masks), n)
        hits = sum(g in train for g in grams)
        hit_total += hits
        gram_total += len(grams)
        per_doc.append({"id": doc_id, "ngrams": len(grams), "in_train": hits,
                        "containment": round(hits / len(grams), 6) if grams else None})
    return OverlapReport(n=n, per_document=per_doc,
                         corpus_containment=round(hit_total / gram_total, 6) if gram_total else 0.0,
                         masked_patterns=len(masks))
