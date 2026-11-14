"""Automatic PROXY indicators for generated stories. Torch-free.

None of these is a CRAFT dimension, a dissertation metric or a substitute for
human judgement. Each result carries ``metric_status = "proxy"`` and says what it
does NOT measure (docs/ar/ANSG_SCIENTIFIC_REVIEW_2026-11.md §6).

* ``copy_overlap`` — originality guard: share of the story's word n-grams found in
  a source corpus, and the longest run of words copied verbatim.
* ``distinct_n`` / ``repetition`` — lexical diversity across / within texts.
* ``arabic_letter_ratio`` — script sanity.
* ``plan_surface_coverage`` — whether planned character names and event triggers
  appear on the surface (NOT whether the plot follows the plan).
* ``marker_rate`` — occurrences of listed dialect markers per 100 words (presence
  proxy only; NOT dialect authenticity).
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..data.leakage import ngram_overlap
from ..data.normalization import simple_tokenize

PROXY = "proxy"


def _words(text: str) -> list[str]:
    return [t for _, _, t in simple_tokenize(text)]


def longest_common_run(text: str, sources: Iterable[str]) -> int:
    """Length (in words) of the longest word sequence of ``text`` that appears verbatim in a source."""
    words = _words(text)
    src = [_words(s) for s in sources]
    index: dict[str, list[tuple[int, int]]] = {}
    for d, toks in enumerate(src):
        for i, w in enumerate(toks):
            index.setdefault(w, []).append((d, i))
    best = 0
    for i in range(len(words)):
        for d, j in index.get(words[i], ()):
            k = 0
            toks = src[d]
            while i + k < len(words) and j + k < len(toks) and words[i + k] == toks[j + k]:
                k += 1
            best = max(best, k)
    return best


def copy_overlap(text: str, sources: Iterable[str], n: int = 8, mask_patterns: Iterable[str] = ()) -> dict:
    sources = list(sources)
    rep = ngram_overlap(sources, {"generated": text}, n=n, mask_patterns=mask_patterns)
    return {"metric_status": PROXY, "n": n, "containment": rep.per_document[0]["containment"],
            "longest_copied_run_words": longest_common_run(text, sources),
            "does_not_measure": "paraphrase or plot-level borrowing"}


def distinct_n(texts: Iterable[str], n: int = 2) -> dict:
    grams, total = set(), 0
    for t in texts:
        w = _words(t)
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        grams.update(g)
        total += len(g)
    return {"metric_status": PROXY, "n": n, "distinct": round(len(grams) / total, 6) if total else None,
            "does_not_measure": "semantic or plot diversity"}


def repetition(text: str, n: int = 3) -> dict:
    w = _words(text)
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return {"metric_status": PROXY, "n": n, "repeated_fraction": round(1 - len(set(g)) / len(g), 6) if g else None}


def arabic_letter_ratio(text: str) -> dict:
    letters = [c for c in text if c.isalpha()]
    arabic = [c for c in letters if "؀" <= c <= "ۿ" or "ݐ" <= c <= "ݿ"]
    return {"metric_status": PROXY, "ratio": round(len(arabic) / len(letters), 6) if letters else None}


def plan_surface_coverage(text: str, plan_graph) -> dict:
    from ..narrative.validation import fold, mention_matches
    words = _words(text)
    folded = fold(text)
    chars = [e for e in plan_graph.entities if e.kind.animate]
    named = [e.name for e in chars if any(mention_matches(" ".join(words[i:i + len(e.name.split())]), e.surface_forms())
                                          for i in range(len(words)))]
    triggers = [ev.trigger for ev in plan_graph.events if fold(ev.trigger) in folded]
    return {"metric_status": PROXY,
            "characters_named": f"{len(named)}/{len(chars)}",
            "event_triggers_on_surface": f"{len(triggers)}/{len(plan_graph.events)}",
            "does_not_measure": "event order, causality or whether the plot follows the plan"}


def marker_rate(text: str, markers: Iterable[str]) -> dict:
    from ..narrative.validation import fold
    words = [fold(w) for w in _words(text)]
    wanted = {fold(m) for m in markers}
    hits = sum(w in wanted or (len(w) > 3 and w[0] in "وف" and w[1:] in wanted) for w in words)
    return {"metric_status": PROXY, "per_100_words": round(100 * hits / len(words), 3) if words else None,
            "does_not_measure": "dialect authenticity, naturalness or correct variety"}


def summarise(text: str, *, sources: Iterable[str] = (), plan_graph=None,
              markers: Iterable[str] = (), mask_patterns: Iterable[str] = ()) -> Mapping[str, dict]:
    out = {"arabic_letter_ratio": arabic_letter_ratio(text), "repetition_3": repetition(text, 3)}
    sources = list(sources)
    if sources:
        out["copy_overlap_8"] = copy_overlap(text, sources, 8, mask_patterns)
    if plan_graph is not None:
        out["plan_surface_coverage"] = plan_surface_coverage(text, plan_graph)
    markers = list(markers)
    if markers:
        out["dialect_marker_rate"] = marker_rate(text, markers)
    return out
