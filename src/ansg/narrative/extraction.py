"""Candidate generation for event annotation — an INSPECTION BASELINE, not an extractor.

What this does
--------------
* :func:`verb_trigger_candidates` marks tokens whose form (after stripping the
  conjunction proclitics و/ف and diacritics) equals a member verb form listed in a
  supplied ArabicVerbNet index. Inflected forms (قال/قالت/يقول) match only if the
  resource lists them; recall is therefore low and unknown. No morphological
  analyser is bundled and none is assumed.
* :func:`causal_cue_candidates` marks occurrences of common Arabic causal / temporal
  connectives. These are *cues for annotators*, not causal relations: "فلما" is
  frequently purely temporal, and "حتى" can mean "until", "so that" or "even".

Every candidate is returned with ``status = automatic`` and an evidence span, and
must be confirmed, completed (participants, preconditions, effects, knowledge)
and adjudicated by human annotators before it can count as silver or gold. No
event, participant or relation is ever produced without its text span.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.normalization import simple_tokenize
from ..data.records import AnnotationSource, AnnotationStatus
from .events import Event, Evidence, Modality
from .validation import fold

# Connectives grouped by their *usual* reading; the grouping is a heuristic for
# annotators, not a claim about each occurrence.
CAUSAL_CUES: dict[str, tuple[str, ...]] = {
    "cause_explicit": ("لأن", "لان", "بسبب", "لذلك", "فلذلك", "لهذا", "من أجل", "كي", "لكي"),
    "consequence_or_temporal": ("فلما", "لما", "حتى", "فإذا", "إذ", "عندئذ", "حينئذ"),
}


@dataclass(frozen=True)
class CueCandidate:
    cue: str
    category: str
    evidence: Evidence


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    stops = ".!?؟\n"
    s = max((text.rfind(c, 0, start) for c in stops), default=-1) + 1
    ends = [i for i in (text.find(c, end) for c in stops) if i != -1]
    e = min(ends) + 1 if ends else len(text)
    while s < e and text[s].isspace():
        s += 1
    return s, e


def verb_trigger_candidates(text: str, verbnet: Any, record_prefix: str = "cand") -> list[Event]:
    """Candidate events for every token matching a listed VerbNet member form."""
    forms: dict[str, list[str]] = {}
    for cls, member in verbnet.members():
        if member.verb:
            forms.setdefault(fold(member.verb), []).append(cls.class_id or cls.source_file)
    out: list[Event] = []
    for start, end, tok in simple_tokenize(text):
        f = fold(tok)
        candidates = [f]
        if len(f) > 3 and f[0] in "وف":
            candidates.append(f[1:])
        hit = next((c for c in candidates if c in forms), None)
        if hit is None:
            continue
        s, e = _sentence_bounds(text, start, end)
        classes = sorted(set(forms[hit]))
        out.append(Event(
            id=f"{record_prefix}-{len(out) + 1:04d}", trigger=tok, evidence=Evidence(s, e, text[s:e]),
            discourse_index=len(out), modality=Modality.ACTUAL,
            verbnet_class=classes[0] if len(classes) == 1 else None,
            status=AnnotationStatus.AUTOMATIC, source=AnnotationSource.AUTOMATIC,
        ))
    return out


def causal_cue_candidates(text: str) -> list[CueCandidate]:
    """Occurrences of connectives that often signal causal or temporal links."""
    folded_tokens = [(s, e, fold(t)) for s, e, t in simple_tokenize(text)]
    out: list[CueCandidate] = []
    for category, cues in CAUSAL_CUES.items():
        for cue in cues:
            parts = [fold(p) for p in cue.split()]
            n = len(parts)
            for i in range(len(folded_tokens) - n + 1):
                window = [folded_tokens[i + j][2] for j in range(n)]
                first = window[0]
                # allow a conjunction proclitic on the first word (و + لما, ف + لذلك)
                if window[1:] == parts[1:] and (first == parts[0] or
                                                (len(first) > len(parts[0]) and first[0] in "وف"
                                                 and first[1:] == parts[0])):
                    s, e = folded_tokens[i][0], folded_tokens[i + n - 1][1]
                    out.append(CueCandidate(cue, category, Evidence(s, e, text[s:e])))
    out.sort(key=lambda c: c.evidence.start)
    # a longer cue (فلما) and its suffix (لما) can both match the same token: keep the longest
    dedup: list[CueCandidate] = []
    for c in out:
        if dedup and dedup[-1].evidence.start == c.evidence.start:
            if len(c.cue) > len(dedup[-1].cue):
                dedup[-1] = c
            continue
        dedup.append(c)
    return dedup
