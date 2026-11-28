"""Seed Adherence Score (SAS): did a story use the cultural elements it was asked to use?

SAS checks a generated story against the elements that
:func:`ansg.generation.seed_prompt.build_story_prompt` sampled for it: the
proverb, the traditional opening and closing, the character and setting, and
the dialect of the dialogue. It is a transparent, lexical **instruction
adherence** measure. It is not a measure of cultural authenticity or story
quality, it is not the dissertation's CMPI, and it has not yet been validated
against human judgments. The validation plan is in
``docs/SEED_ADHERENCE_METRIC.md``.

Equations (all components lie in [0, 1]):

    r(P, W)  = LCS_tokens(P, W) / |P|                       phrase recall
    p        = r(proverb, story)
    o        = r(opening, first |opening| + 10 tokens)
    c        = r(closing, last  |closing| + 10 tokens)
    e        = mean( r(character, story), r(setting, story) )
    d        = h_t / (h_t + h_o)          dialect regions, 0 if h_t + h_o = 0
    d        = 1 / (1 + h_o)              MSA
    SAS      = 0.2 p + 0.2 o + 0.2 c + 0.2 e + 0.2 d

``LCS_tokens`` is the longest common token subsequence after Arabic
normalisation. ``h_t`` counts story tokens that are distinctive markers of the
target region; ``h_o`` counts distinctive markers of the other regions.
Distinctive markers are the tokens of a region's ``dialect_markers`` (plus the
curated ``lexicons/dialect_lexicon.json``, when present) that do not
occur in any MSA phrase of the seed files or in another region's markers,
minus a short declared list of standard or pan-dialect words (``NOT_DISTINCTIVE``).
The equal weights are a declared default, not an empirical result; the
components are always reported separately.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import json

from ..generation.seed_prompt import REGIONS, SEED_DIR, load_seed

WEIGHTS = {"proverb": 0.2, "opening": 0.2, "closing": 0.2, "entities": 0.2, "dialect": 0.2}
EDGE_SLACK = 10

# Tokens that survive the set arithmetic but are standard Arabic or common to
# several dialects, so they would produce false dialect hits. Normalised forms.
NOT_DISTINCTIVE = frozenset({
    "لو", "عارف", "سمحت", "تمام", "كمان", "الحمد", "لله", "مين", "تعمل", "هلا",
    "حلو", "مره", "مرحبا", "شي", "الشي", "حاجه", "ساكن", "ليا", "سمح", "واقع",
    "بعدين", "نروح", "اروح", "بالطريق",
    # found ambiguous on the parallel stories: ليه = "why" (EGY) but "to him" (MAG);
    # مش is also Levantine; نمشي is also Gulf
    "ليه", "مش", "نمشي",
    # شوي is Gulf as well as Levantine
    "شوي",
    # عم: Levantine progressive particle, but also "paternal uncle" in every variety (ولاد عم)
    "عم",
    # found on Phase 1 samples: ابي = "my father" (MSA) vs "I want" (Gulf); عندو is Levantine too
    "ابي", "عندو",
})

_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_LETTERS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text).translate(_LETTERS)
    return " ".join(_NON_WORD.sub(" ", text).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    # tolerate a leading conjunction (و / ف) on either side
    strip = lambda t: t[1:] if len(t) > 2 and t[0] in "وف" else t  # noqa: E731
    return strip(a) == strip(b)


def lcs_recall(phrase: str, window: list[str]) -> float:
    """Share of the phrase's tokens found in order in ``window``."""
    ptoks = tokens(phrase)
    if not ptoks or not window:
        return 0.0
    prev = [0] * (len(window) + 1)
    for p in ptoks:
        cur = [0] * (len(window) + 1)
        for j, w in enumerate(window, 1):
            cur[j] = prev[j - 1] + 1 if _same(p, w) else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1] / len(ptoks)


@lru_cache(maxsize=4)
def distinctive_markers(seed_dir: Path | None = None) -> dict[str, frozenset[str]]:
    seeds = {r: load_seed(r, seed_dir) for r in REGIONS}
    msa_vocab: set[str] = set()
    for seed in seeds.values():
        for m in seed["dialect_markers"]:
            msa_vocab.update(tokens(m["msa"]))
    msa_vocab.update(t for m in seeds["msa"]["dialect_markers"] for t in tokens(m["dialect"]))
    raw = {
        r: {t for m in seeds[r]["dialect_markers"] for t in tokens(m["dialect"])} - msa_vocab
        for r in REGIONS if r != "msa"
    }
    lexicon_path = (seed_dir or SEED_DIR) / "lexicons" / "dialect_lexicon.json"
    if lexicon_path.exists():
        lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
        for r in raw:
            raw[r] |= {t for w in lexicon.get(r, []) for t in tokens(w)} - msa_vocab
    out = {}
    for r, toks in raw.items():
        others = set().union(*(v for k, v in raw.items() if k != r))
        out[r] = frozenset(toks - others - NOT_DISTINCTIVE)
    out["msa"] = frozenset()
    return out


@dataclass
class SeedAdherence:
    proverb: float | None
    opening: float | None
    closing: float | None
    entities: float
    dialect: float
    target_marker_hits: int
    other_marker_hits: int
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def seed_adherence(story: str, region: str, elements: dict, seed_dir: Path | None = None) -> SeedAdherence:
    """Score ``story`` against the ``elements`` of its :class:`StoryPrompt`.

    A component whose element was not requested (``None``) is reported as
    ``None`` and its weight is redistributed over the remaining components.
    """
    toks = tokens(story)
    proverb = lcs_recall(elements["proverb"], toks) if elements.get("proverb") else None
    opening = closing = None
    if elements.get("opening"):
        opening = lcs_recall(elements["opening"], toks[: len(tokens(elements["opening"])) + EDGE_SLACK])
    if elements.get("closing"):
        closing = lcs_recall(elements["closing"], toks[-(len(tokens(elements["closing"])) + EDGE_SLACK):])
    entities = (lcs_recall(elements["character"], toks) + lcs_recall(elements["setting"], toks)) / 2

    markers = distinctive_markers(seed_dir)
    target = markers[region]
    others = frozenset().union(*(v for k, v in markers.items() if k != region))
    h_t = sum(1 for t in toks if t in target)
    h_o = sum(1 for t in toks if t in others)
    if region == "msa":
        dialect = 1.0 / (1 + h_o)
    else:
        dialect = h_t / (h_t + h_o) if (h_t + h_o) else 0.0

    parts = {"proverb": proverb, "opening": opening, "closing": closing, "entities": entities, "dialect": dialect}
    used = {k: v for k, v in parts.items() if v is not None}
    total_w = sum(WEIGHTS[k] for k in used)
    score = sum(WEIGHTS[k] * v for k, v in used.items()) / total_w
    return SeedAdherence(proverb, opening, closing, entities, dialect, h_t, h_o, round(score, 4))
