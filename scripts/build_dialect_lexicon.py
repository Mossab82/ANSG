"""Rebuild data/cultural_seeds/lexicons/dialect_lexicon.json from the parallel TRAINING stories.

A word becomes a marker of a variety when, after normalisation,
  * it occurs in that variety's version of at least MIN_DF training stories, and
  * it occurs in no other variety's version and in no MSA version of any training story,
  * and it is not in seed_adherence.NOT_DISTINCTIVE or in the hand-curated EXCLUDE list below.

Test-split stories (``"split": "test"``) are never read when building. They are used
only afterwards, to report out-of-sample coverage and cross-variety false hits.

Usage: PYTHONPATH=src python scripts/build_dialect_lexicon.py [--min-df 3] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from ansg.evaluation.seed_adherence import NOT_DISTINCTIVE, tokens

ROOT = Path(__file__).resolve().parents[1]
STORIES = ROOT / "data" / "stories" / "parallel" / "parallel_stories.jsonl"
OUT = ROOT / "data" / "cultural_seeds" / "lexicons" / "dialect_lexicon.json"
VARIETIES = ("msa", "egyptian", "levantine", "gulf", "maghrebi")

# Hand-curated removals. Each reason says why the word is not a safe marker even though
# the synthetic training stories happen to use it in one variety only.
_MSA = "also ordinary MSA; a generated MSA story could use it"
_OTHER = "also used in another variety, absent from these stories only by chance"
_SPELL = "spelling-dependent (plural ending without alif, ة/ه, ت/ث), not dialect-specific"
_CONTENT = "story-specific content word, not a dialect feature"
_FORMULA = "opening formula; SAS already scores the opening separately"
EXCLUDE: dict[str, str] = {
    # egyptian
    "واهل": _CONTENT, "ناحيه": _MSA, "فهموا": _MSA, "وفهموا": _MSA, "جنبه": _OTHER,
    "بتلف": _CONTENT, "حصل": _MSA, "كفايه": _MSA, "ومش": "مش is pan-dialect (NOT_DISTINCTIVE)",
    "باله": _OTHER, "بكره": _SPELL,
    # levantine
    "لحاله": _OTHER, "بيعرف": _OTHER, "بيقدر": _OTHER, "بقدر": _OTHER, "بعرف": _OTHER,
    "قاله": _OTHER, "باتجاه": _MSA, "فات": _OTHER, "اخدت": _OTHER, "قالتله": _OTHER,
    "ع": "single letter",
    # gulf
    "ذاك": _MSA, "وبقت": _OTHER, "ثاني": _SPELL, "كذا": _MSA, "الدرب": _MSA, "درب": _MSA,
    # maghrebi
    "ولكن": _MSA, "حجيتك": _FORMULA, "جيتك": _FORMULA, "دار": _MSA, "كاملين": _OTHER,
    "حيث": _MSA, "نهار": _MSA, "كثر": _MSA, "ولي": _MSA, "الربان": _MSA, "المقدم": _MSA,
    "الحجره": _CONTENT, "ا": "single letter", "والا": _MSA, "الثوب": _MSA, "الحجار": _OTHER,
    "مجموعين": _OTHER, "وعرفو": _SPELL, "عرفو": _SPELL, "رجعو": _SPELL, "سمعو": _SPELL,
    "وتعلمو": _SPELL, "ووصلو": _SPELL, "شكرو": _SPELL, "صلحو": _SPELL, "خدمو": _SPELL,
    "جلسو": _SPELL, "تعاونو": _SPELL, "وتعاونو": _SPELL, "واتافقو": _SPELL,
}


def load(split: str) -> list[dict]:
    rows = [json.loads(l) for l in STORIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if (r.get("split") == "test") == (split == "test")]


def build(train: list[dict], min_df: int) -> tuple[dict[str, list[str]], dict[str, Counter]]:
    df = {v: Counter() for v in VARIETIES}
    for r in train:
        for v in VARIETIES:
            df[v].update(set(tokens(r[v])))
    lex = {}
    for v in VARIETIES[1:]:
        others = set().union(*(df[o] for o in VARIETIES if o != v))
        keep = [t for t, c in df[v].items()
                if c >= min_df and t not in others and t not in NOT_DISTINCTIVE and t not in EXCLUDE]
        lex[v] = sorted(keep, key=lambda t: (-df[v][t], t))
    return lex, df


def evaluate(lex: dict[str, list[str]], stories: list[dict]) -> dict:
    out = {}
    for v, words in lex.items():
        own = set(words)
        hits = [len(set(tokens(r[v])) & own) for r in stories]
        cross = sum(bool(set(tokens(r[o])) & own) for r in stories for o in VARIETIES if o != v)
        out[v] = {"stories_with_hit": sum(h > 0 for h in hits), "stories": len(stories),
                  "hits_per_story": hits, "other_variety_versions_hit": cross}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    train, test = load("train"), load("test")
    lex, _ = build(train, args.min_df)
    ev = evaluate(lex, test)
    print({v: len(w) for v, w in lex.items()})
    print(json.dumps(ev, ensure_ascii=False))
    if args.dry_run:
        return
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    changes = list(old.get("provenance", {}).get("changes", []))
    changes.append(f"{date.today().isoformat()}: rebuilt by scripts/build_dialect_lexicon.py from "
                   f"{len(train)} training stories (min_df={args.min_df}); the test split was not read.")
    doc = {
        "provenance": {
            "method": ("Words that occur in one variety's versions of at least min_df parallel TRAINING "
                       "stories and in no other variety's or MSA versions, minus NOT_DISTINCTIVE and a "
                       "hand-curated exclusion list with reasons (scripts/build_dialect_lexicon.py)."),
            "min_df": args.min_df,
            "training_stories": len(train),
            "test_split_read": False,
            "curated_by": "Claude (assistant); not a native speaker",
            "review_status": "pending_native_review",
            "out_of_sample_check_on_test_split": ev,
            "note": ("Built from synthetic (ChatGPT) stories, so it reflects that generator's dialect "
                     "choices. Coverage on the 9 test stories is not a validation of the markers on "
                     "real dialect text."),
            "changes": changes,
        },
        **lex,
    }
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
