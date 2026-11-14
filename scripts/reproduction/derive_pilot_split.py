#!/usr/bin/env python3
"""Derive the heritage pilot's train/validation/test split deterministically from evidence.

Pilot split v1 (PR #25) put kalila_dimna-019 in TEST although 98.5% of its 8-grams occur in
TRAIN page kalila_dimna-007 (evidence/corpus/classical_corpus_audit.json), and excluded the
Sinbad page for an overlap the measurement does not support (0.49% 8-gram containment).

Split v2 rules (no result of any run was looked at):
1. All Alf Layla pages are TRAIN: the frame narrative spans them and tales cross page boundaries.
2. Kalila pages are grouped; groups are merged on textual evidence only:
   - near-duplicate text: 8-gram containment > 50% (019 inside 007);
   - the same story continued: both pages name the chapter's protagonists at least 5 times
     ('دمنة', 'شتربة'; counts recorded below).
3. Groups are assigned with ansg.data.splits.assign_splits (hash_exact, 70/15/15, seed 42 —
   the thesis seed, §5.2.1).

``--check`` verifies that scripts/validate_classical_corpus.py and the Kaggle pilot carry exactly
this split.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
CORPUS = ROOT / "data" / "corpora" / "classical_arabic" / "stories.jsonl"
STORY_NAMES = ("دمنة", "شتربة")
MIN_NAME_COUNT = 5
DUPLICATE_THRESHOLD = 0.5


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def derive() -> dict:
    from ansg.data.leakage import ngram_overlap
    from ansg.data.records import NarrativeRecord, SourceProvenance
    from ansg.data.splits import SplitConfig, assign_splits
    rows = [json.loads(l) for l in CORPUS.read_text("utf-8").splitlines() if l.strip()]
    nights = sorted(r["id"] for r in rows if r["corpus"] == "arabian_nights")
    kal = {r["id"]: r for r in rows if r["corpus"] == "kalila_dimna"}
    parent = {k: k for k in kal}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    evidence = []
    ids = sorted(kal)
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            c_ab = ngram_overlap([kal[b]["text"]], {"x": kal[a]["text"]}, n=8).per_document[0]["containment"] or 0
            c_ba = ngram_overlap([kal[a]["text"]], {"x": kal[b]["text"]}, n=8).per_document[0]["containment"] or 0
            names = {n: [len(re.findall(n, kal[a]["text"])), len(re.findall(n, kal[b]["text"]))] for n in STORY_NAMES}
            shared = [n for n, (x, y) in names.items() if x >= MIN_NAME_COUNT and y >= MIN_NAME_COUNT]
            if max(c_ab, c_ba) > DUPLICATE_THRESHOLD or shared:
                parent[find(b)] = find(a)
                evidence.append({"pages": [a, b], "containment_8gram": [round(c_ab, 6), round(c_ba, 6)],
                                 "shared_story_names": {n: names[n] for n in shared}})
    recs = [NarrativeRecord(record_id=k, text=kal[k]["text"], source=SourceProvenance("w", "w")) for k in ids]
    out = assign_splits(recs, SplitConfig(strategy="hash_exact", ratios={"train": 0.7, "validation": 0.15,
                                                                         "test": 0.15}, seed=42),
                        group_fn=lambda r: find(r.record_id))
    split = {s: sorted(k for k, v in out.items() if v == s) for s in ("train", "validation", "test")}
    split["train"] = sorted(nights + split["train"])
    words = {s: sum(len(r["text"].split()) for r in rows if r["id"] in ids_) for s, ids_ in split.items()}
    return {"version": 2, "split": split, "excluded": [], "words": words, "group_merges": evidence,
            "rules": __doc__.split("Split v2 rules")[1].split("``--check``")[0].strip()}


def check(d: dict) -> list[str]:
    problems = []
    v = _load(ROOT / "scripts" / "validate_classical_corpus.py", "vcc")
    p = _load(ROOT / "kaggle" / "heritage" / "ansg_heritage_pilot.py", "pilot")
    for name, (a, b) in {"validator": (v.TRAIN, v.VALIDATION), "pilot": (p.TRAIN_IDS, p.VALIDATION_IDS)}.items():
        if set(a) != set(d["split"]["train"]) or set(b) != set(d["split"]["validation"]):
            problems.append(f"{name} train/validation differ from the derived split")
    if set(v.TEST) != set(d["split"]["test"]) or set(p.TEST_IDS) != set(d["split"]["test"]):
        problems.append("test split differs from the derived split")
    if set(v.EXCLUDED) or set(p.EXCLUDED_IDS):
        problems.append("split v2 excludes no page")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    d = derive()
    print(json.dumps(d, ensure_ascii=False, indent=2))
    if args.check:
        problems = check(d)
        for x in problems:
            print("SPLIT MISMATCH:", x, file=sys.stderr)
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
