"""Heuristic cultural-motif spans for the parallel stories (input to L_cultural, Eq. 4.18).

The dissertation trains CSAM's bias matrices with L_cultural, whose positive pairs are
"position pairs within the same cultural motif span (from ANHC annotations, §3.5.2)".
This corpus has no such annotations, so this script derives three motif types by rule:

* opening_formula: a traditional opening at the start of the story
  (كان يا ما كان، حُكي أن، يُحكى، حجيتك وما جيتك، بلغني ...), up to the first comma or period;
* wisdom_transmission: the last quoted speech «...» of the story, which in these
  stories states the moral;
* proverb: any proverb from the seed files that occurs verbatim.

Spans are character offsets into the *cleaned* text (diacritics and tatweel removed,
spaces squeezed), i.e. the exact string train_phase1.py tokenises. They are rule-based,
not human annotations, and are recorded as such.

Usage: PYTHONPATH=src python scripts/annotate_motif_spans.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_phase1 import clean  # noqa: E402  (same cleaning as training)

from ansg.generation.seed_prompt import REGIONS, load_seed  # noqa: E402

STORIES = ROOT / "data" / "stories" / "parallel" / "parallel_stories.jsonl"
OUT = ROOT / "data" / "stories" / "parallel" / "motif_spans.jsonl"
OPENINGS = ("كان يا ما كان", "حكي أن", "حكي ان", "يحكى أن", "يحكى ان", "يحكى في", "يروى", "حجيتك وما جيتك",
            "بلغني", "كان يا سيدي")
_QUOTE = re.compile(r"«([^»]{3,})»")


def opening_span(text: str) -> tuple[int, int] | None:
    for f in OPENINGS:
        if text.startswith(f):
            m = re.search(r"[،,.]", text[len(f):])
            end = len(f) + (m.start() if m and m.start() < 80 else 0)
            return 0, end
    return None


def proverbs() -> list[str]:
    out = []
    for r in REGIONS:
        for p in load_seed(r).get("proverbs", []):
            t = clean(p.get("text", "") if isinstance(p, dict) else str(p))
            if len(t.split()) >= 3:
                out.append(t)
    return out


def spans_for(text: str, provs: list[str]) -> list[dict]:
    spans = []
    op = opening_span(text)
    if op and op[1] > op[0]:
        spans.append({"type": "opening_formula", "start": op[0], "end": op[1]})
    quotes = list(_QUOTE.finditer(text))
    if quotes:
        q = quotes[-1]
        spans.append({"type": "wisdom_transmission", "start": q.start(1), "end": q.end(1)})
    for p in provs:
        i = text.find(p)
        if i >= 0:
            spans.append({"type": "proverb", "start": i, "end": i + len(p)})
    for s in spans:
        s["text"] = text[s["start"]:s["end"]]
    return spans


def main() -> None:
    provs = proverbs()
    rows = [json.loads(l) for l in STORIES.read_text(encoding="utf-8").splitlines() if l.strip()]
    out, kinds, empty = [], Counter(), 0
    for r in rows:
        for v in REGIONS:
            text = clean(r[v])
            spans = spans_for(text, provs)
            kinds.update(s["type"] for s in spans)
            empty += not spans
            out.append({"id": r["id"], "variety": v, "split": r.get("split") or "train",
                        "method": "rule-based (scripts/annotate_motif_spans.py)", "spans": spans})
    OUT.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in out), encoding="utf-8")
    print(f"{len(out)} texts; spans by type: {dict(kinds)}; texts without spans: {empty}")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
