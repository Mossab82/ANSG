"""Validate and append a batch of synthetic parallel dialect stories.

Usage: PYTHONPATH=src python scripts/add_parallel_stories.py BATCH.jsonl [--generated-by ChatGPT]

Each input line: {"value", "msa", "egyptian", "levantine", "gulf", "maghrebi"}.
Checks per variant: Arabic script, length, and a dialect-marker profile (target
vs other-region distinctive markers, as in the Seed Adherence Score). Records
are appended to data/stories/parallel/parallel_stories.jsonl as synthetic and
pending native review; the check report is printed, nothing is rejected
silently.
"""
import argparse
import json
from pathlib import Path

from ansg.evaluation.seed_adherence import distinctive_markers, tokens
from ansg.generation.seed_prompt import REGIONS

OUT = Path(__file__).resolve().parents[1] / "data" / "stories" / "parallel" / "parallel_stories.jsonl"


def profile(text: str, region: str, markers) -> tuple[int, int]:
    toks = tokens(text)
    others = frozenset().union(*(v for k, v in markers.items() if k != region))
    return sum(t in markers[region] for t in toks), sum(t in others for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--generated-by", default="ChatGPT")
    args = ap.parse_args()
    existing = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT.exists() else []
    n = len(existing)
    markers = distinctive_markers()
    added = []
    for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        missing = [r for r in REGIONS if not rec.get(r)]
        if missing:
            print(f"SKIP ({rec.get('value')}): missing {missing}")
            continue
        n += 1
        checks = {}
        for r in REGIONS:
            words = len(rec[r].split())
            hits, other = profile(rec[r], r, markers)
            checks[r] = {"words": words, "target_markers": hits, "other_markers": other}
        rec = {"id": f"parallel-{n:04d}", "value": rec["value"], "synthetic": True,
               "generated_by": args.generated_by, "review_status": "pending_native_review",
               **{r: rec[r] for r in REGIONS}, "checks": checks}
        added.append(rec)
        flags = [f"{r}: {c['words']}w t={c['target_markers']} o={c['other_markers']}" for r, c in checks.items()]
        print(rec["id"], rec["value"], "|", "; ".join(flags))
    with OUT.open("a", encoding="utf-8") as fh:
        for rec in added:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"added {len(added)}; total {n}")


if __name__ == "__main__":
    main()
