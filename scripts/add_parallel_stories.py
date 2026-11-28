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


def split_records(line: str) -> list[dict]:
    """Parse one JSONL line; if the generator merged several stories into one object
    (the key "value" repeats), split it back into one record per story."""
    pairs = json.loads(line, object_pairs_hook=lambda kv: kv)
    if isinstance(pairs, dict):
        return [pairs]
    groups, cur = [], []
    for key, val in pairs:
        if key == "value" and cur:
            groups.append(dict(cur))
            cur = []
        cur.append((key, val))
    if cur:
        groups.append(dict(cur))
    if len(groups) > 1:
        print(f"NOTE: one line held {len(groups)} merged stories; split them")
    return groups


def profile(text: str, region: str, markers) -> tuple[int, int]:
    toks = tokens(text)
    others = frozenset().union(*(v for k, v in markers.items() if k != region))
    return sum(t in markers[region] for t in toks), sum(t in others for t in toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--generated-by", default="ChatGPT")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="test = held out from training and from dialect-lexicon building")
    args = ap.parse_args()
    existing = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()] if OUT.exists() else []
    n = len(existing)
    markers = distinctive_markers()
    added = []
    records = []
    for line in Path(args.batch).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.extend(split_records(line))
    for rec in records:
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
        rec = {"id": f"parallel-{n:04d}", "value": rec["value"], "split": args.split, "synthetic": True,
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
