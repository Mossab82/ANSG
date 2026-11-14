#!/usr/bin/env python3
"""Audit adjudicated DIA/MOTIF/MORAL token coverage against corpus text."""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sentencepiece as spm

LAYERS = ("DIA", "MOTIF", "MORAL")
COVERING_DECISIONS = {"label", "reviewed_none"}


def read_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if not out or start > out[-1][1]:
            out.append([start, end])
        elif end > out[-1][1]:
            out[-1][1] = end
    return out


def intersect_intervals(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            out.append((start, end))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", nargs="+", required=True, help="JSONL/JSONL.GZ files or directories")
    ap.add_argument("--annotations-dir", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="SentencePiece model used for token_start/token_end")
    ap.add_argument("--tokenizer-profile", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    story_paths = set()
    for raw in args.stories:
        path = Path(raw)
        if path.is_dir():
            story_paths.update(path.glob("*.jsonl"))
            story_paths.update(path.glob("*.jsonl.gz"))
        elif path.is_file():
            story_paths.add(path)
        else:
            raise SystemExit(f"Stories path not found: {path}")
    story_paths = sorted(story_paths)
    if not story_paths:
        raise SystemExit("No JSONL story files found")

    annotations = {layer: defaultdict(list) for layer in LAYERS}
    ann_rows = Counter()
    ann_adjudicated = Counter()
    malformed = Counter()
    for layer in LAYERS:
        path = args.annotations_dir / f"{layer}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            ann_rows[layer] += 1
            if row.get("layer") != layer or not row.get("story_id"):
                malformed[layer] += 1
                continue
            prov = row.get("provenance", {})
            span = row.get("span", {})
            decision = row.get("decision")
            if prov.get("adjudication_status") != "adjudicated" or decision not in COVERING_DECISIONS:
                continue
            if span.get("tokenizer_profile") != args.tokenizer_profile:
                malformed[layer] += 1
                continue
            start, end = span.get("token_start"), span.get("token_end")
            cstart, cend = span.get("char_start"), span.get("char_end")
            if not all(isinstance(v, int) for v in (start, end, cstart, cend)) or start < 0 or end <= start or cstart < 0 or cend <= cstart:
                malformed[layer] += 1
                continue
            annotations[layer][row["story_id"]].append((start, end, cstart, cend))
            ann_adjudicated[layer] += 1

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    token_totals = Counter()
    covered_totals = Counter()
    triple_total = 0
    stories_total = 0
    stories_by_split = Counter()
    triple_by_split = Counter()
    seen_ids = set()
    orphan_counts = Counter()
    out_of_range = Counter()

    for path in story_paths:
        for row in read_jsonl(path):
            story_id = row.get("story_id") or row.get("record_id")
            text = row.get("text")
            if not story_id or not isinstance(text, str):
                raise ValueError(f"{path}: each story needs story_id/record_id and text")
            if story_id in seen_ids:
                raise ValueError(f"Duplicate story id: {story_id}")
            seen_ids.add(story_id)
            stories_total += 1
            split = row.get("split", "unassigned")
            stories_by_split[split] += 1
            token_count = len(sp.encode(text, out_type=int))
            token_totals["all"] += token_count
            spans = {}
            for layer in LAYERS:
                raw_spans = annotations[layer].get(story_id, [])
                valid = []
                for start, end, cstart, cend in raw_spans:
                    if end > token_count:
                        out_of_range[layer] += 1
                    elif cend > len(text):
                        out_of_range[layer] += 1
                    else:
                        valid.append((start, end))
                spans[layer] = merge(valid)
                covered_totals[layer] += sum(end - start for start, end in spans[layer])
            triple_intervals = intersect_intervals(spans["DIA"], spans["MOTIF"])
            triple_intervals = intersect_intervals(triple_intervals, spans["MORAL"])
            triple = sum(end - start for start, end in triple_intervals)
            triple_total += triple
            triple_by_split[split] += triple
            if stories_total % 50 == 0:
                print(f"Audited {stories_total} stories", flush=True)

    for layer in LAYERS:
        orphan_counts[layer] = len(set(annotations[layer]) - seen_ids)

    report = {
        "schema": "ansg.annotation_coverage_report/1",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer_profile": args.tokenizer_profile,
        "tokenizer_model": str(args.tokenizer),
        "stories": stories_total,
        "stories_by_split": dict(stories_by_split),
        "corpus_tokens": token_totals["all"],
        "annotation_rows": dict(ann_rows),
        "adjudicated_covering_rows": dict(ann_adjudicated),
        "invalid_or_profile_mismatch_rows": dict(malformed),
        "out_of_range_spans": dict(out_of_range),
        "orphan_story_ids": dict(orphan_counts),
        "adjudicated_covered_tokens_by_layer_union": dict(covered_totals),
        "triple_covered_tokens": triple_total,
        "triple_covered_tokens_by_split": dict(triple_by_split),
        "triple_coverage_fraction": triple_total / token_totals["all"] if token_totals["all"] else 0,
        "claim_target_tokens": 23000000,
        "reproduces_thesis_annotation_claim": False,
        "coverage_rule": "union of token intervals per layer; only adjudicated label/reviewed_none spans count; triple coverage is interval intersection across DIA, MOTIF, MORAL"
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
