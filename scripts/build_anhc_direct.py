#!/usr/bin/env python3
"""Build a reproducible ANHC direct-build inventory, grouped splits and reports.

This command creates a new, explicitly direct-build release profile. It never
claims historical thesis counts or manufactures human annotations.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sentencepiece as spm

LAYERS = ("DIA", "MOTIF", "MORAL")
SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(value: str | None) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = "".join(ch for ch in value if not ("\u064b" <= ch <= "\u065f" or ch == "\u0670" or ch == "\u0640"))
    value = value.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ٱ":"ا"}))
    return " ".join("".join(ch if ch.isalnum() else " " for ch in value).split())


def work_lock(record: dict) -> str:
    src = record.get("source", {})
    titles = " ".join(str(src.get(k) or "") for k in ("title", "original_title"))
    folded = titles.casefold()
    if "ألف ليلة" in titles or "alf layla" in folded or "one thousand and one nights" in folded:
        return "alf_layla_work_cycle"
    if "كليلة ودمنة" in titles or "kalila" in folded or "bidpai" in folded:
        return "kalila_1816_frame_cycle"
    author = normalize(src.get("original_author") or src.get("author"))
    title = normalize(src.get("original_title") or src.get("title"))
    identity = "\x1f".join((author, title)) if author and title else record["record_id"]
    return "work_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def select_split(group_id: str, seed: str, train: float, validation: float) -> str:
    point = int(hashlib.sha256(f"{seed}\x1f{group_id}".encode()).hexdigest()[:16], 16) / 0xFFFFFFFFFFFFFFFF
    if point < train:
        return "train"
    if point < train + validation:
        return "validation"
    return "test"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-manifest", type=Path, required=True, help="Source-linked narrative-candidate manifest")
    ap.add_argument("--records-root", type=Path, required=True, help="Directory containing the manifest's compressed text_file paths")
    ap.add_argument("--story-units", type=Path, required=True, help="Provisional Alf Layla/Kalila units JSONL")
    ap.add_argument("--tokenizer-model", type=Path, required=True, help="Pinned direct-build SentencePiece model")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", default="ANHC-direct-v1")
    ap.add_argument("--train-ratio", type=float, default=0.80)
    ap.add_argument("--validation-ratio", type=float, default=0.10)
    args = ap.parse_args()
    if not (0 < args.train_ratio < 1 and 0 <= args.validation_ratio < 1 and args.train_ratio + args.validation_ratio < 1):
        raise SystemExit("Require train_ratio > 0, validation_ratio >= 0, and train_ratio + validation_ratio < 1")
    for path in (args.candidate_manifest, args.story_units, args.tokenizer_model):
        if not path.is_file():
            raise SystemExit(f"Required input not found: {path}")
    if not args.records_root.is_dir():
        raise SystemExit(f"Records directory not found: {args.records_root}")

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer_model))
    source_rows = list(read_jsonl(args.candidate_manifest))
    if not source_rows:
        raise SystemExit("Candidate manifest is empty")
    records, token_words_by_id = {}, {}
    source_metadata = {}
    count_by_scope = Counter()
    words_by_scope = Counter()
    tokens_by_scope = Counter()
    for index, row in enumerate(source_rows, 1):
        rid = row.get("record_id")
        if not rid or rid in records:
            raise ValueError(f"Missing or duplicate record_id in candidate manifest: {rid}")
        rel = row.get("text_file")
        if not rel:
            raise ValueError(f"{rid}: missing text_file")
        path = args.records_root / rel
        materialized = next(read_jsonl(path), None)
        if not materialized or materialized.get("record_id") != rid:
            raise ValueError(f"{rid}: materialized record id mismatch at {path}")
        text = materialized.get("text")
        if not isinstance(text, str):
            raise ValueError(f"{rid}: materialized record has no text")
        digest = digest_text(text)
        expected = row.get("source_text_sha256")
        if digest != expected or digest != materialized.get("source_text_sha256"):
            raise ValueError(f"{rid}: source text SHA-256 mismatch")
        words = len(text.split())
        tokens = len(sp.encode(text, out_type=int))
        src = row.get("source", {})
        # Keep the best available title/author/category metadata and use the same
        # normalized work lock for duplicate editions/translations.
        full = {**row, "source": src}
        lock = work_lock(full)
        scope = row.get("narrative", {}).get("candidate_scope", "unspecified")
        source_metadata[rid] = row
        records[rid] = {
            "record_id": rid,
            "title": src.get("title"),
            "author": src.get("author"),
            "source_text_sha256": digest,
            "candidate_scope": scope,
            "observed_whitespace_words": words,
            "sentencepiece_tokens": tokens,
            "work_group": lock,
            "text_file": rel,
            "license": src.get("license"),
            "attribution": src.get("attribution"),
        }
        token_words_by_id[rid] = (words, tokens)
        count_by_scope[scope] += 1
        words_by_scope[scope] += words
        tokens_by_scope[scope] += tokens
        if index % 100 == 0:
            print(f"Verified and counted {index}/{len(source_rows)} source records", flush=True)

    group_to_split = {}
    group_words, group_tokens = Counter(), Counter()
    for row in records.values():
        group = row["work_group"]
        chosen = select_split(group, args.seed, args.train_ratio, args.validation_ratio)
        if group in group_to_split and group_to_split[group] != chosen:
            raise AssertionError(f"Work group split inconsistency: {group}")
        group_to_split[group] = chosen
        row["split"] = chosen
        row["split_assignment"] = "direct_build_grouped_hash_v1"
        group_words[chosen] += row["observed_whitespace_words"]
        group_tokens[chosen] += row["sentencepiece_tokens"]

    units = list(read_jsonl(args.story_units))
    known_units = {u.get("unit_id") for u in units}
    for rid, summary in records.items():
        unit_id = f"{rid}-whole-book"
        if unit_id in known_units:
            continue
        source_row = source_metadata[rid]
        src = source_row.get("source", {})
        units.append({
            "schema": "ansg.story_unit_candidate/1",
            "unit_id": unit_id,
            "source_record_id": rid,
            "source_cycle_id": summary["work_group"],
            "source_group_split_lock": summary["work_group"],
            "split_lock_basis": "normalized original/title-author work identity; review duplicates before publication",
            "unit_type": "whole_book_source_container",
            "title_candidate": src.get("title"),
            "author_candidate": src.get("author"),
            "source_title_original": src.get("original_title"),
            "source_author_original": src.get("original_author"),
            "candidate_scope": summary["candidate_scope"],
            "source_text_sha256": summary["source_text_sha256"],
            "observed_whitespace_words": summary["observed_whitespace_words"],
            "status": "provisional_unsegmented_source_container",
            "parent_story_id": None,
            "annotations": {layer: None for layer in LAYERS},
            "split": "unassigned",
            "review_required": ["review_narrative_relevance_and_exclusions", "identify_story_boundaries_or_retain_as_context_only", "deduplicate_translations_and_editions_before_split"]
        })
        known_units.add(unit_id)
    units.sort(key=lambda u: (u.get("source_record_id", ""), u.get("unit_type", ""), u.get("unit_id", "")))
    known_units = set()
    for unit in units:
        uid = unit.get("unit_id")
        if not uid or uid in known_units:
            raise ValueError(f"Missing or duplicate unit_id: {uid}")
        known_units.add(uid)
        unit.setdefault("annotations", {layer: None for layer in LAYERS})
        if set(unit["annotations"]) != set(LAYERS):
            raise ValueError(f"{uid}: annotation layers must be exactly {LAYERS}")
        if any(value is not None for value in unit["annotations"].values()):
            raise ValueError(f"{uid}: pre-existing labels are not accepted by this clean direct-build command")
        group = unit.get("source_group_split_lock") or unit.get("source_cycle_id") or unit.get("source_record_id")
        unit["source_group_split_lock"] = group
        unit["split"] = group_to_split.get(group, select_split(group, args.seed, args.train_ratio, args.validation_ratio))
        unit["split_assignment"] = "direct_build_grouped_hash_v1"
    units_by_split = Counter(u["split"] for u in units)

    # Create output atomically after all sources have passed hash validation.
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source_records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"))+"\n" for r in records.values()), encoding="utf-8")
    (args.output / "story_units.jsonl").write_text("".join(json.dumps(u, ensure_ascii=False, separators=(",", ":"))+"\n" for u in units), encoding="utf-8")
    split_rows = [{"record_id": rid, "work_group": row["work_group"], "split": row["split"], "source_text_sha256": row["source_text_sha256"]} for rid, row in records.items()]
    (args.output / "split_assignments.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":"))+"\n" for r in split_rows), encoding="utf-8")
    for split in SPLITS:
        split_ids = sorted(rid for rid, row in records.items() if group_to_split[row["work_group"]] == split)
        (args.output / f"{split}_record_ids.txt").write_text("".join(rid + "\n" for rid in split_ids), encoding="utf-8")
    ann_dir = args.output / "annotations"
    ann_dir.mkdir(exist_ok=True)
    for layer in LAYERS:
        (ann_dir / f"{layer}.jsonl").write_text("", encoding="utf-8")
    split_counts = {split: {"records": sum(1 for r in records.values() if group_to_split[r["work_group"]] == split),
                            "groups": sum(1 for g, val in group_to_split.items() if val == split),
                            "story_unit_rows": units_by_split[split],
                            "observed_whitespace_words": group_words[split],
                            "sentencepiece_tokens": group_tokens[split]} for split in SPLITS}
    report = {
        "schema": "ansg.direct_build_release_report/1",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "profile_id": "ANHC-direct-build-v1",
        "status": "reproducible_direct_build_not_historical_reproduction",
        "inputs": {"candidate_manifest": str(args.candidate_manifest), "candidate_manifest_sha256": hashlib.sha256(args.candidate_manifest.read_bytes()).hexdigest(),
                   "tokenizer_model": str(args.tokenizer_model), "tokenizer_model_sha256": hashlib.sha256(args.tokenizer_model.read_bytes()).hexdigest(),
                   "sentencepiece_version": spm.__version__, "vocabulary_size": sp.get_piece_size()},
        "candidate_sources": len(records),
        "candidate_scope_counts": dict(count_by_scope),
        "candidate_scope_whitespace_words": dict(words_by_scope),
        "candidate_scope_sentencepiece_tokens": dict(tokens_by_scope),
        "grouped_splits": {"algorithm": "SHA-256(seed, work_group) threshold assignment; group rows cannot cross splits",
                           "seed": args.seed, "ratios": {"train": args.train_ratio, "validation": args.validation_ratio, "test": round(1-args.train_ratio-args.validation_ratio, 6)},
                           "realized": split_counts},
        "story_unit_rows": len(units),
        "story_units_by_split": dict(units_by_split),
        "annotation_layers": {layer: {"rows": 0, "adjudicated": False, "status": "empty_template_only"} for layer in LAYERS},
        "triple_covered_tokens": 0,
        "thesis_comparison": {"reported_corpus_tokens": 70404829, "reported_triple_covered_tokens": 23000000,
                              "reproduces_historical_counts": False,
                              "reason": "Direct-build tokenizer, sources, boundaries, and original adjudicated annotation records differ or remain unavailable."},
        "limitations": ["Publisher categories are candidate filters; whole books are not verified story units.",
                        "The 80/10/10 ratios and hash seed are direct-build operational choices, not historical thesis split values.",
                        "DIA/MOTIF/MORAL outputs are empty templates; no labels or human adjudication are fabricated.",
                        "The script emits source-record ID lists and metadata; source text remains in its attributed input files."]
    }
    (args.output / "release_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ann_dir / "README.md").write_text("# Annotation templates\n\nDIA.jsonl, MOTIF.jsonl, and MORAL.jsonl are intentionally empty. The direct-build command creates schema-ready empty files; it does not infer or fabricate human labels. The 23M triple-covered-token thesis figure is not reproduced by this release.\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sources": len(records), "story_unit_rows": len(units), "splits": split_counts, "triple_covered_tokens": 0}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
