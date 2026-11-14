#!/usr/bin/env python3
"""Train and count a pinned, non-root-aware SentencePiece profile for direct intake."""
import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sentencepiece as spm

CORE = {"novels", "detective.fiction", "science.fiction", "children.stories", "plays", "literature"}
ADJACENT = {"history", "biographies", "travel.literature"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True, help="all_books records directory")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--training-sentences", type=int, default=200000)
    ap.add_argument("--sample-seed", type=int, default=20261114)
    ap.add_argument("--kalila-jsonl", type=Path)
    ap.add_argument("--reuse-existing-model", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    training_text = args.output / "training_text.txt"
    record_paths = sorted(args.records.glob("hindawi-*.jsonl.gz"))
    if not record_paths:
        raise SystemExit(f"No Hindawi records found at {args.records}")

    # Materialize source-only text, never JSON metadata or labels.
    with training_text.open("w", encoding="utf-8", newline="\n") as out:
        for path in record_paths:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                record = json.loads(source.readline())
            out.write(record["text"])
            out.write("\n")

    # SentencePiece's in-memory trainer can exceed RAM on the full 66M-word
    # intake. Build a deterministic reservoir sample of lines for vocabulary
    # training; full corpus counts below still cover every record.
    sample_path = args.output / "training_sample.txt"
    rng = random.Random(args.sample_seed)
    reservoir = []
    seen_lines = 0
    with training_text.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                continue
            seen_lines += 1
            if len(reservoir) < args.training_sentences:
                reservoir.append(line)
            else:
                index = rng.randrange(seen_lines)
                if index < args.training_sentences:
                    reservoir[index] = line
    with sample_path.open("w", encoding="utf-8", newline="\n") as out:
        for line in reservoir:
            out.write(line + "\n")
    training_text_sha256 = sha256(training_text)
    training_text.unlink()

    prefix = args.output / "anhc_direct_spm32"
    model_path = prefix.with_suffix(".model")
    if args.reuse_existing_model:
        if not model_path.is_file():
            raise SystemExit(f"Cannot reuse missing model: {model_path}")
    else:
        spm.SentencePieceTrainer.train(
            input=str(sample_path), model_prefix=str(prefix), model_type="unigram",
            vocab_size=args.vocab_size, character_coverage=0.9995,
            normalization_rule_name="nfkc", add_dummy_prefix=True,
            remove_extra_whitespaces=False, input_sentence_size=0,
            shuffle_input_sentence=False, num_threads=1,
            bos_id=1, eos_id=2, unk_id=0, pad_id=-1,
            hard_vocab_limit=False, max_sentence_length=16384,
        )
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    by_scope_tokens = Counter()
    by_scope_words = Counter()
    by_period_tokens = Counter()
    by_period_words = Counter()
    by_translation_tokens = Counter()
    by_translation_words = Counter()
    all_tokens = all_words = 0
    narrative_candidate_tokens = narrative_candidate_words = 0
    alf_layla_tokens = alf_layla_words = 0
    kalila_tokens = kalila_words = 0
    for record_index, path in enumerate(record_paths, start=1):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            record = json.loads(source.readline())
        text = record["text"]
        count = len(sp.encode(text, out_type=int))
        words = len(text.split())
        categories = set(record["narrative"]["publisher_categories"])
        scopes = ["narrative_core"] if categories & CORE else []
        if categories & ADJACENT:
            scopes.append("narrative_adjacent")
        if not scopes:
            scopes.append("non_narrative_general")
        if categories & (CORE | ADJACENT):
            narrative_candidate_tokens += count
            narrative_candidate_words += words
        for scope in scopes:
            by_scope_tokens[scope] += count
            by_scope_words[scope] += words
        period = record["narrative"]["period"]
        by_period_tokens[period] += count
        by_period_words[period] += words
        translation = "translated" if record["source"]["translation"] else "Arabic_original"
        by_translation_tokens[translation] += count
        by_translation_words[translation] += words
        if "ألف ليلة وليلة" in record["source"]["title"]:
            alf_layla_tokens += count
            alf_layla_words += words
        all_tokens += count
        all_words += words
        if record_index % 50 == 0:
            print(f"Counted {record_index}/{len(record_paths)} books", flush=True)

    # Count the already collected Kalila text under this same direct-build profile.
    kalila_path = args.kalila_jsonl or args.records.parent.parent / "kalila_1816.jsonl"
    if kalila_path.exists():
        for line in kalila_path.open(encoding="utf-8"):
            row = json.loads(line)
            kalila_words += row["counts"]["whitespace_words"]
            kalila_tokens += len(sp.encode(row["text"], out_type=int))

    report = {
        "schema": "ansg.token_count_report/1",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "profile_id": "ANHC-direct-SPM32-v1",
        "sentencepiece_version": spm.__version__,
        "model_type": "unigram",
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": sp.get_piece_size(),
        "normalization": "SentencePiece NFKC; source text preserved separately",
        "root_aware": False,
        "training_corpus": "all 1,745 CC-BY Arabic E-Book Corpus books, source text only, lexicographically sorted source filenames",
        "tokenizer_training_sample": {
            "algorithm": "reservoir sampling over non-empty source-text lines",
            "seed": args.sample_seed,
            "sampled_lines": len(reservoir),
            "population_lines": seen_lines,
            "sample_text_sha256": sha256(sample_path)
        },
        "training_text_sha256": training_text_sha256,
        "tokenizer_model_sha256": sha256(model_path),
        "counting": "SentencePiece pieces; excludes BOS/EOS/padding; JSON metadata and annotations excluded",
        "all_books": {"records": len(record_paths), "observed_whitespace_words": all_words, "sentencepiece_tokens": all_tokens},
        "source_published_statistics": {
            "provider": "Språkbanken Text",
            "corpus_version_date": "2025-09-12",
            "provider_reported_tokens": 76486597,
            "provider_reported_sentences": 3629107,
            "license": "CC BY 4.0",
            "source_url": "https://spraakbanken.gu.se/resurser/arabic-ebooks",
            "recomputed_by_this_script": False,
            "comparability_note": "Provider-reported corpus tokens use the provider's own processing; do not equate or add them to this script's SentencePiece counts."
        },
        "narrative_candidates_unique": {
            "record_selection": "publisher category intersects core OR adjacent narrative categories",
            "sentencepiece_tokens": narrative_candidate_tokens,
            "observed_whitespace_words": narrative_candidate_words
        },
        "narrative_scope_sentencepiece_tokens_nonexclusive": dict(by_scope_tokens),
        "narrative_scope_whitespace_words_nonexclusive": dict(by_scope_words),
        "period_sentencepiece_tokens": dict(by_period_tokens),
        "period_whitespace_words": dict(by_period_words),
        "translation_sentencepiece_tokens": dict(by_translation_tokens),
        "translation_whitespace_words": dict(by_translation_words),
        "alf_layla": {"observed_whitespace_words": alf_layla_words, "sentencepiece_tokens": alf_layla_tokens},
        "kalila_1816": {"observed_whitespace_words": kalila_words, "sentencepiece_tokens": kalila_tokens},
        "thesis_comparison": {
            "claimed_tokens": 70400000,
            "claimed_tokens_exact": 70404829,
            "exact_reproduction": False,
            "reason": "The thesis's original root-aware SentencePiece model and preprocessing artifact are not available; this direct profile is reproducible but not the historical tokenizer."
        },
        "artifacts": {"model": model_path.name, "vocab": prefix.with_suffix(".vocab").name, "training_sample": sample_path.name}
    }
    (args.output / "token_count_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
