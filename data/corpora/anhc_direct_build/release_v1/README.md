# ANHC direct-build release v1

This is a runnable direct-build pipeline over source-attributed Arabic e-books. It is not a historical reproduction of the thesis ANHC corpus.

## Build in one command after source preparation

Install Python 3.10+ and `sentencepiece==0.2.0`. From the repository root, first materialize the source families and train the direct tokenizer:

```bash
python scripts/ingest_arabic_ebooks.py --scope narrative_candidates --output build/anhc/candidates
python scripts/ingest_arabic_ebooks.py --scope all_books --output build/anhc/all_books
python scripts/train_anhc_direct_tokenizer.py --records build/anhc/all_books/records --output build/anhc/tokenizer
python scripts/build_anhc_direct.py --candidate-manifest build/anhc/candidates/manifest.jsonl --records-root build/anhc/candidates --story-units data/corpora/anhc_direct_build/story_units/story_units_provisional.jsonl --tokenizer-model build/anhc/tokenizer/anhc_direct_spm32.model --output build/anhc/release_v1 --seed ANHC-direct-v1 --train-ratio 0.8 --validation-ratio 0.1
```

The ingestion script downloads the cited source archive and metadata when they are absent. The pipeline verifies every candidate text against its SHA-256 before counting it. The tokenizer counts pieces without BOS/EOS/padding. Source text stays in its attributed input files; generated release outputs contain source metadata and record IDs, not copies of the books.

## Current direct-build run

- 1,388 source books: 941 narrative-core candidates and 447 narrative-adjacent candidates.
- 51,567,623 whitespace words and 82,705,621 direct-build SentencePiece tokens. Of those, core candidates account for 25,875,270 words / 42,150,781 pieces; adjacent candidates account for 25,692,353 words / 40,554,840 pieces.
- 1,634 provisional inventory rows: 1,388 whole-book containers, 236 Alf Layla heading spans, and 10 Kalila chapter containers.
- Direct-build split choice: group-hash 80/10/10 with seed `ANHC-direct-v1`; no matching work group crosses splits. This is an operational choice for this build, not the thesis's historical split.
- DIA, MOTIF, and MORAL annotation templates are empty. Triple-covered tokens: 0.

## Outputs

- `release_report.json`: source hashes, direct tokenizer counts, split realization, annotation status, and thesis comparison.
- `source_records.jsonl`: source metadata and per-record counts.
- `story_units.jsonl`: provisional units with group-level splits and null annotation layers.
- `split_assignments.jsonl` and `*_record_ids.txt`: deterministic split map and record lists.
- `annotations/{DIA,MOTIF,MORAL}.jsonl`: empty files for future annotation ingestion.

The thesis reports 70,404,829 corpus tokens and 23M triple-covered tokens. This run does not reproduce those figures: it uses a different tokenizer profile and available sources, and contains no adjudicated triple-layer labels. See [The Arabic E-Book Corpus](https://doi.org/10.5878/7rbh-gy93) and the repository's [ANHC source-scope reconciliation](../../../../docs/ar/ANHC_SOURCE_SCOPE_PROGRESS.md).
