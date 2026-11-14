# ANHC direct-build release v1

This is a runnable direct-build pipeline over source-attributed Arabic e-books. It is not a historical reproduction of the thesis ANHC corpus.

## Build in one command after source preparation

Install Python 3.10+ and `sentencepiece==0.2.0`. From the repository root, first materialize the source families and train the direct tokenizer:

```bash
python scripts/ingest_arabic_ebooks.py --scope narrative_candidates --output build/anhc/candidates
python scripts/ingest_arabic_ebooks.py --scope all_books --output build/anhc/all_books
python scripts/build_kalila_1816.py --output build/anhc
python scripts/train_anhc_direct_tokenizer.py --records build/anhc/all_books/records --output build/anhc/tokenizer
python scripts/build_anhc_direct.py --candidate-manifest build/anhc/candidates/manifest.jsonl --records-root build/anhc/candidates --story-units data/corpora/anhc_direct_build/story_units/story_units_provisional.jsonl --tokenizer-model build/anhc/tokenizer/anhc_direct_spm32.model --kalila-jsonl build/anhc/kalila_1816.jsonl --output build/anhc/release_v1 --seed ANHC-direct-v1 --train-ratio 0.8 --validation-ratio 0.1
```

The ingestion script downloads the cited source archive and metadata when they are absent. The pipeline verifies every candidate text against its SHA-256 before counting and emitting it. The tokenizer counts pieces without BOS/EOS/padding. By default, the build writes compressed text records under `corpora/`; these are local model inputs, are not checked into the repository, and are ignored by `.gitignore`. Use `--metadata-only` to generate reports and split manifests without writing text copies. To add the separate count-only user dialect summary, pass `--dialect-audit-report /path/to/privacy-safe-report.json`; the dialect text itself is excluded from story totals and splits.

## Current direct-build run

- 1,398 source text records: 1,388 source books (941 narrative-core candidates and 447 narrative-adjacent candidates) plus 10 Kalila chapter records. Alf Layla volumes are already among the source books and are counted once.
- 51,597,738 whitespace words and 82,757,505 direct-build SentencePiece tokens. Of those, core candidates account for 25,875,270 words / 42,150,781 pieces; adjacent candidates account for 25,692,353 words / 40,554,840 pieces; Kalila accounts for 30,115 words / 51,884 pieces.
- Four auxiliary dialect files add 1,668,817 pieces in a separate audit report; their redistribution terms are unresolved, so the texts are excluded from the story corpus and splits.
- 1,634 provisional inventory rows: 1,388 whole-book containers, 236 Alf Layla heading spans, and 10 Kalila chapter containers.
- Direct-build split choice: group-hash 80/10/10 with seed `ANHC-direct-v1`; no matching work group crosses splits. This is an operational choice for this build, not the thesis's historical split.
- DIA, MOTIF, and MORAL annotation templates are empty. Triple-covered tokens: 0.

## Outputs

- `release_report.json`: source hashes, direct tokenizer counts, split realization, annotation status, and thesis comparison.
- `source_records.jsonl`: source metadata and per-record counts.
- `story_units.jsonl`: provisional units with group-level splits and null annotation layers.
- `split_assignments.jsonl` and `*_record_ids.txt`: deterministic split map and record lists.
- `corpora/pretraining/{train,validation,test}.jsonl.gz`: all source text records, including adjacent categories, for self-supervised pretraining.
- `corpora/narrative_candidates/{train,validation,test}.jsonl.gz`: only core and classical candidates; whole books/chapters remain unsegmented and require story-boundary review before supervised story generation.
- `annotations/{DIA,MOTIF,MORAL}.jsonl`: empty files for future annotation ingestion.

The thesis reports 70,404,829 corpus tokens and 23M triple-covered tokens. This run does not reproduce those figures: it uses a different tokenizer profile and available sources, and contains no adjudicated triple-layer labels. The thesis itself also reports 55,600 narratives in Appendix A.17 versus 5,247 in §5.4.4, and gives different dialect distributions across its tables; see the repository's [thesis alignment report](../../../../docs/ar/ANHC_THESIS_ALIGNMENT.md), [The Arabic E-Book Corpus](https://doi.org/10.5878/7rbh-gy93), and [OpenITI documentation](https://openiti.org/documentation/).

## Provisional heading-bounded sections

A separate inventory extracts 8,838 provisional heading-bounded sections from 478 of 951 narrative-source records (27,639,186 direct SentencePiece pieces). These are section candidates that require content and boundary review; they are not confirmed stories or human annotations. Rebuild them with `scripts/segment_story_boundary_candidates.py`; see [`ANHC_HEADING_SECTION_CANDIDATES.md`](../../../../docs/ar/ANHC_HEADING_SECTION_CANDIDATES.md). Source text remains local and is not committed.
