# Direct-build SentencePiece token-count profile

This is a reproducible count profile for the direct corpus build, not the thesis's historical root-aware tokenizer.

- SentencePiece 0.2.0, Unigram, 32,000 vocabulary, NFKC normalization.
- Vocabulary training uses deterministic reservoir sampling of 200,000 non-empty source-text lines (seed 20261114) from the full 1,745-book Arabic E-Book Corpus.
- Counts are then computed over every book, excluding BOS/EOS, JSON metadata, and annotation payloads.
- Model SHA-256: `651cb8ff3a9086f12959dc788ab5e43dfffdeb20a4c3faa644dc1879e5bedab1`.
- Full-corpus observed count: 66,081,654 whitespace words / 107,032,144 SentencePiece pieces.
- Unique narrative-candidate selection (publisher core or adjacent narrative categories): 51,567,623 whitespace words / 82,707,626 SentencePiece pieces.
- Alf Layla wa-Layla: 707,422 whitespace words / 1,134,099 pieces across the seven included editions/volumes (six Arabic originals, one translation).
- Kalila wa-Dimna, 1816 edition: 30,115 whitespace words / 51,845 pieces.

The tokenizer count differs from the thesis target of 70,404,829 and does not establish a match. The original root-aware model and preprocessing are unavailable. No text was padded or trimmed to reach the target. Annotation coverage and splits remain unbuilt.

## Reproduce

1. Download and materialize the official Arabic E-Book Corpus Version 1 from Researchdata.se (DOI `10.5878/7rbh-gy93`, CC BY 4.0) using `scripts/ingest_arabic_ebooks.py --scope all_books`.
2. Install `sentencepiece==0.2.0`.
3. Run:

```bash
python scripts/train_anhc_direct_tokenizer.py \
  --records PATH/TO/arabic_ebooks_all/records \
  --output PATH/TO/tokenizer_spm32 \
  --kalila-jsonl PATH/TO/kalila_1816.jsonl
```

The generated `token_count_report.json` records source scope, profile parameters, and hashes. Do not compare token totals across tokenizers as if they were the same measurement.
