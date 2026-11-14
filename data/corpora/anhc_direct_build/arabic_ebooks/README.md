# Arabic E-Book Corpus intake

## Source

This intake uses *The Arabic E-Book Corpus (Version 1)* by Andreas Hallberg, University of Gothenburg, DOI [10.5878/7rbh-gy93](https://doi.org/10.5878/7rbh-gy93). The official dataset provides 1,745 Arabic books and is released under CC BY 4.0. Its documentation states that included original works and translations are public domain or CC BY 4.0. The source archive and metadata are downloaded directly from the Researchdata.se deposit.

## Materialized on 2026-11-14

- 1,745 source books in the archive.
- 1,388 books selected as narrative candidates from publisher categories.
- 941 include a core literary category: novels, detective fiction, science fiction, children's stories, plays, or literature.
- 472 include an adjacent narrative category: history, biographies, or travel literature. Twenty-five books fall into both category groups.
- 51,567,623 observed whitespace-delimited words across all selected texts; the source metadata reports 62,891,124 words for those same selected records. Both counts are kept in the report; the difference requires investigation.
- 38,491,864 observed words are from Arabic-original works; 13,075,759 are from translated works.
- No tokenizer-based token total is claimed. These are whole-book candidates, not story-segmented or annotated records. They are primarily edition-specific Arabic, not a dialect corpus.

## Reproduction

From the repository root, run:

```bash
python scripts/ingest_arabic_ebooks.py \
  --zip arabic-corpus-txt.zip \
  --metadata arabic-ebooks-metadata.tsv \
  --output data/corpora/anhc_direct_build/arabic_ebooks
```

If the archive or metadata files are absent, the script downloads them from the source deposit. It preserves each source text, attaches book-level attribution and SHA-256, and writes per-book compressed JSONL records, a manifest, and a build report. It does not split books into story units, annotate them, or assign train/validation/test splits.

## Rights and citation

Attribute Hallberg's corpus and preserve each book's publisher metadata. Dataset DOI and license details are in the Researchdata.se record. Article describing the corpus: Hallberg (2025), *An 81-million-word multi-genre corpus of Arabic books*, DOI [10.1016/j.dib.2025.111456](https://doi.org/10.1016/j.dib.2025.111456).
