# Arabic E-Book Corpus intake

## Source

This intake uses *The Arabic E-Book Corpus (Version 1)* by Andreas Hallberg, University of Gothenburg, DOI [10.5878/7rbh-gy93](https://doi.org/10.5878/7rbh-gy93). The official dataset provides 1,745 Arabic books and is released under CC BY 4.0. Its documentation states that included original works and translations are public domain or CC BY 4.0. The source archive and metadata are downloaded directly from the Researchdata.se deposit.

## Materialized on 2026-11-14

The intake has two explicit scopes: `narrative_candidates` selects books in publisher story/literature categories for story segmentation and annotation; `all_books` retains the full licensed collection as a broad language-model pretraining pool. The latter is not treated as 1,745 verified stories.

- 1,745 source books in the archive.
- 1,388 books selected as narrative candidates from publisher categories.
- 941 include a core literary category: novels, detective fiction, science fiction, children's stories, plays, or literature.
- 472 include an adjacent narrative category: history, biographies, or travel literature. Twenty-five books fall into both category groups.
- 51,567,623 observed whitespace-delimited words across all selected texts; the source metadata reports 62,891,124 words for those same selected records. Both counts are kept in the report; the difference requires investigation.
- 38,491,864 observed words are from Arabic-original works; 13,075,759 are from translated works.
- No tokenizer-based token total is claimed. These are whole-book candidates, not yet adjudicated story segments or annotations. They are primarily edition-specific Arabic, not a dialect corpus.
- This intake includes six Arabic-original volumes of *Alf Layla wa-Layla* (692,899 observed whitespace words) plus one Arabic translation of an English-origin edition (14,523 observed words). A first rule-based pass found **236 standalone heading candidates** in the six Arabic-original volumes. They are provisional boundary markers requiring review; nested story links remain unresolved. See `alf_layla_boundaries/`.

## Reproduction

From the repository root, run:

```bash
python scripts/ingest_arabic_ebooks.py \
  --zip arabic-corpus-txt.zip \
  --metadata arabic-ebooks-metadata.tsv \
  --scope narrative_candidates \
  --output data/corpora/anhc_direct_build/arabic_ebooks
```

For the broad pretraining pool, change the scope to `all_books` and use a different output directory.

If the archive or metadata files are absent, the script downloads them from the source deposit. It preserves each source text, attaches book-level attribution and SHA-256, and writes per-book compressed JSONL records, a manifest, and a build report. It does not split books into story units, annotate them, or assign train/validation/test splits.

To materialize the entire 1,745-book base for pretraining, pass `--scope all_books`; to materialize only the story/literature candidate subset, pass `--scope narrative_candidates` (the default). Use separate output directories so the broad pretraining pool cannot be mistaken for the story-only subset.

## Rights and citation

Attribute Hallberg's corpus and preserve each book's publisher metadata. Dataset DOI and license details are in the Researchdata.se record. Article describing the corpus: Hallberg (2025), *An 81-million-word multi-genre corpus of Arabic books*, DOI [10.1016/j.dib.2025.111456](https://doi.org/10.1016/j.dib.2025.111456).
