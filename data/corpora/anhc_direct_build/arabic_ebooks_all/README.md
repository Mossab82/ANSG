# Arabic E-Book Corpus — full 1,745-book intake

This is the broad, licensed text pool for Arabic language-model pretraining. It is kept separate from the 1,388-book narrative-candidate subset. Publisher categories identify 941 core literary candidates and 472 adjacent narrative candidates (25 overlap); the remaining 357 books are other general genres. The pool must not be described as 1,745 stories.

## Materialized from the official Version 1 archive on 2026-11-14

- 1,745/1,745 source files materialized; no archive files missing.
- 66,081,654 observed whitespace-delimited words from extracted source text.
- The provider metadata reports 80,587,361 words for these books. The 14.5M difference is retained for investigation; neither figure is substituted for the other.
- 49,805,829 observed words are from Arabic-original books; 16,275,825 from translations.
- No ANSG-tokenizer count is claimed. Språkbanken reports its own corpus count separately; do not substitute it for our tokenizer count.
- All records have source-book attribution and text SHA-256 in the reproducible materializer output.

The full text is not copied into this GitHub branch. The repository contains the downloader/materializer and report; the source archive is available directly from Researchdata.se under CC BY 4.0. Run the script with `--scope all_books` and a separate output directory to reproduce the records.
