# ANHC direct build

This branch is assembling a reproducible corpus release from openly licensed source texts and source-attributed classical narratives. Thesis numbers remain comparison targets until the original tokenizer, corpus manifest, codebook, and split artifacts are available.

## Current measured intake

- Arabic E-Book Corpus Version 1: 1,745 materialized books; 66,081,654 observed whitespace words.
- Narrative candidate subset: 1,388 whole-book candidates; 51,567,623 observed whitespace words. Publisher categories contain 25 books that overlap core and adjacent narrative classes.
- Direct SentencePiece profile: 107,029,217 pieces across all books and 82,705,621 pieces in the unique narrative candidate subset. This is a non-root-aware profile and does not reproduce the thesis's 70,404,829-token count.
- Classical source records: Alf Layla wa-Layla includes six Arabic originals and one Arabic translation; Kalila wa-Dimna includes ten chapters from the 1816 edition.

## Data and reproducibility map

- `arabic_ebooks/`: narrative-candidate materializer report and Alf Layla heading-boundary candidates.
- `arabic_ebooks_all/`: full 1,745-book pool, source report, and reproducible materializer.
- `tokenizer_spm32/`: token-count report, tokenizer profile, and SHA-256 provenance.
- `story_units/`: 236 provisional Alf Layla spans plus 10 Kalila chapter containers, work-cycle locks, and unsegmented-prefix record.
- `../../../../scripts/ingest_arabic_ebooks.py`: download and materialize source records.
- `../../../../scripts/train_anhc_direct_tokenizer.py`: fit direct tokenizer and count every source text.
- `../../../../scripts/build_story_unit_review_manifest.py`: regenerate source-linked provisional story units.
- `../../../../scripts/audit_anhc_annotations.py`: count adjudicated single-layer and triple-covered token intervals.
- `../../../../docs/ar/ANHC_RELEASE_PROTOCOL.md`: source rules, annotation codebook policy, thesis conflicts, and split protocol.

The full Arabic E-Book Corpus text is not duplicated in this repository. Its official Version 1 record is available from Researchdata.se (DOI: 10.5878/7rbh-gy93, CC BY 4.0). Kalila source pages retain their edition, revision IDs, attribution, and CC BY-SA 4.0 metadata.

## Release state

Story-boundary candidates are provisional, source-cycle split labels remain unassigned, and human-adjudicated DIA/MOTIF/MORAL coverage is zero. The branch does not claim the thesis's 23 million triple-covered tokens, 847,000 segments, or final train/validation/test population. No text is padded or trimmed to fit a target.
