# Cultural seed files

One JSON file per region (`msa`, `egyptian`, `levantine`, `gulf`, `maghrebi`). Each file lists values, proverbs, dialect markers, stock characters, settings and traditional story openings and closings.

These files serve two purposes:
- they condition story prompts;
- they seed the small cultural knowledge graph planned for a later stage.

They are **not** evaluation ground truth.

## Provenance and review
- Each file records its origin under `provenance`. The current files were drafted with ChatGPT and have not yet been checked by a native speaker.
- `review_notes` lists items that a native speaker should confirm, reword or drop.
- When a file has been reviewed, set `provenance.review_status` to `native_reviewed` and remove the notes that were resolved.
- Every proverb carries a `confidence` field. Prompt builders should use only `high` and `medium` items.

## Schema
`tests/unit/test_cultural_seeds.py` checks the structure:
- required keys;
- minimum counts;
- the allowed `confidence` values;
- Arabic script in every dialect field.
