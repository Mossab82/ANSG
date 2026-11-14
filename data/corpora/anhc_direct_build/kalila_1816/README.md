# Directly collected Arabic narrative texts

## Batch 1 — Kalila wa Dimna (1816)

- Source index: [كليلة ودمنة (1816)](https://ar.wikisource.org/wiki/كليلة_ودمنة_(1816)), Paris, 1816; edition associated with Sylvestre de Sacy.
- Source pages: 10 chapter records, each pinned to its own Wikisource revision in the JSONL records.
- Extracted amount: 30,115 whitespace-delimited words. This is **not** a tokenizer count.
- Six linked chapter pages currently return `missingtitle`; one front-matter page was excluded. The build report records each case.
- The records are chapter-level. Nested stories have not yet been segmented or annotated. All splits remain unassigned.
- The text pages identify CC BY-SA 4.0. Reuse must retain attribution, link the license, indicate extraction changes, and preserve share-alike terms.

## Files

- `kalila_1816.jsonl.gz`: directly fetched chapter texts with source revision, attribution, checksum, counts, and empty annotation fields.
- `build_report.json`: exact retrieval totals, missing pages, and exclusions.
- `build_kalila_1816.py`: standard-library fetch and extraction program.

## Attribution

“كليلة ودمنة (1816),” transcribed by Wikisource contributors from the 1816 Paris edition associated with Sylvestre de Sacy. Page-specific revision URLs and SHA-256 checksums are in each record. The collection release of these extracted texts is made available under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Changes: rendered HTML was converted to text and whitespace was separated; source wording was not editorially corrected.

This is a new, directly collected source batch. It is not the thesis-era ANHC, does not establish the thesis corpus counts, and is not ready for model training until the annotation, segmentation, and split work is completed.
