# ANHC three-layer annotation demo

This directory contains a small, illustrative sample of the DIA, MOTIF, and
MORAL annotation file shapes. It is intended to make the repository's planned
annotation structure visible after the original annotation files were lost.

## What is included

- `DIA.jsonl`: one whole-span register proxy for each sample unit. Classical
  edited Arabic is mapped to `MSA` for display; this is not dialect
  identification.
- `MOTIF.jsonl`: candidate spans from a small set of lexical patterns.
- `MORAL.jsonl`: candidate spans from a small set of value-related terms.
- `manifest.json`: source IDs, licenses, tokenizer fingerprint, sample size,
  and generation method.

Each annotation records character and SentencePiece offsets and carries
`adjudication_status: provisional`, with `annotator_id: demo_rule_engine_v1`.
The values are intentionally marked as rule suggestions. They are not human
labels, adjudicated labels, or reconstructed historical annotations.

## Scope and limits

The sample includes two provisional story spans from each of six Arabic
e-book records and two chapter containers from the 1816 edition of *Kalila wa
Dimna*. A chapter container may contain several nested tales. The files do not
include source text; annotations point to source IDs and offsets only. Splits
are unassigned.

This demo does not represent the thesis's claimed 23-million-token, three-layer
annotation set, reproduce its counts or results, or provide train, validation,
or test data. It is a visual/schema example only. Do not count it as ANHC gold
annotation coverage.

## Sources

- Arabic E-Book Corpus, University of Gothenburg, DOI:
  <https://doi.org/10.5878/7rbh-gy93> (dataset license and edition attribution
  are retained in `manifest.json`).
- *Kalila wa Dimna* (1816), Wikisource transcription, CC BY-SA 4.0; page
  revisions and source links are retained in `manifest.json`.
