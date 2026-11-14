# Alf Layla wa-Layla story-boundary candidates

This directory contains 236 provisional boundary markers found in six Arabic-original volumes included in the CC BY 4.0 Arabic E-Book Corpus. Each row gives a source record ID, Unicode character offsets, line number, heading text, and source-text SHA-256.

The detector only selects standalone lines beginning with `حكاية`, `حكايات`, `قصة`, or `قصّة`. These are candidate headings for human review. They are not gold story segments, do not assign parent-child relations, and do not count toward the 23-million tri-annotated-token target.

`boundary_report.json` maps each volume to its candidate files and source checksum. Recreate them after materializing the source books with:

```bash
python scripts/segment_alflayla_heading_candidates.py \
  --records data/corpora/anhc_direct_build/arabic_ebooks/records \
  --output data/corpora/anhc_direct_build/arabic_ebooks/alf_layla_boundaries
```
