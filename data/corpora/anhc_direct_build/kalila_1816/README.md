# Directly collected Arabic narrative texts

## Batch 1 — Kalila wa Dimna (1816)

This is the first collected source batch for the direct ANHC build. It contains ten chapter-level records from the Arabic Wikisource transcription of the Paris 1816 edition of *Kalila wa Dimna*, associated with Sylvestre de Sacy. The records preserve the source transcription; no OCR or spelling correction has been applied.

- Source index: [كليلة ودمنة (1816)](https://ar.wikisource.org/wiki/كليلة_ودمنة_(1816)); index revision 638949.
- Each record pins its own source page revision and permanent URL, and includes source-text SHA-256, extraction note, whitespace word count, and attribution.
- Source license: CC BY-SA 4.0 as shown on the Wikisource page. Reuse requires attribution and share-alike; see <https://creativecommons.org/licenses/by-sa/4.0/>.
- 17 narrative chapter links were requested; 10 pages were available and collected, six linked pages were missing, and one front-matter page was excluded.
- Collected size: 30,115 whitespace-delimited words. No tokenizer-based token count is claimed.
- Records are source chapters, and may contain nested tales. They are not segmented into story units, annotated, or split for training/evaluation.
- This is a small pilot batch in the new direct build. It is not the historical thesis ANHC and does not establish or reproduce its reported 70.4M/23M counts or experimental results.

## Files

- `build_report.json`: collection counts, missing pages, and limitations.
- `records/kalila-1816-XX.jsonl`: one UTF-8 JSONL source record per available chapter, in source order. Files are plain text and can be concatenated for ingestion.
- `../../../../scripts/build_kalila_1816.py`: reproducible MediaWiki API fetcher; stores both plain JSONL and compressed local outputs.
