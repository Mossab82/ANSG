# ANHC source scope and progress

This note separates size evidence from actual narrative data and dialect support.

## Current scale measures

| Scope | Measure | Meaning |
| --- | ---: | --- |
| Complete Arabic E-Book Corpus | 76,486,597 provider-reported tokens | Provider-native measure across all 1,745 books; close to the thesis's approximate 72M scale, but not a story-only count |
| Same complete corpus | 107,029,217 local SentencePiece pieces | Direct-build SPM32 count; a different tokenizer and counter |
| Publisher-selected narrative candidates | 82,705,621 local SentencePiece pieces | Candidate subset; includes adjacent classes and still requires story-level review |
| Thesis target | 70,404,829 tokens | Historical target; exact tokenizer/preprocessing are unavailable, so direct-build counts do not claim exact reproduction |
| Human-adjudicated triple-covered text | 0 tokens | DIA, MOTIF, and MORAL coverage must all be human-adjudicated before counting |

## Added dialect intake

The four user-supplied files were counted with the pinned direct-build SentencePiece model. They contain **66,247 non-empty short lines**, **924,493 whitespace words**, and **1,668,817 SentencePiece pieces** across Jordanian, Palestinian, Syrian, and Lebanese labels. Inspection shows short social-media-style lines, not long stories. Their text is not published in this repository; redistribution rights remain unresolved. The detailed count and SHA-256 report is at `data/corpora/anhc_direct_build/user_provided_dialect_count_report.json`.

MADAR is retained for external dialect evaluation only: it covers 25 city varieties, including Amman and Salt, but its travel-domain parallel sentences are not narratives and its access terms prohibit modification and redistribution. QADI, NADI, DART, and Shami are likewise separate dialect-identification resources, not story data.

## New narrative source

The Layla Witheeb Jordanian Arabic dataset provides orthographic transcriptions for 109 native speakers reading and retelling an adapted Little Red Riding Hood story. Its page identifies CC BY 4.0. It is well matched to Jordanian spoken narrative and can inform dialect modeling and evaluation. The repetitions are renditions of one plot, so they must share a story-cycle split. The published archive is 1.55 GB; only the text transcription files are needed for this corpus workflow, while audio and speaker metadata are unnecessary for the first text-only intake. No token count is recorded until those text files are ingested and checked.

## Completion gate

The current release has a 70M-scale source pool and substantial classical/narrative candidates, but it does **not** yet establish the thesis's 55,600 narrative units or 23M human-adjudicated DIA–MOTIF–MORAL tokens. We will keep the full-corpus scale result, story-only counts, and annotated coverage as separate audited totals. Story boundaries, source attribution, annotation evidence, adjudication, and cycle-safe split assignment are required before the corpus stage can be represented as complete.
