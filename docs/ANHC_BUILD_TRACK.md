# ANHC construction track

**ANHC is a project deliverable we are building. It is not an external corpus the author is expected to already possess.** This track turns the currently collected sources into a traceable narrative corpus, then adds reviewed annotations and splits. It distinguishes that new corpus from the thesis-reported ANHC so we do not present reconstructed data as the historical dataset.

## Starting inventory

The repository currently has a seed collection at `data/corpora/classical_arabic/`: 20 Wikisource source-page/cycle records, 195,197 words total (five One Thousand and One Nights pages and fifteen Kalila wa Dimna pages). The records are not yet extracted into independent nested stories, annotated, or assigned a leakage-safe story split. The build report explicitly marks rights and story-independent splitting as unverified.

This is a useful bootstrap source inventory. It is not yet the finished ANHC and its word count must not be equated with the thesis's 70.4M-token corpus.

## Populated staging build

The first construction pass now materializes **27 ANHC-schema records with their Arabic text** in `data/corpora/anhc/staging_records.jsonl`: 12 candidate spans segmented from the five Nights source pages using the repository's configurable title-boundary extractor, plus 15 titled Kalila wa Dimna source-page candidates. Each record includes the source-page id, exact revision id, source text digest, URL, and character span. `story_review_queue.jsonl` lists every candidate for structural review; the extraction is a proposal and nested fables remain unreviewed.

All three annotation layers (DIA, MOTIF, MORAL) remain `unannotated`; rights are `unknown`, `redistributable` is false, and splits remain unassigned. The build has 0 schema errors and expected warnings for unresolved rights. This is populated staging data for the ANHC we are constructing, not a rights-cleared training release or the thesis-reported 70.4M-token / 23M-annotated-token dataset. Re-run with `python scripts/build_anhc_staging.py` after updating the seed.

## User-supplied reference books

The newly supplied **Kalila wa Dimna** PDF is 101 pages. Its searchable text contains 45,726 whitespace-separated words after NFKC/format-control normalization, though 715 private-use glyphs indicate character-mapping defects. Its contents page lists 19 section starts; all 15 current seed titles now cross-reference to printed page starts in the PDF, while four are front matter. A local-only, git-ignored quarantine extraction now holds 19 page-range candidates (15 fable chapters/story cycles and four front-matter sections), with 45,714 normalized whitespace-separated words from printed pages 2–101. This is still a structural proposal: nested tales are not independently segmented, and title/page matches do not establish text equivalence with Wikisource. The PDF edition and reuse rights remain unverified, so the extracted text is neither redistributable nor cleared for training.

The supplied **One Thousand and One Nights** PDF is a 1,342-page scan. It yields no usable Arabic text through `pdftotext`; no Arabic Tesseract language pack is installed in this environment, so Arabic OCR and bibliographic identification remain required. A visible W. R. Taylor Collection stamp is archival provenance, not a publication/rights statement. Both files therefore expand the source evidence and structural review work; only the Kalila file is immediately text-extractable, and neither is rights-cleared.

## Linked Arabic language resources

The three linked resources support separate ANSG components; they are not story corpora:

- [QCRI Dialectal Arabic Resources](https://alt.qcri.org/resources/da_resources/) and its [dataset repository](https://github.com/qcri/dialectal_arabic_resources) provide segmentation/POS resources. The four-dialect POS set has 350 manually segmented and tagged tweets per dialect (Egyptian, Levantine, Gulf, Maghrebi); the provider page states Apache-2.0. Use only as auxiliary DIA/morphology data and keep the official folds separate from narrative training/evaluation.
- [Arabic VerbNet](https://github.com/JaouadMousser/Arabic-Verbnet) is an XML verb lexicon with roots, forms, syntactic frames, semantic roles/predicates, and examples. It may support CKG/verb-frame work, but it is not a story or dialect corpus; its reuse license remains unverified, so the registry does not admit its data yet.
- [Arabic Dialect Identification](https://github.com/Lafifi-24/arabic-dialect-identification) describes a sentence-level classifier project combining QADI, IADD, MADAR, the QCRI POS set, and 10,000 Wikipedia MSA sentences. Keep it as a baseline/discovery reference until constituent licenses and partitions are traced. Do not copy its aggregate data into ANHC or treat its labels as story-level gold.

The source roles and admission gates are recorded in `configs/data/anhc_build.yaml` and `data/sources/dialect_registry.json`. This preserves the distinction between narrative training data, morphological resources, and dialect-classification benchmarks.

Two further linked resources have complementary roles:

- [Sanaa AI Stories](https://www.sanaa.ai/stories/index.php/) displays short modern stories with site-provided dialect labels, including Palestinian-Jordanian, Palestinian, Yemeni, and Moroccan examples. The site shows a rotating 200-story window. No reuse license or authorship/generation statement was found on the reviewed story pages, so these texts remain discovery references only; do not scrape, add, or train on them until permission and provenance are established. Site labels are candidate labels, not ANHC gold.
- [Wojood / ArabicNER](https://github.com/SinaLab/ArabicNER) is an entity-recognition resource, not a story corpus. Its README reports about 550K MSA/dialect tokens, 21 entity types, about 75K entities, and nested annotations for 22.5% of entities; the full corpus is available upon request for academic/commercial use. The repository code is MIT-licensed. It is a promising candidate for checking character/place/entity consistency, including nested mentions, but full corpus access and model terms must be reviewed before importing labels or weights.

## Source admission rules

A source is admitted to a training split only after its exact edition/revision and reuse rights are recorded. Work-level public-domain status does not automatically clear a later edition's notes, editorial additions, images, or formatting. Keep only the text whose rights have been verified; store attribution and source revision in every record. If rights remain uncertain, keep the source in the registry as `review_required` and keep its text out of distributable training exports.

Current source notes:

- The Wikisource page for **One Thousand and One Nights, Part 1 (Hindawi 2022)** states that the original work text is public domain and separately identifies the cover/design as CC BY 4.0. The build uses text only and must preserve that distinction. Verify the exact revision used by the repository collector and each remaining volume before marking records trainable.
- The Wikisource scan for **Kalila wa Dimna, Egypt 1355 AH** is marked public domain. The individual story pages in the current seed are categorized as unknown-source pages, so map them to the verified scan/revision or retain `review_required`.
- **AraStories** is a promising MSA/Egyptian/Moroccan story source. Its repository describes three story/prompt files and an MIT project licence; its paper says the data include machine-translated and GPT-4-generated stories. Preserve those origin types separately and review per-row source/redistribution terms before importing text. This source can supplement the new ANHC, but it is not automatically thesis ANHC or gold data.
- The **Arabic Stories Corpus** repository reports an Apache 2.0 licence but its files include material gathered from Mawdoo3 and Qisasse. The repository licence alone does not establish permission for every underlying story; keep it out of training until the underlying rights are confirmed.
- Short dialect corpora such as Shami/MADAR can support dialect metadata or separate DAL experiments, but they are not long-form narrative stories. Do not mix them into the narrative corpus as story examples.

## Record structure to build

Each narrative unit will keep:

- stable record id, source-work id, exact source URL/revision, source-file digest, edition, rights status, and attribution;
- narrative title, text span in the source, parent/child story relation, nesting level, and the night/chapter/page where it occurs;
- genre and period where supported by the source;
- language variety and geographic granularity (MSA, Egyptian, Levantine, Gulf, Maghrebi; country/city only when evidenced);
- speaker/dialogue spans separated from narrator text when available;
- DIA, MOTIF, and MORAL labels with annotator, guideline version, evidence span, confidence, and annotation status.

The existing `ansg.anhc-record/0.1` schema already provides provenance, parent-story, nesting, spans, splits, and the three annotation layers. Use it as the first implementation contract; do not invent thesis label inventories that are absent from the supplied dissertation.

## Annotation and split policy

1. **Segment first, annotate second.** Keep source pages/cycles as parents. Extract child stories only where headings or a reviewed segmentation decision support the boundary; uncertain boundaries remain marked for review.
2. **Separate proposal from gold.** Automated/agent-generated labels are `automatic`; one unadjudicated human label is `silver`; only human-adjudicated labels under a versioned guide are `gold`. A model cannot promote its own output to gold.
3. **Keep layers independent.** DIA is not inferred from a motif or moral label. A dialect guess must retain its evidence and remain unannotated/automatic until a reviewer confirms it.
4. **Split by narrative family.** All nested tales, retellings, parallel translations, and repeated source cycles must stay together. The thesis's 80/10/10 and seed 42 are target settings; use them only after group identities and corpus coverage are audited.
5. **Record what is actually built.** Report counts by source, rights, variety, genre, nesting, annotation layer/status, and split. Keep the thesis's 70.4M / 23M token figures as reference targets, never as current counts.

## Build order

- **B0 — Source registry:** done as metadata in `data/corpora/classical_arabic/index.json`; this track adds rights/origin decisions without silently changing the source text.
- **B1 — Rights pass:** pin exact revisions and classify each source/page as `public_domain_text`, `licensed`, `restricted`, or `review_required`.
- **B2 — Narrative segmentation:** initial candidate spans and a boundary-review queue are built; preserve parent-page ids and inspect nesting before accepting boundaries or splitting.
- **B3 — Dialect/story intake:** add rights-cleared narrative sources for missing varieties; keep dialect benchmarks and story-generation corpora distinct by origin and task.
- **B4 — Annotation pilot:** create a small, versioned guide and blind double annotation for DIA/MOTIF/MORAL; measure agreement; adjudicate disagreements; then expand.
- **B5 — Leakage-safe splits:** cluster source variants and narrative families before assigning train/validation/test; freeze the test set before model selection.
- **B6 — ANHC release candidate:** produce the manifest, text export according to rights, annotation exports by status, split manifest, and token statistics for the selected tokenizer.

## Current blockers for the full thesis-scale target

The existing seed is far below 70.4M tokens; full source coverage and the original 23M human-annotated token layer do not exist in the repository. MOTIF identifiers and several CRAFT/CMPI definitions are absent from the supplied thesis transcription. These are construction tasks and must be tracked as such; they do not mean the author should already have an ANHC package.

## References

- [Existing seed inventory and build limitations](../data/corpora/classical_arabic/build_report.json)
- [Existing ANHC record schema](../src/ansg/data/records.py)
- [User-supplied PDF intake report](../data/corpora/anhc/source_intake.json)
- [Wikisource One Thousand and One Nights, Part 1 (Hindawi 2022)](https://ar.wikisource.org/wiki/ألف_ليلة_وليلة_(مؤسسة_هنداوي،_2022)/الجزء_الأول)
- [Wikisource Kalila wa Dimna scan, Egypt 1355 AH](https://ar.wikisource.org/wiki/ملف:كليلة_ودمنة.pdf)
- [AraStories data/code repository](https://github.com/UBC-NLP/arastories)
- [AraStories paper](https://aclanthology.org/2024.arabicnlp-1.13/)
- [Arabic Stories Corpus repository](https://github.com/motazsaad/Arabic-Stories-Corpus)
- [Dialect source registry](../data/sources/dialect_registry.json)
