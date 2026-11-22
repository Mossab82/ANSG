# ANHC narrative source review

**Review date:** 2026-11-21  
**Purpose:** identify real Arabic narrative material that can contribute to the ANHC being built. Discovery is not the same as downloading, rights clearance, or admission into training.

## What the linked resources contribute

| Resource | Material | ANHC role |
|---|---|---|
| [Arabic VerbNet](https://github.com/JaouadMousser/Arabic-Verbnet) | Arabic verb classes, frames, semantic roles and examples | Lexical/semantic support for morphology and CKG; not stories. Verify its license before copying records. |
| [QCRI Dialectal Arabic Resources](https://alt.qcri.org/resources/da_resources/) | Dialect segmentation and POS data, including short tweets in four varieties | Auxiliary support for dialect and morphology components; not narrative training. |
| [Arabic dialect identification](https://github.com/Lafifi-24/arabic-dialect-identification) | Classifier code and aggregated sentence-level data | Baseline/discovery only; constituent provenance and test leakage must be checked. |
| [Sanaa AI Stories](https://www.sanaa.ai/stories/index.php/) | Short stories and proverb explanations labelled with several dialects | Useful for discovering dialect categories and planning human review. No reuse permission or authorship was verified, so text is not ingested. |
| [ArabicNER / Wojood](https://github.com/SinaLab/ArabicNER) | Nested named-entity annotations and tools | Possible character/location consistency component; not a story source. Check corpus access and model terms separately. |

These resources are useful, but they do not by themselves supply the long-form stories ANHC needs.

## Best narrative candidates found

1. **Moroccan Darija — DarijaStory.** The dataset card describes 4,392 long stories or chapters scraped from 9esa.com, with one train split and an ODC-BY declaration. This is the strongest direct story-generation candidate so far. The source stories' own terms and author rights still need independent review, and chapters must be linked before any split.
2. **Moroccan Darija — Elfilali's Darija Stories.** The card lists 6,142 chapters and claims more than 70 million tokens under CC-BY-NC-4.0. It includes source URLs and author fields, which makes record-level review possible. The 70M is a source claim, not an ANHC count; non-commercial terms and the rights to each underlying story need checking.
3. **Jordanian Arabic — Layla Witheeb dataset.** The 2026 data paper describes recordings and transcripts for 109 native speakers reading and telling a short story based on an adapted Little Red Riding Hood. This is unusually relevant for Jordanian dialect and retelling evidence, but it is one story family, not diverse plots. The dataset-specific license must be checked on the Mendeley record before taking the transcripts.
4. **Palestinian Arabic — Curras/Currasat story subset.** The broader corpus is small and morphologically useful; its documentation/paper report a limited story subset gathered from different sources, including Cinderella. Extract only the story subset, verify the exact release and its source-level rights, and do not count all Curras tokens as stories.
5. **Algerian Darija — TinyStories Algerian Darija.** The card lists 11,326 parallel child-story pairs, machine translated/adapted from TinyStories, with MIT claimed for the derivative and CDLA-Sharing-1.0 for the original. Keep this in a synthetic/adapted partition, separate from human-authored or heritage stories, and review derivative obligations.
6. **Algerian Darija — Algerian Darja Corpus.** The card describes 11,151 long-form conversational transcripts and CC-BY-4.0, but it is conversation rather than narrative, reports missing per-row source identifiers/licences, and has only a train split. Do not count it as story data; consider only as quarantined dialect-style support if provenance can be resolved.

The registry also retains AraStories and the MSA Arabic Stories Corpus as candidates, with their generated/translated origins or underlying website rights flagged.

## Current decision

No external narrative text has been copied into ANHC or marked training-cleared. The immediate sequence is:

1. Get the dataset-specific terms and pin an immutable revision.
2. Inspect metadata and compute quality/genre/dialect summaries; inspect only within the dataset's terms.
3. Resolve story/chapter parents, duplicates, parallel retellings and source URLs.
4. Admit text into a separate staging area only when the applicable rights permit that use.
5. Keep human-authored, heritage, synthetic, translated, spoken-retelling and conversational material distinguishable.
6. Annotate DIA, MOTIF and MORAL with automatic/silver/gold status kept separate; group narrative families before freezing splits.

The thesis-reported 70.4M total and 23M annotated tokens remain targets for reference. None of the candidate datasets' advertised counts are current ANHC measurements.

## Sources

- [DarijaStory dataset card](https://huggingface.co/datasets/MBZUAI-Paris/DarijaStory)
- [Elfilali Darija Stories dataset card](https://huggingface.co/datasets/alielfilali01/Darija-Stories-Dataset)
- [Jordanian storytelling data paper and Mendeley DOI](https://www.sciencedirect.com/science/article/pii/S2352340926004270) · [dataset record](https://data.mendeley.com/datasets/v9n7g7ns49/1)
- [Currasat documentation](https://sina.birzeit.edu/currasat/about-en.html) · [Curras + Baladi paper](https://aclanthology.org/2022.lrec-1.82/)
- [TinyStories Algerian Darija card](https://huggingface.co/datasets/touati-kamel/TinyStories-Algerian-Darija)
- [Algerian Darja Corpus card](https://huggingface.co/datasets/algerian-nlp/algerian-darja-corpus)
