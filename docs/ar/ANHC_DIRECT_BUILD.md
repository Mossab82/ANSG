# ANHC Direct Build

## Purpose

This workstream collects narrative texts and annotations directly from identified sources. It does not reconstruct missing historical files or claim that newly collected material is the exact historical ANHC release described in the dissertation.

The dissertation reports 70.4 million tokens, 23 million tri-layer-annotated tokens, five Arabic varieties, and period totals of 24.6M / 18.3M / 27.5M. Those figures are recorded as validation targets only. The build reports measured totals from pinned source files and a pinned tokenizer; it never pads, trims, or fabricates text to hit the targets.

## Intake groups

1. **Classical stories and animal fables:** begin with identified editions of *Alf Layla wa-Layla* and *Kalila wa-Dimna*. Discover page order from the source index and pin every page revision. Page availability alone does not establish the edition's training or redistribution rights; human review is required before either use.
2. **Modern narratives (1800–2000):** admit a work only after its exact edition and rights status are recorded.
3. **Contemporary dialect stories:** collect original oral or written stories from contributors under explicit, recorded consent for the intended research uses. Do not scrape Wattpad, forums, books, subtitles, or podcasts.

## Record requirements

Each admitted story must have a stable story ID; source and edition; author/narrator where available; source URL or consent record; retrieval date and revision/version; source-text checksum; language/dialect label with evidence; period and genre; story-cycle and parent/child links only when supported by source evidence; rights and permitted-use state; split assignment; and annotation status.

The source register in [anhc_direct_build.yaml](../../configs/data/anhc_direct_build.yaml) is the admission gate. Unknown rights or unknown edition identity mean the text is not admitted to training or public redistribution.

## Build milestones

- **M0 — direct intake:** enumerate the named source editions and acquire stable source snapshots.
- **M1 — rights and identity review:** approve each source file for the specific use (research, training, public release).
- **M2 — story segmentation:** preserve nested tales and story cycles, retain ambiguous boundaries as unresolved.
- **M3 — dialect and period coverage:** measure observed counts against the thesis targets; document gaps.
- **M4 — annotation pilot:** publish DIA/MOTIF/MORAL guidelines and adjudicate a small human-checked sample before scaling.
- **M5 — scale and evaluation:** scale only from approved sources and validated annotation workflow; create story-cycle splits before model training.

No milestone is complete merely because a target count is reached. Each release must include a source manifest, annotation report, tokenizer/version, observed counts, checksums, rights decisions, and split-leakage report.
