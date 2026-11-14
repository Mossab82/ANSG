# Threats to validity

## Internal validity

| Threat | Mitigation |
| --- | --- |
| **Length and call-count confound:** B/C make several calls and may produce longer text than A/D | Equal total token budget; story length reported; length as a covariate |
| **Information confound:** segment prompts may carry information that the flat prompt lacks | Both prompts are rendered from the same plan. Information parity is audited before the study by listing the plan facts each prompt contains |
| **Prompt-format effect** | H6 (depth-1 equivalence) detects effects that are unrelated to nesting |
| **Controller trace reported as evidence** | Only the blind annotation of text counts (METRICS.md §0) |
| **Seed and sampling variance** | Several seeds per cell; seeds fixed and recorded; mixed models |

## Construct validity

- **Annotation subjectivity.** "Level", "narrator" and "resumption point" in free text are judgements. Mitigations: guideline, training, κ, adjudication. Low κ is reported, not hidden.
- **Metrics are new operational definitions.** METRICS.md says so explicitly. They have not been validated against reader perception; the human items provide that check.
- **Proxies** (dialect markers, plan coverage) are never primary. The marker lists are unreviewed drafts.
- **Cultural fidelity and dialect naturalness** can only be judged by native speakers of the specific variety. Southern Jordanian, Alexandrian, Kuwaiti and Fassi varieties are not interchangeable.

## External validity

- The scenarios were written by one person, are few, and are unreviewed. More scenarios and scenario authors are needed.
- Results for one generator may not transfer to others. Reporting at least two generator sizes is advisable.
- The stories are original, inspired by frame-tale structure. They say nothing about heritage-text generation.

## Statistical conclusion validity

- The single primary hypothesis and the Holm correction limit false positives.
- Power is set by simulation after the pilot.
- Effect sizes are reported with CIs, not only p-values.

## Reproducibility and integrity

- **Contamination.** A model fine-tuned on the Nights may reproduce heritage plots. M8 and human originality checks address this, against the full training text.
- **Leakage.** The heritage pilot's night-level split is not story-independent (`docs/ar/DATA_INTEGRITY_AND_TALE_BOUNDARIES.md`). It must not be used for any HNC claim.
- **Separation from the thesis.** HNC results are reported separately and never as ANSG-Original results.

## Novelty

Narrative embedding as a **stack** has prior art in narrative theory (Ryan 1990) and in computational story *interpretation* by Gervás (2021 onward), who co-supervises this dissertation. Any novelty claim must be limited to what a documented literature review supports (LITERATURE_REVIEW.md §4). No novelty is claimed in this repository.
