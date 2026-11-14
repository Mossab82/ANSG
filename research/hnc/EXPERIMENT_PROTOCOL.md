# Experiment protocol (draft for pre-registration)

**Status:** not run. No numbers below are results. Sample sizes are fixed by a pilot (§7).

## 1. Conditions

| ID | Description | `scripts/hnc_generate.py --variant` |
| --- | --- | --- |
| A | ANSG-Original generation path, HNC disabled, flat plan prompt | `A` |
| B | Full HNC: stack control, resumption, and cross-level knowledge validation of the plan | `B` |
| C | Level tracking only: no knowledge validation | `C` |
| D | Knowledge validation of the flattened plan only: no stack or resumption control | `D` |

**Same for all conditions:**
- generator, checkpoint (hash-pinned), tokenizer;
- chat template use;
- temperature 0.7 and top-p 0.95 (Ch. 6 §6.1.2 defaults);
- seeds;
- plan content.

**Token budget parity:** A and D get the total budget that B or C spends over all segments. This removes a length confound (THREATS_TO_VALIDITY.md).

**External baselines:** none planned. No published system for Arabic nested-narrative generation with available code was found (LITERATURE_REVIEW.md). A general long-story planner such as Re3 or DOC may be added as an exploratory baseline only if it runs in Arabic without modification. Otherwise it is not reported.

## 2. Generators

- A local, hash-verified instruction-tuned model, run through `ansg-generate` infrastructure. The Qwen2.5-1.5B-Instruct heritage LoRA pilot is a candidate once its data split is repaired.
- ANSG cultural modules only if trained modules exist (`docs/GENERATION_READINESS.md`). Otherwise the generator is labelled `backbone_only`, and the paper must say so.

## 3. Factors

| Factor | Levels |
| --- | --- |
| Condition | A, B, C, D |
| Depth | 1, 2, 3, ≥4 |
| Setting | Aqaba, Alexandria, Kuwait City, Fes |
| Register | R1 (MSA narration, dialect dialogue), R2 (dialect narration and dialogue) |
| Seeds | k per cell (from the pilot) |

## 4. Materials (`research/hnc/scenarios/`)

| Depth | Scenario(s) |
| --- | --- |
| 1 | `*_1level` for all four settings (frame only: control) |
| 2 | `aqaba_2level`, `alexandria_2level`, `fes_2level` |
| 3 | `kuwait_3level` |
| ≥4 | `fes_4level_stress` |

**Requirements met by every scenario:**
- the embedded tale has a stated function;
- it has a declared time context;
- it conveys knowledge that the frame story uses when it resumes;
- all of this is validated by `ansg.hnc.validation`.

**Before human evaluation:**
1. Cultural and linguistic review of every scenario by native speakers from the setting. The status is currently `unreviewed_draft`.
2. Additional culturally grounded scenarios for depths 3 and ≥4 in every setting. `fes_4level_stress` is a structural stress test with generic names, not a cultural story.
3. A check that no scenario reproduces a heritage plot. Originality is checked by people, and against the source with M8.

## 5. Annotation of generated stories

- **Annotators:** two trained Arabic-speaking annotators per story, blind to the condition. Generation records are stripped of labels and stories are shuffled.
- **What they annotate:** the realised trace (METRICS.md §0) and the Event Graph knowledge and causal violations of each frame.
- **Training:** a guideline with worked examples on pilot stories that are not used in the analysis.
- **Agreement:** κ per field, with disagreements adjudicated by a third annotator.

## 6. Human evaluation

- **Raters:**
  - native speakers of each setting's variety, at least 3 per setting;
  - not involved in system development;
  - recruited with informed consent;
  - paid, if applicable, per institutional rules.
- **Design:**
  - each rater sees stories from all conditions for the same scenario, in random order, without system identity;
  - each story is rated by at least 3 raters;
  - attention checks use planted defective stories.
- **Items:** H-PLOT, H-TRANS, H-TRACK, H-RETURN, H-LANG, H-CULT, H-DIAL, H-ORIG (METRICS.md §3), plus an optional pairwise preference B vs A.
- **Ethics:** institutional approval is required before recruitment. No participant data has been collected, and none is invented.

## 7. Pilot

A small pilot sets the inputs of the main study. Pilot data is not used in confirmatory analyses.

1. Check the pipeline end to end.
2. Estimate variances for a **simulation-based power analysis**, which fixes k and the number of raters.
3. Fix δ and the TOST margins (HYPOTHESES.md).

## 8. Analysis

| Analysis | Method |
| --- | --- |
| Structural measures (M1–M3, M6, M7; proportions per story) | Mixed-effects logistic regression: condition × depth (+ register), random intercepts for scenario and seed |
| Count measures (M4, M5) | Negative-binomial mixed model |
| Ratings | Cumulative-link mixed model, random intercepts for rater, scenario and story |
| Confidence intervals | 95%, cluster bootstrap over scenarios as a robustness check |
| Multiplicity | H1 confirmatory; Holm across H2–H7 |

All analysis code is written and frozen before data collection.
