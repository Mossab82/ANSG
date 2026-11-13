# Experimental specifications

> **Provenance of this document.** The dissertation was not available when this
> file was created (2026-11-13). Statements in §1 are copied from `README.md`
> as written by the repository owner in commit `faaa3c0`; they have **not**
> been re-verified against the thesis text by the author of this file.
> Everything else is a slot to be filled by transcription, with an exact
> section/table/equation reference per value.

## 1. Known inter-chapter disagreements (preserved, not resolved)

| Topic | Alternative A | Alternative B | Alternative C |
| --- | --- | --- | --- |
| Backbone | GPT-2 Medium (Ch. 4) | AraGPT2-large (App. A) | "774M base parameters" (Ch. 5) |
| Parameter counts | 18M trainable modules (Ch. 4) | 847M total / 73M additions (Ch. 5) | — |
| Training phases | Three phases (Ch. 4–5) | Four stages (App. A.26) | — |
| GPU | A100 40GB (Ch. 5) | A100 80GB (App. A) | — |
| Reward | Three-term reward (Ch. 5) | Four-term reward (App. A.3) | — |
| CSAM dimensions | TO TRANSCRIBE | TO TRANSCRIBE | — |
| Software versions | TO TRANSCRIBE | TO TRANSCRIBE | — |

### Observations that may help reconciliation (public facts, not thesis claims)

These are facts about the public models, offered only to help the author check
the text. They do not decide which chapter is correct.

* Public GPT-2 *Medium* has ≈355M parameters; GPT-2 *Large* has ≈774M. The
  Ch. 5 "774M base" figure therefore matches GPT-2 Large, not GPT-2 Medium (Ch. 4).
* Ch. 5 is internally consistent: 774M + 73M = 847M.
* AraGPT2-large (aubmindlab) is reported by its authors at ≈792M parameters,
  i.e. close to but not equal to 774M.
* An Arabic-trained backbone (AraGPT2) and the English GPT-2 tokenizer behave
  very differently on Arabic; the choice affects every downstream number.

**Decision required from the author:** which configuration is the
*executable* reference for the reimplementation, and how the others will be
reported (e.g. as ablations or as errata). Until then, Phase 2 configs will
carry all variants side by side with their chapter reference.

## 2. Data specifications (Ch. 3) — TO TRANSCRIBE

| Item | Value | Thesis reference | Status |
| --- | --- | --- | --- |
| Nights edition(s) used | | | missing |
| Number of tales / records in ANHC | | | missing |
| Other ANHC sources and their licences | | | missing |
| DIA label inventory (dialect codes) | | | missing |
| MOTIF inventory (motif identifiers; relation to CKG) | | | missing |
| MORAL framework categories | | | missing |
| Annotation guidelines version; annotators; agreement | | | missing |
| Normalisation steps applied | | | missing |
| Segmentation unit for training (tale / night / passage / sentence) | | | missing |
| Train/validation/test ratios or partition file | | | missing (configured 80/10/10 placeholder) |
| Random seed(s) | | | missing |

## 3. Knowledge resources — TO TRANSCRIBE

| Item | Value | Thesis reference | Status |
| --- | --- | --- | --- |
| ArabicNameNet: version, size, fields, entity types, relations | | | missing |
| ArabicVerbNet: version, #classes, #verbs, #frames | | | missing |
| CKG: node kinds and counts | | | missing |
| CKG: relation (schema relationship) types | | | missing |
| CST-Onto: number of classes, top-level structure | | | missing |
| Schema activation function | | | missing |

## 4. Model (Ch. 4) — TO TRANSCRIBE

Backbone, tokenizer, CSAM equations and tensor shapes (hidden size, number of
schema slots, heads, insertion layers), DAL/LoRA rank/alpha/target modules,
morphological conditioning inputs, parameter counts per module.

## 5. Training (Ch. 5, App. A) — TO TRANSCRIBE

Per phase/stage: objective, data, epochs/steps, batch size, learning rate and
schedule, optimiser, precision, gradient clipping, seeds; RL algorithm, reward
terms, reward weights, KL coefficient, rollout settings; hardware and software
versions.

## 6. Evaluation (Ch. 6)

See `docs/CRAFT_EVALUATION_SPECIFICATION.md`.

## 7. Reproducibility conventions adopted by this reimplementation

* All new runs write a `manifest.json` with real UTC timestamps, git commit and
  dirty flag, config hash, input and output SHA-256.
* Preprocessing is deterministic: no RNG; splits are SHA-256-based.
* New results are reported as **new**; no new run is presented as, or dated
  as, a historical experiment.
