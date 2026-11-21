# Chapter 4–5 reconciliation for the implementation track

**Status:** working reconstruction proposal; not an erratum and not proof of historical execution.  
**Source:** author-supplied dissertation transcription, checked against the glossary and the cited Chapter 4–5 sections.  
**Implementation rule:** preserve source variants, label the selected build path, and keep training blocked while required evidence or scientific definitions are missing.

## Decisions recorded

| Topic | Textual evidence | Working implementation interpretation | Status |
|---|---|---|---|
| CSAM name | Chapter 0 glossary lists **Cultural Schema Attention Mechanism (CSAM)**. Chapter 4 §4.3 is headed “Cultural Semantic Alignment Module.” | Use the glossary expansion as the canonical name in new documentation and user-facing reports. Keep the Chapter 4 title as a recorded variant; do not rename existing code or claim the terminology conflict is historically resolved. | Terminology choice adopted for this implementation track. |
| Training stages | Chapter 4 Table 4.1 and §4.1.6 describe pretraining / cultural fine-tuning / RL variants. Chapter 5 overview (printed p.132) and §§5.6.1–5.6.3 (pp.149–150) describe supervised training, adversarial GAN refinement, then PPO. Appendix A.2.6 describes a four-phase variant. | Use Chapter 5’s detailed three-stage schedule as the first end-to-end reconstruction target. Treat Chapter 4’s cultural objective (Eq. 4.16–4.19) as part of the supervised stage where compatible. Keep Appendix A’s multi-agent stage as a separate, unselected variant. Do not overwrite the thesis register. | Provisional build target; historical protocol unresolved. |
| GAN | Chapter 5 §5.6.2 explicitly describes a cultural discriminator and adversarial phase between supervised training and PPO. | Implement GAN as the middle stage in the Chapter 5 candidate. The existing mean-pooled discriminator is not equivalent to the thesis’s reported 3-layer CNN/feature-matching setup and must remain labelled as a prototype until aligned. | Direction adopted; implementation fidelity incomplete. |
| Backbone lineage | Chapter 4 §4.1.6 states GPT-2 Medium (355M) continued pretraining on 70.4M ANHC tokens; Chapter 5 Table 5.1 reports 774M base + 73M cultural modules = 847M; Appendix A.3 names AraGPT2-large. These do not establish a checkpoint transition. | Preserve both size/model variants. Treat “GPT-2 Medium was an earlier development checkpoint” as a testable reconstruction hypothesis only. Do not resize, swap, or chain checkpoints without original configs, checkpoint metadata, or author-supplied evidence. | Unresolved; blocks a faithful full-model run. |
| Reward hierarchy | Chapter 4 Eqs. 4.26–4.29 and Chapter 5 report top-level cultural / linguistic / coherence weights 0.45 / 0.35 / 0.20. The internal cultural definitions and Appendix A.3 differ. | Keep top-level weights distinct from sub-reward weights. Do not label a surrogate as CMPI or claim a final reward reproduction until the missing sub-metric definitions and authoritative variant are supplied. | Top-level transcription supported; full reward unresolved. |

## Concrete build sequence

1. **Freeze traceability, not the science.** Keep `configs/specification/thesis_variants.yaml` and `docs/SPECIFICATION_CONFLICTS.md` unchanged as the verbatim variant register. This document records the working interpretation alongside them.
2. **Establish inputs and provenance.** Validate the actual ANHC source files, licences, split manifest, gold annotations, CKG/CST-Onto export, and tokenizer/model checkpoint. The required-input list is in `docs/IMPLEMENTATION_GAPS.md`.
3. **Make the candidate executable but gated.** Store the Chapter 5 schedule in a reconstruction profile marked non-historical and non-training-ready. Validate required evidence before any real run.
4. **Connect the model path.** Load the selected Arabic backbone and tokenizer from explicit local artifacts; integrate morphology, CSAM, CKG activation, and DAL with shape/mask contracts. Keep any non-thesis bridge or approximation named as an engineering variant.
5. **Implement training stages separately.** First supervised objective (including Eq. 4.16 terms whose labels/formulae exist), then GAN matching §5.6.2, then PPO matching §5.6.3/Table 5.9. Log phase transitions, checkpoints, seed, data digest, and hardware from actual runs.
6. **Unblock scientific reward terms.** Add only terms with complete equations and usable labels. Keep undefined CRAFT/CMPI terms unavailable rather than substituting similarly named metrics.
7. **Evaluate evidence in layers.** Run data/interface checks, synthetic smoke runs, licensed real-data runs, then agent-assisted critique. Agents may find errors or rank outputs; their ratings are not human-evaluation results. Collect the thesis’s human ratings separately before making human-study claims.
8. **Release a reproducible candidate.** Publish exact source/model/data identifiers and licences, config and checkpoint digests, manifests, logs, evaluation scripts, and limitations. Call it a full ANSG reconstruction only after the end-to-end training and evaluation evidence is independently auditable.

## Current repository evidence and limits

The repository already contains data/knowledge infrastructure, thesis variant transcription, model components, GAN/PPO prototypes, and evaluation interfaces. That is useful implementation work, but it does not establish that the reported 847M model was trained or that the reported thesis metrics were reproduced. The Chapter 5 result claims (including final CMPI and human ratings) remain reported thesis values until the corresponding artifacts and evaluation are available.

## Traceability pointers

- Dissertation: glossary CSAM entry (Chapter 0); Chapter 4 §§4.1.6, 4.3, 4.5 and Table 4.1; Chapter 5 Table 5.1, overview (printed p.132), §§5.6.1–5.6.3 and Table 5.9; Appendix A.2.6 and A.3.
- Repository transcription: `configs/specification/thesis_variants.yaml`.
- Conflict ledger: `docs/SPECIFICATION_CONFLICTS.md`.
- Missing source material and author decisions: `docs/IMPLEMENTATION_GAPS.md`.
- Model/data contract and ordered integration work: `docs/INTEGRATION_EXECUTION_PLAN.md`.


## Current data and checkpoint audit

At the audited `master` head `e053bef`, the repository contains a 20-record classical-Arabic collection (`data/corpora/classical_arabic/stories.jsonl`, 195,197 words). Its own build report says model-training rights have not been verified, the split is not story-independent, and its rows are source pages/cycles rather than nested-story annotations. It is therefore a useful research pilot corpus after rights and split review, but it is **not the thesis's ANHC** and must not be used to claim the Chapter 5 run.

The repository readiness documentation also reports no bundled pretrained Arabic causal-LM checkpoint, no trained CSAM/DAL weights, and no real CKG/CST-Onto export. The current data examples and CKG fixtures are explicitly synthetic. The immediate technical gate is consequently source eligibility and provenance, followed by the real model/tokenizer and ontology artifacts; launching a thesis-scale training run before those checks would produce an untraceable variant.
