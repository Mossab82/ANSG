# ANSG author implementation decisions

This log records choices made for the current implementation. A selected
implementation path is not evidence that the thesis's original experiment used
that exact variant. Source statements remain preserved in
`configs/specification/thesis_variants.yaml` and
`docs/SPECIFICATION_CONFLICTS.md`.

## D7 — Training sequence

- **Decision:** Use the detailed Chapter 5 sequence for the next implementation: supervised fine-tuning (SFT), adversarial GAN training, then PPO.
- **Training-stage numbering:** SFT is stage 1; GAN is stage 2; PPO is stage 3. A base pretrained checkpoint is required before SFT.
- **Project-roadmap numbering:** “Phase 2” remains model-component implementation; “Phase 3” is training. These roadmap phases are not the training-stage numbers.
- **Source:** Chapter 5 §5.1.3 / §5.6 as transcribed under `training_protocol_variants.ch5_5_1_3` in `configs/specification/thesis_variants.yaml`.
- **Rationale:** This is the user's selected detailed account of the final training workflow, including GAN between supervised training and reinforcement learning.
- **Status:** Selected for implementation; not executed.
- **Scope:** Resolves which sequence to implement, not the historical conflict across Chapters 3–4, Chapter 5, and Appendix A. It does not choose the backbone, tokenizer, reward definition, or corpus provenance.
- **Execution gates:** Rights-cleared/frozen training data, provenance-verified base checkpoint, approved supervised targets, defined discriminator inputs/objective, validated PPO reward, compute provenance, and an operationally defined CMPI stopping metric.
- **Configuration:** `configs/experiments/ansg_ch5_selected.yaml`.

The currently available ANHC staging data and dialect inventory do not meet these
execution gates. The selected protocol must not be launched as thesis
reproduction until the gates are met; small engineering dry-runs may use toy
data and must be labeled as such.
