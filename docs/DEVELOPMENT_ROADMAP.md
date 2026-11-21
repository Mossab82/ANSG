# Development roadmap

Each phase ends with tests that run on CPU and a PR. No expensive training
happens before Phase 3, and none before the specification decisions in
`docs/IMPLEMENTATION_GAPS.md §C` are made.

## Phase 1 — Data and knowledge infrastructure (this PR)
- [x] Repository audit (`docs/REPOSITORY_AUDIT.md`)
- [x] Package skeleton, YAML configs, secure config loading, run manifests
- [x] Offset-preserving Arabic normalisation; segmentation
- [x] Nights ingestion (tales, nights, formulae) and CLI pipeline
- [x] ANHC record schema, validation, annotation import, export policy
- [x] Deterministic, leakage-safe splits (incl. predefined partitions)
- [x] ArabicVerbNet parser; ArabicNameNet configurable adapter
- [x] CKG data model + CST-Onto loader + CSAM-facing index/adjacency
- [x] CRAFT interfaces and specification template
- [x] Transcribe thesis specifications (Ch. 3–6, App. A) with references; conflict register (`SPECIFICATION_CONFLICTS.md`)
- [x] Implement fully specified procedures: Eq. 3.1, 4.2, 4.20/A.4, 6.1, A.1, AHP, §5.4.4 clustering, A.8 profile, Table 5.4 formulae

## Phase 1c — Author decisions and real-data validation
- Author resolves `IMPLEMENTATION_GAPS.md §C` (backbone, protocol, reward, CSAM form, normalisation).
- Author supplies missing definitions (`§A`: CMPI, codebook, lexicons) where they exist.
- Run ingestion on the real Nights witnesses with DBSCAN clustering; tune patterns; compare counts with Tables 3.2–3.3.
- Load real ArabicVerbNet / ArabicNameNet / CKG / CST-Onto; fill `declared` counts; check `count_report`.
- Import gold annotations; compute label distributions per split.
- Exit criterion: dataset statistics either match the thesis or the differences are documented.

## Phase 2 — Model components (CPU-testable)
- Backbone loading for each reported variant (config-selectable), tokenizer audit on Arabic.
- CSAM module: shape tests, parameter-count tests against each chapter's numbers (reporting mismatches rather than forcing them).
- DAL/LoRA, morphological conditioning.
- Schema activation per thesis definition.

This project-roadmap phase is **not** the same as a training-stage number. The
selected training protocol is recorded as decision D7: its Stage 1 is SFT, Stage
2 is GAN, and Stage 3 is PPO. See `docs/AUTHOR_DECISIONS.md` and
`configs/experiments/ansg_ch5_selected.yaml`.

## Phase 3 — Training
- Implement the selected Chapter 5 SFT → GAN → PPO schedule; retain other thesis variants as source evidence.
- Run CPU/tiny-config dry-runs and record experiment manifests.
- GPU runs only after the selected data, checkpoint, objective, reward, and compute gates are satisfied; never backfill historical run claims.

## Phase 4 — Reward and RL
- Reward terms per Ch. 5 / App. A.3 (both variants configurable); RL loop.

## Phase 5 — CRAFT and human evaluation
- Implement metrics with status `specified`; unit tests from the test specifications; human evaluation tooling.
