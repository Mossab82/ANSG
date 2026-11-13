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
- [ ] Transcribe thesis specifications (blocked: dissertation not available)

## Phase 1b — Specification transcription and real-data validation
- Transcribe items A1–A10 into `EXPERIMENTAL_SPECIFICATIONS.md`, configs and `CRAFT_EVALUATION_SPECIFICATION.md`, each with references.
- Run ingestion on the real Nights edition; tune patterns; report counts vs. thesis.
- Load real ArabicVerbNet / ArabicNameNet / CKG / CST-Onto; fill `declared` counts; check `count_report`.
- Import gold annotations; compute label distributions per split.
- Exit criterion: dataset statistics either match the thesis or the differences are documented.

## Phase 2 — Model components (CPU-testable)
- Backbone loading for each reported variant (config-selectable), tokenizer audit on Arabic.
- CSAM module: shape tests, parameter-count tests against each chapter's numbers (reporting mismatches rather than forcing them).
- DAL/LoRA, morphological conditioning.
- Schema activation per thesis definition.

## Phase 3 — Training
- Implement the staged training schedule(s); dry-runs on tiny configs; experiment logging.
- Only then: GPU runs, with manifests and real dates.

## Phase 4 — Reward and RL
- Reward terms per Ch. 5 / App. A.3 (both variants configurable); RL loop.

## Phase 5 — CRAFT and human evaluation
- Implement metrics with status `specified`; unit tests from the test specifications; human evaluation tooling.
