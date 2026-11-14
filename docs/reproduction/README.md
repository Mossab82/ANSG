# ANSG-Original reproduction — status and evidence

| Question | Answer | Where |
| --- | --- | --- |
| Are thesis results reproduced? | **No.** Data, weights, CRAFT code and human ratings are unavailable (cited DOIs and repositories return 404), and CMPI is undefined | [RESULTS_COMPARISON.md](RESULTS_COMPARISON.md) |
| What does each Chapter 6 result need? | per-result inputs and status | [THESIS_RESULTS_TRACEABILITY.md](THESIS_RESULTS_TRACEABILITY.md) |
| Which components and equations are implemented? | component matrix. Correction: its AHP row (6.5, "MATCH") is PARTIAL; see DISCREPANCY_REPORT D-RPT and C-REWARD-4 | [../THESIS_TO_CODE_TRACEABILITY.md](../THESIS_TO_CODE_TRACEABILITY.md) |
| Where do thesis claims conflict with evidence or with each other? | 18 arithmetic inconsistencies among reported values; artefact, corpus and MADAR discrepancies | [DISCREPANCY_REPORT.md](DISCREPANCY_REPORT.md) |
| Is the repository corpus the ANHC? | **No**: about 0.28% of the claimed size, different sources, no dialects or annotations | [CORPUS_EQUIVALENCE.md](CORPUS_EQUIVALENCE.md) |
| Which resources may be used, and how? | source and licence registry, rights gate | `data/sources/source_registry.json`, [RIGHTS_GATE.md](RIGHTS_GATE.md) |
| How do I run everything from a fresh clone? | pinned install, tests, preflight, audits | [RUN_GUIDE.md](RUN_GUIDE.md) |
| Kaggle pilot? | four failed runs, `KeyError: 'night'` fixed, GPU gated on rights | [KAGGLE_PILOT_STATUS.md](KAGGLE_PILOT_STATUS.md) |

## Three categories, kept apart

- **Reproduced:** formula-level computations from thesis inputs only (Eq. 3.1, 4.2, 4.29/4.30, Eq. 6.1, and the AHP method), each pinned by a test. For AHP, only Table A.10's weights reproduce: its CR and the second matrix do not.
- **Unavailable:** every experimental result of Chapter 6, together with the data and artefacts it needs.
- **New or supplementary experiments:**
  - the Wikisource classical corpus and its LoRA pilot;
  - the optional dialect-ID interface;
  - ANSG-HNC (separate PR #27, disabled by default).

  None of these is reported as a thesis result.
