# Results comparison — thesis value vs reproduced value (2026-11-14)

**Bottom line:**
- **No Chapter 6 experimental result has been reproduced.** The data (ANHC, test set), the trained weights, the CRAFT implementation and the original human ratings are unavailable: the cited DOIs and repositories return 404 (DISCREPANCY_REPORT D-ART-1). CMPI has no defining equation in the thesis (C-EVAL-2).
- Nothing in this table was produced by training or evaluating a model. No score was tuned or estimated to approach a thesis value.
- Values in the "Recomputed" rows are **arithmetic from the thesis's own reported inputs**. They test internal consistency, not experimental reproduction (`evidence/reproduction/reported_values_audit.json`).

| # | Quantity | Thesis value (ref) | Reproduced / recomputed | Difference | Explanation and evidence |
| --- | --- | --- | --- | --- | --- |
| 1 | ANSG CMPI | 0.92 (Table 6.4, p. 170; §6.1.4) | **not reproduced** | — | CMPI is undefined (C-EVAL-2). CRAFT code, weights and test set are unavailable (D-ART-1) |
| 2 | ANSG BLEU-4 | 35.8 (§6.1.6, p. 169) | **not reproduced** | — | Needs the 500-narrative test references and the trained model (both unavailable) |
| 3 | ALDA | 92.3% (Table 6.9, p. 180) | **not reproduced** | — | ALDA sub-measures lack a combining rule (C-EVAL-6). Dialect test data unavailable |
| 4 | Human overall rating | 4.5 / 5 (Table 6.7, p. 176) | **not reproduced** | — | Requires a new human study. The original ratings are "available upon request" and were not obtained |
| 5 | Fleiss κ (human) | 0.82 (§6.1.3, p. 165) | **not reproduced** | — | Same as row 4 |
| 6 | Ablation, all rows | Table 6.9 (p. 180) | **not reproduced** | — | Same as rows 1–3 |
| 7 | Gap closure (Eq. 6.1) | 92.3% (p. 166) | recomputed **92.31%** (GPT-4 = 0.68) / **90.48%** (GPT-4 = 0.73 per Table 6.4) | +0.01 / −1.82 | Two different GPT-4 CMPI values in the thesis (C-EVAL-11) |
| 8 | ANSG–MM-GTA improvement row | Table 6.4 | all six recomputed within rounding | ≤ 0.05 | consistent |
| 9 | Cohen's d, ANSG vs GPT-4, CMPI | 1.89 (Table 6.3) / 1.58 (Table 6.5) / 2.14 (summary) | 1.58 = t·√(2/200) from Table 6.5's t = 15.8 | — | Consistent only with n = 200 per system. With the n = 500 test set, d = 1.00 |
| 10 | Table 6.5 ΔCMPI (ArabicGPT, GPT-4, AraT5) | +0.38, +0.24, +0.21 | 0.24, 0.19, 0.18 from Table 6.4 | −0.14, −0.05, −0.03 | Table 6.5 does not use Table 6.4's baseline CMPI values |
| 11 | Table 6.7 d vs MM-GTA, GPT-4 (overall) | 0.67, 1.23 | 3.00, 4.31 if ± is SD | +2.33, +3.08 | Spread type unstated |
| 12 | Table 6.6 "weighted average" consistency | 84.3 | 84.80 (n-weighted); 84.34 (unweighted) | +0.50 / +0.04 | The printed value is an unweighted mean |
| 13 | Ablation percentages (Table 6.9) | −15.2 / −15.1 / −33.7 / −8.7 / −8.4 / −37.0 | all recomputed within rounding | ≤ 0.05 | consistent arithmetic; Figure 6.4's Morph bar (8.4) is the BLEU, not the CMPI, reduction (5.4) |
| 13b | AHP weights from the printed comparison matrices | 0.35 / 0.30 / 0.25 / 0.10 (A.10, A.23) | Table A.10 → 0.350 / 0.300 / 0.249 / 0.100 (CR ≈ 0.0001 vs stated 0.023); second matrix → 0.37 / 0.31 / 0.22 / 0.10 | second matrix: +0.02 / +0.01 / −0.03 / 0.00 | `evaluation/ahp.py`; `tests/unit/test_thesis_computations.py`; C-REWARD-4 |
| 14 | Eq. 4.30 reward example | 0.854 (p. 119) | 0.85455 (`training.cultural_reward`; `tests/test_thesis_fidelity.py`) | +0.0005 | printed value truncated; formula reproduced |
| 15 | ANHC size | 70,404,829 tokens (A.20, p. 289) | available corpus: 195,197 words | about 0.28% | not the same corpus ([CORPUS_EQUIVALENCE.md](CORPUS_EQUIVALENCE.md)) |
| 16 | Kalīla wa-Dimna in ANHC | 3.1M tokens (§3.2.2, p. 57) | 15 available chapters: 33,168 words | about 1.1% | claim unexplained (D-DATA-4) |
| 17 | MADAR | "2.1M", "26 cities" (p. 55) | official: 25 cities + MSA; ≈ 0.73M words (estimate) | — | D-RES-1. The thesis DIA classifier (94.3% on MADAR) is not reproduced |
| 18 | ANHC narratives | 55,600 (A.20) vs 5,247 (§5.4.4) | — | — | internal conflict (D-DATA-1) |

## What *is* reproduced (formula level, from thesis inputs)

- These calculations follow the thesis exactly and are pinned by tests:
  - CRD Eq. 3.1 (lexicon not supplied);
  - cultural weight Eq. 4.2 with the §4.2.1 worked examples;
  - AHP weight derivation (the method is reproduced; Table A.10's weights match, its CR and the second matrix do not; row 13b);
  - reward aggregation Eq. 4.29 and Eq. 4.30;
  - gap closure Eq. 6.1.

  None of these is an experimental result.
- Component-level status of CSAM, DAL, the generator and RL is in `docs/THESIS_TO_CODE_TRACEABILITY.md`.

## What would change this table

The items below are listed in [THESIS_RESULTS_TRACEABILITY.md](THESIS_RESULTS_TRACEABILITY.md):
- ANHC (or at least the test set) with its split;
- the CRAFT implementation and the CMPI definition;
- the trained checkpoints;
- the baselines' configurations or outputs;
- the human-evaluation protocol materials.

With them, `evaluate.py`-style runs could fill the "Reproduced" column with measured values and their differences.
