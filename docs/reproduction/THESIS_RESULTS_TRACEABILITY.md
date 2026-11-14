# Chapter 6 (and related) results: what each needs, and its reproduction status

Component-level traceability (ANHC, CKG, CSAM, DAL, generator, RL, CRAFT) is in `docs/THESIS_TO_CODE_TRACEABILITY.md`. This table covers each **reported experiment or metric**.

Pages come from the PDF footers of the supplied thesis. The status vocabulary is the same as in that file. "Blocked by" names the unavailable inputs, which are listed in `data/sources/source_registry.json`.

## Common inputs of every Chapter 6 result (none available)

| Input | Thesis reference | Status |
| --- | --- | --- |
| Test set: 500 narratives (Table A.12, p. 283; §6.1.2, p. 163) vs 525 (§5.4.4, p. 143) | ANHC test split | **unavailable** (D-ART-1); size conflict C-EVAL-12 |
| ANHC training data and split: TF-IDF + DBSCAN ε = 0.3, min_samples 2, seed 42 | §5.4.4, p. 143 | **unavailable**; metric unstated (C-SPLIT-1) |
| Preprocessing: normalisation per App. A.8; SAHDA | A.8 (p. 276); §3.4 (p. 60) | A.8 implemented as a profile; SAHDA not implemented |
| Model configuration | §4.1.6 (p. 89); Table 5.1 (p. 134); A.3 (p. 270–271) | three conflicting backbones (C-MODEL-1); 1280/12 not divisible |
| Seeds | primary 42; 123, 456, 789 and 1011 for consistency (§5.2.1, p. 135) | recorded; nothing to run |
| Software | Python 3.10.12, PyTorch 2.0.1+cu118, Transformers 4.30.2, CAMeL Tools 1.5.2 (§5.2.1, p. 135) | recorded (D-RES-2) |
| Trained checkpoints (phases 1–4) | Table 9.6 (p. 254) | **unavailable** (404) |
| CRAFT implementation: CMPI, ALDA, CulturalScore | Table 9.6 (p. 254); Ch. 6 | **unavailable** (404); CMPI undefined (C-EVAL-2) |
| Baselines: ArabicGPT, AraBART, GPT-4, NarrativeMind, MM-GTA, AraT5-Cultural, CULTURA | Tables 6.1–6.2 (p. 162–164); A.5 (p. 272–273) | not implemented; configurations conflict (C-EVAL-9) |

## Per result

| Result | Reference | Metric definition | Data and annotations | Evaluation code | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Main comparison: CMPI, BLEU-4, human | §6.1.4 (p. 166); Table 6.4 (p. 170) | CMPI: none (C-EVAL-2); BLEU-4: standard | test set; system outputs | `craft-eval` (404) | UNVERIFIABLE | reported_values_audit: GAP-CLOSURE-*, ADVANCE-*, T6.4-* |
| Effect sizes | Table 6.3 (p. 168) | Cohen's d; the sample unit is not stated | per-item scores | — | UNVERIFIABLE; conflicts with Table 6.5 | T6.5-VS-T6.3-D-GPT4 |
| Significance (Bonferroni) | Table 6.5 (p. 171); A.12 (p. 281) | t-test; 9 vs 6 comparisons (C-EVAL-3) | per-item scores | `evaluation/statistics.py` (method only) | PARTIAL (method), results UNVERIFIABLE | T6.5-DELTA-*, T6.5-D-FROM-T-* |
| Dialect performance (ALDA) | Table 6.6 (p. 173) | four sub-measures, no combining rule (C-EVAL-6) | per-dialect test subsets (Σn = 900 vs 500) | — | UNVERIFIABLE | T6.6-*, A12-VS-T6.6-GULF-N |
| Human evaluation | §6.1.3 (p. 165); Table 6.7 (p. 176); A.24 (p. 295) | 1–5 Likert; Fleiss κ 0.82; ICC 0.87 | 5 experts × 200 narratives/system | — | UNVERIFIABLE (ratings unavailable) | T6.7-* |
| Ablation | Table 6.9 (p. 180); Figure 6.4 | CMPI, BLEU-4, ALDA, human | test set n = 500, seed 42 | `ablation.py` (not found; D-ART-2) | UNVERIFIABLE | T6.9-*, F6.4-* |
| Component-to-gain mapping | Table 6.10 (p. 181) | derived from the ablation | — | — | UNVERIFIABLE | — |
| Failure cases and error taxonomy | Tables 6.11–6.12 (p. 182–183) | manual categorisation (n = 500) | error annotations | — | UNVERIFIABLE | — |
| Fluency–authenticity trade-off | §6.1.6 (p. 169) | BLEU d = −0.71 (vs −0.67 in §6.1.4, C-EVAL-10) | system outputs | — | UNVERIFIABLE | — |
| CRD of the corpus | §3.6.1, Eq. 3.1 (p. 69–70) | defined | 2,847-entry lexicon (not supplied) | `evaluation/corpus_metrics.py` | PARTIAL (formula MATCH) | `test_crd.py` |
| AHP reward weights | §4.5.2 (p. 117); A.11 and A.23 (p. 280, 293–294) | defined | expert matrices (printed) | `evaluation/ahp.py` | PARTIAL: the Table A.10 matrix reproduces its stated weights, but CR ≈ 0.0001 vs the stated 0.023; the second printed matrix gives 0.37/0.31/0.22/0.10, not 0.35/0.30/0.25/0.10 (C-REWARD-4) | `test_thesis_computations.py` |
| Reward example | Eq. 4.30 (p. 119) | defined | printed inputs | `training.cultural_reward` | MATCH | `test_thesis_fidelity.py` |
| Dialect classifier (DIA) | §4.2.4 (p. 94); §5.4.3 (p. 140) | CNN over character trigrams; 94.3% on MADAR | MADAR (licensed, not committed) | — | NOT IMPLEMENTED (model unavailable) | registry `madar`; `evaluation/dialect_id.py` (experimental, not thesis) |
| Corpus statistics | Tables 3.2–3.3 (p. 56); A.20 (p. 289) | counts | ANHC | — | UNVERIFIABLE; internal conflicts | D-DATA-1/2, T3.*, A20-* |

## Order of work to make a result reproducible

1. **Obtain ANHC (or the test set) and the split file.** This unblocks rows 1–9 at the data level. Record in the registry: the corpus hash, its licence and the permission status.
2. **Obtain or define CMPI and the CRAFT implementation.** The definition must be the author's; it cannot be invented here.
3. **Obtain the checkpoints**, or retrain with the thesis configuration once C-MODEL-1 is resolved.
4. **Run the evaluation with seed 42 and the four consistency seeds.** Report measured values beside the thesis values in RESULTS_COMPARISON.md, including differences and diagnosis, without tuning on the test set.
