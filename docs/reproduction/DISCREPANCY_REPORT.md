# Discrepancy report — thesis claims vs evidence (2026-11-14)

**Scope.**
- Claims in the dissertation (`Tesis_Ibrahim`, the supplied Markdown conversion of the PDF; pages from its PDF footers) that conflict with:
  - primary external sources;
  - the thesis itself;
  - data available to this repository.
- Each discrepancy is **recorded, not resolved**. Fixing the numbers is the author's decision.
- Engineering conflicts between chapters (C-xxx codes) are in `docs/SPECIFICATION_CONFLICTS.md`; this report adds what this audit established.

**Evidence files:**

| File | Content |
| --- | --- |
| `evidence/external/external_evidence.json` | CI run [43564658606](https://github.com/Mossab82/ANSG/actions/runs/43564658606) (earlier [43564395466](https://github.com/Mossab82/ANSG/actions/runs/43564395466)) |
| `evidence/reproduction/reported_values_audit.json` | Arithmetic audit of reported values |
| `evidence/corpus/classical_corpus_audit.json` | Corpus measurements |

## D-ART — thesis artefacts that cannot be located

**D-ART-1 — DOIs and repositories in Table 9.6 (p. 254) and App. A.20.2 (p. 290) do not resolve.**

| Item | Result | Evidence |
| --- | --- | --- |
| 5 Zenodo records (ANHC, weights, CRAFT, training pipeline, CKG) | Zenodo API → 404; doi.org → 404 | CI run 43564658606 |
| GitHub `mossab-ansg` account | 404 (GitHub API and github.com) | CI run 43564658606 |
| Its 7 repositories (`anhc-test-set`, `sahda`, `ansg-framework`, `ansg-weights`, `craft-eval`, `ansg-train`, `ckg-arabic`) | 404 | CI run 43564658606 |

- Records 8847291, 8847342, 8847356, 8847367 and 8847378 all returned 404.
- The commit `a7b3c9d` (§5.2.1, p. 135) cannot be checked.
- The thesis also gives **conflicting access statements** for ANHC:
  - Table 9.6: "controlled access", via the DOI;
  - §3.7.1, p. 75: "10% sample … Zenodo DOI [to be assigned]", with the full corpus "available upon request";
  - A.20.2: "Full corpus: Zenodo archive".
- **Consequence:** neither the original data, nor the weights, nor the CRAFT implementation is available. Every Chapter 6 number is therefore unreproducible from public artefacts.

**D-ART-2 — Reproduction-guide references do not match Chapter 6.**
- App. A.26, Table A.23 (p. 302) maps:
  - "Table 6.2 (System Comparison)" (Table 6.2 is actually the baseline configurations);
  - "Table 6.4 (Dialectal Results)" (dialect results are Table 6.6);
  - "Table 6.8 (Ablation)" (the ablation is Table 6.9);
  - "Figure 6.3 (Attention Visualization)" (Figure 6.3 is the dialect consistency chart).
- The scripts it names (`evaluate_dialects.py`, `ablation.py`, `visualize_attention.py`) do not exist in this repository. They are attributed to the unreachable `ansg-framework` repository.

## D-DATA — ANHC

**D-DATA-1 — Number of narratives:** 55,600 (A.20 Table A.17, p. 289) vs 5,247 (§5.4.4, p. 143; 4,197 / 525 / 525 split).
- 55,600 × 1,266 tokens = 70.39M agrees with the 70.4M token count.
- 5,247 × 1,266 = 6.64M does not.

**D-DATA-2 — Dialect distribution:** Table 3.3 (p. 56) vs A.20 Table A.17 (p. 289).

| Variety | Table 3.3 | A.20 |
| --- | --- | --- |
| MSA | 40.8% | 28.3% |
| Egyptian | 32.4% | 22.1% |
| Levantine | 14.5% | 18.7% |
| Gulf | 7.2% | 14.9% |
| Maghrebi | 5.1% | 11.2% |
| Mixed / code-switching | — | 4.8% |

**D-DATA-3 — Alf Layla size.**
- §3.2.2 (p. 57) says the Mahdi critical edition "provides 8.2M tokens".
- The five Wikisource Nights pages in this repository cover 147 of the 1,001 nights and contain 162,029 words. That is about 2% of 8.2M.
- From general knowledge, not measured here: Mahdi's edition covers only the oldest core of the Nights, a few hundred nights, so 8.2M tokens is implausible for one edition. This needs the author's explanation, for example several witnesses (§3.2.4), a different unit, or a typo.

**D-DATA-4 — Kalīla wa-Dimna size.**
- §3.2.2 claims 3.1M tokens.
- The 15 chapters available here contain 33,168 words, about 1.1% of 3.1M.
- A complete Kalīla wa-Dimna is of the same order as these chapters, not ~100× larger.
- **Unexplained.**

**D-DATA-5 — Corpus statistics vs splits.**
- A.20 reports 4,237,451 sentences.
- §5.4.4 reports a narrative-level split; the cluster method is in `SPECIFICATION_CONFLICTS.md` C-SPLIT-1.
- The test set is 500 narratives in §6.1.2 and Table A.12 (p. 283), but 525 in §5.4.4 (C-EVAL-12).

**D-DATA-6 — The repository's classical corpus is not ANHC.** See [CORPUS_EQUIVALENCE.md](CORPUS_EQUIVALENCE.md).

## D-RES — External resources as described in the thesis

**D-RES-1 — MADAR.**
- **Thesis description:** the comparison table in §3.1.5 (p. 55) lists "MADAR 2.1M, Dialectal labels, 26 cities". Chapter 2 Table 2.1 (p. 25) describes "the MADAR corpus (25,000 sentences spanning MSA and four major dialects)".
- **Official documentation** (CAMeL Lab download page; LREC 2018 paper; CI-fetched snippets in `evidence/external/external_evidence.json`):
  - 25 city dialects plus MSA (also English and French);
  - **Corpus-26** = 2,000 BTEC sentences × (25 cities + MSA);
  - **Corpus-6** = 12,000 sentences × (5 cities + MSA);
  - average sentence length about 6.5 words.

| Thesis statement | Status |
| --- | --- |
| "26 cities" | **Incorrect**. It is 25 cities + MSA; "Corpus-26" counts varieties, not cities |
| "2.1M" | No unit is given and no official figure matches. Estimate from the official numbers: (2,000 × 26 + 10,000 × 6) sentences × 6.5 words ≈ **0.73M words** (`MADAR-SIZE-WORDS-ESTIMATE`). **Unverified; source of "2.1M" unknown** |
| "25,000 sentences … MSA and four major dialects" (Ch. 2) | Inconsistent with both the official sizes and the §3.1.5 entry |

- **Licence** (official page): internal research and evaluation only; further distribution, modification or commercial use requires CMU's permission. Recorded in `data/sources/source_registry.json`; no MADAR file is committed.

## D-RES-2 — Software environment

- §5.2.1 (p. 135) declares Python 3.10.12, PyTorch 2.0.1+cu118, Transformers 4.30.2, CAMeL Tools 1.5.2 and SentencePiece 0.1.99.
- The pinned environment verified here is Python 3.11, PyTorch 2.15 and Transformers 5.23 (`requirements/constraints-py311.txt`). That environment checks the software path only. **It is not the thesis environment, and no thesis result is run in it.**

## D-RPT — Internal inconsistencies of reported results

From `evidence/reproduction/reported_values_audit.json`, which recomputes 46 values from the thesis's own inputs: 28 are consistent and 18 are inconsistent. The material ones:

| Check | Reported | Recomputed from the thesis's own inputs |
| --- | --- | --- |
| Gap closure Eq. 6.1 (p. 166) with Table 6.4's GPT-4 CMPI 0.73 | 92.3% | 90.5% (the text and Eq. 6.1 use 0.68) |
| Table 6.5 ΔCMPI vs Table 6.4 | ArabicGPT +0.38; GPT-4 +0.24; AraT5 +0.21 | 0.24; 0.19; 0.18 (NarrativeMind and MM-GTA agree) |
| Table 6.5 Cohen's d vs t | 1.58 | Equals t·√(2/200), so it is consistent only if n = 200 per system (the human-evaluation sample), not with the n = 500 test set |
| Table 6.3 vs Table 6.5 d, ANSG vs GPT-4 (CMPI) | 1.89 | 1.58 in Table 6.5 (2.14 in the executive summary) |
| Table 6.7 d from means ± spread (if ± is SD) | 0.67 (MM-GTA); 1.23 (GPT-4) | 3.0; 4.31. The table does not say whether ± is SD or SE |
| Table 6.6 "Weighted Average" consistency | 84.3 | n-weighted mean is 84.8; 84.3 is the *unweighted* mean |
| Table 6.6 per-dialect n vs test set | Σn = 900; Gulf n = 150 | Test set 500 (Table A.12); Gulf 75 |
| AHP (A.10; second matrix, A.23) — `tests/unit/test_thesis_computations.py` | weights 0.35/0.30/0.25/0.10; CR 0.023 and 0.078 | A.10 weights match but CR ≈ 0.0001; second matrix gives 0.37/0.31/0.22/0.10 with CR ≈ 0.004 (C-REWARD-4) |
| Figure 6.4 "Morph Pipeline" CMPI bar | 8.4 | 5.4% CMPI reduction (8.4% is the BLEU reduction) |
| A.20 vs §5.4.4 narratives; A.20 vs Table 3.3 dialect shares | — | see D-DATA-1 and D-DATA-2 |

The headline arithmetic is consistent: gap closure with 0.68, the Table 6.4 percentage row, the Table 6.9 ablation percentages, and Eq. 4.30.

## D-PILOT — Repository pilot (supplementary experiment, not thesis)

**D-PILOT-1 — The split v1 test page leaked into training.**
- `kalila_dimna-019` (test) has 98.48% of its 8-grams in train page `kalila_dimna-007`.
- **Fix:** split v2, derived deterministically (`scripts/reproduction/derive_pilot_split.py`).
- Split v2 has its own limitation: its test set is only 839 words (two short chapters). Any test CE from it is a weak diagnostic.

**D-PILOT-2 — The Sinbad exclusion was unsupported.** The Sinbad page shares 0.49% of 8-grams with the other Nights pages; split v2 includes it.

**D-PILOT-3 — Tales cross page boundaries.**
- Nights part 1 ends mid-tale and part 2 opens with its continuation (381 words; 3,711 words before the first title of part 3).
- Night sequences regress: 88→87 in part 3, 553→531 on the Sinbad page.
- **Consequence:** page groups are not story-independent, so all Nights pages stay in one split.

**D-PILOT-4 — `KeyError: 'night'`** (run 43561944084). Fixed and preflighted; see [KAGGLE_PILOT_STATUS.md](KAGGLE_PILOT_STATUS.md).
