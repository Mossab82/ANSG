# Implementation gaps and required inputs

Status after Phase 1b (2026-11-13): the dissertation has been read and transcribed
(Chapters 3–6, Appendix A; glossary; Appendix B where relevant). What remains open
falls into four groups.

## A. Information the dissertation does not contain

These cannot be transcribed because they are absent or illegible in the thesis.
The author must supply them (e.g. from the original code, data or notes) or decide
how to proceed.

| # | Missing | Blocks | Ref |
| --- | --- | --- | --- |
| A1 | MOTIF codebook (87 archetypes / 847 concepts with identifiers); the 47 boundary cases; calibration gold set — cited as App. A.7 but not there | MOTIF label validation, CSAM supervision | C-ANHC-3 |
| A2 | CMPI definition (equation, sub-scores) | CRAFT, reward, every reported result | C-EVAL-2 |
| A3 | Definitions of CulturalScore, ALDA, MFA, PCI, Table 6.4 dimensions, reward sub-metrics (virtue, rhetorical, register, coreference, temporal, causal) | CRAFT, reward | C-EVAL-1, C-REWARD-5 |
| A4 | R_i (rhetorical bias) computation; context-embedding construction for activation; DAL compatibility rules; dialogue detector | CSAM, DAL | C-CSAM-5/6, C-DAL-2 |
| A5 | Cultural marker lexicon (2,847 entries); formulaic database (437); moral ontology (127); Cultural Motif Database (698); Historical Lexicon (3,247) | CRD, formula detection, Stage 4 | §3.6.1, §5.4.2, §4.2.5 |
| A6 | SAHDA rules (45 FSTs) and models | thesis preprocessing | §3.4 |
| A7 | Arabic text of all examples (export corrupted) | verification of reconstructed formulae | T-1 |

## B. Data/resources from the author

| # | Resource | Location (git-ignored) | Notes |
| --- | --- | --- | --- |
| B1 | Nights witnesses: Mahdi (1984), Bulaq (1835), Calcutta II (1839–42) files + licences | `data/private/alf_layla/` | one `sources` entry per witness; enable `cluster_dbscan` |
| B2 | Other ANHC sources and the ANHC release (5,247 or 55,600 narratives) | `data/private/` | C-ANHC-1, C-ANHC-7 |
| B3 | Gold DIA / MOTIF / MORAL annotations | `data/private/anhc/` | importer exists |
| B4 | Original train/val/test partition (4,197/525/525) and the 500-narrative test set | `data/private/anhc/` | `strategy: predefined` |
| B5 | CST-Onto (OWL) and CKG export (12,341 segments / 898 tags) | `data/private/ckg/` | count checks ready |
| B6 | Original code (`mossab-ansg/ansg-framework`), checkpoints, configs | — | identifiers in thesis cannot be valid (C-REPRO-1) |
| B7 | ArabicVerbNet / ArabicNameNet files — only if they are to be used, since the thesis does not use them for ANSG | `data/private/` | C-RES-1 |

## C. Decisions required from the author

| # | Decision | Options in the thesis |
| --- | --- | --- |
| C1 | Reference backbone | GPT-2 Medium 355M / 774M base / AraGPT2-large (C-MODEL-1) |
| C2 | Tokenizer | backbone BPE vs SentencePiece-32k root-aware (C-MODEL-3) |
| C3 | Training protocol | 2-phase / 3-phase (Table 4.1) / 3-phase (§4.1.6) / 3-phase with GAN / 4-phase (C-TRAIN-1) |
| C4 | Reward | five definitions (C-REWARD-1); which AHP matrix is authoritative (C-REWARD-4) |
| C5 | CSAM formulation | Eq. 4.12–4.13 vs §4.3.6 vs A.25 code; head partition (C-CSAM-2…4) |
| C6 | Activation | τ 0.5 / 768-d vs 0.75 / 256-d / max 50 (C-CSAM-5) |
| C7 | Normalisation of model input | A.8 (keep diacritics, NFKC, alef) vs Ch. 5 dual stream (C-NORM-1) |
| C8 | Human evaluation protocol | §6.1.3 vs A.13.1 vs A.24 (C-EVAL-8) |
| C9 | Software stack | Ch. 5 (PyTorch 2.0.1 …) vs App. A.2 (PyTorch 1.12 …) or current versions with documented deviation (C-ENV-1) |
| C10 | How to report thesis inconsistencies (errata, ablation variants) | — |
| C11 | Git-history purge of leaked credentials; code licence; moving legacy scripts to `legacy/` | — |

## D. Known limitations of the current implementation

* Tale nesting is not inferred; titles are regex-detected (warnings reported).
* No morphological analyser integrated (C-MORPH-1 decision pending).
* DBSCAN clustering uses an assumed cosine metric and scikit-learn TF-IDF defaults (C-SPLIT-1).
* CRD needs a lexicon and an explicit multi-word policy.
* `CosineThresholdActivator` needs supplied embeddings; no OWL2Vec* training.
* Reconstructed Table 5.4 formulae are unverified (T-1).
* Legacy root scripts remain untouched and non-functional (`REPOSITORY_AUDIT.md`).
