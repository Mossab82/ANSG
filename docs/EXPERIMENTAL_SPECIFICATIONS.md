# Experimental specifications (transcribed from the dissertation)

*Transcribed 2026-11-13 from the author-supplied Markdown export of the dissertation.
Section/table/equation numbers are the thesis'. Where the thesis gives several
values, all are listed with their location and a conflict id
(`SPECIFICATION_CONFLICTS.md`). Machine-readable form:
`configs/specification/thesis_variants.yaml`. "—" = not stated in the thesis.*

> These are the specifications **as written in the dissertation**. Reproducing them
> in this repository produces **new** runs with new dates; no value below is a
> record of an experiment executed by this reimplementation.

## 1. Data — ANHC (Ch. 3, App. A.7, A.17, A.20; Ch. 5 §5.4)

| Item | Value | Location |
| --- | --- | --- |
| Corpus size | 70.4M tokens (70,404,829), 847K segments; 23M tokens tri-annotated | §3.1, Tables 3.2–3.3, A.17 |
| Narratives | 5,247 (§5.4.4) **or** 55,600 (Table A.17) | C-ANHC-1 |
| Temporal strata | Classical 24.6M (35.0%), Modern 18.3M (26.0%), Contemporary 27.5M (39.1%) | Table 3.2; C-ANHC-9 |
| Dialect strata | MSA 28.7M (40.8%), EGY 22.8M (32.4%), LEV 10.2M (14.5%), GULF 5.1M (7.2%), MAG 3.6M (5.1%) | Table 3.3; C-ANHC-5 |
| Nights sources | Mahdi critical edition (1984), 8.2M tokens; parallel Bulaq (1835) and Calcutta II (1839–42) witnesses | §3.2.2, §3.2.4; C-ANHC-8 |
| Other classical sources | Kalīla wa-Dimna 3.1M; adab (Maqāmāt al-Ḥarīrī, Kitāb al-Aghānī, al-Jāḥiẓ) 13.3M | §3.2.2 |
| DIA layer | 5 classes (MSA, Egyptian, Levantine, Gulf, Maghrebi), segment level | §3.5.1 |
| MOTIF layer | 87 CST-Onto archetypes (or 847 concepts / 437 categories) — codebook not in thesis | §3.5.2; C-ANHC-3 |
| MORAL layer | 8 classes: Justice, Wisdom, Courage, Temperance, Faith, Charity, Patience, Gratitude | §3.5.3; C-ANHC-4 |
| Annotation density | 2.4 annotations per segment | §3.5.3 |
| IAA (Ch. 3) | Cohen's κ / Fleiss' κ: DIA 0.891/0.876, MOTIF 0.762/0.741, MORAL 0.824/0.807; 500 segments per layer, 3 annotators each, 25 experts | Table 3.9; C-ANHC-6 |
| IAA bootstrap | 1,000 resamples, percentile 95% CI; widths 0.028/0.040/0.032 | Table 3.13 |
| Annotation time | DIA 3.1, MOTIF 7.8, MORAL 12.3 min/segment | Table 3.10 |
| Preprocessing (SAHDA) | 45 FST rules (12 novel); neural verification threshold 0.85 (grid search on 5,000 samples); BALD active learning; 3,847 of 46,951 low-confidence cases refined | §3.4 |
| Normalisation | NFKC; alef normalisation; tashkeel preserved; tatweel removed except calligraphic; punctuation preserved; dialect exceptions | Table A.8, Listing A.3; C-NORM-1 |
| Diacritics in model input | primary undiacritised stream + auxiliary diacritic stream | §5.4.1; C-NORM-1 |
| Cultural density (CRD) | Eq. 3.1; lexicon 2,847 entries (not supplied); overall 2.41 (SD 1.18); classical 3.82, modern 2.67, contemporary 1.94 | §3.6.1, Table 3.12 |
| Splits | 80/10/10 = 4,197/525/525 narratives; TF-IDF + DBSCAN (ε 0.3, min_samples 2) clusters kept together; stratify dialect 35/25/20/15/5, density <1.5/1.5–3/>3 = 30/50/20, period 35/30/35 | §5.4.4; C-SPLIT-1 |
| Anti-memorisation | paraphrase augmentation 10%; formula masking 15%; synthetic augmentation ≤ 15%; 2,000 synthetic temporal examples | §5.4.4, §5.7.1, §5.7.3 |
| Formulaic database | 437 patterns; 6 examples in Table 5.4 (exact match with morphological normalisation, 94.2% precision) | §5.4.2 |
| Licence/access | research licence on request vs CC BY-NC-SA 4.0 on Zenodo | §3.7.1 vs A.20.2; C-ANHC-7 |

## 2. Knowledge resources (Ch. 3–4, App. A.4, A.19)

| Item | Value | Location |
| --- | --- | --- |
| CST-Onto | 87 archetypes, 847 concepts, 1,847 entities (rows sum to 2,047), 143 object / 89 datatype properties, 234 SWRL rules, 647 annotation instances | §3.1.4, Table A.16; C-CKG-2 |
| SWRL example | Character(?c) ∧ hasArchetype(?c, hakim) → exhibitsVirtue(?c, justice) ∧ exhibitsVirtue(?c, forbearance) ∧ exhibitsVirtue(?c, eloquence) | Listing A.5 |
| CKG | 12,341 segments from 6 listed sources, 898 tags (847 + 51); κ = 0.84 curation; 47,823 candidates → 18,472 reviewed | §4.3.2, Table 4.8 |
| Schema embeddings | OWL2Vec*: 5,431 triples, 500,000 walks (length 10), skip-gram 300-d, window 5, 15 negatives, 25 epochs, projected 300→768 (Eq. 4.8) — or 256-d (A.4) | §4.3.2; C-CSAM-5 |
| Activation | cosine > 0.5 (Eq. 4.20) or ≥ 0.75 with ≤ 50 active (A.4) | C-CSAM-5 |
| Stage-4 sources | Arabic WordNet 23,481 synsets; Cultural Motif Database 698; Historical Lexicon 3,247 | §4.2.5 |
| ArabicVerbNet / ArabicNameNet | not part of ANSG (App. B.3.7, AI-Cinema only) | C-RES-1 |

## 3. Model (Ch. 4)

| Item | Value | Location |
| --- | --- | --- |
| Backbone | GPT-2 Medium 355M / 774M base / AraGPT2-large | C-MODEL-1 |
| Morphological features | m_t ∈ R⁶⁴ = [root 16; pattern 12; voice/aspect 8; dialect 12; cultural 16] | Eq. 4.6 |
| Cultural weight | w_c = 0.6·f/f_max + 0.4·w_expert | Eq. 4.2 |
| Sequence pooling | M_seq = Σ α_t m_t, α = softmax(wᵀm_t), w ∈ R⁶⁴ | Eq. 4.7 |
| CSAM attention | softmax(QKᵀ/√d_k + βH_i + γR_i)V, β = 0.347, γ = 0.283, d_k = 64 | Eq. 4.9; C-CSAM-2…6 |
| CSAM loss | L_LM + 0.1·L_cultural + 0.05·L_schema | Eq. 4.16–4.19 |
| Heads | 12 cultural heads (ablation 6/8/12/16/20 in Table 4.10) | §4.3.6; C-CSAM-3 |
| DAL | LoRA r = 8, α = 16, dropout 0.05, targets q_proj, v_proj; gate Eq. 4.25 | §4.4, Table 5.8; C-DAL-1/2 |
| Schema inventory | |S| = 898; S_active 3–5 typical; τ = 0.5 | Table 4.2 |

## 4. Training (Ch. 4–5, App. A.3, A.26)

Training phases, compute, schedules: **see C-TRAIN-1…5** — the thesis gives five
incompatible protocols (2-phase, two different 3-phase, 3-phase with GAN, 4-phase).
Values common to Ch. 5 Tables 5.8–5.9:

| Item | Value |
| --- | --- |
| Optimiser | AdamW, LR 5e-5 (TPE optimum 4.7e-5), β = (0.9, 0.999), ε = 1e-8, wd 0.01, 1,000 warm-up steps, cosine decay |
| Batch | 8 per GPU × 4 GPUs × 4 accumulation = 128 |
| Clipping | 1.0 (0.5 during RL) |
| Dropout | attention 0.15, hidden 0.10, embedding 0.05 |
| PPO | clip 0.20, value 0.5, entropy 0.01, GAE λ 0.95, 4 epochs, minibatch 32, rollout 128, KL 0.01, LR 1e-5 |
| Precision | AMP FP16, FP32 master weights; cultural embeddings & discriminator FP32; loss scale 2¹⁶, backoff 0.5 |
| Memory | gradient checkpointing; CKG cache (500 motifs on GPU, LRU 200); sliding window 256/64 beyond 768 tokens |
| Seeds | 42 (primary); 123, 456, 789, 1011 (validation); cudnn deterministic |
| Early stopping | val CMPI gain < 0.01 for 10 epochs; checkpoint every 5 epochs |
| HPO | grid 36 configs → TPE 50 iterations (Optuna) |

## 5. Reward (Ch. 4 §4.5, Ch. 5 §5.6.3, App. A.3/A.11/A.23)

Five different definitions: **C-REWARD-1**. Sub-metric definitions absent: C-REWARD-5.
AHP re-computation: C-REWARD-4.

## 6. Evaluation (Ch. 6, App. A.12–A.14, A.24)

See `CRAFT_EVALUATION_SPECIFICATION.md`. Fully specified procedures implemented in
`ansg.evaluation.statistics`: gap closure (Eq. 6.1), scale normalisation (Eq. A.1),
Bonferroni, Cohen's/Fleiss' κ, percentile bootstrap.

## 7. Environment

C-ENV-1 (two incompatible software stacks and GPU sizes), C-ENV-2 (inference).

## 8. Reproducibility conventions adopted by this reimplementation

* Every run writes `manifest.json` with real UTC timestamps, git commit, config hash and input/output SHA-256.
* Preprocessing is deterministic; splits are hash- or cluster-based.
* Thesis artefact identifiers that cannot be verified (C-REPRO-1) are recorded as stated and never used as provenance for runs here.
