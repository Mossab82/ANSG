# Specification conflicts register

Source: the dissertation as supplied by the author on 2026-11-13 (Markdown export of
the PDF, 13,146 lines). Section, table and equation numbers refer to the dissertation.
Every entry quotes the competing statements and their locations. **Nothing here is
resolved**: resolution is an author decision (see `IMPLEMENTATION_GAPS.md §C`).
Machine-readable variants: `configs/specification/thesis_variants.yaml`.

Entries marked **[recomputed]** were checked arithmetically
(`tests/unit/test_thesis_computations.py` reproduces them). Entries marked
**[public fact]** compare a thesis statement with documented properties of public
models; they inform, but do not decide, the reconciliation.

Additional entries marked **[author audit]** were raised in the author's own
*ANSG Technical Specification Audit* (v0.1, supplied 2026-11-13) and verified here
against the dissertation text. Arithmetic for shapes and parameter counts is
generated reproducibly in `PARAMETER_AUDIT.md`.

**Scope:** Chapters 3–6 and Appendix A were transcribed systematically. Chapters
7–9 also restate specifications (e.g. §8.7.1, Table 8.6) and have only been
spot-checked; further conflicts may exist there.

## T — Transcription limitations

| Id | Issue |
| --- | --- |
| T-1 | Arabic script in the export is largely corrupted (PDF font-encoding damage), e.g. Tables 3.5–3.8, 4.4–4.7, 5.3–5.5, A.8 and all examples. Only transliterations/translations are reliable. Arabic strings in this repository that come from the thesis (Table 5.4 formulae) are **reconstructed from transliteration** and must be checked against the PDF. |
| T-2 | Some symbols are lost (e.g. "rating ￿2 or ￿5" in §6.1.3; Table A.8 dialect exceptions). These are recorded as illegible, not guessed. |

## Model and architecture

| Id | Topic | Statements |
| --- | --- | --- |
| C-MODEL-1 | Backbone | Ch.4 §4.1.6: GPT-2 Medium, 355M, English-pretrained, continued Arabic pretraining. Ch.5 Table 5.1: 847M total = 774M base + 73M cultural. App. A.3 Listing A.2: `aubmindlab/aragpt2-large`, 847M, hidden 1280, 20 heads. Ch.4 §4.4.1/§4.6.1 compute with d = 768 and 12 layers. **[public fact]** GPT-2 Medium: 355M, d=1024, 24 layers, 16 heads; GPT-2 Large: 774M; GPT-2 small: d=768, 12 layers; AraGPT2-large ≈ 792M, d=1280, 20 heads. |
| C-MODEL-2 | Parameter counts | §8.7.1: "CSAM alone requires 73M parameters" **[author audit]** — vs Table 5.1 (73M = all cultural modules) and §4.1.6 (CSAM 16M). §4.1.6 calls H_i, R_i "16M parameters", but Eq. 4.12 recomputes them per input from S_active and §4.6.1 counts them as activation memory (24n² floats) **[author audit]**. Ch.4 §4.1.6: trainable 16M (CSAM) + 1.2M (DAL) + 0.8M (weight predictors) = 18M, base frozen. Ch.4 §4.4.1/Table 4.12: DAL 737,280 / 738K = 0.21%. Ch.3, Ch.4 text, §4.9: DAL "0.3%" (§4.9: "0.3% … (738K of 355M)" — 738K/355M = 0.21% **[recomputed]**). Ch.5 Table 5.1: cultural modules 73M; trainable 847M full / 2.5M LoRA-only. Table 4.10: "Params (M) 847" at 12 heads. |
| C-MODEL-3 | Tokenizer | Ch.5 §5.4.1: SentencePiece 32,000 root-aware vocabulary. Ch.4 Table 4.11 / §4.6: "standard GPT-2" forward pass. **[public fact]** GPT-2/AraGPT2 ship fixed BPE vocabularies (50,257 / 64,000); replacing the tokenizer requires new embeddings, which no chapter describes. |
| C-CSAM-1 | Name | "Cultural Semantic Alignment Module" (Ch.4 §4.3, terminology note) vs "Cultural Schema Attention Mechanism" (Ch.3 §3.1.1, glossary, A.3, A.25). |
| C-CSAM-2 | Heritage bias H_i | Eq. 4.12–4.13: H_i[j,k] = Σ_{s∈S_active} w_s·σ(v_{s,i}ᵀ[pos_j; pos_k; c_s]). §4.3.6: H_i(q,k) = Σ_{p∈P_i} w_p·sim(q,b_p)·sim(k,b_p), b_p ∈ R^{d_k}. Table 4.2: n×n learned matrices. Worked example §4.3.7 uses a negative relevance (−0.254), impossible under the sigmoid of Eq. 4.13, and applies H without β although Eq. 4.9/4.11 scale it by β. |
| C-CSAM-3 | Head organisation | §4.3.6 (first list): heads 1–3 virtue (214 patterns), 4–5 structure (178), 6–7 archetypes (165), 8–9 rhetorical (147), 10–11 dialectal (97), 12 cross-schema (46) [sum 847 **recomputed**]. §4.3.6 (second list) and §4.9: h1–h4 heritage, h5–h8 rhetorical (jinās, sajʿ, ṭibāq, tashbīh), h9–h12 dialectal (EGY, LEV, GULF, MAG). A.25 code: heads 4–7 rhetorical = sajʿ, tawriya, iltifāt. |
| C-CSAM-4 | Code vs equations | A.25 Listing A.6 implements cross-attention from hidden states to 1,847×256 schema embeddings and returns 0.370·h + β·mean(heritage) + γ·mean(rhetorical); the three coefficients sum to **1.000**, i.e. a convex combination of hidden states **[author audit]**; dialectal heads are computed but unused; `alpha = 0.370` appears nowhere in Ch.4; the helper `compute_multihead_attention` is called but never defined, so the listing is not runnable **[author audit]**. Visible modules total 2,770,432 parameters at hidden 1280 (`PARAMETER_AUDIT.md`). Eq. 4.9 instead adds βH_i + γR_i to the self-attention logits. |
| C-CSAM-5 | Schema activation | Eq. 4.20 / Table 4.2: cosine > τ = 0.5, "3–5 typical"; §5.7.1: τ = 0.5 reduced active schemas per token from 8.4 to 2.1. App. A.4 / Table A.5: cosine ≥ 0.75 on 256-d embeddings, max 50 active, plus filters (domain exact match, era, dialect) without operational definitions. Embedding dims: 768 (Eq. 4.8, W_proj ∈ R^{300×768}) vs 256 (A.4, A.25). |
| C-CSAM-7 | Head size **[author audit]** | A.25 sets `head_dim = hidden_size // 12`. With A.3's hidden 1280 this is 106 remainder 8; with GPT-2 Medium's 1024 it is 85 remainder 4. Only d = 768 yields the d_k = 64 stated in §4.3.6. |
| C-CSAM-6 | Rhetorical bias R_i | Purpose described (sajʿ, jinās, ṭibāq) but no computation is given anywhere. |
| C-DAL-1 | Adapter placement | Eq. 4.24 adapts "W" generically; §4.4.1 counts one 2·r·d adapter per layer; Table 5.8 targets q_proj and v_proj (two matrices). **[public fact]** HF GPT-2/AraGPT2 have a fused `c_attn`, no `q_proj`/`v_proj`. |
| C-DAL-2 | Algorithm 1 | Line 18: P_d(y_t) = P(y_t) + g_d·ΔW_d h_t adds a hidden-size vector to a vocabulary distribution; line 20: P_final = Σ_d g_d·P_d is not normalised. The dialogue detector ("trained LSTM", §4.4.4) is not specified. |
| C-MORPH-1 | Morphological tools | Ch.4 §4.2.2: Stanford Arabic Parser + AraMorph 2.0. Ch.5 §5.4.1: CAMeL Tools 1.5.2 (calima-msa-s31, calima-egy-s31), MADAMIRA (§5.5.2). App. A.2: CAMeL Tools 1.4.0. Ch.6 Table 6.10: "CAMeL Tools for Arabic morphology". |

## Training and reward

| Id | Topic | Statements |
| --- | --- | --- |
| C-TRAIN-1 | Training phases | Ch.3 §3.1.2: two phases (masked-LM pretraining on 70.4M; SFT on 23M). Ch.4 Table 4.1: pretrain (MLM) → fine-tune L_CSAM → RL. Ch.4 §4.1.6: causal-LM pretraining 100k steps → cultural SFT 50k → PPO 20k. Ch.5 §5.1.3/§5.6: supervised → **GAN adversarial** → RL (epochs 1–30 / 31–65 / 66–100). App. A.26 Listing A.9: four phases incl. "multi-agent refinement" (contradicting Ch.4 §4.1.5's replacement of multi-agent design). |
| C-TRAIN-2 | Compute | Ch.3: ~150 GPU-h for phase 1 alone. Ch.4 §4.1.6: ~90 GPU-h total, 4×A100 80GB. Ch.4 Table 4.1: 4×A100 40GB. Ch.5: 72 h × 4 = 288 GPU-h, 24/28/20 h. A.26: 24/12/24/12 h. |
| C-TRAIN-3 | Schedules / LR | Table 4.1: LR 3e-4 (3 epochs, batch 32), 5e-5 (5 epochs), 1e-5 (10k steps). §4.1.6: steps 100k/50k/20k. Ch.5: epochs, LR 5e-5 (TPE optimum 4.7e-5), RL 1e-5. A.3: phase-1 10 epochs, RL LR 1e-6. Grad accumulation fixed at 4 (Table 5.2/5.8) vs phase-specific 4/2/8 (§5.7.2). Phase 2→3 trigger: cultural F1 > 0.75 (§4.1.2) vs discriminator accuracy < 70% (§5.1.3). |
| C-TRAIN-4 | Symbol reuse | β, γ denote CSAM coefficients (0.347, 0.283) and, in Table 5.7, "motif weight" 0.28 and "temporal weight" 0.18; α is Eq. 4.2's 0.6, Table 5.7's cultural weight 0.42, A.3's `alpha_base` 0.370, and CULTURA's gate 0.7 (Eq. 4.1). |
| C-TRAIN-5 | Objective type | Ch.3 and Table 4.1 call phase 1 masked-LM; §4.1.6 and Eq. 4.17 use causal next-token prediction (GPT-2 is causal). |
| C-REWARD-1 | Reward definition | Eq. 4.26–4.29: R = 0.45 R_cult + 0.35 R_ling + 0.20 R_coh with sub-weights (CMPI .35 / virtue .30 / rhetorical .25 / register .10; PPL⁻¹ .40 / BLEU .35 / BERTScore .25; coref .50 / temporal .30 / causal .20). Table 4.1 & §4.1.1: R = 0.45 CMPI + 0.35 BLEU + 0.20 PCI. §5.6.3: R_cult = CMPI, R_ling = norm. BLEU + (1 − norm. PPL), R_coh = embedding coherence, plus penalties Eq. 5.2 (−0.15·CoherencePenalty), Eq. 5.3 (−0.1·ProverbRepetition) and KL 0.01. App. A.3/A.11: four terms 0.35 cultural / 0.30 linguistic / 0.25 coherence / 0.10 diversity. |
| C-REWARD-2 | Same numbers, different criteria | 0.35/0.30/0.25/0.10 is assigned to {CMPI, virtue, rhetorical, register} (§4.5.1), {rhetorical, virtue, narrative structure, register} (§5.5.3, A.23) and {cultural, linguistic, coherence, diversity} (A.11). Ch.5 §5.1 gives CMPI-internal weights 0.35/0.25/0.25/0.15. |
| C-REWARD-3 | "No ground-truth references" | RQ4 and §4.5 claim reference-free RL, but R_ling includes BLEU against ANHC classical-stratum references (Eq. 4.27) and Table 4.1 feeds BLEU into the reward. |
| C-REWARD-4 | AHP verification **[recomputed]** | Table A.10 matrix → eigenvector 0.350/0.300/0.249/0.100 ✓, but CR ≈ 0.0001 (stated 0.023). Table A.20 matrix → **0.37/0.31/0.22/0.10**, not the stated 0.35/0.30/0.25/0.10; CR ≈ 0.004 (stated 0.078). Panel: 12 experts with 3 Delphi rounds and CR 0.02 (§4.5.2), 5 experts (A.11), 12 experts (A.23). |
| C-REWARD-5 | Normalisation | Perplexity⁻¹ = 0.065 "normalized to [0,1]: 0.82" (§4.5.4) — rule not given. Virtue, rhetorical, register, coreference, temporal, causal sub-metrics have no definitions. |
| C-REWARD-6 | Worked example **[recomputed]** | §4.5.4: 0.35·0.91 + 0.30·0.88 + 0.25·0.73 + 0.10·0.95 = **0.860**, printed 0.867; R_total with recomputed parts = 0.852 (printed 0.854). |
| C-REWARD-7 | Ch.5 reward terms and penalties | Eq. 5.1 (p.151): R_ling = normalised BLEU-4 + (1 − normalised perplexity), printed **without weights**, so its range is [0, 2]; R_cult = CMPI and R_coh = sentence-embedding cosine coherence, each a single term, whereas Ch.4 Eq. 4.26–4.28 define both as weighted composites, and both chapters cite §4.5 as the source. Eq. 5.2 (p.152) subtracts 0.15·CoherencePenalty from **R_total**; Eq. 5.3 (p.153) subtracts 0.1·ProverbRepetition from **R_cult**. The KL coefficient 0.01 (p.151) has no stated placement (reward-side or loss-side). Machine-readable: `configs/specification/reward_rl_variants.yaml`. |
| C-REWARD-8 | A.21 ratio | A.21.3 (p.291): cultural fidelity 0.63 : fluency 0.37, then in the same section cultural components 'collectively receive 65% weight versus linguistic quality ... at 35%'; neither matches the top-level cultural weight 0.45 (Eq. 4.29, Eq. 5.1). |

## Corpus (ANHC) and annotation

| Id | Topic | Statements |
| --- | --- | --- |
| C-ANHC-1 | Corpus units | 70.4M tokens / 847K segments (Tables 3.2–3.3); 5,247 narratives split 4,197/525/525 (§5.4.4); 55,600 narratives, 70,404,829 tokens, 4,237,451 sentences (Table A.17). |
| C-ANHC-2 | Annotated share | 23M of 70.4M tokens tri-annotated (§3.1.2, §3.7); 100% of narratives with cultural annotations and dialect markers, 94.1% with moral tags (Table A.17). |
| C-ANHC-3 | MOTIF inventory | §3.5.2 / Fig. 3.2: 87 archetypes. §3.1.4: the 847 specialized concepts are "the primary annotation vocabulary used in ANHC tri-layer annotation". Table A.7: motifs 437 categories + 847 archetypes. The codebook that §3.5 says is in App. A.7 (87 motifs, 8 moral classes, 5 dialects, 47 boundary cases) **is not in the dissertation**. |
| C-ANHC-4 | MORAL inventory | §3.5.3: 8 classes (Justice, Wisdom, Courage, Temperance, Faith, Charity, Patience, Gratitude). Table 3.8 also uses Generosity, Honor, Truthfulness. §5.4.2: 127-concept moral ontology. |
| C-ANHC-5 | Dialect distribution | Table 3.3: MSA 40.8 / EGY 32.4 / LEV 14.5 / GULF 7.2 / MAG 5.1 %. Table A.17: 28.3 / 22.1 / 18.7 / 14.9 / 11.2 / Mixed 4.8 %. §5.4.4 stratification target 35/25/20/15/5. §6.2.3: "Gulf and Egyptian … combined 37% of ANHC" (Table 3.3 gives 39.6%; A.17 gives 37.0%). Table 8.6 (§8.7.2): Gulf 18.6 / EGY 25.0 / MSA 35.0 / LEV 7.1 / MAG 14.3 / **Iraqi 7.2** % (a sixth dialect, ALDA 74.8%; rows sum to **107.2%** **[recomputed]**), plus Sudanese and Yemeni < 1%. |
| C-ANHC-6 | Annotation layers & team | Ch.3: DIA / MOTIF / MORAL, Cohen's κ 0.891 / 0.762 / 0.824; 25 experts (5 per region) for IAA; 54 community annotators. Table A.7: Linguistic / Cultural / Dialectal layers, κ 0.89 / 0.84 / 0.81, 23 annotators; caption says all layers > 0.80 "except Dialectal (0.81)". §5.7.3: consensus raised Fleiss' κ 0.67 → 0.84. |
| C-ANHC-7 | Licensing & access | §3.7.1: full corpus on request (research licence); publisher-copyright sources cannot be redistributed; 10% sample on Zenodo "DOI [to be assigned]". App. A.20.2: whole corpus CC BY-NC-SA 4.0 on Zenodo DOI 10.5281/zenodo.8847291; §A.26 box: some resources "not publicly redistributed". |
| C-ANHC-8 | Nights source | §3.2.2: Mahdi critical edition (1984) provides 8.2M tokens; §3.2.4: parallel Bulaq (1835) and Calcutta II (1839–42) witnesses. Note for checking: Mahdi's edition covers the early Syrian-branch nights only, so the token figure should be verified against the actual files. |
| C-ANHC-9 | Temporal strata | Table 3.2: Classical 35.0 / Modern 26.0 / Contemporary 39.1 %. §5.4.4: Classical 35 / "Renaissance" 30 / Contemporary 35 %. |
| C-ANHC-10 | Validation counts **[recomputed]** | Table 3.4 regional correction rates weight to 8.16 %, printed 8.3 %. |
| C-NORM-1 | Normalisation | App. A.8 / Listing A.3: NFKC, alef normalisation, **preserve** tashkeel, remove tatweel (except calligraphic). Ch.5 §5.4.1: primary stream **undiacritised**, diacritics as an auxiliary token stream. Ch.3 §3.4: SAHDA FST rules (45; hamza normalisation, ta-marbuta restoration) — not supplied. |
| C-SPLIT-1 | Split method | §5.4.4: TF-IDF + DBSCAN (ε 0.3, min_samples 2), whole clusters per split, plus stratification; distance metric and TF-IDF settings unstated. Listing A.3: "Train/validation/test split generation" as preprocessing stage 6, no method. |

## Knowledge resources

| Id | Topic | Statements |
| --- | --- | --- |
| C-CKG-1 | Counts | CST-Onto: 87 archetypes, 847 concepts, 1,847 OWL entities, 234 SWRL rules (§3.1.4). CKG: 12,341 segments, 898 tags = 847 + 51 relational patterns (§4.3.2). "1,847 motifs" / "437 fine-grained categories" (Table A.2, A.3, Table 5.1, Table 6.10). Cultural Motif Database: 698 patterns (§4.2.5). Formulaic database: 437 patterns (§5.4.2). |
| C-CKG-2 | Table A.16 sum **[recomputed]** | 87 + 847 + 143 + 89 + 234 + 647 = **2,047**, stated total 1,847. |
| C-CKG-3 | Sources | §4.3.2 text: "eight canonical Arabic narrative corpora"; Table 4.8 lists six. |
| C-RES-1 | ArabicVerbNet / ArabicNameNet | Not part of the ANSG architecture in the dissertation. Their only mention (line ~13060) is App. B.3.7, describing the separate AI-Cinema paper. ANSG's knowledge sources are Arabic WordNet (23,481 synsets), the Cultural Motif Database and a Historical Lexicon (§4.2.5), CST-Onto and the CKG. The adapters in this repository exist because the Phase 1 brief requested them. |

## Evaluation

| Id | Topic | Statements |
| --- | --- | --- |
| C-EVAL-1 | What CRAFT is | Ch.6 intro: ontology-driven test over 847 concepts (virtue concepts, archetypes, formulaic expressions). §6.2.4: 4 dimensions = CMPI, CulturalScore, ALDA, MFA. Table 6.4: Rhetoric, Formulaic, Metaphor, Prosody, Moral Framework. §4.1.1 RQ3: AHP-derived multi-dimensional metrics. No CRAFT formula anywhere. |
| C-EVAL-2 | CMPI | No equation. "Cultural Motif Preservation Index" (Ch.4, Ch.6, glossary) vs "Cultural Meaning Preservation Index" (Ch.5 §5.1, Table A.2, which says "defined in Chapter 4"; Ch.4 points to §3.6.1, which defines CRD). Ch.5 §5.1 weights 0.35/0.25/0.25/0.15. Table 6.4 overall CMPI (0.92) is not derivable from its dimension scores. |
| C-EVAL-3 | Multiple comparisons | Bonferroni for 9 comparisons, α' = 0.0056 (§6.2.1) vs 6 comparisons (Table A.11). |
| C-EVAL-4 | Bootstrap | Percentile, 1,000 resamples (§3.6.2) vs BCa, 10,000 (Table A.11, A.13.2). |
| C-EVAL-5 | CulturalScore | Automatic 0–1 rhetoric score from NarrativeMind (Ch.6) vs expert AHP-based rating normalised to 0–100 (§3.6.1). |
| C-EVAL-6 | ALDA | Reported as percentage (92.3 %, Table 6.9) and as 0.89 (Listing A.11); four sub-measures in Table 6.6 without a combining rule. |
| C-EVAL-7 | Generation | temperature 0.7, top-p 0.95, max 512, seed 42 (§6.1.2) vs GPT-4 top-p 0.9 (Table A.6); test narratives average 712 tokens (Table A.12) vs 512-token generation cap. |
| C-EVAL-8 | Human evaluation | §6.1.3: 5 raters (2 classical-literature PhDs, sociolinguist, comparative literature, ḥakawātī), 4 dimensions. A.13.1: 5 PhDs by region (2 LEV, 1 EGY, 1 GULF, 1 MAG). A.24: 30 evaluators, 3 dimensions, mixed-effects models. §4.4.4: 12 Syrian raters, 10-point scale. |
| C-EVAL-9 | Baselines | Table 6.1: ArabicGPT 135M zero-shot (no ANHC), NarrativeMind BLOOMZ-3B λ = 0.7, MM-GTA 245M. Table A.6: ArabicGPT 1.5B fine-tuned on ANHC, NarrativeMind 350M λ = 0.4, MM-GTA 420M, Jais-30B (absent from Ch.6). CULTURA appears in Table 6.3 but not among the nine baselines. |
| C-EVAL-10 | Effect sizes | ANSG vs GPT-4 on CMPI: d = 2.14 (exec. summary), 1.89 (Table 6.3), 1.58 (Table 6.5). vs MM-GTA: 0.89 (§6.1.4), 1.23 (Table 6.3), 0.94 (Table 6.5). BLEU vs MM-GTA: d = −0.67 (§6.1.4) vs −0.71 (§6.1.6). |
| C-EVAL-11 | Reported values | GPT-4 CMPI 0.68 (summary, Fig. 6.1, Eq. 6.1) vs 0.73 (Table 6.4, Table A.9). Prosody gain +8.7 % (§6.2.2, Table 6.11) vs +12.5 % (Table 6.4); ANSG prosody 0.72 (Table 6.4) vs 0.78 (§6.2.2). Latency 147 ms/token (§4.6.1) vs 42 ms/token (§4.1.5) vs 147 ms per 512 tokens (Table 5.1, A.2). |
| C-EVAL-12 | Test set | 500 narratives (§6.1.2, A.14) vs 525 test split (§5.4.4); Table 6.6 per-dialect n sums to 900; A.14 per-dialect counts (100/100/100/75/75/50) differ from Table 6.6. |

## Environment and reproducibility

| Id | Topic | Statements |
| --- | --- | --- |
| C-ENV-1 | Hardware/software | Ch.5: Python 3.10.12, PyTorch 2.0.1+cu118, Transformers 4.30.2, CAMeL 1.5.2, Optuna 3.2.0, CUDA 11.8; 4×A100 **40GB**, 512 GB RAM, InfiniBand. App. A.2: Python 3.9.7, PyTorch 1.12.0+cu116, Transformers 4.24.0, CAMeL 1.4.0, TRL 0.4.7, PEFT 0.2.0, CUDA 11.6; 4×A100 **80GB**, 1 TB RAM, NVLink. |
| C-ENV-2 | Inference | 16 GB min VRAM and 482 MB footprint (Table 5.1) vs 16 GB min VRAM with an 18.7 GB footprint (Table 8.5) vs RTX 3090 with 18.7 GB VRAM (A.2) vs 1.4 GB base model (§4.6.1). **[public fact]** 847M parameters occupy ≈ 1.7 GB in fp16. |
| C-REPRO-1 | Artifact identifiers | "Verified commit" `a7b3c9d2e5f8g1h4j6k8m0n2p4q6r8s0` and checkpoint "SHA-256" values contain letters g–z, so **as printed they cannot be valid git object ids or SHA-256 digests**; the configuration "SHA-256" has 32 hex characters (128 bits; SHA-256 has 64). Repository URL differs from this repository (`mossab-ansg/ansg-framework` vs `Mossab82/ANSG`). These could not be verified from this environment and are not treated as historical artefacts. |
| C-ETHICS-1 | Ethics | §3.3.3: "Research Ethics Compliance … did not involve … human subjects research". App. A.17: no formal approval from a research ethics committee; human participation as voluntary consultation. §3.3.1: 54 compensated community annotators; A.7: 23 annotators. |
