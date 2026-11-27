# Thesis-aligned ANSG implementation plan

**Goal:** a new implementation that stays as close as practical to the architecture the dissertation describes in Chapters 4–5. Section and equation numbers below refer to the submitted dissertation.

**Two caveats apply to the whole plan:**
- This is a **new implementation**. It does not reproduce, verify or replace any number reported in the dissertation.
- Where the dissertation gives several specifications for one component, we choose one, say which section we follow, and record the choice under "Decisions".

## Components

| # | Component | Dissertation | This implementation | Data needed | Status |
|---|---|---|---|---|---|
| 1 | Backbone | §4.1.6: GPT-2 class causal LM. The Ch. 4 equations use hidden size 768 and 12 layers (Eq. 4.8, §4.4.1) | AraGPT2, size per decision D1 | — | to choose |
| 2 | Morphological pipeline | §4.2: 6 stages (decomposition → feature vectors) | CAMeL Tools analyser feeding token features | text | not started |
| 3 | Cultural Knowledge Graph | §4.3.2: schema embeddings c_s = W_proj · OWL2Vec(s), from CST-Onto | Small graph from `data/cultural_seeds`, Arabic VerbNet and Arabic WordNet, embedded with OWL2Vec*-style walks or a trainable lookup (D3) | seeds + lexicons | seeds ready |
| 4 | CSAM | Eq. 4.9: softmax(QKᵀ/√d_k + βH_i + γR_i)V; 12 heads split heritage 4 / rhetorical 4 / dialectal 4; β = 0.347, γ = 0.283 (§4.3.4) | `src/ansg/csam.py`, extended to bias the backbone's own attention logits. β and γ start from the dissertation values, then are tuned | motif-span labels | scaffold exists, untrained |
| 5 | CSAM training loss | Eqs. 4.16–4.19: L = L_LM + λ_align · L_cultural + λ_schema · L_schema | Same three terms | motif spans (P_pos) and schema labels | not started |
| 6 | DAL | Eq. 4.24: LoRA rank 8 per dialect (MSA, EGY, LEV, GULF, MAG). Eq. 4.25: schema-conditioned gate g_d = σ(v_dᵀ[h; s_ctx]) · 𝟙[compatible] | `src/ansg/dal.py`, five rank-8 adapters, gate as in Eq. 4.25, compatibility rules from §4.4.2 | dialect-labelled text | scaffold exists, untrained |
| 7 | Phase 1: supervised | Table 5.1 / §5.1.3: epochs 1–30 | LM + CSAM losses on the open corpus | corpus | not started |
| 8 | Phase 2: GAN refinement | Table 5.1 / §5.1.3: epochs 31–65 | Discriminator details to be read from Ch. 5 before design (D4) | corpus | not started |
| 9 | Phase 3: RL | PPO (§4.1.6); R_total = 0.45 R_cult + 0.35 R_ling + 0.20 R_coh (Eq. 4.29); R_cult = 0.35 CMPI + 0.30 Virtue + 0.25 Rhetorical + 0.10 Register (Eq. 4.26); R_ling (Eq. 4.27); R_coh (Eq. 4.28) | PPO with the dissertation's top-level structure and weights. Sub-metrics without equations in the dissertation get explicit, documented operationalisations (D5) | prompts | not started |
| 10 | Evaluation | CMPI, CRAFT (Ch. 6) | SAS (`docs/SEED_ADHERENCE_METRIC.md`) now. A documented CMPI operationalisation later (D5), validated against human ratings | human ratings | SAS done |

## Decisions for Mossab

- **D1. Backbone size.** AraGPT2 sizes are base, medium, large and mega. Exact layer counts and dimensions must be confirmed on the model cards before use.
  - *base*: about 768 hidden units and 12 layers. Matches the Ch. 4 equations and the DAL parameter count in §4.4.1.
  - *medium*: matches the name "GPT-2 Medium".
  - *large*: matches Listing A.2 (`aragpt2-large`, hidden 1280).

  **Proposal:** build and debug on *base*, where every Ch. 4 dimension holds exactly, then scale to *medium* or *large*.
- **D2. Corpus.** ANHC is not available. Use open-licensed heritage and dialect text, keep a source and licence register, and keep restricted data out of the public repository.
- **D3. Schema embeddings.** The CST-Onto OWL file is needed for real OWL2Vec* initialisation. Is it available? If not, use a trainable lookup initialised from the seed graph.
- **D4. GAN phase.** Read the Ch. 5 description of the discriminator and losses before designing Phase 2.
- **D5. Sub-metrics without equations.** CMPI, Virtue, Rhetorical, Register, Coreference, Temporal and Causal each get a written formula in `docs/`. These formulas are this implementation's choices, not the dissertation's.

## Build order
1. Backbone loads, and CSAM bias hooks into its attention. Unit test: zero bias gives exactly the original logits (Eq. 4.10 = Eq. 4.11 with B = 0).
2. Seed graph and schema embeddings, with schema activation.
3. DAL adapters and gate on top of the backbone.
4. Phase 1 training on a small open corpus on Kaggle or Modal. Verify the loss falls and the stories are sensible.
5. Reward functions with documented sub-metrics, validated on human ratings.
6. Phase 3 PPO. Phase 2 GAN after D4.
