# Thesis-to-Code Fidelity Audit — ANSG-Original (2026-11-14)

> **ملخص بالعربية.** تربط هذه المصفوفة كل مكوّن أو معادلة في الرسالة بموضعها، وبالملف والدالة المقابلة، وبحالة التنفيذ، وبالاختبار المطلوب، وبأثر الاختلاف على إعادة إنتاج النتائج.
> - **النتيجة العامة:** لا يمكن حالياً إعادة إنتاج أي رقم من الفصل السادس.
>   - البيانات غير متاحة: مدوّنة ANHC، وCKG وCST-Onto، ومعجم العلامات الثقافية.
>   - تعريفات CRAFT غير مكتملة: CMPI بلا معادلة.
>   - بعض المكونات غير منفذة: خط المعالجة الصرفية، وخسائر CSAM المساعدة، ومكونات المكافأة.
>   - توجد تعارضات بين فصول الرسالة نفسها، موثّقة في `docs/SPECIFICATION_CONFLICTS.md`.
> - **المطابق فعلاً:** معادلات حسابية محدودة: Eq 3.1 وEq 4.2 وAHP وEq 4.29/4.30 وEq 6.1.
> - **ANSG-HNC خارج هذه المصفوفة تماماً:** امتداد بحثي مستقل لا يغيّر أي سطر هنا.

## Method and limits

- **Source.** The dissertation as supplied (`Tesis_Ibrahim.md`, a Markdown conversion of the PDF; not committed to the repository).
- **Page numbers.** Pages are read from the PDF page footers preserved in that conversion. A footer line `## N` closes page N, and an item gets the number of the first footer after it. Items near a page break can be off by one. Section start pages agree with the thesis table of contents (for example §4.3.5 → p. 103, §4.6 → p. 119). "First mention" marks a table or figure whose caption could not be told apart from an earlier reference to it.
- **No reproduction attempted.** Nothing here was checked against original checkpoints, logs or the ANHC corpus, because they are not available.
- **Code references.** File and function names are those of `master` at `9a3e69d`.

**Status:**

| Status | Meaning |
| --- | --- |
| MATCH (مطابق) | The code computes what the thesis specifies, and a test pins it |
| PARTIAL (جزئي) | Part of it is implemented, or an unstated choice is made explicit as a parameter |
| DIVERGENT (غير مطابق) | Implemented differently from the thesis |
| NOT IMPLEMENTED (غير منفذ) | — |
| UNVERIFIABLE (غير قابل للتحقق) | The thesis lacks the definition or data needed to check it, or its chapters conflict |

**Impact on reproducing reported results:**

| Impact | Meaning |
| --- | --- |
| BLOCKING | No reported number can be reproduced while this gap remains |
| HIGH | Changes the reported numbers materially |
| MEDIUM | — |
| LOW | — |
| NONE | — |

## 1. ANHC corpus and data pipeline (Ch. 3, Ch. 5 §5.4)

| # | Thesis element | Reference (chapter § / page) | Code | Status | Test (existing → required) | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | ANHC corpus, splits 4,197 / 525 / 525 | §3.1–3.2 (p. 49–58); §5.4.4 (p. 143); App. A.20 (p. 288) | — (corpus not supplied) | UNVERIFIABLE | — → counts and hashes once the corpus is supplied | BLOCKING |
| 1.2 | SAHDA preprocessing, 3 stages | §3.4 (p. 60–62) | — | NOT IMPLEMENTED | — → stage-wise filter tests | MEDIUM |
| 1.3 | Text normalisation protocol | App. A.8, Table A.8 (p. 276) | `data/normalization.py` (profile `thesis_appendix_a8`) | PARTIAL | `test_thesis_profiles.py` | LOW |
| 1.4 | Annotation layers DIA, MOTIF, MORAL | §3.5.1–3.5.3 (p. 65) | `data/records.py`, `data/anhc.py` | PARTIAL: the MOTIF codebook (87 archetypes) is absent | `test_records.py` → codebook validation | HIGH |
| 1.5 | IAA and bootstrap validation | §3.5.4, §3.6.2 (p. 66, 71) | `evaluation/statistics.py` | MATCH, with the percentile vs BCa choice explicit | `test_thesis_computations.py` | LOW |
| 1.6 | Cultural Reference Density, Eq. 3.1 | §3.6.1, Eq. 3.1 (p. 70) | `evaluation/corpus_metrics.cultural_reference_density` | PARTIAL: the 2,847-entry lexicon is not supplied | `test_crd.py` | HIGH |
| 1.7 | Splits 80/10/10, seed 42, TF-IDF + DBSCAN clusters | §5.4.4 (p. 143) | `data/splits.py` (`cluster_records_dbscan`) | PARTIAL: metric and TF-IDF settings unstated | `test_splits.py`, `test_thesis_profiles.py` | MEDIUM |
| 1.8 | Formulaic markers (437 patterns) | §5.4.2, Table 5.4 (p. 141, first mention) | `data/nights.THESIS_TABLE_5_4_FORMULAE` | PARTIAL: 6 of 437 | `test_thesis_profiles.py` | MEDIUM |
| 1.9 | Tokenizer: SentencePiece 32k, root-aware | §5.4.1 (p. 138) | — (byte tokenizer or the backbone's tokenizer) | DIVERGENT (C-MODEL-3) | — → tokenizer artefact and hash | HIGH |
| 1.10 | Tale-level data integrity *(not in the thesis)* | — | `data/mediawiki.py`, `data/tale_structure.py`, `data/leakage.py` | engineering addition | `test_mediawiki_and_tales.py`, `test_leakage.py` | — |

## 2. Morphology and cultural knowledge (Ch. 4 §4.2–4.3.2)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 2.1 | Root–pattern cultural weight, Eq. 4.2 | §4.2 (p. 92) | `knowledge/cultural_weight.py` | MATCH: α = 0.6; Likert→[0,1] made explicit | `test_thesis_computations.py` (worked examples) | MEDIUM |
| 2.2 | Morphological conditioning pipeline, Eq. 4.3–4.7 (64-dim vector) | §4.2.1–4.2.7 (p. 91–95) | — | NOT IMPLEMENTED; the analyser is unresolved (C-MORPH-1) | — → per-stage tests | HIGH |
| 2.3 | CKG construction and declared counts | §4.3.2 (p. 98); Table 4.8 (p. 98) | `knowledge/ckg.py` | PARTIAL: four conflicting declarations kept (C-CKG-1/2); no graph export supplied | `test_ckg.py` | BLOCKING |
| 2.4 | CST-Onto | App. A.19 (p. 286) | `knowledge/ckg.load_ontology_rdf` | PARTIAL: loader only; the ontology file is not supplied | `test_ckg.py` | BLOCKING |
| 2.5 | OWL2Vec* schema embeddings, Eq. 4.8 | §4.3.2 (p. 100) | — | NOT IMPLEMENTED | — | HIGH |
| 2.6 | Schema activation: cosine > τ, Eq. 4.20 / App. A.4 | §4.3.7 (p. 108); A.4 (p. 272) | `knowledge/ckg.CosineThresholdActivator` (`LexicalTriggerEvidence` is a non-thesis baseline) | PARTIAL: τ 0.5 vs 0.75 (C-CSAM-5); embeddings supplied externally | `test_thesis_profiles.py` | HIGH |

## 3. CSAM (Ch. 4 §4.3.3–4.3.7, App. A.25)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 3.1 | Cultural attention with pre-softmax bias, Eq. 4.9–4.11, 4.14 | §4.3.3 (p. 100–102) | `model.CausalAttention` (bias added before softmax) with `csam.PairwiseCulturalBias` | PARTIAL: in-layer in the byte-level research model only. 12 cultural planes are projected to the transformer's head count (an engineering decision). The pretrained bridge is post-backbone (see 3.6) | `test_core.py`, `test_schema_padding.py` → equivalence with Eq. 4.14 on a fixed input | HIGH |
| 3.2 | Heritage matrix H_i, Eq. 4.12–4.13 | §4.3.4 (p. 102) | `csam.PairwiseCulturalBias` (σ of [pos_j; pos_k; c_s], weighted by w_s) | PARTIAL: w_s is a learned softmax, not Eq. 4.12's activation weights (owner decision D4); C-CSAM-2 | `test_core.py` | HIGH |
| 3.3 | Rhetorical matrix R_i | §4.3.4 (p. 102) | `csam.PairwiseCulturalBias.relevance_r` | UNVERIFIABLE: no computation given (C-CSAM-6) | — | MEDIUM |
| 3.4 | CSAM training objective: L_LM + λ_align·L_cultural + λ_schema·L_schema, Eq. 4.16–4.19 | §4.3.5 (p. 103–104) | only L_LM (`model.ANSG.forward`) | NOT IMPLEMENTED: no alignment loss, no schema-prediction loss | — → loss-term unit tests | HIGH |
| 3.5 | Specialised heads (12 heads, pattern groups) | §4.3.6 (p. 104) | `csam.CulturalSchemaAttention` (3 groups × 4 heads) | DIVERGENT (C-CSAM-3) | — | MEDIUM |
| 3.6 | Listing A.6: 0.370·h + β·heritage + γ·rhetorical | App. A.25 (p. 297–298) | `csam.CulturalSchemaAttention.forward`; `pretrained.PretrainedANSG` (bridge v2) | PARTIAL: concat + projection instead of the listing's means; 1280/12 widened to 1284 (C-CSAM-7). The bridge is post-backbone, not in-layer | `test_parameter_audit.py`, `test_generation_stub_backbone.py` (identity at initialisation) | HIGH |
| 3.7 | Worked example: wisdom schema, Eq. 4.20–4.23 | §4.3.7 (p. 108–109) | — | UNVERIFIABLE (needs the CKG embeddings) | — | LOW |

## 4. DAL (Ch. 4 §4.4)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 4.1 | LoRA dialect adapters, Eq. 4.24 (r = 8, α = 16) | §4.4.1 (p. 110–111); A.3 (p. 270) | `dal.DialectalAdaptation` | PARTIAL: applied to hidden states, not to q_proj/v_proj (C-DAL-1, Table 5.8, p. 147 first mention) | `test_model_data_contract.py` | HIGH |
| 4.2 | Schema-conditioned gating with compatibility indicator, Eq. 4.25 | §4.4.2 (p. 111–112) | `dal.DialectalAdaptation.gate` (MLP + softmax) | DIVERGENT: the compatibility indicator 1[s active compatible with d] is absent | — → gating test against Eq. 4.25 | MEDIUM |
| 4.3 | Rank-selection ablation | §4.4.3 (p. 112) | — | NOT IMPLEMENTED (an experiment, not code) | — | LOW |
| 4.4 | Dialect inventory (5 classes) | §3.5.1; A.3 | `config.DIALECTS`, `data/training_examples.DIALECT_TABLE` | MATCH (order fixed) | `test_training_examples.py` | LOW |

## 5. Neural Narrative Generator (Ch. 4 §4.1.6, §4.6; Ch. 5)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 5.1 | Backbone: GPT-2 Medium (§4.1.6) vs 774M base (Table 5.1) vs AraGPT2-large (A.3 Listing A.2) | §4.1.6 (p. 89); Table 5.1 (p. 134 first mention); A.3 (p. 271) | `generation/runtime.load_backbone` (any local causal LM), `pretrained.PretrainedANSG` | UNVERIFIABLE: three conflicting specifications (C-MODEL-1) | `test_generation_stub_backbone.py` | BLOCKING |
| 5.2 | Algorithm 1: morphology → schema activation → CSAM → DAL → sampling | §4.6, Algorithm 1 (p. 120) | `model.ANSG.generate`; `generation/runtime.run_generation` | PARTIAL: lines 1–6 (morphology, ActivateSchemas) are absent; schemas are supplied explicitly | `test_generation_runtime.py` | HIGH |
| 5.3 | Generation settings: temperature 0.7, top-p 0.95, max 512, seed 42 | §6.1.2 (p. 163) | `generation/runtime.SamplingConfig` defaults | MATCH (the conflict with Table A.6 is documented: C-EVAL-7) | `test_generation_runtime.py` | LOW |
| 5.4 | Training configuration (A.3) | A.3 (p. 270–271) | `configs/thesis_appendix_a3_reported.json` | UNVERIFIABLE: reported-only; 1280 is not divisible by 12 | `test_parameter_audit.py` | HIGH |

## 6. Cultural Reinforcement Learning (Ch. 4 §4.5; Ch. 5 §5.6)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 6.1 | R_cult = 0.35·CMPI + 0.30·Virtue + 0.25·Rhetorical + 0.10·Register, Eq. 4.26 | §4.5.1 (p. 115) | — | NOT IMPLEMENTED; UNVERIFIABLE (CMPI has no equation: C-EVAL-2) | — | BLOCKING |
| 6.2 | R_ling = 0.40·PPL⁻¹ + 0.35·BLEU + 0.25·BERTScore, Eq. 4.27 | §4.5.1 (p. 116) | `evaluation/basic_metrics.bleu4` only | NOT IMPLEMENTED | — | HIGH |
| 6.3 | R_coh = 0.50·Coreference + 0.30·Temporal + 0.20·Causal, Eq. 4.28 | §4.5.1 (p. 117) | — (`narrative.validation` gives causal/temporal checks for annotated graphs, which is not the thesis scorer) | NOT IMPLEMENTED | — | HIGH |
| 6.4 | R_total = 0.45·R_cult + 0.35·R_ling + 0.20·R_coh, Eq. 4.29, worked example Eq. 4.30 = 0.854 | §4.5.3–4.5.4 (p. 118–119) | `training.cultural_reward` | MATCH for the aggregation. A.3 lists different weights (0.35/0.30/0.25/0.10, four components: C-REWARD) | `test_thesis_fidelity.py::test_eq_4_30_worked_example` | HIGH |
| 6.5 | AHP weight derivation | §4.5.2 (p. 117); A.11 (p. 280); A.23 (p. 293–294) | `evaluation/ahp.py` | MATCH | `test_thesis_computations.py` | LOW |
| 6.6 | Phases: supervised (epochs 1–30), adversarial (31–65), PPO (66–100) | §5.6.1–5.6.3 (p. 149–150) | `training.py`, `train_pipeline.py` | PARTIAL: algorithms exist; the reward is a surrogate (`SURROGATE_NOT_CRAFT`); schedule and hyper-parameters are not reproduced | `test_core.py` | BLOCKING |

## 7. CRAFT and evaluation (Ch. 6, App. A.13–A.14, A.24)

| # | Thesis element | Reference | Code | Status | Test | Impact |
| --- | --- | --- | --- | --- | --- | --- |
| 7.1 | CRAFT dimensions: CMPI, CulturalScore, ALDA, MFA | §6.2 (p. 169–174) | `evaluation/craft.py` (registry refuses undefined metrics) | UNVERIFIABLE: definitions or combining rules missing (C-EVAL-1/2/5/6) | `test_craft_interfaces.py` | BLOCKING |
| 7.2 | Gap closure, Eq. 6.1 / Eq. A.1 | §6.1.4 (p. 166) | `evaluation/statistics.py` | MATCH | `test_thesis_computations.py` | LOW |
| 7.3 | Bonferroni and bootstrap tests | §6.2.1; A.12 (p. 281) | `evaluation/statistics.py` | PARTIAL: 9 vs 6 comparisons; percentile vs BCa (C-EVAL-3/4) | `test_thesis_computations.py` | MEDIUM |
| 7.4 | Human evaluation protocol | §6.1.3 (p. 165); A.13.1 (p. 282); A.24 (p. 295–296) | `configs/evaluation/craft.yaml` (protocol fields) | PARTIAL: interface only; raters and ratings unavailable (C-EVAL-8) | — | BLOCKING |
| 7.5 | Test set: 500 narratives vs 525 | §6.1.2; §5.4.4; A.14 (p. 283) | — | UNVERIFIABLE (C-EVAL-12) | — | BLOCKING |
| 7.6 | Baselines | §6.1.1 (p. 161); A.5 (p. 272–273) | — | NOT IMPLEMENTED; inconsistent between chapters (C-EVAL-9) | — | HIGH |

## 8. What reproduction needs, in order

1. **Owner artefacts:**
   - the ANHC corpus with its split file;
   - a CKG / CST-Onto export with schema embeddings;
   - the cultural-marker lexicon and the MOTIF codebook;
   - the original checkpoints, if they exist;
   - the evaluation test set.

   These items turn the BLOCKING rows into testable ones.
2. **Owner decisions on conflicting specifications:**
   - backbone (C-MODEL-1);
   - tokenizer (C-MODEL-3);
   - τ (C-CSAM-5);
   - reward weights (Eq. 4.29 vs A.3);
   - CMPI definition (C-EVAL-2);
   - activation weights (D4).

   Each choice is recorded as a variant in `configs/specification/thesis_variants.yaml`. None is chosen silently.
3. **Code that must exist before a reproduction is attempted:**
   - the morphology pipeline (2.2);
   - the CSAM auxiliary losses (3.4);
   - the Eq. 4.25 gating (4.2);
   - the reward components (6.1–6.3), once defined.

**Rule.** Every such change is ANSG-Original work and must follow the thesis text. Any deviation is recorded here as DIVERGENT. ANSG-HNC never substitutes for any of it.
