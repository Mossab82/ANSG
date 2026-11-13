# ANSG architecture

> Updated 2026-11-13 after the dissertation was supplied. Section/equation numbers are the
> thesis'. Where chapters disagree the competing versions are referenced by conflict id
> (`SPECIFICATION_CONFLICTS.md`); none is chosen here. Inputs/outputs of implemented
> components are taken from the code.

## 1. Component overview

```text
                ┌──────────────────────── Phase 1 (implemented) ────────────────────────┐
 licensed text ─▶ normalisation ─▶ tale/night segmentation ─▶ ANHC records ─▶ splits ─▶ JSONL + manifest
 (Nights, …)     (offset-tracked)   (+ formula detection)      (DIA/MOTIF/MORAL     (hash/predefined)
                                                                slots, provenance)
 ArabicVerbNet XML ─▶ VerbNet parser ──┐
 ArabicNameNet export ─▶ adapter ──────┼─▶ lexical / morphological conditioning (Phase 2)
 CKG JSON + CST-Onto ─▶ CKG interface ─┴─▶ index + adjacency + ActivationScores ─▶ CSAM (Phase 2)
                └───────────────────────────────────────────────────────────────────────┘
 Phase 2+: backbone LM + CSAM + DAL/LoRA ─▶ staged training ─▶ RL with cultural reward ─▶ CRAFT evaluation
```

## 2. Implemented components (Phase 1)

### 2.1 Arabic normalisation — `ansg.data.normalization`
* **Input:** raw Unicode string; `NormalizationConfig`.
* **Output:** `NormalizedText(text, raw, offset_map, operations)`; `offset_map[i]` = raw index of `text[i]`.
* **Algorithm:** (1) cluster base+combining marks, fold Arabic presentation forms (NFKC on U+FB50–FDFF, U+FE70–FEFF only), apply NFC/NFKC per cluster; (2) per-character deletions/substitutions (invisible controls, tatweel; opt-in: diacritics, dagger alef, Qur'anic marks, alef/alef-maqsura/ta-marbuta folding, digit script); (3) whitespace collapsing with newline cap.
* **Defaults:** non-destructive for orthography and diacritics (rhetorical/cultural preservation).
* **Thesis reference:** App. A.8 profile available (`profile: thesis_appendix_a8`); default stays non-destructive because Ch. 5 §5.4.1 conflicts (C-NORM-1).

### 2.2 Segmentation — `ansg.data.segmentation`
* Paragraphs (blank line) and sentences (`. ! ? ؟ ؛ …`; `،` optional), offsets preserved; decimal points protected.

### 2.3 Nights ingestion — `ansg.data.nights`, `ansg.data.arabic_numbers`
* **Input:** one UTF-8 file per edition; `SourceProvenance`; `NightsConfig` (regexes for night headings, tale titles, frame formulae).
* **Output:** one `NarrativeRecord` per tale (+ optional preamble) with `night_numbers`, title, source spans (normalised and raw), detected formulae; `IngestionReport` (heading counts, unparsed headings, gaps/regressions in night numbering).
* **Night numbers:** digits (Western/Arabic-Indic) or ordinal words up to 1001; strict — unknown words give `None` and a warning, never a guess.
* **Record id:** `{source_id}:{index:05d}:{sha256(text)[:12]}`.

### 2.4 ANHC records — `ansg.data.records`, `ansg.data.anhc`
* `NarrativeRecord`: `text`, `source` (provenance & copyright), `narrative` metadata, layers `dia`/`motif`/`moral`, per-layer `annotation_status ∈ {unannotated, automatic, silver, gold}`, `split`.
* Annotation types: `DialectAnnotation(label)`, `MotifAnnotation(motif_id)`, `MoralAnnotation(framework, value)`; all have `span | None` (document-level), `annotator_id`, `source`, `confidence`, `guideline_version`.
* Validation: spans in range, confidence ∈ [0,1], gold layers contain only adjudicated/imported human annotations, automatic annotations never in silver/gold, labels ∈ configured inventory (if supplied).
* I/O: JSONL; text of non-redistributable sources withheld unless explicitly allowed; `import_annotations` merges supplied gold/silver files.

### 2.5 Splitting — `ansg.data.splits`
* Strategies: `predefined` (reproduce original partition), `hash_exact` (exact ratios, order-independent), `hash_bucket` (stable under corpus growth). Group-level (default: tale) to prevent leakage.

### 2.6 ArabicVerbNet — `ansg.knowledge.arabic_verbnet`
* **Input:** VerbNet-style XML class files. **Output:** `ArabicVerbNet` (classes with members `{verb, root, deverbals, participles, attributes}`, thematic roles + selectional restrictions with inheritance, frames `{description, examples, syntax, semantics}`, subclasses); indexes by verb and root; `legacy_rows()`.

> ArabicVerbNet (2.6) and ArabicNameNet (2.7) are **not ANSG components in the dissertation** (C-RES-1); they were built because the Phase 1 brief requested them.

### 2.7 ArabicNameNet — `ansg.knowledge.arabic_namenet`
* **Input:** CSV/TSV/JSONL export + YAML column mapping (format unverified). **Output:** entities (id, name, type, variants, attributes), relations, folded-name lookup, longest-match gazetteer over text.

### 2.8 CKG interface — `ansg.knowledge.ckg`
* **Input:** CKG JSON (nodes = schemas/motifs, edges = schema relationships, ontology classes) and optionally CST-Onto as OWL/Turtle (via `rdflib`).
* **Output for CSAM:** `index()` (stable id → row), `adjacency()` (sparse `(i, j, w)`), `ActivationScores.as_vector(index)`.
* **Declared vs supplied:** `DeclaredStructure` (from config/thesis) is only compared against supplied data (`count_report`, `validate`).
* **Activation:** `SchemaActivator` protocol; `CosineThresholdActivator` implements Eq. 4.20 / App. A.4 given supplied embeddings. `LexicalTriggerEvidence` is a labelled inspection utility, not the thesis function.
* **Declared variants:** `configs/knowledge/ckg.yaml` keeps the four incompatible count statements (Ch. 3, Table A.16, Table 4.8, App. A) side by side.

### 2.9 CRAFT — `ansg.evaluation.craft`
* Interfaces and a transcribed registry; every metric `incomplete` (no definitions in the thesis).

### 2.10 Thesis-specified procedures implemented in Phase 1b
* `evaluation.statistics` (Eq. 6.1, Eq. A.1, Bonferroni, κ, bootstrap, Cohen's d), `evaluation.ahp` (AHP eigenvector/CR), `evaluation.corpus_metrics` (CRD, Eq. 3.1), `knowledge.cultural_weight` (Eq. 4.2), `knowledge.ckg.CosineThresholdActivator` (Eq. 4.20 / A.4), `data.splits.cluster_records_dbscan` (§5.4.4), `NormalizationConfig.thesis_appendix_a8()` (A.8), Table 5.4 formula set (reconstructed, T-1).

## 3. Thesis components and their specifications (Phase 2+)

| Component | Thesis | Inputs → outputs | Specification as written | Conflicts |
| --- | --- | --- | --- | --- |
| Backbone | §4.1.6, Table 5.1, A.3 | token ids → hidden states/logits | GPT-2 Medium 355M **or** 774M base **or** AraGPT2-large (hidden 1280, 20 heads) | C-MODEL-1/2/3 |
| Morphological conditioning | §4.2, Eq. 4.2–4.7 | tokens → m_t ∈ R⁶⁴, M_seq | 6 stages: decomposition (Root, Pattern, POS, Voice, Aspect, Case; Eq. 4.3); 47 patterns (Eq. 4.4); dialect vector d_t (Eq. 4.5); WordNet / motif DB / historical lexicon; m_t = [r 16; p 12; v 8; d 12; c 16] (Eq. 4.6); attention pooling (Eq. 4.7) | C-MORPH-1 |
| Cultural weight | Eq. 4.2 | (root, pattern) counts + expert Likert → w_c | α = 0.6 — **implemented** (`knowledge.cultural_weight`) | — |
| CKG / schema embeddings | §4.3.2, Eq. 4.8 | CST-Onto → c_s | OWL2Vec* 300-d → W_proj → 768 (or 256-d, A.4) | C-CKG-1/2, C-CSAM-5 |
| Schema activation | Eq. 4.20, A.4 | context, c_s → w_s, S_active | cosine vs τ — **implemented** as `CosineThresholdActivator` (both variants; embeddings supplied) | C-CSAM-5 |
| CSAM attention | Eq. 4.9–4.14 | Q, K, V, S_active → biased heads | softmax(QKᵀ/√d_k + βH_i + γR_i)V, β 0.347, γ 0.283 | C-CSAM-1…6 |
| CSAM objective | Eq. 4.16–4.19 | annotated spans, schema labels → loss | L_LM + 0.1 L_cultural (BCE on motif-span pairs vs sampled negatives) + 0.05 L_schema (multi-label BCE) | C-TRAIN-1 |
| DAL | Eq. 4.24–4.25, Alg. 1 | h, s_ctx → gated LoRA | W_d = W + B_d A_d, r = 8; g_d = σ(v_dᵀ[h; s_ctx])·1[compatible] | C-DAL-1/2 |
| Training | Ch. 3–5, A.26 | ANHC → checkpoints | five incompatible protocols | C-TRAIN-1…5 |
| Reward / RL | Eq. 4.26–4.30, 5.1–5.3, A.3 | generated text → scalar | five incompatible definitions; PPO settings Table 5.9 | C-REWARD-1…6 |
| CRAFT | Ch. 6 | outputs, references, CKG, ratings → scores | no computable metric definitions | C-EVAL-1…12 |

### Data flow as described in §4.1.1 / Algorithm 1

prompt → morphological conditioning (m_t) → schema activation (S_active, τ) →
per-step CSAM bias (H_i, R_i) → transformer → P(y_t) → DAL gating per dialect →
P_final → sample → update S_active. RL (training only) scores outputs with the reward.
Algorithm 1 is dimensionally inconsistent at lines 18–20 (C-DAL-2).
