# ANSG architecture

> Component names come from `README.md` (owner, commit `faaa3c0`) and the
> Phase 1 brief. Inputs/outputs of **implemented** components are taken from
> the code. For **not-yet-implemented** components, formulas, shapes and
> hyperparameters are marked `TO TRANSCRIBE` because the dissertation was not
> available in this session (see `docs/REPOSITORY_AUDIT.md` §0).

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
* **Thesis reference:** Ch. 3 (preprocessing) — exact steps TO TRANSCRIBE.

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

### 2.7 ArabicNameNet — `ansg.knowledge.arabic_namenet`
* **Input:** CSV/TSV/JSONL export + YAML column mapping (format unverified). **Output:** entities (id, name, type, variants, attributes), relations, folded-name lookup, longest-match gazetteer over text.

### 2.8 CKG interface — `ansg.knowledge.ckg`
* **Input:** CKG JSON (nodes = schemas/motifs, edges = schema relationships, ontology classes) and optionally CST-Onto as OWL/Turtle (via `rdflib`).
* **Output for CSAM:** `index()` (stable id → row), `adjacency()` (sparse `(i, j, w)`), `ActivationScores.as_vector(index)`.
* **Declared vs supplied:** `DeclaredStructure` (from config/thesis) is only compared against supplied data (`count_report`, `validate`).
* **Activation:** `SchemaActivator` protocol. Thesis activation function: TO TRANSCRIBE. `LexicalTriggerEvidence` is a labelled inspection utility, not the thesis function.

### 2.9 CRAFT — `ansg.evaluation.craft`
* Interfaces only; see `docs/CRAFT_EVALUATION_SPECIFICATION.md`.

## 3. Planned components (Phase 2+) — specifications TO TRANSCRIBE

| Component | Inputs | Outputs | Formulas / hyperparameters |
| --- | --- | --- | --- |
| Backbone LM | token ids | hidden states, logits | Backbone choice disputed (see EXPERIMENTAL_SPECIFICATIONS §1) |
| Morphological conditioning | tokens, roots/patterns (ArabicVerbNet) | conditioning features | TO TRANSCRIBE (Ch. 4) |
| CSAM | hidden states; CKG schema embeddings; activation scores | culturally-conditioned hidden states | TO TRANSCRIBE (Ch. 4): attention equations, dims, insertion layers |
| DAL / LoRA | backbone weights; dialect id | adapted weights | TO TRANSCRIBE: rank, alpha, target modules |
| Training stages | ANHC splits | checkpoints | 3 phases (Ch. 4–5) vs 4 stages (App. A.26) |
| RL / reward | generated stories, CKG, references | policy update | 3-term (Ch. 5) vs 4-term (App. A.3) reward; weights TO TRANSCRIBE |
| CRAFT | generated stories, references, CKG, human ratings | metric scores | Ch. 6, TO TRANSCRIBE |
