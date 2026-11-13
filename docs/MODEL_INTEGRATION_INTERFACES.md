# Model integration interfaces (Issue #6, Step 4)

Exact interfaces needed to connect the data/knowledge layer now in this branch
to the PyTorch model on `implementation/ansg-core-2026`. That branch is **not
merged** here. Signatures in §1 are quoted from core head **0958e30**; §3 also
describes the unmerged branch `chatgpt/fix-csam-schema-padding` at **8ea90d6**
(six commits on top of 0958e30), which was verified locally but is under review
and not part of this PR. Contract ids `K*` and conflict ids `I-*` refer
to `INTEGRATION_EXECUTION_PLAN.md` and `INTEGRATION_COMPATIBILITY_AUDIT.md`;
`C-*` refer to `SPECIFICATION_CONFLICTS.md`.

Nothing here supplies missing cultural annotations, CRAFT formulas or checkpoints.

## 1. What each side exposes today

### Data/knowledge layer (this branch, torch-free)

| Purpose | Interface |
| --- | --- |
| Corpus → records | `ansg.data.pipeline.run_pipeline(config, base_dir) -> manifest dict`; writes `<out>/{train,validation,test}.jsonl`, `split_assignment.json`, `manifest.json` |
| Records I/O | `ansg.data.anhc.read_jsonl(path) -> Iterator[NarrativeRecord]` (raises if text was withheld) |
| Record | `ansg.data.records.NarrativeRecord(record_id, text, source: SourceProvenance, narrative: NarrativeMetadata, dia, motif, moral, annotation_status, split, normalization_config_sha256, schema_version)` |
| Annotation provenance | `_AnnotationBase(span, annotator_id, source: AnnotationSource, confidence, guideline_version, attributes)`; layer status `AnnotationStatus ∈ {unannotated, automatic, silver, gold}` |
| Label inventories | `configs/data/nights.yaml: label_inventories` — DIA `[MSA, Egyptian, Levantine, Gulf, Maghrebi]`, MORAL 8 classes, MOTIF `null` (codebook absent, C-ANHC-3) |
| CKG | `ansg.knowledge.ckg.CulturalKnowledgeGraph.index(kinds=None) -> dict[node_id, int]` (**0-based**, sorted by id); `adjacency(index, relations=None, default_weight=1.0) -> list[(i, j, w)]` |
| Activation | `SchemaActivator.activate(text, graph) -> ActivationScores(scores: dict[node_id, float], method: str, evidence)`; implementations `LexicalTriggerEvidence` (inspection baseline), `CosineThresholdActivator` (Eq. 4.20 / A.4; embeddings supplied) |
| Run provenance | `ansg.utils.provenance.RunManifest` (git commit + dirty flag, config SHA-256, input/output SHA-256, real UTC time) |

### Model (core branch, requires torch)

| Purpose | Interface (verbatim) |
| --- | --- |
| Dialect inventory | `ansg.config.DIALECTS = ("msa", "egyptian", "levantine", "gulf", "maghrebi")` — integer id = position |
| Config | `ModelConfig(vocab_size=256, context_length=128, hidden_size=96, transformer_heads=4, layers=2, cultural_heads=12, schema_count=64, schema_dim=64, lora_rank=8, lora_alpha=16.0, dropout=0.0, alpha_base=0.370, beta_heritage=0.347, gamma_rhetorical=0.283)` |
| Dataset | `NarrativeDataset(filename, context_length=128, default_schema=0)`; JSONL rows `{"text": str, "schema_ids": [int], "dialect_id": int}`; **defaults** `schema_ids=[0]`, `dialect_id=0` when absent |
| Batch | `collate(batch) -> {"input_ids", "labels", "schema_ids", "dialect_ids"}` — `schema_ids` zero-padded, **no mask** |
| Model | `ANSG.forward(input_ids, schema_ids, dialect_ids=None, labels=None)`; `ANSG.generate(input_ids, schema_ids, dialect_ids=None, max_new_tokens=32, temperature=1.0, top_k=None)` |
| CSAM | `CulturalSchemaAttention.forward(hidden_states, schema_ids, schema_mask=None)` — mask supported here but **never passed** by `ANSG.forward`; raises if a row is fully masked |
| Pairwise bias | `PairwiseCulturalBias.forward(schema_ids, seq_len) -> [B, transformer_heads, T, T]` — **no mask**; `softmax(schema_importance)` includes padding |
| DAL | `DialectalAdaptation.forward(hidden, schema_summary, dialect_ids=None)`; `schema_summary = csam.schema_embeddings(schema_ids).mean(dim=1)` (includes padding) |
| Training | `supervised_step(model, batch, optimizer)` calls **`model(**batch)`** — every batch key must be a `forward` keyword |
| Provenance | `write_run_manifest(destination, config, source_data=None, seed=42)` |

## 2. Schema indexing and padding (I-CKG-2, I-CSAM-1)

**Conflict.** `CulturalKnowledgeGraph.index()` assigns 0 to the first real node;
core reserves 0 for padding / "no schema" and pads with 0 in `collate`.

**Interface (new, torch-free, in the data layer):**

```python
@dataclass(frozen=True)
class SchemaVocabulary:
    PAD: ClassVar[int] = 0
    node_to_id: Mapping[str, int]        # = {n: i + 1 for n, i in graph.index(kinds).items()}
    source_sha256: str                   # SHA-256 of the CKG file the ids were built from
    kinds: tuple[str, ...] | None        # which node kinds were indexed

    @property
    def schema_count(self) -> int: ...   # len(node_to_id) + 1  -> ModelConfig.schema_count
    def encode(self, node_ids) -> list[int]: ...   # unknown id -> KeyError, never 0
    def decode(self, ids) -> list[str]: ...        # PAD -> skipped
```

* `ModelConfig.schema_count` must equal `vocab.schema_count`; a checkpoint must store `source_sha256` so that ids are never reinterpreted against a different CKG.
* Thesis sizes for `schema_count - 1`: 898 (Table 4.2) or 1,847 (A.3/A.25) — C-CKG-1; no default is chosen.

**Tests:** PAD never assigned; ids stable under input order; `encode` of an unknown node raises; round-trip.

## 3. CSAM attention masks (I-CSAM-1, I-CKG-4)

Measured defect at core 0958e30: the same example alone vs batched differs by up to 0.874 in logits.

**State on `chatgpt/fix-csam-schema-padding` (8ea90d6, unmerged, under review):**

```text
collate(batch) -> {..., "schema_mask": Bool[B, S]}      # ids != 0; empty schema lists allowed
ANSG.forward(input_ids, schema_ids, dialect_ids=None, labels=None, schema_mask=None)
ANSG.generate(..., top_k=None, schema_mask=None)
CulturalSchemaAttention.forward(hidden_states, schema_ids, schema_mask=None)
PairwiseCulturalBias.forward(schema_ids, seq_len, schema_mask=None)   # masked softmax
PretrainedANSG.forward(..., attention_mask=None, schema_mask=None)
default schema_mask = schema_ids.ne(0); an id 0 marked active -> ValueError
row with no active schema: CSAM returns its input unchanged, pairwise bias = 0,
DAL summary = zero vector (masked mean with clamp)
```

Verified here (Python 3.11, torch 2.15 CPU): that branch's suite **24 passed**
(18 original + 6 in `tests/test_schema_padding.py`), and `tools/integration_probe.py`
step 7 measures **1.4e-6** (was 0.874).

Consequences for this layer:

* PAD = 0 in `SchemaVocabulary` (§2) matches the branch's `schema_ids.ne(0)` default exactly.
* `schema_mask` is a `forward` keyword, so `supervised_step`'s `model(**batch)` keeps working.
* The empty-row behaviour (identity CSAM, zero bias) is a concrete proposal for decision D2; the owner still has to accept it scientifically.
* Still open on that branch: `NarrativeDataset` keeps `dialect_id=0` as a default for rows without a dialect (I-DATA-4); CSAM query/key receive no gradient with a single active schema (I-CSAM-2, a design property, unchanged).

## 4. Dialect identifiers for DAL (I-DATA-2, I-DAL-1/2)

| Thesis label (§3.5.1, data inventory) | DAL code (core `DIALECTS`) | id |
| --- | --- | ---: |
| MSA | msa | 0 |
| Egyptian | egyptian | 1 |
| Levantine | levantine | 2 |
| Gulf | gulf | 3 |
| Maghrebi | maghrebi | 4 |

* Source of a training example's dialect: **exactly one document-level DIA annotation** (`span is None`). Otherwise `dialect_id = None` → DAL soft gate. Never core's default `0`.
* Span-level DIA annotations (code-switching, §4.4.4) cannot be represented by core's per-example `dialect_ids: Long[B]`. Open question.
* Iraqi (Table 8.6) and "Mixed" (A.17) are not in either inventory (C-ANHC-5).
* With explicit ids the gate is bypassed and never trained (measured; D6).

**Tests:** all five mappings; unknown label → error; conflicting document-level labels → `None` + warning.

## 5. Cultural annotation provenance (governance)

Every training example and batch must remain traceable to its record and to the
status of each label used:

```python
@dataclass(frozen=True)
class TrainingExample:            # plan contract K4
    record_id: str
    text: str
    split: str
    dialect: str | None           # DAL code
    dialect_label_status: str     # AnnotationStatus of the DIA layer: unannotated|automatic|silver|gold
    schema_node_ids: tuple[str, ...]
    schema_weights: tuple[float, ...]
    activation_method: str        # ActivationScores.method, e.g. "lexical_trigger_count"
    source_sha256: str            # SourceProvenance.file_sha256
```

* Activated schemas are **model inputs derived by an activator**, not MOTIF annotations; they must never be written back into `record.motif` or reported as gold.
* The training run manifest records: data `manifest.json` hash, `split_assignment.json` hash, CKG `source_sha256`, activator method and parameters, normalisation config hash, tokenizer, and the label-status counts actually used.
* Records whose text was withheld (non-redistributable) are skipped and counted, never filled.

## 6. Training and validation dataset contracts (I-DATA-4, I-DATA-7)

```text
data pipeline  -> <out>/{train,validation,test}.jsonl   (NarrativeRecord JSONL, split fixed by §5.4.4 settings)
adapter        -> <out>/model/{train,validation,test}.examples.jsonl  (TrainingExample JSONL, torch-free)
core dataset   <- reads TrainingExample JSONL; tokenises; builds schema_ids via SchemaVocabulary
```

* Splits are decided once by the data pipeline (seed 42, ratios 0.8/0.1/0.1, optional DBSCAN clusters) and never re-drawn by the model code.
* Core `NarrativeDataset` must reject rows without explicit `schema_node_ids`/`dialect` fields instead of defaulting to `0`.
* Test split is held out from every training phase; the thesis test-set definitions conflict (500 vs 525 narratives, C-EVAL-12).

## 7. Recommendations for connecting CKG → CSAM → DAL (ordered, small PRs)

1. **Data-side adapter (torch-free):** `SchemaVocabulary`, dialect table, `TrainingExample` writer + tests. Safe to land before the model.
2. **Model PR on the core line:** review and land `chatgpt/fix-csam-schema-padding` (mask threading + padding-invariance tests, verified in §3); additionally remove the `dialect_id=0` default in `NarrativeDataset`.
3. **Namespace unification (plan PR 2)** before bringing the model into `master`: core's `data.py`, `evaluation.py`, `knowledge.py` would otherwise be shadowed by this branch's packages (topology guard reports the 3 collisions).
4. **Activation weights:** optional `schema_weights` tensor end to end; whether CSAM uses Eq. 4.12's `w_s` or learned weights is an owner decision (D4).
5. **Embeddings:** `CosineThresholdActivator` requires schema/context embeddings; OWL2Vec* over CST-Onto (Eq. 4.8) needs the ontology file, which is not available.
6. **Losses:** L_cultural (Eq. 4.18) needs MOTIF spans, which require the MOTIF codebook (C-ANHC-3); L_schema (Eq. 4.19) needs gold schema labels. Neither can be trained on activation output without labelling it as automatic.
