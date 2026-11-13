# Integration execution plan (Issue #3)

Companion to `INTEGRATION_COMPATIBILITY_AUDIT.md` (conflict ids `I-*`). Principles,
from `PROJECT_GOVERNANCE.md`:

* every step is a **separate, reviewable PR from `master`**; no large merge before
  the topology guard and both test suites pass in one tree;
* `implementation/ansg-core-2026` and `claude/relaxed-knuth-q63k61` are never
  rewritten; their commits are brought in by merge (history preserved);
* **all thesis specifications, experimental configurations and evaluation
  protocols are carried forward unchanged** — in particular
  `configs/specification/thesis_variants.yaml`, `configs/evaluation/craft.yaml`,
  `configs/knowledge/ckg.yaml`, `configs/data/nights.yaml`,
  `docs/SPECIFICATION_CONFLICTS.md` (data branch) and
  `configs/thesis_appendix_a3_reported.json`, `configs/thesis_ch5_reported.json`,
  `docs/thesis_traceability.md` (core branch). Probe step 10 checks that the two
  records keep agreeing (18 values, 0 mismatches);
* nothing invents annotations, CRAFT formulas, reward definitions or checkpoints;
  all test data are synthetic and labelled.

## 0. Status (Issue #11)

| Plan PR | Status |
| --- | --- |
| 0 Audit (Issue #3) | merged (#5) |
| 1 Data branch into `master` (Issue #6) | merged (#10) |
| 2 Namespace unification | **done in Issue #11**: `git mv` of `data.py → data/training_dataset.py`, `evaluation.py → evaluation/basic_metrics.py`, `knowledge.py → knowledge/lexical_baseline.py`; lazy PEP 562 re-exports; unified `pyproject.toml`; topology guard 0 findings |
| 3 Schema-mask fix | done by ChatGPT (#9, merged into the model branch) and brought in by Issue #11 |
| 4 ANHC → training adapter | **done in Issue #11**: `ansg.data.training_examples` (`SchemaVocabulary` PAD = 0, dialect table, `TrainingExample`, `ansg-build-examples`); dataset keeps missing dialect/schema unknown; DAL treats dialect id −1 as unknown (soft gate) |
| 5 Activation weights in the model | open (decision D4) |
| 6 Evaluation separation | partly: `weighted_cmpi` labelled surrogate (docstring + `is_dissertation_cmpi = False`); CRAFT registry still refuses CMPI; rename is decision D3 |
| 7 CI consolidation | **done in Issue #11**: single `.github/workflows/ci.yml` |

## 1. Ordered PRs

| # | PR (scope) | Content | Non-destructive because | Gate (must pass) |
| --- | --- | --- | --- | --- |
| 0 | **This PR** — audit | Audit + plan docs; `tools/check_package_topology.py` (+ tests); `tools/integration_probe.py`; CI `integration-audit.yml` | Adds new paths only; no `src/` change on `master` | CI green; topology 0 on `master`; probe 0 FAIL |
| 1 | Bring the data branch into `master` | Merge `claude/relaxed-knuth-q63k61` (dry run `git merge-tree origin/master origin/claude/relaxed-knuth-q63k61`: no conflicts; each branch alone also merges into `master` cleanly — conflicts appear only when **both** are combined) | Merge commit keeps history | data suite green under CI; topology 0 |
| 2 | Namespace unification for core (I-PKG-1/2/3/8) | On a branch from updated `master`: merge `implementation/ansg-core-2026`, then in the **same PR, separate commits**: (a) `git mv src/ansg/data.py src/ansg/data/training_dataset.py`, (b) `evaluation.py → evaluation/basic_metrics.py`, (c) `knowledge.py → knowledge/lexical_baseline.py`, (d) lazy re-export shims (§2), (e) unified `pyproject.toml`/`__init__.py` | `git mv` keeps blame; shims keep every existing import path working | topology 0; core 18 + data 130 tests green in one tree; import-regression test (§4.1) |
| 3 | Schema-mask fix (I-CSAM-1, I-CKG-4) | `collate` returns `schema_mask`; thread through `ANSG.forward`, `CulturalSchemaAttention`, `PairwiseCulturalBias` (masked softmax), DAL summary (masked mean); defined behaviour for an all-masked row | Default behaviour unchanged for unpadded batches | regression: same example alone vs batched, \|Δlogit\| < 1e-5 |
| 4 | ANHC → training adapter (I-DATA-1/2/4/7, I-CKG-2) | `ansg.integration.contracts` (dataclasses below), `DialectCode` table, `SchemaVocabulary` (0 = PAD), adapter from split JSONL to `TrainingExample` | Pure addition; torch-free | adapter unit + edge tests (§4.2) |
| 5 | Activation → model (I-CKG-3) | Optional `schema_weights` tensor end to end; model ignores it unless configured (Eq. 4.12 vs learned weights is an owner decision) | Opt-in flag | shape/gradient tests both modes |
| 6 | Evaluation separation (I-EVAL-1) | Keep `CraftRegistry` as the only CRAFT entry point; core `weighted_cmpi` renamed or flagged per reviewer decision | Old name kept as deprecated alias if retained | registry refuses CMPI; no CMPI-named output in reports |
| 7 | CI consolidation (I-CI-1) | One workflow running both suites, topology, probe (until removed), smoke pipeline | — | green on PR head |

PRs 1–2 are the "large merge"; they happen only after this PR is reviewed.

## 2. Shim design for PR 2 (lazy, torch-free import of the data layer)

`src/ansg/data/__init__.py` (added lines only):

```python
_CORE = {"NarrativeDataset": "training_dataset", "ByteTokenizer": "training_dataset", "collate": "training_dataset"}

def __getattr__(name):            # PEP 562: imports torch only when a core symbol is requested
    if name in _CORE:
        from importlib import import_module
        return getattr(import_module(f".{_CORE[name]}", __name__), name)
    raise AttributeError(name)
```

Same pattern for `ansg.evaluation` (`bleu4`, `summarize`, `weighted_cmpi` → `basic_metrics`)
and `ansg.knowledge` (`Motif`, `CulturalKnowledgeGraph` → `lexical_baseline`).

`ansg.knowledge.CulturalKnowledgeGraph` keeps resolving to **core's** lexical class,
because core code and tests import it under that path; the data branch's class stays
at `ansg.knowledge.ckg.CulturalKnowledgeGraph`. The identical names are a known
hazard (I-CKG-1); the shim emits `DeprecationWarning` pointing to
`ansg.knowledge.lexical_baseline` so that a later PR can rename without breakage.

Unified `pyproject.toml` (I-PKG-3): name decided by owner; `dependencies = ["PyYAML>=6.0"]`;
extras `model = ["torch>=2.1", "numpy>=1.24"]`, `metrics`, `test`, `ontology`, `splits`, `legacy`;
`[tool.pytest.ini_options] pythonpath = ["src"]`, `testpaths = ["tests"]`, marker `integration`;
console script `ansg-ingest`.

Installation:

```bash
python -m pip install -e ".[test]"                    # data/knowledge/evaluation layer, no torch
python -m pip install -e ".[test,model]"              # + core model and training
python -m pytest -q                                   # all suites
python tools/check_package_topology.py                # must report 0 findings
```

## 3. Data-flow contracts

Target flow: **Arabic narrative sources → preprocessing → ANHC annotations → CKG
activation → CSAM → DAL → generation → CRAFT evaluation (reported separately)**.
Each contract names its producer, consumer, types and invariants. "Owner" marks a
value the dissertation does not fix.

### K1 Source → ingestion
* **In:** licensed UTF-8 file + `SourceProvenance` (edition, licence, `redistributable`). Nights witnesses: Mahdi 1984 / Bulaq 1835 / Calcutta II as separate sources (§3.2.2, §3.2.4).
* **Producer:** `ansg.data.nights.ingest_file` (data branch).
* **Out:** `NarrativeRecord` per tale + `IngestionReport`.
* **Invariants:** no network; missing file → error; no annotation created.

### K2 Preprocessing
* **In:** raw text; `NormalizationConfig` (default non-destructive; `thesis_appendix_a8` profile available).
* **Out:** `NormalizedText(text, offset_map)`; record keeps `normalization_config_sha256`.
* **Invariants:** offsets map back to raw text; newlines preserved (core `normalize_arabic` is **not** used for corpus text, I-DATA-3). Model-input diacritic policy: owner (C-NORM-1).

### K3 ANHC annotations
* **Type:** data `NarrativeRecord` with `dia`, `motif`, `moral` lists and `annotation_status ∈ {unannotated, automatic, silver, gold}` (§3.5, Table 3.7).
* **Labels:** DIA ∈ {MSA, Egyptian, Levantine, Gulf, Maghrebi} (§3.5.1); MORAL ∈ 8 classes (§3.5.3); MOTIF ids = CKG node ids — inventory **not available** (C-ANHC-3).
* **Invariants:** gold only from supplied files; automatic never upgraded; text withheld for non-redistributable sources.

### K4 Training example (new, PR 4)
```python
@dataclass(frozen=True)
class TrainingExample:
    record_id: str
    text: str                       # never None; records with withheld text are skipped and counted
    split: Literal["train", "validation", "test"]
    dialect: DialectCode | None     # from exactly one document-level DIA annotation, else None
    schema_node_ids: tuple[str, ...]   # CKG node ids from activation (K5), ranked
    schema_weights: tuple[float, ...]  # same length; activation scores w_s
    label_status: Mapping[str, str]    # copied from record.annotation_status
```
* `DialectCode`: `MSA→("msa",0)`, `Egyptian→("egyptian",1)`, `Levantine→("levantine",2)`, `Gulf→("gulf",3)`, `Maghrebi→("maghrebi",4)` — integer ids follow core `DIALECTS`.
* **Invariants:** no default dialect or schema is ever substituted (core's `schema_ids=[0]` / `dialect_id=0` fallbacks are not used).

### K5 CKG activation
* **In:** text (or context embedding) + `CulturalKnowledgeGraph` (data branch) + activator: `LexicalTriggerEvidence` (inspection baseline), `CosineThresholdActivator` (Eq. 4.20: τ 0.5 strict; A.4: 0.75 inclusive, ≤ 50) — embeddings supplied, not trained here.
* **Out:** `ActivationScores(scores: dict[node_id, float], method, evidence)`.
* **Invariants:** empty result allowed and explicit; `method` recorded in the run manifest.

### K6 Schema vocabulary and batch (PR 3/4)
```text
SchemaVocabulary: PAD = 0; node_id -> 1..N in CulturalKnowledgeGraph.index() order; schema_count = N + 1
Batch = {
  input_ids:     Long[B, T]            tokenizer declared in config (byte tokenizer = smoke only)
  labels:        Long[B, T]            -100 at padding
  schema_ids:    Long[B, S]            0 = PAD
  schema_mask:   Bool[B, S]            False at PAD (new)
  schema_weights:Float[B, S] | None    activation w_s (optional, PR 5)
  dialect_ids:   Long[B] | None        None -> soft gate
}
```
* **Invariants:** `schema_ids < schema_count`; `T ≤ context_length`; a row with no schema has `schema_mask` all False and a defined model behaviour (owner decision, I-CKG-4).

### K7 CSAM (core)
* **In:** hidden `Float[B,T,D]`, `schema_ids`, `schema_mask`; config `cultural_heads=12`, `D % 12 == 0` (1280 rejected; bridge pads to 1284, I-CSAM-4).
* **Out:** `Float[B,T,D]`.
* **Invariants:** masked schemas have zero attention weight; output independent of batch padding (I-CSAM-1); gradients reach q/k only when ≥ 2 schemas are active (I-CSAM-2, reported).

### K8 DAL (core)
* **In:** hidden `Float[B,T,D]`, schema summary `Float[B,C]` (masked mean), `dialect_ids` or None.
* **Out:** `Float[B,T,D]`.
* **Invariants:** five dialects only; explicit ids bypass the gate (I-DAL-2); span-level code-switching not represented (I-DAL-1).

### K9 Generation (core)
* `ANSG.generate(input_ids, schema_ids, dialect_ids, max_new_tokens, temperature, top_k) -> Long[B, T+new]`. Thesis decoding: temperature 0.7, top-p 0.95, max 512, seed 42 (§6.1.2) — top-p is not implemented in core (top-k only).

### K10 Evaluation (reported separately)
* **CRAFT:** only via `CraftRegistry` built from `configs/evaluation/craft.yaml`; every CRAFT metric is `incomplete` and raises `MetricNotSpecifiedError` (C-EVAL-1/2).
* **Implemented statistics:** gap closure (Eq. 6.1), Eq. A.1, κ, bootstrap, Bonferroni, AHP (data branch); BLEU-4 via sacrebleu (core).
* **Reports** state `synthetic | real | not run`; no output is labelled CMPI until the owner supplies a definition (I-EVAL-1).

## 4. End-to-end and edge-case tests (all synthetic, labelled)

### 4.1 Import regression (PR 2)
For every symbol imported by core `train.py`, `train_pipeline.py`, `evaluate.py`,
`scripts/*.py`, `tests/*.py`: `from ansg.<path> import <name>` succeeds; importing
`ansg.data` does not import torch (`"torch" not in sys.modules` after import).

### 4.2 Adapter (PR 4)
* DIA label mapping for all five labels; unknown label → error; two conflicting document-level DIA → `dialect=None` + warning.
* Record with withheld text → skipped and counted.
* No activation → empty schema tuple (no substituted id).
* Schema vocabulary: PAD never assigned to a node; ids stable under input order.
* Split leakage: no `split_cluster` spans two splits after the adapter.

### 4.3 Model contracts (PR 3, 5)
* Shapes: `[B,T,D]` preserved by CSAM and DAL; logits `[B,T,V]`.
* Padding invariance: same example alone vs batched, |Δlogit| < 1e-5.
* All-masked schema row: defined output, finite loss.
* Gradients: non-zero for CSAM q/k with ≥ 2 schemas; DAL gate gets gradient iff `dialect_ids is None` (after a non-zero LoRA `up`).
* `schema_weights` mode: shape/gradient tests; off by default.

### 4.4 End-to-end (after PR 4)
`synthetic Nights fixture → ingest → synthetic DIA (AUTOMATIC) → CKG activation →
TrainingExample → batch → ANSG forward/backward → generate 4 tokens → CraftRegistry
refuses CMPI → report {synthetic: true, metrics: "not run"}`. This replaces
`tools/integration_probe.py`, which is retired at that point.

## 5. Decisions required from the owner (blocking specific PRs)

| # | Decision | Blocks |
| --- | --- | --- |
| D1 | Package/distribution name (`ansg` vs `ansg-research`) and version string | PR 2 |
| D2 | Behaviour of a row with no active schema | PR 3 |
| D3 | Whether core's `weighted_cmpi` is renamed, flagged, or removed from reporting | PR 6 |
| D4 | Eq. 4.12 activation weights vs learned schema weights | PR 5 |
| D5 | Thesis variants to reproduce first (backbone, protocol, reward, normalisation: C-MODEL-1, C-TRAIN-1, C-REWARD-1, C-NORM-1) | reproduction phase |
| D6 | Supervised training of the DAL gate when dialect labels exist (I-DAL-2) | reproduction phase |
