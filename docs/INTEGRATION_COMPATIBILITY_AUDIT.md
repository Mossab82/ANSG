# Integration compatibility audit (Issue #3)

*Audit date 2026-11-13. Branch heads audited:* `master` **725c20d**,
`implementation/ansg-core-2026` **0958e30** ("core"), `claude/relaxed-knuth-q63k61`
**361aeb0** ("data"). Neither source branch was modified. Every finding below was
reproduced with the commands in §7; the CI workflow
`.github/workflows/integration-audit.yml` re-computes them on each run.

Conflict ids `I-*` are integration conflicts (this document). Ids `C-*` refer to
dissertation specification conflicts in `docs/SPECIFICATION_CONFLICTS.md` on the
data branch, which this PR does not copy. All test data are **synthetic**.

## 1. Summary

| Area | Severity | Finding |
| --- | --- | --- |
| Packaging | **Blocking** | `ansg.data`, `ansg.evaluation` and `ansg.knowledge` are *modules* on core and *packages* on data. Git merges them without a conflict; Python then resolves the packages and core's modules become unreachable. Core's tests, `train.py`, `train_pipeline.py` and `evaluate.py` fail with `ImportError` (I-PKG-1). |
| Correctness (core) | **High** | Zero-padded schema slots are attended as real schemas: the same example's logits change by up to **0.874** depending on its batch-mates (I-CSAM-1). |
| Data contract | High | No adapter exists from the data branch's ANHC records to core's training rows. Dialect labels (`MSA` vs `msa`), id spaces (0 = padding on core, 0 = first node on data) and normalisers are incompatible (I-DATA-*, I-CKG-2). |
| Science | Open | Several behaviours are faithful to *one* thesis variant only (CSAM placement, DAL gate, reward, CMPI naming). These are listed, not resolved (§3). |
| Legacy | None | Root legacy scripts are byte-identical on all three branches; both branches made the same security edits independently. |

Textual git conflicts in a trial merge (`master` + data + core) are only four:
`.gitignore`, `README.md`, `pyproject.toml`, `src/ansg/__init__.py`. Everything
else merges "cleanly", which is exactly why the topology guard in this PR exists.

## 2. Packaging, imports and repository layout

| Id | Conflicting files | Cause | Evidence | Recommended minimal resolution | Required tests |
| --- | --- | --- | --- | --- | --- |
| I-PKG-1 | core `src/ansg/{data,evaluation,knowledge}.py` vs data `src/ansg/{data,evaluation,knowledge}/__init__.py` | Same dotted name used for a module and a package; packages take precedence on import | Trial merge: `import ansg.data` → `data/__init__.py`, `NarrativeDataset` absent; `pytest` stops at collection of `tests/test_core.py` (`ImportError: cannot import name 'CulturalKnowledgeGraph'`). `tools/check_package_topology.py` reports 3 `module-package` findings. | Move each core module into the corresponding package under a new, non-clashing name (`data/training_dataset.py`, `evaluation/basic_metrics.py`, `knowledge/lexical_baseline.py`) and re-export core's public names from the package `__init__` through lazy `__getattr__` shims (so the torch-free data layer does not import torch). Exact names: Execution Plan §2. | Topology guard = 0 findings; import-path regression test for every symbol core's scripts/tests import; both suites green in one tree. |
| I-PKG-2 | `src/ansg/__init__.py` (both) | add/add; `__version__` `"0.1.0"` vs `"0.1.0.dev0"`; core `provenance.py` imports `__version__` | trial merge conflict | One `__init__` exporting `__version__` only; version string is an owner decision. | `import ansg; ansg.__version__` |
| I-PKG-3 | `pyproject.toml` (both) | add/add. Name `ansg-research` vs `ansg`; core requires `torch>=2.1, numpy>=1.24, PyYAML`; data requires PyYAML only with extras `model, ontology, splits, legacy`; core `[tool.pytest] pythonpath=["src"]`, data relies on install + `integration` marker; data defines console script `ansg-ingest` | trial merge conflict; data `test_parameter_audit.py` spawns a subprocess that needs `ansg` importable (pytest `pythonpath` does not propagate to subprocesses) | Union: base deps `PyYAML`; extras `model=[torch, numpy]`, `metrics=[sacrebleu, rouge-score]`, `test`, `ontology`, `splits`, `legacy`; keep `pythonpath=["src"]` **and** install `-e .[test,model]` in CI; register the `integration` marker; keep `ansg-ingest`. | `pip install -e .[test]` imports data layer without torch; `.[test,model]` runs all suites. |
| I-PKG-4 | core `src/ansg/provenance.py` vs data `src/ansg/utils/provenance.py` | Two run-manifest implementations (core imports torch; data records git commit/dirty flag and config hash) | read | Keep both for now (different callers); converge in a later PR. | — |
| I-PKG-5 | core `src/ansg/config.py` (`ModelConfig`, JSON) vs data `src/ansg/utils/config.py` (YAML loader with credential guard) | Same word, different concept; no path clash | read | Keep; document `ansg.config` = model shapes, `ansg.utils.config` = file loading. | — |
| I-PKG-6 | core `scripts/parameter_audit.py`, `scripts/validate_spec.py` vs data `scripts/audit_parameters.py`, `src/ansg/utils/parameter_audit.py`, `docs/PARAMETER_AUDIT.md` | Two audits: core counts an instantiated model; data does thesis arithmetic | both report `1280 // 12 = 106 r 8` | Keep both; they answer different questions. | existing |
| I-PKG-7 | core `configs/thesis_*_reported.json` vs data `configs/specification/thesis_variants.yaml` | Duplicate records of the same thesis values | probe step 10 compares 18 values mechanically: **0 mismatches** — A.3 (α 0.370, β 0.347, γ 0.283, r 8, α_LoRA 16, phase-1 10 epochs, LR 5e-5, RL LR 1e-6, PPO 4, reward 0.35/0.30/0.25/0.10) and Ch.5 (847M = 774M + 73M, 30/35/35 epochs, 0.45/0.35/0.20, A100 40GB, torch 2.0.1, transformers 4.30.2) **agree** | Preserve both unchanged; YAML is the full register, JSON files are core's executable subsets. A consistency test should pin them together. | value-equality test |
| I-PKG-8 | `README.md`, `.gitignore` | both edited | trial merge conflict | Hand merge; `.gitignore` = union (`data/processed/`, `*.egg-info/`, `.pytest_cache/`). | — |
| I-PKG-9 | `docs/PROJECT_GOVERNANCE.md` | exists only on `master` (both branches predate it) | `git diff` shows "D" relative to master — not a deletion by either branch | Integration branches must start from `master` (this PR does). | — |

## 3. Interface and scientific conflicts along the target data flow

| Id | Files | Conflict | Evidence | Recommendation | Thesis | Tests |
| --- | --- | --- | --- | --- | --- | --- |
| I-DATA-1 | core `legacy_bridge.NarrativeRecord` vs data `data.records.NarrativeRecord` | Same class name, different schema: core = `story_id, text, dialect, motif_ids: tuple[str], moral_labels: tuple[str]`; data = `record_id, text, source provenance, narrative metadata, dia/motif/moral` annotation objects with spans, annotator, source and per-layer status (`unannotated/automatic/silver/gold`) | read | Data record is the canonical ANHC record (it can represent gold vs automatic and provenance); core's record stays the *legacy-input* type with an explicit converter. Never alias the two names. | §3.5, Table 3.7 | round-trip converter tests; converter refuses to upgrade status |
| I-DATA-2 | core `config.DIALECTS`, `legacy_bridge.VALID_DIALECTS`; data `configs/data/nights.yaml: label_inventories.dia` | `msa, egyptian, levantine, gulf, maghrebi` (core, lowercase, integer id = position) vs `MSA, Egyptian, Levantine, Gulf, Maghrebi` (data, thesis §3.5.1 spelling) | probe step 3: core rejects data labels without mapping | One canonical dialect table (code, thesis label, integer id) with explicit aliases; integer ids fixed by core's order. Iraqi (Table 8.6) / Mixed (A.17) not added until decided (C-ANHC-5). | §3.5.1, §4.4.1, §5.4.1 | mapping tests; unknown label → error |
| I-DATA-3 | core `legacy_bridge.normalize_arabic` vs data `data/normalization.py` | Core collapses *all* whitespace (newlines lost: probe 8 → 0) and its regex `[ً-ٰٟ]` also removes the dagger alef; data is offset-preserving, configurable, has the A.8 profile | probe step 2 | Use data normaliser for corpus ingestion; keep core function only for legacy SQL rows. Which profile feeds the model is an owner decision (C-NORM-1). | App. A.8; §5.4.1 | offset/round-trip tests exist on data branch |
| I-DATA-4 | core `data.NarrativeDataset` vs data `data/anhc.py` JSONL | Core reads `{text, schema_ids:int[], dialect_id:int}` and silently substitutes `schema_ids=[0]`, `dialect_id=0` when absent; data writes full ANHC records and **withholds text** (`null`) for non-redistributable sources | read; probe | Adapter `ANHC record → TrainingExample` (Plan §3, contract K4) that never defaults a label: missing DIA → `dialect_id=None` (soft gate), no activation → explicit empty schema list + mask; withheld text → record skipped and counted. | §3.5, §5.4.4 | adapter unit tests incl. withheld text, no DIA, no schemas |
| I-DATA-5 | core `examples/demo.jsonl`, `examples/ckg_demo.json` | Not labelled synthetic; `demo.jsonl` uses schema ids **4, 5, 6** that `ckg_demo.json` does not define (it has 1–3) | probe pre-check | Label as synthetic; add an id-validity test against the CKG. | governance | id-validity test |
| I-DATA-6 | core `data.ByteTokenizer` | 256-symbol byte tokenizer vs thesis SentencePiece-32k root-aware (C-MODEL-3); data branch has none | read | Tokenizer becomes a declared component of the run config; byte tokenizer stays for smoke tests only. | §5.4.1 | — |
| I-DATA-7 | core `train.py` vs data `data/splits.py` | Core trains on a whole file; data implements 80/10/10, seed 42, DBSCAN clustering (§5.4.4) | read | Core training consumes the split files produced by the data pipeline. | §5.4.4 | leakage test across the adapter |
| I-CKG-1 | core `knowledge.CulturalKnowledgeGraph` vs data `knowledge.ckg.CulturalKnowledgeGraph` | Same name. Core: integer motif ids, keywords, untyped relations, lexical `activate()`. Data: string node ids, kinds, typed edges, ontology classes, declared-count variants, `SchemaActivator` protocol, cosine activator (Eq. 4.20 / A.4) | read | Data class canonical; core class kept as a named lexical baseline. | §4.3.2, Eq. 4.20, A.4 | — |
| I-CKG-2 | core `data.collate`, `csam.py` vs data `ckg.CulturalKnowledgeGraph.index()` | Core treats id **0 as padding / "no-schema"**; data `index()` starts at **0** for the first real node | probe step 4 | `SchemaVocabulary`: 0 = PAD, real nodes 1..N in data-index order; `schema_count = N + 1`. | Table 4.2 (S = 898) / A.3 (1,847) | vocabulary tests; out-of-range id → error |
| I-CKG-3 | core `PairwiseCulturalBias.schema_importance` vs data `ActivationScores.scores` | Eq. 4.12 weights each schema by its activation `w_s`; core ignores activation scores and learns its own softmax weights; core lexical `activate()` defaults to top-8, data A.4 variant allows 50, Ch.4 3–5 typical | read | Carry `schema_weights` through the batch as an optional tensor; whether the model uses `w_s` (Eq. 4.12) or learned weights is a scientific decision. | Eq. 4.12, 4.20, C-CSAM-5 | shape test with weights |
| I-CKG-4 | core `CulturalSchemaAttention.forward` vs data activators | Data activation can return **no** schema (probe: 1 of 2 synthetic tales); core CSAM requires ≥ 1 schema and raises if all are masked; `PairwiseCulturalBias` has no mask at all | probe step 4 | Contract: explicit "no active schema" = row of PAD with mask all-false; model must define behaviour (e.g. skip cultural branch). Decision required. | — | all-masked row test |
| I-CSAM-1 | core `model.ANSG.forward`, `csam.PairwiseCulturalBias`, `data.collate` | **Bug:** `collate` pads `schema_ids` with 0, returns no mask; `ANSG.forward` never passes `schema_mask` to CSAM; pairwise bias and DAL summary average over padding | probe step 7: max \|Δlogit\| = **0.874** for the same example alone vs batched | Add `schema_mask` to `collate` and thread it through CSAM, pairwise bias (masked softmax) and the DAL summary (masked mean). | — | regression: alone == batched (tol 1e-5) |
| I-CSAM-2 | core `csam.CulturalSchemaAttention` | With one active schema the cross-attention softmax is over a single key: query/key projections and the β/γ schema biases get **no gradient** (probe step 6; they do with 2 schemas) | probe step 6 | Not a code bug — a property of the A.25 schema-as-keys design. Thesis expects 3–5 (Table 4.2) but §5.7.1 reports 2.1 per token. Report to owner. | A.25 vs Eq. 4.9 (C-CSAM-4) | gradient test for 1 vs ≥ 2 schemas |
| I-CSAM-3 | core `model.py` | CSAM cross-attention applied once after all blocks + pairwise bias in every block through a learned 12→4 head projection (core documents this as a new decision); Eq. 4.9 adds bias inside each head | read | Keep as a labelled variant; do not describe as Eq. 4.9. | Eq. 4.9, C-CSAM-2/3/4 | — |
| I-CSAM-4 | core `pretrained.py`, `docs/thesis_traceability.md` | Doc says the bridge projects 1280→**1296**→1280; the code `((d+h-1)//h)*h` gives **1284** | arithmetic | Correct the doc on the core branch (owner/PR #1). | C-CSAM-7 | width test |
| I-CSAM-5 | core `ModelConfig.schema_dim` (64 smoke; 256 in `PretrainedANSG`) vs data `ckg.yaml` (768 Ch.4 / 256 A.4) | schema embedding width not fixed | read | Width is a config field; OWL2Vec* initialisation is unavailable (no ontology supplied). | Eq. 4.8 | — |
| I-DAL-1 | core `dal.DialectalAdaptation` | One dialect per example; thesis §4.4.4 describes token-level code-switching; data DIA layer supports span-level labels | read | Contract carries document-level dialect only; span-level dialect is an open question. | §4.4.4 | — |
| I-DAL-2 | core `dal.DialectalAdaptation` | Gate = MLP(schema summary) — ignores hidden state `h` and the compatibility indicator of Eq. 4.25; explicit `dialect_ids` replace the gate by a one-hot, so the gate is **never trained** when labels exist | probe step 6 (gate gradient only without dialect ids) | Report; decision needed on supervised gate training. | Eq. 4.25 | gate-gradient test |
| I-DAL-3 | core `dal.py` | Adapter applied once to final hidden states, not per-layer q/v LoRA (Table 5.8, Eq. 4.24) | read | Labelled variant. | C-DAL-1 | — |
| I-TRN-1 | core `training.cultural_reward` | Hard-codes the Ch.5 three-term weights (0.45/0.35/0.20) and rejects the A.3 four-term variant | read | Reward weights from the variants register; selection is an owner decision (C-REWARD-1). | §5.6.3, A.3 | — |
| I-TRN-2 | core `training.py` | Only L_LM is optimised; L_cultural (Eq. 4.18) and L_schema (Eq. 4.19) are absent. They need MOTIF spans (data branch provides spans) and the MOTIF codebook (absent, C-ANHC-3) | read | Define the loss interface now; implement after codebook decision. | Eq. 4.16–4.19 | — |
| I-TRN-3 | core `training.NarrativeDiscriminator` | Mean-pooled embedding MLP; §5.7.4 reports a 3-layer CNN with feature matching | read | Labelled variant. | §5.7.4 | — |
| I-EVAL-1 | core `evaluation.weighted_cmpi` vs data `configs/evaluation/craft.yaml` | Same metric name, different status: core computes 0.35/0.25/0.25/0.15 (Ch.5 §5.1) under the name CMPI; data registers CMPI as **undefined** and refuses to compute it (C-EVAL-2). Governance P4: "do not silently replace CMPI" | probe step 9 | Owner/reviewer decision: rename to an explicit Ch.5-aggregate name, or keep with a status flag. Never report its output as CMPI. | Ch.5 §5.1; Ch.6 | — |
| I-EVAL-2 | core `evaluate.py` | Generates samples only and states it computes no CRAFT metric | read | Compatible with data-branch policy. | Ch.6 | — |

## 4. Dependencies and Python versions

| Item | core | data | Thesis | Note |
| --- | --- | --- | --- | --- |
| Python | ≥ 3.10; CI 3.11 | ≥ 3.10; tested 3.10, 3.13 | 3.10.12 (Ch.5) / 3.9.7 (A.2) | A.2's 3.9.7 is outside both ranges (C-ENV-1). |
| torch | required, ≥ 2.1 | optional extra | 2.0.1 (Ch.5) / 1.12.0 (A.2) | Audit ran torch 2.15.0+cu132 on CPU. |
| transformers | optional (bridge) | optional | 4.30.2 / 4.24.0 | — |
| numpy | required | — | — | used by core `train.py` |
| PyYAML | required | required | — | — |
| sacrebleu, rouge-score | `metrics` extra | — | — | — |
| rdflib, scikit-learn | — | `ontology`, `splits` extras | — | — |

## 5. Tests and CI

| Id | Finding | Recommendation |
| --- | --- | --- |
| I-TST-1 | Layouts: core `tests/test_*.py`; data `tests/unit/`, `tests/integration/`, `tests/conftest.py`. No duplicate test basenames (checked). | Keep both layouts. |
| I-TST-2 | Core tests assert gradients are *not None*; with zero-initialised LoRA or a single schema they can be all-zero (I-CSAM-2, I-DAL-2). | Assert non-zero gradients where the design requires learning. |
| I-CI-1 | Core workflow `test.yml` triggers on push to its branch and PRs to `master`, installs torch from PyPI (CUDA wheel). Data branch has no CI; its earlier test results were local runs only. | This PR adds `integration-audit.yml` (CPU torch index). Merge the two workflows when the code is integrated. |

## 6. Measured results (local, 2026-11-13; CI re-runs them)

| Command | Result |
| --- | --- |
| core suite, isolated (Python 3.11.18, torch 2.15.0 CPU) | **18 passed** |
| core CI steps `validate_spec.py`, `train.py --epochs 1` (synthetic demo data) | ran; `REQUIRES DESIGN DECISION` (1280/12); smoke loss printed — not a result |
| data suite, isolated (Python 3.11.18) | **130 passed, 2 skipped** (rdflib, scikit-learn not installed) |
| topology guard on `master`, core, data | 0 findings each |
| trial merge `master` + data + core | 4 textual conflicts; after trivial resolution: 3 `module-package` findings; core suite → collection `ImportError` |
| `tools/integration_probe.py` | 10 steps: 5 PASS, 5 FINDING, 0 FAIL (details in §3) |

## 7. Reproduce

```bash
git fetch origin
git worktree add --detach ../core origin/implementation/ansg-core-2026
git worktree add --detach ../data origin/claude/relaxed-knuth-q63k61
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch && python -m pip install numpy PyYAML pytest
(cd ../core && python -m pytest -q)
(cd ../data && PYTHONPATH=src python -m pytest -q)
python tools/check_package_topology.py --root ../core
python tools/integration_probe.py --core ../core --data ../data --json probe.json
```
