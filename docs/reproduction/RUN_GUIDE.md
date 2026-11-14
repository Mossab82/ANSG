# Fresh-clone run guide

Everything below runs on CPU without credentials, except the steps marked **owner**.

**What this guide does not do.** None of these commands reproduces a thesis result: the thesis data, weights and evaluation code are unavailable ([RESULTS_COMPARISON.md](RESULTS_COMPARISON.md)). The commands verify the software, the data provenance and the audits, and prepare the gated supplementary pilot.

## 1. Clone and install (pinned)

```bash
git clone https://github.com/Mossab82/ANSG.git
```

```bash
cd ANSG
```

```bash
python3.11 -m venv .venv && . .venv/bin/activate
```

CPU-only PyTorch: narrow the torch pin to the `+cpu` build of the same release. PyPI stays the primary index because the CPU wheel index does not mirror every pinned dependency (for example `filelock==4.2.10`).

```bash
sed 's/^torch==2\.15\.0$/torch==2.15.0+cpu/' requirements/constraints-py311.txt > constraints-cpu.txt
```

```bash
python -m pip install -c constraints-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu torch
```

```bash
python -m pip install -c constraints-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu -e ".[test,model,ontology,splits,pretrained]"
```

- The pins were verified with PyPI's `torch 2.15.0+cu132` wheel running on CPU; CI (`full-stack-pinned`) runs the same release as `2.15.0+cpu`. On a CUDA machine, use `requirements/constraints-py311.txt` directly without the extra index.

- The data/knowledge layer alone needs only `python -m pip install -e ".[test]"`, with no PyTorch.
- The thesis's own environment (§5.2.1: Python 3.10.12, PyTorch 2.0.1, Transformers 4.30.2) is recorded in DISCREPANCY_REPORT D-RES-2. It is not what is pinned here.

## 2. Test suite

```bash
python -m pytest -rs
```

Expected:
- **full install:** every test passes, with no skips;
- **data-only install:** only tests marked "model extra (PyTorch) not installed" are skipped.

## 3. CPU preflight and audits (no network)

```bash
python scripts/validate_classical_corpus.py
```

```bash
python scripts/reproduction/derive_pilot_split.py --check
```

```bash
python scripts/preflight_heritage_pilot.py
```

```bash
python scripts/reproduction/audit_classical_corpus.py --check
```

```bash
python scripts/reproduction/build_story_records.py --check
```

```bash
python scripts/reproduction/audit_reported_values.py
```

```bash
python data/sources/build_source_registry.py && git diff --exit-code data/sources/source_registry.json
```

- `preflight_heritage_pilot.py` runs the Kaggle pilot's real data path (corpus verification, split, chunking, leakage check) on CPU.
- With `--require-rights` it also applies the rights gate, and it **exits 2** until the owner records the licence review ([RIGHTS_GATE.md](RIGHTS_GATE.md)).

## 4. Network evidence (CI)

`.github/workflows/reproduction-evidence.yml` (CPU, metadata only) regenerates:
- `evidence/external/external_evidence.json`: thesis DOIs and repositories, MADAR documentation, and the Wikisource pages at their pinned revisions;
- `data/corpora/classical_arabic/story_segments.json`.

To run it:
- **owner:** trigger it with `workflow_dispatch`; or
- push a change to the collector on the reproduction branch.

## 5. Rebuilding the classical corpus from Wikisource (network)

The frozen text in `data/corpora/classical_arabic/stories.jsonl` is verified against its pinned revisions by the evidence workflow, which got 20/20 byte-identical matches.

```bash
python scripts/build_classical_corpus.py
```

This rebuilds from the **current** revisions, which may change the text and hashes. Compare the result with `index.json`, and never overwrite the frozen corpus without a new review.

## 6. Gated GPU pilot (owner)

`.github/workflows/kaggle-heritage-pilot.yml` runs on `workflow_dispatch`, or on the one-time trigger files. **Before any Kaggle push** it requires all three:
1. the corpus integrity check;
2. the CPU preflight of the real data path;
3. the **rights gate**.

The pilot is a supplementary experiment (Qwen2.5-1.5B LoRA). Its outputs are never thesis results. Status: [KAGGLE_PILOT_STATUS.md](KAGGLE_PILOT_STATUS.md).

## 7. Reproducing ANSG-Original (blocked)

Requirements are in [THESIS_RESULTS_TRACEABILITY.md](THESIS_RESULTS_TRACEABILITY.md): ANHC or its test set, CMPI/CRAFT, checkpoints, and baselines. When they are available:
1. register them in `data/sources/build_source_registry.py` (with hash, licence and permission);
2. ingest them with `ansg-ingest`;
3. evaluate with seed 42 and the four consistency seeds;
4. add the measured values and differences to RESULTS_COMPARISON.md.
