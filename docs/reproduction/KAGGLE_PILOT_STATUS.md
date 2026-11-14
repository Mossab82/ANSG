# Kaggle heritage pilot — status (supplementary experiment, not ANSG-Original)

**What it is.** A Qwen2.5-1.5B-Instruct LoRA trained on the public classical corpus (`kaggle/heritage/ansg_heritage_pilot.py`). It does not train or evaluate CSAM, DAL, CKG, RL or CRAFT. Its numbers are never thesis results.

## Run history

The failure reports below are transcribed from the job logs into `evidence/kaggle/heritage_pilot_run_history.json`.

| GitHub run | Outcome | Optimizer steps | Root cause | Fixed by |
| --- | --- | --- | --- | --- |
| 43509787560 | failure | — | watcher polled the wrong kernel slug | PR #21 |
| 43557375454 | failure | 0 | `ValueError: Page has insufficient Arabic narrative text`: wikitext with transcluded text stripped | PR #23 (rendered HTML) |
| 43559807003 | failure | 0 | `missingtitle` at night 3: page titles constructed instead of discovered | PR #24/#25 (frozen corpus); `src/ansg/data/mediawiki.py` discovers titles |
| 43561944084 | failure | 0 | `KeyError: 'night'`: `chunks()` read a field the frozen corpus does not have; the preflight never ran that path | this branch |

No successful run exists, and no adapter, loss curve or sample from this pilot is available.

## Fix in this branch

- **Data-path fix.** `chunks()` tags each window with its source group (`row["id"]`). `prepare_data()` refuses any source group that appears in two splits.
  - Validation CE uses all validation chunks.
  - Test CE is measured once before training and once after, and is never used for selection.
- **CPU preflight.** `scripts/preflight_heritage_pilot.py` imports the pilot and runs `load_verified_corpus` → `prepare_data` on the committed corpus.
  - The `master` version of `chunks()` reproduces `KeyError: 'night'` on this data. The fixed version passes, which `tests/unit/test_reproduction_audits.py` checks.
- **Split v2** (`scripts/reproduction/derive_pilot_split.py`). Split v1 put test page `kalila_dimna-019` in the test set although 98.5% of its 8-grams occur in train page `kalila_dimna-007`. v2 is derived deterministically and is leak-free by the same measure.
  - **Limitation:** the v2 test set is 839 words.
- **GPU gate.** The workflow now runs the preflight with `--require-rights` before pushing to Kaggle.

## Why no new GPU run was started

The rights gate does not pass: the corpus is `pending_owner_review` ([RIGHTS_GATE.md](RIGHTS_GATE.md)). The task rules forbid a real run before the data and rights gates pass. Once the owner records the review, the existing `workflow_dispatch` trigger can start one run. Its outputs (logs, `final_report.json`, adapter) are preserved by the workflow's artifact upload. They belong in a supplementary-results table, never in the thesis comparison.
