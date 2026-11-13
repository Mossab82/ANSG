# ANSG experiment provenance — reproducibility contract

This module is an **independent engineering tool**. It does not generate stories,
recompute dissertation evaluation scores, or imply that a dissertation result was
reproduced.

## Code

`ansg.reproducibility` supplies:

- `sha256_file(path)`: a streaming SHA-256 for data/checkpoints.
- `assert_disjoint_splits({train, validation, test})`: checks exact record-ID overlap.
- `create_manifest(...)`: a source-code SHA, data fingerprints, configuration,
  seed, timestamp, optional checkpoint fingerprint, and an explicit real/synthetic label.
- `save_manifest(...)`: creates a JSON manifest without overwriting existing runs.
- `compare_measurement(...)`: separates the **thesis claim** and a **new measured
  value**; missing results are represented as `not_run`, with no invented delta.

Example (synthetic smoke run; use an actual SHA and an existing data file):

```python
from ansg.reproducibility import create_manifest, save_manifest

manifest = create_manifest(
    run_id="synthetic-smoke-001",
    git_sha="<insert a real 40-character commit SHA>",
    configuration={"model": "tiny-smoke", "seed": 42},
    seed=42,
    data_files={"synthetic_example": "tests/fixtures/nights/synthetic_nights.txt"},
    kind="synthetic",
)
save_manifest(manifest, "runs/synthetic-smoke-001/manifest.json")
```

The example placeholder SHA is intentionally **not executable** until a real commit
is supplied. Do not store example placeholder hashes as provenance.

## Scientific and engineering limitations

This guard checks **exact record IDs**, not story-level near-duplicates or
semantically equivalent passages across ANHC train/test files. For final thesis
experiments, supplement it with corpus clustering, near-duplicate detection and
human validation of split integrity.

Manifest `git_sha` and `kind` are caller-supplied and must be verified by the
experiment runner. The module does not prove that a checkpoint came from the
specified commit, that a git working tree was clean, or that a measurement is valid.
Environment, tokenizer, corpus licences, metric implementation/version, GPU and
sample size must be added to each full experiment protocol.

No metric has been fabricated or replaced. Where the dissertation does not define
a computational metric such as CMPI unambiguously, report `not_run` until an
authoritative scoring implementation is obtained and validated.
