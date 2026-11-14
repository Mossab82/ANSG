# Arabic generation readiness (Issue #15)

This document covers the Arabic story-generation runtime (`ansg-generate`), the capability gate
(`scripts/check_generation_readiness.py`) and what the repository can and cannot show to the
examination committee today. It is an engineering milestone. **It reproduces no result from the
dissertation and computes no CRAFT/CMPI metric.**

## 1. Current status (as of this pull request)

| Question | Answer | Evidence |
| --- | --- | --- |
| Readiness level of the repository as distributed | **`smoke_only`** | `python scripts/check_generation_readiness.py` (CI runs it in every data-only job) |
| Pretrained Arabic causal-LM checkpoint in the repository | **None.** Large checkpoints are never committed or downloaded in CI | `.gitignore` excludes `*.safetensors`, `*.pt`; `ansg-generate` accepts local directories only |
| Original author checkpoints (backbone or cultural modules) | **Not available to this work.** No conversion was attempted or claimed (§6) | — |
| Trained CSAM/DAL weights | **None.** Only the training code exists; nothing was trained on real data | `scripts/train_cultural_modules.py` |
| Generation genuinely exercised with a real pretrained model | **No.** CI exercises the full path on a tiny randomly initialised *synthetic stub* only | `scripts/generation_smoke.py`, `tests/integration/test_generation_stub_backbone.py` |
| Data / CKG / schema vocabulary consistency | Checkable. Synthetic fixtures pass in CI. No real CKG export or licensed ANHC corpus is present | `readiness.json` CI artefact |

To reach `backbone_only`, someone with the right to use a pretrained Arabic causal LM must place
it locally and declare it (§4). To reach `trained_ansg`, cultural modules must be trained on the
licensed ANHC data with a real CKG (§5). Neither step can be faked through the CLI. The label is
derived from hashes and manifests, and no flag sets it.

## 2. Readiness levels

`ansg.generation.labels.ReadinessLevel`, from weakest to strongest:

| Level | Required evidence (all checked by the gate) |
| --- | --- |
| `smoke_only` | Anything less than the next row. This covers the in-repository byte-level research model (`train.py`), synthetic smoke checkpoints and the synthetic stub backbone |
| `backbone_only` | A local Hugging Face-format causal-LM directory with weights, `config.json` and tokenizer files. The directory digest must match an expected digest. `ansg_backbone.json` must declare name, source and licence, with `"synthetic": false`. The declared language, if any, must include `ar`. Tokenizer size must not exceed the embedding rows |
| `ansg_untrained` | The above, plus a cultural-module checkpoint whose manifest says `status: untrained`. The checkpoint hash, backbone digest, tokenizer digest and schema-vocabulary hash must all match |
| `ansg_trained_synthetic_only` | As above, but `status: trained` with optimiser steps > 0 and training-data hashes, and `data_kind: synthetic` |
| `trained_ansg` | As above with `data_kind: real`. In addition, the schema vocabulary must match the supplied CKG (`source_sha256`), and the vocabulary and CKG must appear in the `ansg-build-examples` manifest. Every training-data hash must be an output of that manifest. A trained checkpoint that fails any of these is reported as `backbone_only` |

The gate also checks whether generation has been **exercised**. That requires at least one
`ansg-generate` record that re-validates, was made with the same backbone, has generated tokens,
and is not a synthetic-stub run.

Freshly initialised CSAM/DAL can always be attached at run time (`--untrained-cultural-modules`).
That never raises the level.

```bash
python scripts/check_generation_readiness.py \
    --backbone /models/aragpt2 --cultural-checkpoint checkpoints/cultural.pt \
    --schema-vocabulary data/processed/examples/schema_vocabulary.json --ckg data/private/ckg.json \
    --examples-dir data/processed/examples --generation-record runs/story.json \
    --require trained_ansg            # exit 1 if the evidence is weaker; --json for machine output
```

## 3. Generation modes and visible labels

Every record carries `label.mode`, `label.banner` and `label.warnings`. The banner is also printed
to stderr. `reproduces_dissertation` and `dissertation_result` are always `false`. A record is
re-validated against its evidence before it is written (`labels.validate_label`), and the gate
rejects any record whose label is stronger than its evidence.

| Mode | When | Banner (abridged) |
| --- | --- | --- |
| `synthetic_stub` | Backbone declares `"synthetic": true` | SYNTHETIC STUB: the text is meaningless |
| `backbone_only` | No cultural modules | BACKBONE-ONLY: no CSAM, no DAL; not ANSG output |
| `ansg_untrained` | `--untrained-cultural-modules`, or an untrained checkpoint | UNTRAINED RESEARCH PROTOTYPE: conditioning not meaningful |
| `ansg_trained_synthetic_only` | Checkpoint trained on synthetic data | Research prototype: no cultural capability claimed |
| `ansg_trained` | Checkpoint trained on declared real data | Trained cultural modules: a new generation, not a reproduction |

Dialect: a `--dialect` passed to any DAL that is not `ansg_trained` produces the warning *"no
dialect adaptation is claimed"*, and `dialect_adaptation_claimed` stays `false`.

## 4. Committee demonstration: local setup

Prerequisites: `python -m pip install -e ".[model,pretrained]"` (PyTorch and transformers). A GPU
is optional; use `--device cuda`. A 774M–800M-parameter backbone in float32 needs about 3–4 GB of
memory. Long generations on CPU are slow.

1. **Obtain a checkpoint yourself.** `ansg-generate` never downloads; the CLI sets
   `HF_HUB_OFFLINE=1`, and loading uses `local_files_only=True` and `trust_remote_code=False`.
   Appendix A.3 (Listing A.2) names `aubmindlab/aragpt2-large`. Chapter 4 §4.1.6 and Table 5.1
   name other backbones (conflict C-MODEL-1). Read the model card and licence before use. Then
   download it outside this tool into a directory such as `/models/aragpt2-large`.
   `ansg-generate` applies only Unicode NFC to the prompt. If the model card recommends other
   preprocessing, apply it to the prompt yourself and note it in the run log.
2. **Declare it.** Create `/models/aragpt2-large/ansg_backbone.json`:

   ```json
   {"name": "aubmindlab/aragpt2-large", "source": "https://huggingface.co/aubmindlab/aragpt2-large",
    "revision": "<commit you downloaded>", "license": "<licence from the model card>",
    "synthetic": false, "language": "ar", "directory_digest": "<from step 3>"}
   ```

3. **Pin its digest.** `python scripts/check_generation_readiness.py --backbone /models/aragpt2-large`
   prints the computed directory digest. This is SHA-256 over sorted `path<TAB>sha256` lines of
   every file except hidden entries and `ansg_backbone.json`. Copy it into `directory_digest`, or
   pass it each time with `--backbone-sha256`. Rerun the gate and expect `backbone_only`.
4. **Generate (backbone only).**

   ```bash
   ansg-generate --backbone /models/aragpt2-large --prompt "كان يا ما كان في قديم الزمان" \
       --device cuda --seed 42 --num-samples 3 --output runs/committee/backbone_only.json
   ```

   Defaults follow Ch. 6 §6.1.2: temperature 0.7, top-p 0.95, 512 new tokens, seed 42 (see
   C-EVAL-7 for the conflicting Table A.6 value). `--temperature 0` gives greedy decoding. Other
   options are `--top-k`, `--repetition-penalty`, `--max-new-tokens`, `--prompt-file`, `--json`,
   `--dtype` and `--deterministic`.
5. **Optional: cultural modules.** Steps 5 and 6 require the licensed corpus and a real CKG export:

   ```bash
   ansg-ingest --config configs/data/nights.yaml
   ansg-build-examples --records-dir data/processed/nights --ckg data/private/ckg.json \
       --output data/processed/examples
   python scripts/train_cultural_modules.py --backbone /models/aragpt2-large \
       --examples data/processed/examples/train.examples.jsonl \
       --schema-vocabulary data/processed/examples/schema_vocabulary.json \
       --data-kind real --authorization-manifest data/private/training_authorization.json \
       --run-plan configs/training/training_run_plan.json \
       --profile configs/training/approved_reconstruction_profile.yaml \
       --steps <N> --device cuda --output checkpoints/cultural.pt
   ansg-generate --backbone /models/aragpt2-large --cultural-checkpoint checkpoints/cultural.pt \
       --schema-vocabulary data/processed/examples/schema_vocabulary.json \
       --schema-node <CKG node id> --dialect Egyptian --prompt "..." --output runs/committee/ansg.json
   ```

6. **Record the state.** Run the gate with every input and generation record (§2). Archive the
   JSON records and the gate's `--json` output together with the git commit.

## 5. Formats

**Generation record** (`ansg.generation_record/1`). It contains:

- `label` (§3); `created_utc`; `git` (commit and dirty flag); `software` (versions of ANSG,
  Python, torch, CUDA and transformers).
- `runtime` (device, dtype, deterministic flag); `generation_config`.
- `backbone`: path, directory digest, `hash_verified`, digest source, per-file SHA-256, declared
  provenance, model type, sizes and parameter count.
- `tokenizer`: class, size, digest, embedding rows, alignment.
- `cultural_modules`: checkpoint path and hash, full manifest, architecture, schema-vocabulary
  hashes, schema node ids and ids, and dialect.
- `prompt`: NFC text, SHA-256, Arabic-letter ratio, token count, exact round-trip.
- `samples`: seed, text, continuation, new-token count, finish reason, SHA-256 of the token ids.
- `timing_seconds`.

Records are written create-only and are never overwritten.

**Cultural-module checkpoint**: `<file>` holds the tensors (every `PretrainedANSG` parameter
except `backbone.*`) and is loaded with `torch.load(weights_only=True)`. `<file>.manifest.json`
(`ansg-cultural-modules`, version 2 = bridge v2; version-1 files are refused) records:

- checkpoint SHA-256 and architecture;
- backbone directory digest, tokenizer digest, and schema-vocabulary SHA-256 and source SHA-256;
- `training`: status, `optimizer_steps`, `data_kind`, `data_sha256`, objective, hyper-parameters
  and the measured `loss_per_step`;
- git and software versions.

A checkpoint without a manifest, or with any mismatched hash or key set, is refused.

## 6. Original author checkpoints

If the author's original checkpoints exist, they can be demonstrated only after these steps:

1. The backbone directory is declared and hashed as in §4.
2. The cultural weights are mapped onto `PretrainedANSG` parameter names by an explicit, reviewed
   conversion script. That script verifies shapes and records the source file hashes.
3. The converted file gets a manifest whose `training` block describes the original training
   honestly, from the author's own logs.

No such converter exists yet, because the original checkpoint layout is unknown to this work.
Original checkpoints must never be relabelled or approximated by newly trained weights.

## 7. Known limitations and missing evidence

- **Integration point.** `PretrainedANSG` applies CSAM and DAL to the backbone's final hidden
  states. This is not the in-layer pre-softmax modification of Eq. 4.9 (Ch. 4 §4.3), and DAL does
  not adapt `q_proj`/`v_proj` (C-DAL-1, Table 5.8). Since bridge v2 (scientific review 2026-11)
  the bridge adds only a gated increment to the input of the backbone's output head (gate
  initialised to 0, DAL up-projection zero), so untrained modules reproduce the backbone's logits
  exactly (tested). The first bridge added an extra LayerNorm and changed the backbone's output
  even when untrained. Untrained modules are still labelled `ansg_untrained`, and their output is
  the backbone's.
- **Width.** For a hidden size of 1280 with 12 cultural heads, the bridge projects to 1284
  (I-CSAM-4, C-CSAM-7).
- **Tokenizer.** The dissertation's SentencePiece 32k vocabulary (Ch. 5 §5.4.1) is not used. The
  backbone's own tokenizer is required (C-MODEL-3).
- **Schema activation.** Schema nodes are supplied explicitly with `--schema-node`. The Eq. 4.20 /
  App. A.4 activation function needs schema embeddings (C-CSAM-5), and activation weights are
  owner decision D4.
- **Training.** Only stage-1 next-token cross-entropy with a frozen backbone is implemented for
  the pretrained path. The adversarial and PPO stages and the CRAFT reward are not implemented
  here (C-TRAIN-1, C-EVAL-2).
- **Declarations.** `data_kind` and `ansg_backbone.json` are recorded declarations. Hashes make
  them traceable, but the gate cannot verify that data is genuinely real or that a licence is
  correct.
- **Repeatability.** Repeatability is bitwise only for the same seed, hardware and software
  stack. Sampling uses a CPU `torch.Generator` seeded with `seed + i` for sample `i`. Use
  `--deterministic` on GPU.
- **Metrics.** No generation quality, cultural or dialect metric is computed.

## 8. Dissertation cross-reference

| Topic | Dissertation | Code | Conflict register |
| --- | --- | --- | --- |
| Backbone | Ch. 4 §4.1.6; Ch. 5 Table 5.1; App. A.3 Listing A.2 | `runtime.load_backbone`, `ansg_backbone.json` | C-MODEL-1, C-MODEL-2 |
| Tokenizer | Ch. 5 §5.4.1; Ch. 4 Table 4.11 | tokenizer digest and alignment check | C-MODEL-3 |
| CSAM | Ch. 4 §4.3, Eq. 4.9; App. A.25 | `ansg.csam`, `ansg.pretrained.PretrainedANSG` | C-CSAM-1…7 |
| Schema activation | Eq. 4.20; App. A.4 | `--schema-node`, `SchemaVocabulary` | C-CSAM-5, D4 |
| DAL | Ch. 4 §4.4, Eq. 4.24, Algorithm 1; Table 5.8 | `ansg.dal` (−1 = unknown dialect) | C-DAL-1, C-DAL-2 |
| Training stages | Ch. 3 §3.1.2; Ch. 4 Table 4.1, §4.1.6; Ch. 5 §5.1.3 | `generation.cultural` (stage 1 only) | C-TRAIN-1…5 |
| Generation settings | Ch. 6 §6.1.2; Table A.6 | `SamplingConfig` defaults | C-EVAL-7 |
| Evaluation | Ch. 6 (CRAFT, CMPI, ALDA) | not computed here | C-EVAL-1, C-EVAL-2 |
