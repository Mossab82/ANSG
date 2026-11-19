# ANHC-backed stepwise story pilot

## 1. Planning and validation pilot

This pilot uses ANSG's local `StoryPlan` and event-graph validator plus deterministic story-proxy metrics. The script `pilot.py` does not call a language model or train any weights.

Source anchor: candidate `hindawi-27482482-b003` (“حكاية الملك جليعاد والشماس”), from the ANHC direct-build branch's Arabic E-Book Corpus material. The candidate's segmentation remains provisional, and its source annotation fields DIA, MOTIF, and MORAL are null. The manually authored illustrative story uses only a broad ruler/adviser consultation motif; it does not copy or retell the source text.

Run the planning pilot from the repository root:

```bash
PYTHONPATH=src python experiments/anhc_story_pilot/pilot.py
```

The deliberately flawed plan fails with `E-KNOWLEDGE-PREMATURE`; the repaired plan passes with zero errors and zero warnings. The final manually authored story covers 3/3 planned characters and 7/7 event triggers according to repository surface-coverage proxies. It has 0.0 8-gram source containment and a dialect-marker proxy of 1.25 per 100 words. These proxies do not measure plot-level borrowing or dialect authenticity.

The final story in `pilot.py` is manually authored against the validated prompt. It is not model output, and it does not demonstrate neural-symbolic story generation.

## 2. Frozen-model realization experiment

The separate `model_comparison.py` runs actual inference with a frozen `Qwen/Qwen2.5-1.5B-Instruct` backbone and compares three conditions: direct prompting, a validated ANSG plan prompt, and a repair pass conditioned on the plan and first draft. All use the same chat template and deterministic greedy decoding. No model weights are trained or updated.

Reproduce from the repository root:

```bash
PYTHONPATH=src python experiments/anhc_story_pilot/model_comparison.py \
  --output-dir model_generation_review --max-new-tokens 480
```

A successful CPU run is recorded at [GitHub Actions run 44334254762](https://github.com/Mossab82/ANSG/actions/runs/44334254762). Its artifact contains `model_comparison.json`, `blind_story_review.csv`, and a separate key.

### Observed result

- The ANSG plan passes validation with 0 errors and 0 warnings.
- The direct generation is complete at 171 whitespace-delimited words, but its surface-trigger proxy matches 1/7 planned event triggers. Its dialect-marker proxy is 0.
- The plan-conditioned draft and the repair pass are identical, stop mid-story at 202 words, match 2/7 event triggers on the surface, and have a dialect-marker proxy of 0. Both include a Chinese token.
- The 8-gram source-copy proxy is 0.0 for all outputs. That does not establish plot-level originality.

This run does **not** show excellent generation or an improvement. It shows a working symbolic plan validator and a gap in realization with this small frozen backbone. It is a component-level experiment, not a reproduction of thesis results.

### Limits

- The backbone is Qwen, not the thesis AraGPT2 checkpoint.
- CSAM, DAL, CKG retrieval, and PPO/RL are not executed.
- The source candidate boundary is provisional; DIA/MOTIF/MORAL labels are null.
- Automatic scores are proxies, not evidence of coherence, dialect authenticity, cultural quality, or CRAFT performance.
- No human ratings have been collected. The included blinded CSV is for a future human review; reviewers should not open the separate key until scoring is complete.
- Do not add experimenter-provisional tags to ANHC training data.
