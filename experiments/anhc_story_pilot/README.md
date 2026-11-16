# ANHC-backed stepwise story pilot

## What was run

This pilot uses ANSG's local `StoryPlan` and event-graph validator plus its deterministic story-proxy metrics. It does not call a language model or train any weights.

Source anchor: candidate `hindawi-27482482-b003`, titled **حكاية الملك جليعاد والشماس**, from the ANHC direct-build branch's Arabic E-Book Corpus material. The candidate's segmentation remains provisional, and its source annotation fields DIA, MOTIF, and MORAL are null. The new story below uses only the broad ruler/adviser consultation motif; it does not copy or retell the source text.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src python experiments/anhc_story_pilot/pilot.py
```

The script writes `pilot_report.json`, both plan graphs, the generation prompt, and the before/after story texts beside itself.

## Run result

- The deliberately flawed plan fails with `E-KNOWLEDGE-PREMATURE`: the adviser gives advice based on the blocked-channel fact before the planned inspection establishes it.
- The repaired plan passes with zero errors and zero warnings.
- The final story covers 3/3 planned characters and 7/7 event triggers according to the repository's surface-coverage proxy.
- The 8-gram copy-overlap proxy against the selected source candidate is 0.0; its longest exact shared run is 2 words. This does not measure paraphrase or plot-level borrowing.
- The dialect-marker proxy is 1.25 per 100 words using the repository's unreviewed Aqaba setting markers. It does not measure dialect accuracy or naturalness.

## Interpretation limits

DIA/MOTIF/MORAL tags in `pilot_report.json` are experimenter-provisional, not ANHC annotations. The repaired story text was manually authored against a prompt generated from a validated plan because this checkout has no verified local Arabic backbone or trained CSAM/DAL checkpoint, and PyTorch is unavailable. Therefore no generated-model output, CSAM inference, DAL inference, CKG retrieval, PPO/RL run, or human evaluation is claimed. The experiment demonstrates plan validation and traceable stage inputs, not successful neural-symbolic generation or a thesis result.

The source record's attribution and license are retained in the JSON report. Do not add the provisional tags to ANHC training data.
