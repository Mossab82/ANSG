# ANSG heritage LoRA pilot (experimental)

## What is actually trained

- **Qwen/Qwen2.5-1.5B-Instruct** is a pretrained foundation model. This pilot trains a *LoRA adapter* for generic next-token prediction on authentic heritage prose.
- Source: Arabic Wikisource's Hindawi 2022 *One Thousand and One Nights*, part 1, nights 1–18, fetched at Kaggle run time from MediaWiki API. **Nights 1–14 are used for training; nights 15–18 are excluded from gradient updates but are NOT story-independent because tales cross night boundaries.** Their CE values are continuation diagnostics only, not evidence of generalization. Original text is identified by Wikisource as public domain; edition and digital-source attribution and licensing should be checked before redistribution.
- The workflow does not depend on the older Kaggle notebook's ephemeral `/kaggle/working` files.
- Initial run: up to 140 actual AdamW optimizer steps (batch size one, 4-step gradient accumulation), LoRA rank 8, context 256 tokens, seed 42, T4 GPU.
- **Do not claim the thesis' ANSG, CSAM/DAL, Arabic VerbNet, CST-Onto, RL or CRAFT is trained or validated by this pilot.** No dialect gold labels are supplied. The dialect prompts are only qualitative probes.
- Reported numbers: training CE loss, held-out night CE loss, generated MSA/Jordanian/Egyptian/Moroccan examples, basic Arabic-letter and repeated-trigram diagnostics. These do NOT prove cultural, dialectal, event-structure or human perceptual fidelity.
- Stop on invalid/insufficient source data and network failures rather than substitute fabricated corpus examples.
- Base model may need HF network connectivity; Kaggle model download and GPU quotas apply.

## Persistence and review

- New **private** Kaggle script: [ansg-heritage-lora-pilot](https://www.kaggle.com/code/mossabibrahim/ansg-arabic-heritage-lora-pilot-experimental).
- Checkpoints and reports are written in `/kaggle/working/ansg_heritage_pilot/` and downloaded via Kaggle version outputs to GitHub Artifacts after COMPLETE.
- Required outputs: `source_manifest.json`, `data_report.json`, `baseline_metrics.json`, `baseline_samples.json`, `adapted_samples.json`, `train_progress.json`, `adapter_final/`, `final_report.json`.
- Failure creates `failure_report.json`, and GPU run is marked failed. No paper-quality claim from a success badge.
- Runs are manual through GitHub workflow_dispatch. An explicit one-time commit trigger for the repaired pilot is separately documented in the workflow.
- The user-owned older private Kaggle notebook `mossabibrahim/notebook6367edd95f` remains unchanged.

Official docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
Base model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
Wikisource edition: https://ar.wikisource.org/wiki/ألف_ليلة_وليلة_(مؤسسة_هنداوي،_2022)/الجزء_الأول

## Research validity gate

This is a **technical training pilot only**. Source retrieval uses rendered MediaWiki HTML, because the original pages include transcluded text that is missing from stripped wikitext. Each night and Arabic-character count is printed for diagnostics.

**Leakage caution:** held-out nights 15–18 may continue the very same tales as nights 1–14. Do not call these independent stories or use their CE for thesis-quality comparisons. A later research-grade experiment must group by complete parent story/nested cycle, verify n-gram contamination, reserve a separate untouched test set, and compare human-evaluated dialectal/cultural/event-graph quality.
