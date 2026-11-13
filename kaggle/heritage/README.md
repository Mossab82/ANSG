# ANSG heritage LoRA pilot (experimental)

## What is actually trained

- **Qwen/Qwen2.5-1.5B-Instruct** is a pretrained foundation model. This pilot trains a *LoRA adapter* for generic next-token prediction on authentic heritage prose.
- Source: Arabic Wikisource's Hindawi 2022 *One Thousand and One Nights*, part 1, nights 1–18, fetched at Kaggle run time from MediaWiki API. **Nights 1–14 only for training; nights 15–18 strictly held out.** Original text is identified by Wikisource as public domain; edition and digital-source attribution and licensing should be checked before redistribution.
- The workflow does not depend on the older Kaggle notebook's ephemeral `/kaggle/working` files.
- Initial run: up to 140 actual AdamW optimizer steps (batch size one, 4-step gradient accumulation), LoRA rank 8, context 256 tokens, seed 42, T4 GPU.
- **Do not claim the thesis' ANSG, CSAM/DAL, Arabic VerbNet, CST-Onto, RL or CRAFT is trained or validated by this pilot.** No dialect gold labels are supplied. The dialect prompts are only qualitative probes.
- Reported numbers: training CE loss, held-out night CE loss, generated MSA/Jordanian/Egyptian/Moroccan examples, basic Arabic-letter and repeated-trigram diagnostics. These do NOT prove cultural, dialectal, event-structure or human perceptual fidelity.
- Stop on invalid/insufficient source data and network failures rather than substitute fabricated corpus examples.
- Base model may need HF network connectivity; Kaggle model download and GPU quotas apply.

## Persistence and review

- New **private** Kaggle script: [ansg-heritage-lora-pilot](https://www.kaggle.com/code/mossabibrahim/ansg-heritage-lora-pilot).
- Checkpoints and reports are written in `/kaggle/working/ansg_heritage_pilot/` and downloaded via Kaggle version outputs to GitHub Artifacts after COMPLETE.
- Required outputs: `source_manifest.json`, `data_report.json`, `baseline_metrics.json`, `baseline_samples.json`, `adapted_samples.json`, `train_progress.json`, `adapter_final/`, `final_report.json`.
- Failure creates `failure_report.json`, and GPU run is marked failed. No paper-quality claim from a success badge.
- Automatic trigger: merge the branch to master (workflow `push` path filter); subsequent repeated run is **manual** unless heritage files change.
- The user-owned older private Kaggle notebook `mossabibrahim/notebook6367edd95f` remains unchanged.

Official docs: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
Base model: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
Wikisource edition: https://ar.wikisource.org/wiki/ألف_ليلة_وليلة_(مؤسسة_هنداوي،_2022)/الجزء_الأول
