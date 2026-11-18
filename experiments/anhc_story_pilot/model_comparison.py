#!/usr/bin/env python3
"""Compare direct Qwen realization with realization conditioned on a validated ANSG plan.

This is a small experimental integration, not a reproduction of the thesis model:
Qwen is an external frozen backbone; CSAM, DAL, CKG and PPO are not loaded.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from pilot import ROOT, SOURCE_ID, SOURCE_PATH, build_plan
from ansg.evaluation.story_proxies import summarise
from ansg.narrative.plan import build_prompt, load_settings, validate_plan

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DIRECT_PROMPT = """اكتب قصة عربية أصلية قصيرة تدور في العقبة عن الملك سالم والوزير عودة وأهل الحي. يواجه أهل الحي مشكلة في وصول الماء، ويجب أن تنتهي القصة بحل عادل يشارك فيه الناس. اجعل الحوار طبيعيًا بلهجة أهل العقبة والسرد بالفصحى. لا تنقل نصًا من أي قصة معروفة. اكتب القصة فقط."""


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("model_generation_review"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(args.seed)
    settings = load_settings(ROOT / "configs/generation/settings.yaml")
    plan = build_plan(repaired=True)
    plan_report = validate_plan(plan, settings)
    if not plan_report.ok:
        raise RuntimeError("ANSG plan failed validation: " + repr(plan_report.to_dict()))

    records = [json.loads(line) for line in SOURCE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    source = next(row for row in records if row["candidate_id"] == SOURCE_ID)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()

    plan_prompt = build_prompt(plan, settings) + "\n\nاكتب النص القصصي فقط، من دون شرح أو قائمة أحداث. أظهر الأحداث السبعة بالترتيب، واجعل معرفة كل شخصية متوافقة مع ما شهدته أو سمعته."
    direct_text = generate(model, tokenizer, DIRECT_PROMPT, args.max_new_tokens)
    plan_text = generate(model, tokenizer, plan_prompt, args.max_new_tokens)

    results = {}
    for condition, text in (("direct", direct_text), ("ansg_plan", plan_text)):
        results[condition] = {
            "text": text,
            "proxy_metrics": summarise(
                text, sources=[source["text"]], plan_graph=plan.graph,
                markers=settings["aqaba"].dialect_markers,
            ),
            "word_count_whitespace": len(text.split()),
        }

    report = {
        "experiment": "ANSG plan-conditioned model realization comparison",
        "status": "completed",
        "model": args.model,
        "weights_updated": False,
        "decoding": {
            "template": "tokenizer.apply_chat_template",
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "plan_validation": plan_report.to_dict(),
        "source_candidate": {
            "candidate_id": SOURCE_ID,
            "title": source.get("title_candidate"),
            "annotation_status": source.get("annotations"),
            "boundary_status": source.get("status"),
        },
        "conditions": results,
        "interpretation_limits": [
            "This uses a frozen Qwen backbone, not the thesis AraGPT2 checkpoint.",
            "CSAM, DAL, CKG retrieval and PPO/RL are not executed.",
            "All automatic scores are proxies; they do not prove story coherence, dialect authenticity or cultural quality.",
            "The ANHC candidate boundary and source labels are provisional/unannotated.",
            "A human comparison is required before claiming an improvement.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rng = random.Random(args.seed)
    swap = bool(rng.getrandbits(1))
    a, b = (("ansg_plan", "direct") if swap else ("direct", "ansg_plan"))
    with (args.output_dir / "blind_story_review.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.DictWriter(f, fieldnames=[
            "prompt_id", "output_A", "output_B",
            "coherence_A_1to5", "coherence_B_1to5",
            "causal_consistency_A_1to5", "causal_consistency_B_1to5",
            "Aqaba_dialect_naturalness_A_1to5", "Aqaba_dialect_naturalness_B_1to5",
            "cultural_plausibility_A_1to5", "cultural_plausibility_B_1to5",
            "originality_A_1to5", "originality_B_1to5", "preferred", "notes",
        ])
        w.writeheader()
        w.writerow({
            "prompt_id": "aqaba-water-story",
            "output_A": results[a]["text"], "output_B": results[b]["text"],
        })
    (args.output_dir / "blind_review_key.json").write_text(
        json.dumps({"aqaba-water-story": {"A": a, "B": b}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Generated direct and ANSG-plan-conditioned stories; report and blind review files written.")


if __name__ == "__main__":
    main()
