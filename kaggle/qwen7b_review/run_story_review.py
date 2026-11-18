"""Kaggle GPU inference comparison: direct prompt vs validated ANSG plan and repair.
Uses a frozen Qwen2.5-7B-Instruct checkpoint; it does not train thesis components.
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path("/kaggle/working/ansg_qwen7b_review")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_COMMIT = "6e0c8377679c1b0b80c7b6415fe5cea143be7285"
ARCHIVE_URL = f"https://codeload.github.com/Mossab82/ANSG/zip/{SOURCE_COMMIT}"
DIRECT_PROMPT = """اكتب قصة عربية أصلية قصيرة تدور في العقبة عن الملك سالم والوزير عودة وأهل الحي. يواجه أهل الحي مشكلة في وصول الماء، وتنتهي القصة بحل عادل يشارك فيه الناس. اجعل الحوار طبيعيًا بلهجة أهل العقبة والسرد بالفصحى. لا تنقل نصًا من أي قصة معروفة. اكتب القصة فقط."""


def fetch_ansg_source() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "ansg-source.zip"
    request = urllib.request.Request(
        ARCHIVE_URL, headers={"User-Agent": "ANSG-Research-Review/1.0"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        archive.write_bytes(response.read())
    extracted = OUT / "source"
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extracted)
    root = next(extracted.glob("ANSG-*"))
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root


def generate(model, tokenizer, prompt: str, torch, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = {name: tensor.to(input_device) for name, tensor in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.0, no_repeat_ngram_size=0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True).strip()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(42)
    source_root = fetch_ansg_source()
    from experiments.anhc_story_pilot.pilot import build_plan
    from ansg.evaluation.story_proxies import summarise
    from ansg.narrative.plan import build_prompt, load_settings, validate_plan

    settings = load_settings(source_root / "configs/generation/settings.yaml")
    plan = build_plan(repaired=True)
    validation = validate_plan(plan, settings)
    if not validation.ok:
        raise RuntimeError("ANSG plan validation failed: " + repr(validation.to_dict()))

    plan_prompt = build_prompt(plan, settings) + (
        "\n\nاكتب قصة كاملة لا قائمة أحداث. التزم بتسلسل الأحداث السبعة وبالمعرفة المقررة لكل شخصية. "
        "الشخصيات: سالم ملك رجل، وعودة وزير رجل، وأهل الحي جماعة. استخدم العربية فقط، "
        "ولا تضع كلمات إنجليزية أو أجنبية. اجعل السرد فصيحًا والحوار بلهجة أهل العقبة، "
        "واختم بحل واضح. الطول نحو 220 إلى 300 كلمة."
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU unavailable")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
        max_memory={i: "13GiB" for i in range(torch.cuda.device_count())},
        low_cpu_mem_usage=True, trust_remote_code=False,
    )
    model.eval()

    direct_text = generate(model, tokenizer, DIRECT_PROMPT, torch, 700)
    plan_text = generate(model, tokenizer, plan_prompt, torch, 700)
    repair_prompt = (
        plan_prompt
        + "\n\nأعد كتابة القصة كاملة كنص متماسك بعد مراجعة المسودة. لا تختصرها ولا تتوقف قبل النهاية. "
        + "احذف الكلمات غير العربية وصحح الأدوار. أظهر الأحداث السبعة بالترتيب: ملاحظة ضعف الماء؛ "
        + "فحص القناة واكتشاف الحجر؛ استشارة عودة بعد اكتشاف السبب؛ نصيحة عودة؛ إزالة الحجر؛ "
        + "توزيع الماء بالدور؛ اتفاق أهل الحي على جدول ومراجعته. أخرج القصة النهائية فقط.\n\nالمسودة:\n"
        + plan_text
    )
    repaired_text = generate(model, tokenizer, repair_prompt, torch, 700)

    conditions = {}
    for name, story in (
        ("direct", direct_text),
        ("ansg_plan", plan_text),
        ("ansg_plan_repaired", repaired_text),
    ):
        conditions[name] = {
            "text": story,
            "word_count_whitespace": len(story.split()),
            "proxy_metrics": summarise(
                story, plan_graph=plan.graph,
                markers=settings["aqaba"].dialect_markers,
            ),
            "ends_with_sentence_punctuation": story.rstrip().endswith(
                ("؟", "!", ".", "۔", "…", "»", '"')
            ),
        }

    report = {
        "experiment": "Frozen Qwen2.5-7B direct vs ANSG-plan-conditioned story realization",
        "status": "completed",
        "model": MODEL_ID,
        "model_weights_updated": False,
        "gpu_names": gpu_names,
        "decoding": {"method": "greedy", "do_sample": False, "max_new_tokens": 700, "seed": 42},
        "plan_validation": validation.to_dict(),
        "plan_origin": "ANSG experimental pilot build_plan(repaired=True); no heritage source text supplied to the model",
        "conditions": conditions,
        "limits": [
            "This is frozen-backbone inference, not training or a reproduction of thesis experiments.",
            "CSAM, DAL, CKG retrieval and PPO/RL are not executed.",
            "The Aqaba dialect setting is an unreviewed draft; marker counts are proxies, not dialect authenticity.",
            "Surface trigger counts do not establish event order, causality or plot adherence.",
            "No human ratings have been collected; do not claim excellent results until blinded native-speaker and expert review.",
        ],
    }
    (OUT / "model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rng = random.Random(42)
    a, b = (("ansg_plan_repaired", "direct") if rng.getrandbits(1)
            else ("direct", "ansg_plan_repaired"))
    with (OUT / "blind_story_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "prompt_id", "output_A", "output_B",
            "coherence_A_1to5", "coherence_B_1to5",
            "causal_consistency_A_1to5", "causal_consistency_B_1to5",
            "Aqaba_dialect_naturalness_A_1to5", "Aqaba_dialect_naturalness_B_1to5",
            "cultural_plausibility_A_1to5", "cultural_plausibility_B_1to5",
            "originality_A_1to5", "originality_B_1to5", "preferred", "notes",
        ])
        writer.writeheader()
        writer.writerow({
            "prompt_id": "aqaba-water-story",
            "output_A": conditions[a]["text"], "output_B": conditions[b]["text"],
        })
    (OUT / "blind_review_key.json").write_text(
        json.dumps({"aqaba-water-story": {"A": a, "B": b}}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"], "model": MODEL_ID,
        "gpus": gpu_names, "plan_ok": validation.ok,
        "word_counts": {k: v["word_count_whitespace"] for k, v in conditions.items()},
        "output_dir": str(OUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
