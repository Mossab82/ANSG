"""Controlled Kaggle GPU comparison for direct vs ANSG-plan story realization.
Uses frozen Qwen2.5-7B-Instruct. This is inference only, not thesis reproduction.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

OUT = Path("/kaggle/working/ansg_qwen7b_review")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_COMMIT = "6e0c8377679c1b0b80c7b6415fe5cea143be7285"
ARCHIVE_URL = f"https://codeload.github.com/Mossab82/ANSG/zip/{SOURCE_COMMIT}"
TARGET_WORDS = 180
MIN_TARGET_WORDS = 130
MAX_TARGET_WORDS = 240
MAX_NEW_TOKENS = 700

SYSTEM_PROMPT = (
    "أنت كاتب قصص باللغة العربية. أخرج قصة عربية فقط، بلا مقدمة أو شرح أو تعريف بنفسك. "
    "لا تكتب بالصينية أو الإنجليزية أو أي لغة أخرى. اكتب نص القصة وحده، "
    "بسرد فصيح وحوار طبيعي بلهجة أهل العقبة في جنوب الأردن."
)

DIRECT_PROMPT = """اكتب قصة أصلية بعنوان «القناة والعدل»، تقع في العقبة بالأردن، عن الملك سالم والوزير عودة وأهل العقبة. اجعل الحبكة والأحداث الآتية أساس القصة، مع الحفاظ على ترتيبها وعلاقاتها السببية:
1. يلاحظ سالم ضعف الماء في القناة.
2. يفحص سالم وعودة القناة ويكتشفان أن حجرًا يسدها.
3. بعد اكتشاف السبب، يستشير سالم عودة.
4. ينصح عودة بإزالة الحجر ثم توزيع الماء بالدور.
5. يزيل أهل الحي الحجر فيعود الماء إلى المجرى.
6. يوزع سالم وأهل الحي الماء بالدور بعدل.
7. يتفق أهل الحي على جدول معلن لتوزيع الماء ومراجعته معًا.
اجعل طول القصة نحو 180 كلمة. لا تنقل نصًا من قصة معروفة. أخرج القصة فقط."""

COMMON_PLAN_STYLE = (
    "اكتب قصة مكتملة ومتماسكة، لا قائمة أحداث. اجعل طولها نحو 180 كلمة. "
    "التزم بالأحداث السبعة وترتيبها. لا تنقل نصًا من قصة معروفة. "
    "أخرج القصة العربية النهائية فقط، بلا تعليق أو عناوين إضافية."
)


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
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
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


def arabic_seed(text: str) -> str:
    """Keep Arabic draft text while removing foreign-script contamination."""
    kept = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF0-9\s،؛؟.!…«»():\-]", " ", text)
    return re.sub(r"[ \t]+", " ", kept).strip()


def diagnostics(story: str, proxies: dict) -> dict:
    words = len(story.split())
    arabic_letters = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", story))
    latin_letters = len(re.findall(r"[A-Za-z]", story))
    cjk_characters = len(re.findall(r"[\u3400-\u4DBF\u4E00-\u9FFF]", story))
    punctuation = story.rstrip().endswith(("؟", "!", ".", "۔", "…", "»", '"'))
    return {
        "whitespace_word_count": words,
        "arabic_letter_count": arabic_letters,
        "latin_letter_count": latin_letters,
        "cjk_character_count": cjk_characters,
        "ends_with_sentence_punctuation": punctuation,
        "heuristic_gates": {
            "target_length_130_to_240_whitespace_words": MIN_TARGET_WORDS <= words <= MAX_TARGET_WORDS,
            "no_latin_or_cjk_letters": latin_letters == 0 and cjk_characters == 0,
            "sentence_punctuation_at_end": punctuation,
        },
        "proxy_metrics": proxies,
    }


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
    plan.target_words = TARGET_WORDS
    validation = validate_plan(plan, settings)
    if not validation.ok:
        raise RuntimeError("ANSG plan validation failed: " + repr(validation.to_dict()))

    plan_prompt = build_prompt(plan, settings) + "\n\n" + COMMON_PLAN_STYLE

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

    # Same backbone, system instruction, seed, and greedy decoding across conditions.
    direct_text = generate(model, tokenizer, DIRECT_PROMPT, torch, MAX_NEW_TOKENS)
    plan_text = generate(model, tokenizer, plan_prompt, torch, MAX_NEW_TOKENS)

    draft_seed = arabic_seed(plan_text)
    repair_prompt = (
        plan_prompt
        + "\n\nأعد كتابة المسودة الأولية التالية لتصبح قصة عربية مكتملة ومتماسكة. "
        + "قد تكون المسودة مبتورة؛ لا تكرر أي نص أجنبي أو تعليق عن النموذج. "
        + "استخدم الخطة أعلاه لإكمال جميع الأحداث السبعة بالترتيب، واجعلها نحو 180 كلمة. "
        + "أخرج القصة النهائية وحدها.\n\nالمسودة العربية المنقحة آليًا:\n"
        + draft_seed
    )
    repaired_text = generate(model, tokenizer, repair_prompt, torch, MAX_NEW_TOKENS)

    conditions = {}
    for name, prompt, story in (
        ("direct", DIRECT_PROMPT, direct_text),
        ("ansg_plan", plan_prompt, plan_text),
        ("ansg_plan_repaired", repair_prompt, repaired_text),
    ):
        conditions[name] = {
            "prompt_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
            "text": story,
            **diagnostics(story, summarise(
                story, plan_graph=plan.graph,
                markers=settings["aqaba"].dialect_markers,
            )),
        }

    report = {
        "experiment": "Controlled frozen Qwen2.5-7B direct vs ANSG-plan-conditioned story realization",
        "status": "completed",
        "model": MODEL_ID,
        "model_weights_updated": False,
        "gpu_names": gpu_names,
        "controlled_task": {
            "setting": "Aqaba, Jordan",
            "characters": ["سالم", "عودة", "أهل العقبة"],
            "target_words": TARGET_WORDS,
            "target_length_gate_whitespace_words": [MIN_TARGET_WORDS, MAX_TARGET_WORDS],
            "same_system_prompt_and_decoding_for_all_conditions": True,
            "decoding": {"method": "greedy", "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "seed": 42},
            "conditions": {
                "direct": "Natural-language prompt with the same seven events as the ANSG plan.",
                "ansg_plan": "Validated ANSG StoryPlan converted to a generation prompt.",
                "ansg_plan_repaired": "Same ANSG plan plus a repair request using only the draft's Arabic-script text.",
            },
        },
        "plan_validation": validation.to_dict(),
        "plan_origin": "ANSG experimental pilot build_plan(repaired=True); no heritage source text supplied to the model",
        "conditions": conditions,
        "limits": [
            "Frozen-backbone inference only; no model training and no reproduction of thesis experiments.",
            "CSAM, DAL, CKG retrieval and PPO/RL are not executed.",
            "The Aqaba dialect setting remains an unreviewed draft; marker counts are proxies, not dialect authenticity.",
            "Surface trigger counts do not establish event order, causality or plot adherence.",
            "Heuristic gates are screening signals, not quality scores.",
            "Only one prompt and one deterministic seed were run; no human ratings have been collected.",
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
        "conditions": {k: {
            "whitespace_word_count": v["whitespace_word_count"],
            "arabic_letter_count": v["arabic_letter_count"],
            "latin_letter_count": v["latin_letter_count"],
            "cjk_character_count": v["cjk_character_count"],
            "heuristic_gates": v["heuristic_gates"],
        } for k, v in conditions.items()},
        "output_dir": str(OUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
