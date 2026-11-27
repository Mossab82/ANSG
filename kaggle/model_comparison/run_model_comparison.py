"""Compare open Arabic-capable instruct models for dialectal story generation.

Inference only, on a Kaggle T4 GPU. This is a backbone selection step for a new
ANSG implementation. It reproduces no dissertation result and computes no
CRAFT/CMPI metric. Stories are written to a blind review file so that a native
speaker can judge dialect authenticity without knowing which model wrote what.
"""
from __future__ import annotations

import gc
import json
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/ansg_model_comparison")
SEED = 42
MAX_NEW_TOKENS = 450

MODELS = [
    {"id": "Qwen/Qwen2.5-7B-Instruct", "trust_remote_code": False},
    {"id": "inception42/Jais-3-7B", "trust_remote_code": True},
    {"id": "humain-ai/ALLaM-7B-Instruct-preview", "trust_remote_code": False},
]

DIALECTS = {
    "msa": "بالعربية الفصحى في السرد والحوار",
    "egyptian": "بسرد فصيح وحوار باللهجة المصرية (القاهرية)",
    "levantine": "بسرد فصيح وحوار باللهجة الشامية (الأردنية أو الفلسطينية)",
    "gulf": "بسرد فصيح وحوار باللهجة الخليجية (النجدية أو الإماراتية)",
    "maghrebi": "بسرد فصيح وحوار بالدارجة المغربية",
}

SEEDS = {
    "karam": (
        "اكتب قصة قصيرة أصلية (نحو 150 كلمة) عن رجل فقير يستضيف غريبًا في ليلة باردة "
        "ويقدّم له آخر ما عنده من طعام، ثم يتبيّن له أن لكرمه أثرًا لم يتوقعه. "
        "اجعل في القصة حوارًا بين الشخصيتين، واختمها بعبرة واضحة."
    ),
    "juha": (
        "اكتب حكاية شعبية قصيرة أصلية (نحو 150 كلمة) عن جحا وجاره الطمّاع الذي يستعير "
        "منه قِدرًا، فيلقّنه جحا درسًا بحيلة ذكية. اجعل في الحكاية حوارًا طريفًا، "
        "واختمها بمثل أو عبرة."
    ),
}

SYSTEM = (
    "أنت راوٍ عربي ماهر. اكتب القصة وحدها بالعربية، بلا عنوان ولا مقدمة ولا شرح، "
    "ولا تستخدم أي لغة أخرى."
)

ARABIC = re.compile(r"[؀-ۿ]")
LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def sh(*cmd: str) -> None:
    subprocess.run(list(cmd), check=False)


def arabic_ratio(text: str) -> float:
    letters = LETTER.findall(text)
    return round(sum(1 for c in letters if ARABIC.match(c)) / len(letters), 3) if letters else 0.0


def build_prompts() -> list[dict]:
    prompts = []
    for seed_id, seed_text in SEEDS.items():
        for dialect_id, dialect_text in DIALECTS.items():
            prompts.append({
                "prompt_id": f"{seed_id}-{dialect_id}",
                "seed": seed_id,
                "dialect": dialect_id,
                "text": f"{seed_text} اكتبها {dialect_text}.",
            })
    return prompts


def chat_inputs(tokenizer, prompt: str, torch):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    except Exception:
        # Some templates reject a system role; fold it into the user turn.
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": SYSTEM + "\n\n" + prompt}],
            add_generation_prompt=True, return_tensors="pt",
        )
    return ids


def run_model(spec: dict, prompts: list[dict], torch, transformers) -> dict:
    model_id = spec["id"]
    record = {"model": model_id, "status": "started", "samples": []}
    t0 = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=spec["trust_remote_code"]
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=spec["trust_remote_code"],
    )
    model.eval()
    record["load_seconds"] = round(time.time() - t0, 1)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for item in prompts:
        random.seed(SEED)
        torch.manual_seed(SEED)
        ids = chat_inputs(tokenizer, item["text"], torch).to(model.device)
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=pad_id,
            )
        new = out[0, ids.shape[1]:]
        text = tokenizer.decode(new, skip_special_tokens=True).strip()
        record["samples"].append({
            "prompt_id": item["prompt_id"],
            "seed": item["seed"],
            "dialect": item["dialect"],
            "text": text,
            "new_tokens": int(new.shape[0]),
            "seconds": round(time.time() - start, 1),
            "arabic_letter_ratio": arabic_ratio(text),
            "whitespace_words": len(text.split()),
        })
        print(f"[{model_id}] {item['prompt_id']}: {len(text.split())} words", flush=True)
    record["status"] = "completed"
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return record


def write_blind_review(results: list[dict], prompts: list[dict]) -> None:
    completed = [r for r in results if r["status"] == "completed"]
    rng = random.Random(SEED)
    labels = "ABCDEFG"
    key = {}
    lines = [
        "# مراجعة عمياء: قصص النماذج",
        "",
        "لكل طلب قصة، القصص مرتبة بحروف عشوائية. قيّم كل قصة من 1 إلى 5 في:",
        "(1) صحة اللهجة في الحوار، (2) سلامة اللغة، (3) تماسك القصة، (4) الروح الثقافية.",
        "المفتاح في `blind_key.json`. لا تفتحه قبل التقييم.",
        "",
    ]
    for item in prompts:
        lines += [f"## {item['prompt_id']}", "", f"> {item['text']}", ""]
        entries = []
        for r in completed:
            sample = next(s for s in r["samples"] if s["prompt_id"] == item["prompt_id"])
            entries.append((r["model"], sample["text"]))
        rng.shuffle(entries)
        key[item["prompt_id"]] = {}
        for label, (model_id, text) in zip(labels, entries):
            key[item["prompt_id"]][label] = model_id
            lines += [f"### {label}", "", text, "", "التقييم: لهجة __ / لغة __ / تماسك __ / ثقافة __", ""]
    (OUT / "stories_blind_review.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sh(sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "accelerate")
    import torch
    import transformers

    report = {
        "purpose": "backbone selection for a new ANSG implementation; not a dissertation reproduction",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "sampling": {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1,
                     "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
        "results": [],
    }
    prompts = build_prompts()
    for spec in MODELS:
        try:
            report["results"].append(run_model(spec, prompts, torch, transformers))
        except Exception as exc:  # one failing model must not stop the comparison
            report["results"].append({
                "model": spec["id"], "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-3000:],
            })
            print(f"[{spec['id']}] FAILED: {exc}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)
        (OUT / "model_comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    write_blind_review(report["results"], prompts)
    report["status"] = "completed" if any(r["status"] == "completed" for r in report["results"]) else "failed"
    (OUT / "model_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("DONE", json.dumps({r["model"]: r["status"] for r in report["results"]}), flush=True)


if __name__ == "__main__":
    main()
