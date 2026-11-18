#!/usr/bin/env python3
"""Blind, matched-prompt baseline vs LoRA generation review for the heritage pilot.

Uses the model's chat template for both conditions and identical deterministic
decoding. It performs inference only; it does not train or update model weights.

Example:
  python compare_generation.py \
    --adapter-path ./ansg_heritage_pilot/adapter_final \
    --output-dir ./generation_review

For the default CPU run, install the pinned requirements from the pilot environment
(torch, transformers==4.51.3, peft==0.15.2). The adapter directory is in the
ansg-heritage-training-evidence workflow artifact.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
from pathlib import Path
from typing import Any

MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = {
    "msa": "اكتب بداية قصة عربية أصلية عن صياد وجد صندوقًا غامضًا ثم اكتشف صاحبه الحقيقي:",
    "levantine": "احكِ قصة قصيرة بلهجة أهل العقبة عن شاب لقى صندوق ورجع يدور على صاحبه:",
    "egyptian": "احكي حكاية قصيرة بالمصري عن صياد لقى أمانة وقرر يرجعها لصاحبها:",
    "moroccan": "حكي ليا حكاية قصيرة بالدارجة المغربية على واحد لقى أمانة وبغا يرجعها:",
}


def arabic_letter_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return round(sum("\u0600" <= c <= "\u06ff" for c in letters) / len(letters), 4)


def repeated_trigram_fraction(text: str) -> float:
    words = text.split()
    n = 3
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


def generate(model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


def load_base(model_name: str, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def collect_samples(model: Any, tokenizer: Any, device: str, max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for prompt_id, prompt in PROMPTS.items():
        generated = generate(model, tokenizer, prompt, device, max_new_tokens)
        rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "generated": generated,
                "words": len(generated.split()),
                "arabic_letter_ratio": arabic_letter_ratio(generated),
                "repeated_trigram_fraction": repeated_trigram_fraction(generated),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("generation_review"))
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if not args.adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_path}")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the baseline, then release it before loading the adapted model.
    # This keeps CPU memory use lower on ordinary workstations.
    baseline_model, tokenizer = load_base(args.model, args.device)
    baseline = collect_samples(baseline_model, tokenizer, args.device, args.max_new_tokens)
    del baseline_model
    gc.collect()

    import torch
    from peft import PeftModel

    dtype = torch.float32 if args.device == "cpu" else torch.float16
    adapted_base = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    adapted_base.to(args.device)
    adapted_model = PeftModel.from_pretrained(
        adapted_base, str(args.adapter_path), is_trainable=False
    )
    adapted_model.eval()
    adapted = collect_samples(adapted_model, tokenizer, args.device, args.max_new_tokens)

    result = {
        "status": "completed",
        "model": args.model,
        "adapter_path": str(args.adapter_path),
        "device": args.device,
        "generation": {
            "chat_template": "tokenizer.apply_chat_template(user message, add_generation_prompt=True)",
            "do_sample": False,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "note": "Greedy decoding is deterministic; seed is recorded for reproducibility.",
        },
        "baseline": baseline,
        "adapted": adapted,
        "interpretation_warning": (
            "Automatic ratios are screening proxies only. They do not establish fluency, "
            "plot coherence, dialect fidelity, cultural quality, CRAFT, or ANSG performance."
        ),
    }
    (args.output_dir / "matched_generation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Blind A/B sheet. Keep the key in a separate file for the evaluator.
    rng = random.Random(args.seed)
    key: dict[str, dict[str, str]] = {}
    by_id = {
        "baseline": {row["prompt_id"]: row for row in baseline},
        "adapted": {row["prompt_id"]: row for row in adapted},
    }
    columns = [
        "prompt_id", "prompt", "output_A", "output_B",
        "fluency_A_1to5", "fluency_B_1to5",
        "coherence_A_1to5", "coherence_B_1to5",
        "dialect_fidelity_A_1to5", "dialect_fidelity_B_1to5",
        "cultural_appropriateness_A_1to5", "cultural_appropriateness_B_1to5",
        "repetition_A_1to5", "repetition_B_1to5",
        "preferred_output", "notes",
    ]
    with (args.output_dir / "blind_human_review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for prompt_id, prompt in PROMPTS.items():
            swap = bool(rng.getrandbits(1))
            source_a, source_b = (("adapted", "baseline") if swap else ("baseline", "adapted"))
            key[prompt_id] = {"A": source_a, "B": source_b}
            writer.writerow(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "output_A": by_id[source_a][prompt_id]["generated"],
                    "output_B": by_id[source_b][prompt_id]["generated"],
                }
            )
    (args.output_dir / "blind_review_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote matched_generation.json and blind review files to {args.output_dir}")


if __name__ == "__main__":
    main()
