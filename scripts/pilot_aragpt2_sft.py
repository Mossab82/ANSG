#!/usr/bin/env python3
"""Small, provenance-recorded Arabic story-continuation SFT pilot.

This is an engineering pilot, not the selected Chapter 5 reproduction: it uses
one pretrained Arabic GPT-2 checkpoint and quarantined synthetic story data,
without ANSG CSAM/DAL, GAN, PPO, or a thesis-sized dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "aubmindlab/aragpt2-base"
MODEL_REVISION = "257d7289dac3605a6cd903ff71d67a52fc5d09aa"
DATASET_REVISION = "82760dc9de372ff56a531766194f91f78e270520"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path, tokenizer, max_length: int):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            ids.append(tokenizer.eos_token_id)
        ids = ids[:max_length]
        if len(ids) < 12:
            continue
        prefix_len = min(max(8, round(len(ids) * 0.30)), len(ids) - 4)
        labels = [-100] * prefix_len + ids[prefix_len:]
        rows.append({"record_id": row["record_id"], "input_ids": ids, "labels": labels,
                     "prefix_len": prefix_len})
    return rows


def batch_for(row, device):
    return {
        "input_ids": torch.tensor([row["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(row["input_ids"])), dtype=torch.long, device=device),
        "labels": torch.tensor([row["labels"]], dtype=torch.long, device=device),
    }


@torch.no_grad()
def mean_loss(model, rows, device):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    for row in rows:
        batch = batch_for(row, device)
        result = model(**batch)
        n = int((batch["labels"][:, 1:] != -100).sum())
        loss_sum += float(result.loss) * n
        token_count += n
    return loss_sum / max(token_count, 1)


@torch.no_grad()
def generate(model, tokenizer, prompt, device, seed, max_new_tokens):
    model.eval()
    torch.manual_seed(seed)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    result = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=True,
                            top_k=50, top_p=0.9, temperature=0.8,
                            repetition_penalty=1.15, no_repeat_ngram_size=3,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(result[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Local snapshot of the pinned AraGPT2-base revision")
    ap.add_argument("--data", required=True, help="Small JSONL subset with record_id and text")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-count", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir, data_path, out = Path(args.model_dir), Path(args.data), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                 trust_remote_code=False).to(device)
    rows = read_rows(data_path, tokenizer, args.max_length)
    if len(rows) < args.train_count + 1:
        raise ValueError(f"need at least train_count + 1 rows; got {len(rows)}")
    train_rows, valid_rows = rows[:args.train_count], rows[args.train_count:]
    prompt = tokenizer.decode(valid_rows[0]["input_ids"][:valid_rows[0]["prefix_len"]],
                              skip_special_tokens=True)

    initial_train_loss = mean_loss(model, train_rows, device)
    initial_valid_loss = mean_loss(model, valid_rows, device)
    baseline = generate(model, tokenizer, prompt, device, args.seed, args.max_new_tokens)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    training_trace = []
    for epoch in range(args.epochs):
        order = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(order)
        losses = []
        for i in order:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch_for(train_rows[i], device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        training_trace.append({"epoch": epoch + 1, "mean_batch_loss": sum(losses) / len(losses),
                               "optimizer_steps": len(losses)})

    final_train_loss = mean_loss(model, train_rows, device)
    final_valid_loss = mean_loss(model, valid_rows, device)
    adapted = generate(model, tokenizer, prompt, device, args.seed, args.max_new_tokens)
    checkpoint_dir = out / "checkpoint"
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    (out / "baseline_generation.txt").write_text(baseline + "\n", encoding="utf-8")
    (out / "finetuned_generation.txt").write_text(adapted + "\n", encoding="utf-8")

    model_hashes = {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file() and p.name != ".gitattributes"}
    result = {
        "run_id": "anhc_aragpt2_sft_pilot",
        "status": "completed",
        "historical_experiment": False,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION,
                  "directory_sha256_by_file": model_hashes,
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "license_metadata": "no SPDX license declared on model card; research-only disclaimer observed"},
        "dataset": {"name": "TinyStories Algerian Darija", "revision": DATASET_REVISION,
                    "license_claim": "derivative MIT; upstream CDLA-Sharing-1.0", "data_kind": "synthetic",
                    "rights_status": "unresolved; local quarantined pilot only",
                    "source_file": str(data_path), "source_sha256": sha256(data_path),
                    "examples_total": len(rows), "train_examples": len(train_rows),
                    "validation_examples": len(valid_rows),
                    "train_record_ids": [r["record_id"] for r in train_rows],
                    "validation_record_ids": [r["record_id"] for r in valid_rows]},
        "training": {"task": "story continuation with prompt-token loss masked",
                     "epochs": args.epochs, "learning_rate": args.learning_rate,
                     "optimizer": "AdamW", "seed": args.seed, "device": str(device),
                     "torch": torch.__version__, "transformers": "4.51.3",
                     "max_length": args.max_length, "trace": training_trace,
                     "initial_train_loss": initial_train_loss,
                     "final_train_loss": final_train_loss,
                     "initial_validation_loss": initial_valid_loss,
                     "final_validation_loss": final_valid_loss},
        "generation": {"prompt": prompt, "seed": args.seed,
                       "baseline_file": "baseline_generation.txt",
                       "finetuned_file": "finetuned_generation.txt",
                       "decoding": {"top_k": 50, "top_p": 0.9, "temperature": 0.8,
                                    "repetition_penalty": 1.15, "no_repeat_ngram_size": 3}},
        "limits": ["No CSAM/DAL conditioning, GAN, PPO, or CRAFT reward.",
                   "Small synthetic sample; loss/generation quality is exploratory only.",
                   "No claims about dialect authenticity or generalization.",
                   "Not the selected dissertation-scale Chapter 5 training reproduction."],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out / "pilot_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"status": result["status"], "device": str(device),
                      "initial_train_loss": initial_train_loss,
                      "final_train_loss": final_train_loss,
                      "initial_validation_loss": initial_valid_loss,
                      "final_validation_loss": final_valid_loss,
                      "elapsed_seconds": result["elapsed_seconds"],
                      "baseline": baseline, "finetuned": adapted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
