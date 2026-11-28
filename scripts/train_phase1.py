"""Phase 1 (supervised) training of CSAM + DAL on a frozen AraGPT2 backbone.

Follows the dissertation's Phase 1 (Table 5.1, §5.1.3) and objective (Eq. 4.16)
in this new implementation:

    L = L_LM + λ_align · L_cultural + λ_schema · L_schema

* Backbone (AraGPT2-base by default) is frozen; CSAM and DAL are trained.
* Data: Wikisource heritage text (MSA), IADD dialect sentences (CC-BY-4.0;
  EGY/LEV/GLF/MGH), and the synthetic parallel stories (all five varieties,
  with their value as schema label).
* L_cultural needs motif-span annotations, which do not exist yet, so
  λ_align defaults to 0 and the run says so in its report.
* Schema weights: for parallel stories the gold value schema is used
  (teacher forcing, weight 1.0); otherwise Eq. 4.20 activation.

Outputs in --out: adapters.pt (CSAM + DAL weights), report.json (losses,
held-out loss per variety with and without the adapters), samples.md
(generated story openings per variety).

Kaggle notebook (GPU, Internet on):
    !git clone -b experiment/base-model-comparison https://github.com/Mossab82/ansg.git
    %cd ansg
    !pip install -q -e . datasets
    !python scripts/train_phase1.py --out /kaggle/working/phase1

Local smoke test (no downloads, tiny random backbone):
    PYTHONPATH=src python scripts/train_phase1.py --smoke --out /tmp/phase1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias
from ansg.csam_bias import CSAMSettings, CulturalSchemaBias, schema_prediction_loss
from ansg.dal_lora import DIALECTS, DialectalAdaptation, formal_schema_indices
from ansg.evaluation.seed_adherence import seed_adherence
from ansg.generation.seed_prompt import build_story_prompt
from ansg.knowledge.ckg import CulturalKnowledgeGraph
from ansg.knowledge.graph_embeddings import load_schema_embeddings

ROOT = Path(__file__).resolve().parents[1]
IADD_REGION = {"EGY": "egyptian", "LEV": "levantine", "GLF": "gulf", "MGH": "maghrebi"}
# Keep Moroccan rows only for the Maghrebi variety (IADD MGH also holds Algerian and
# Tunisian text), and the four Levantine countries; Gulf rows have no country code.
IADD_COUNTRIES = {"egyptian": {"EGY"}, "levantine": {"JOR", "PSE", "SYR", "LBN"}, "maghrebi": {"MAR"}, "gulf": None}
_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_LATIN = re.compile(r"[A-Za-z]")


def clean(text: str) -> str:
    """Strip diacritics and tatweel (AraGPT2 was trained on undiacritised text) and squeeze spaces."""
    return " ".join(_DIAC.sub("", text).split())


def arabic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("\u0600" <= c <= "\u06FF" for c in letters) / len(letters) if letters else 0.0


# ---------------------------------------------------------------- data
def load_examples(args, schema_ids: list[str]) -> list[dict]:
    """Return [{"text", "dialect", "schema" (index or None), "source"}]."""
    rng = random.Random(args.seed)
    examples = []
    for line in (ROOT / "data/corpora/classical_arabic/stories.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        paras = [p.strip() for p in rec["text"].split("\n") if len(p.split()) > 20]
        for p in paras:
            examples.append({"text": clean(p), "dialect": "msa", "schema": None, "source": "wikisource"})
    sid = {s: i for i, s in enumerate(schema_ids)}
    for line in (ROOT / "data/stories/parallel/parallel_stories.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        schema = sid.get(f"schema:{rec['value']}")
        for d in DIALECTS:
            for _ in range(args.parallel_repeat):
                examples.append({"text": clean(rec[d]), "dialect": d, "schema": schema, "source": "parallel",
                                 "story_id": rec["id"]})
    if not args.smoke and args.iadd_per_region > 0:
        from datasets import load_dataset
        iadd = load_dataset("evageon/IADD", split="train")
        by_region: dict[str, list[str]] = {}
        for row in iadd:
            region = IADD_REGION.get(row["Region"])
            text = (row["Sentence"] or "").strip()
            if not region or len(text.split()) < 4 or _LATIN.search(text) or arabic_ratio(text) < 0.95:
                continue
            allowed = IADD_COUNTRIES[region]
            if allowed is not None and row["Country"] not in allowed:
                continue
            by_region.setdefault(region, []).append(clean(text))
        for region, sents in by_region.items():
            rng.shuffle(sents)
            sents = sents[: args.iadd_per_region]
            # pack short sentences of one variety into longer sequences
            for i in range(0, len(sents), args.iadd_pack):
                examples.append({"text": " . ".join(sents[i:i + args.iadd_pack]), "dialect": region,
                                 "schema": None, "source": "iadd"})
    return examples


def split(examples, seed):
    """Hold out 5% per source; parallel stories are held out by story, not by copy."""
    rng = random.Random(seed)
    stories = sorted({e["story_id"] for e in examples if e["source"] == "parallel"})
    held_story = set(rng.sample(stories, max(1, len(stories) // 6))) if len(stories) > 1 else set()
    train, held = [], []
    for e in examples:
        if e["source"] == "parallel":
            (held if e["story_id"] in held_story else train).append(e)
        else:
            (held if rng.random() < 0.05 else train).append(e)
    # one copy per held-out parallel variant is enough for evaluation
    seen, dedup = set(), []
    for e in held:
        key = (e["text"], e["dialect"])
        if key not in seen:
            seen.add(key)
            dedup.append(e)
    return train, dedup


class Batcher:
    def __init__(self, tok, n_schema, seq, device):
        self.tok, self.n_schema, self.seq, self.device = tok, n_schema, seq, device

    def __call__(self, batch):
        ids = [self.tok.encode(e["text"])[: self.seq] for e in batch]
        t = max(len(x) for x in ids)
        pad = self.tok.pad_id
        input_ids = torch.full((len(ids), t), pad, dtype=torch.long)
        labels = torch.full((len(ids), t), -100, dtype=torch.long)
        mask = torch.zeros(len(ids), t)
        for i, x in enumerate(ids):
            input_ids[i, : len(x)] = torch.tensor(x)
            labels[i, : len(x)] = torch.tensor(x)
            mask[i, : len(x)] = 1
        dialect = torch.tensor([DIALECTS.index(e["dialect"]) for e in batch])
        schema = torch.zeros(len(batch), self.n_schema)
        has_schema = torch.zeros(len(batch), dtype=torch.bool)
        for i, e in enumerate(batch):
            if e["schema"] is not None:
                schema[i, e["schema"]] = 1
                has_schema[i] = True
        to = lambda x: x.to(self.device)  # noqa: E731
        return to(input_ids), to(labels), to(mask), to(dialect), to(schema), to(has_schema)


# ---------------------------------------------------------------- model
class SmokeTokenizer:
    """Character-level tokenizer for the offline smoke test."""
    def __init__(self, texts):
        chars = sorted({c for t in texts for c in t})
        self.itos = ["<pad>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.pad_id = 0
        self.eos_id = 0

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i)


class HFTokenizer:
    def __init__(self, name):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.eos_id = self.tok.eos_token_id
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)


def build_backbone(args, examples):
    if args.smoke:
        tok = SmokeTokenizer([e["text"] for e in examples])
        spec = GPT2Spec(vocab_size=len(tok.itos), n_positions=512, n_embd=96, n_layer=2, n_head=12)
        return GPT2WithAttentionBias(spec), tok
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model)
    spec = GPT2Spec.from_hf_config(hf.config.to_dict())
    model = GPT2WithAttentionBias.from_hf_state_dict(spec, hf.state_dict())
    del hf
    return model, HFTokenizer(args.model)


USE = {"csam": True, "dal": True}  # set from --no-csam / --no-dal (ablation)


def adapted_forward(model, csam, dal, ids, labels, mask, dialect, schema, has_schema, use_adapters=True):
    if not use_adapters:
        return model(ids, labels=labels), None, None
    context = (model.wte(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    weights, _ = csam.activation(context)
    weights = torch.where(has_schema[:, None], schema, weights)            # teacher forcing for gold labels
    out = model(ids, labels=labels,
                bias_fn=csam.bias_fn(model.wpe.weight, weights) if USE["csam"] else None,
                delta_fn=dal.delta_fn(dialect, csam.schema_embeddings(), weights) if USE["dal"] else None)
    return out, context, weights


@torch.no_grad()
def heldout_loss(model, csam, dal, batcher, held, bs, use_adapters):
    totals: dict[str, list[float]] = {}
    for i in range(0, len(held), bs):
        batch = held[i:i + bs]
        ids, labels, mask, dialect, schema, has_schema = batcher(batch)
        for j, e in enumerate(batch):
            one = [x[j:j + 1] for x in (ids, labels, mask, dialect, schema, has_schema)]
            out, _, _ = adapted_forward(model, csam, dal, *one, use_adapters=use_adapters)
            totals.setdefault(e["dialect"], []).append(out["loss"].item())
            totals.setdefault(f"{e['source']}/{e['dialect']}", []).append(out["loss"].item())
    return {d: {"loss": round(sum(v) / len(v), 4), "n": len(v)} for d, v in sorted(totals.items())}


@torch.no_grad()
def generate(model, csam, dal, tok, prefix, dialect, schema_vec, max_new, temperature, top_p, device,
             repetition_penalty=1.3, no_repeat_ngram=3):
    ids = torch.tensor([tok.encode(clean(prefix))], device=device)
    d = torch.tensor([DIALECTS.index(dialect)], device=device)
    w = schema_vec[None].to(device)
    for _ in range(max_new):
        ctx = ids[:, -model.spec.n_positions:]
        logits = model(ctx, bias_fn=csam.bias_fn(model.wpe.weight, w) if USE["csam"] else None,
                       delta_fn=dal.delta_fn(d, csam.schema_embeddings(), w) if USE["dal"] else None
                       )["logits"][0, -1]
        seen = ids[0].unique()
        logits[seen] = torch.where(logits[seen] > 0, logits[seen] / repetition_penalty,
                                   logits[seen] * repetition_penalty)
        hist = ids[0].tolist()
        if no_repeat_ngram > 1 and len(hist) >= no_repeat_ngram - 1:
            prefix_ng = tuple(hist[-(no_repeat_ngram - 1):])
            banned = [hist[i + no_repeat_ngram - 1] for i in range(len(hist) - no_repeat_ngram + 1)
                      if tuple(hist[i:i + no_repeat_ngram - 1]) == prefix_ng]
            if banned:
                logits[banned] = float("-inf")
        logits = logits / temperature
        probs = torch.softmax(logits, -1)
        sorted_p, order = probs.sort(descending=True)
        keep = sorted_p.cumsum(0) - sorted_p < top_p
        choice = order[keep][torch.multinomial(sorted_p[keep] / sorted_p[keep].sum(), 1)]
        if choice.item() == tok.eos_id and not isinstance(tok, SmokeTokenizer):
            break
        ids = torch.cat([ids, choice.view(1, 1)], dim=1)
    return tok.decode(ids[0].tolist())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aubmindlab/aragpt2-base")
    ap.add_argument("--out", default="phase1_out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda-schema", type=float, default=0.1)
    ap.add_argument("--lambda-align", type=float, default=0.0)
    ap.add_argument("--iadd-per-region", type=int, default=8000)
    ap.add_argument("--iadd-pack", type=int, default=6)
    ap.add_argument("--parallel-repeat", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-csam", action="store_true", help="ablation: train and evaluate without CSAM bias")
    ap.add_argument("--no-dal", action="store_true", help="ablation: train and evaluate without DAL adapters")
    args = ap.parse_args()
    USE["csam"], USE["dal"] = not args.no_csam, not args.no_dal
    if args.lambda_align:
        raise SystemExit("L_cultural needs motif-span annotations, which this corpus does not have yet.")
    if args.smoke:
        args.steps, args.batch, args.seq, args.log_every = 20, 4, 64, 5

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = CulturalKnowledgeGraph.from_json(ROOT / "data/ckg/seed_ckg.json")
    schema_ids, g = load_schema_embeddings(ROOT / "data/ckg/seed_ckg_embeddings.npz", graph)
    examples = load_examples(args, schema_ids)
    train, held = split(examples, args.seed)
    model, tok = build_backbone(args, examples)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    spec = model.spec
    csam = CulturalSchemaBias(
        CSAMSettings(n_head=spec.n_head, d_model=spec.n_embd, layers=tuple(range(spec.n_layer))),
        torch.tensor(g)).to(device)
    dal = DialectalAdaptation(n_layer=spec.n_layer, d_model=spec.n_embd,
                              formal_schema_index=formal_schema_indices(schema_ids)).to(device)
    params = list(csam.parameters()) + list(dal.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    batcher = Batcher(tok, len(schema_ids), args.seq, device)
    counts = {}
    for e in train:
        counts[f"{e['source']}/{e['dialect']}"] = counts.get(f"{e['source']}/{e['dialect']}", 0) + 1
    print("train examples:", json.dumps(counts, ensure_ascii=False), "| held-out:", len(held), flush=True)

    base_loss = heldout_loss(model, csam, dal, batcher, held, args.batch, use_adapters=False)
    print("held-out loss, backbone only:", base_loss, flush=True)

    rng = random.Random(args.seed)
    log, start = [], time.time()
    for step in range(1, args.steps + 1):
        batch = rng.sample(train, args.batch)
        ids, labels, mask, dialect, schema, has_schema = batcher(batch)
        out, context, _ = adapted_forward(model, csam, dal, ids, labels, mask, dialect, schema, has_schema)
        loss = out["loss"]
        s_loss = torch.tensor(0.0, device=device)
        if has_schema.any():
            s_loss = schema_prediction_loss(csam, context[has_schema], schema[has_schema]) / has_schema.sum()
            loss = loss + args.lambda_schema * s_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps:
            rec = {"step": step, "lm_loss": round(out["loss"].item(), 4), "schema_loss": round(s_loss.item(), 4),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - start)}
            log.append(rec)
            print(rec, flush=True)

    csam.eval()
    dal.eval()
    adapted_loss = heldout_loss(model, csam, dal, batcher, held, args.batch, use_adapters=True)
    print("held-out loss, with CSAM+DAL:", adapted_loss, flush=True)

    torch.save({"csam": csam.state_dict(), "dal": dal.state_dict(), "schema_ids": schema_ids,
                "backbone": args.model, "args": vars(args)}, out_dir / "adapters.pt")

    samples, lines = [], ["# Phase 1 samples", "", f"backbone: {args.model}; steps: {args.steps}", ""]
    for region in DIALECTS:
        prompt = build_story_prompt(region, rng_seed=1)
        e = prompt.elements
        schema_vec = torch.zeros(len(schema_ids))
        key = f"schema:{e['value']}"
        if key in schema_ids:
            schema_vec[schema_ids.index(key)] = 1.0
        text = generate(model, csam, dal, tok, e["opening"] or "كان يا ما كان", region, schema_vec,
                        max_new=60 if args.smoke else 220, temperature=0.8, top_p=0.9, device=device)
        sas = seed_adherence(text, region, e).as_dict()
        samples.append({"region": region, "value": e["value"], "text": text, "sas": sas})
        lines += [f"## {region} (value: {e['value']})", "", text, "", f"SAS: {sas['score']} {sas}", ""]
    (out_dir / "samples.md").write_text("\n".join(lines), encoding="utf-8")

    report = {
        "purpose": "Phase 1 of a new thesis-aligned implementation; not a dissertation reproduction",
        "backbone": args.model, "device": device, "args": vars(args),
        "trainable_parameters": {"csam": sum(p.numel() for p in csam.parameters()),
                                 "dal_adapters": dal.adapter_parameter_count(),
                                 "dal_total": sum(p.numel() for p in dal.parameters())},
        "objective": "L_LM + lambda_schema * L_schema; L_cultural disabled (no motif-span annotations)",
        "components": dict(USE),
        "decoding": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.3, "no_repeat_ngram": 3},
        "train_counts": counts, "heldout_size": len(held),
        "heldout_loss_backbone_only": base_loss, "heldout_loss_with_adapters": adapted_loss,
        "log": log, "samples": samples,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", sorted(p.name for p in out_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
