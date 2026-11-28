"""Phase 1 (supervised) training of CSAM + DAL on a frozen AraGPT2 backbone.

Follows the dissertation's Phase 1 (Table 5.1, §5.1.3) and objective (Eq. 4.16)
in this new implementation:

    L = L_LM + λ_align · L_cultural + λ_schema · L_schema

* Backbone (AraGPT2-base by default) is frozen; CSAM and DAL are trained.
* Data: Wikisource heritage text (MSA), IADD dialect sentences (CC-BY-4.0;
  EGY/LEV/GLF/MGH), and the synthetic parallel stories (all five varieties,
  with their value as schema label).
* L_cultural (Eq. 4.18) uses rule-based motif spans from
  data/stories/parallel/motif_spans.jsonl (scripts/annotate_motif_spans.py):
  positive pairs lie inside one motif span, negatives are random pairs. λ_align = 0.1
  and λ_schema = 0.05 follow §4.3.5; --lambda-align 0 switches L_cultural off.
* --csam-init-std and --csam-lr-mult are engineering settings (option B): a larger
  initial scale for the Eq. 4.13 parameters and a larger learning rate for CSAM.
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
# Country values as stored in IADD (checked with the datasets-server statistics endpoint).
IADD_COUNTRIES = {"egyptian": {"Egypt"}, "levantine": {"Palestine", "Jordan", "Syria", "Lebanon"},
                  "maghrebi": {"Morocco"}, "gulf": None}
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
    spans_path = ROOT / "data/stories/parallel/motif_spans.jsonl"
    spans = {}
    if spans_path.exists():
        for line in spans_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            spans[(rec["id"], rec["variety"])] = [(s["start"], s["end"]) for s in rec["spans"]]
    for line in (ROOT / "data/stories/parallel/parallel_stories.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        schema = sid.get(f"schema:{rec['value']}")
        if rec.get("split") == "test":   # reserved test stories: evaluation only, never trained on
            for d in DIALECTS:
                examples.append({"text": clean(rec[d]), "dialect": d, "schema": schema,
                                 "source": "parallel_test", "story_id": rec["id"]})
            continue
        for d in DIALECTS:
            for _ in range(args.parallel_repeat):
                examples.append({"text": clean(rec[d]), "dialect": d, "schema": schema, "source": "parallel",
                                 "story_id": rec["id"], "spans": spans.get((rec["id"], d), [])})
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
        if e["source"] == "parallel_test":
            held.append(e)
        elif e["source"] == "parallel":
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

    def token_spans(self, batch):
        """Character motif spans -> token index spans [start, end), clipped to --seq."""
        out = []
        for e in batch:
            spans = []
            for a, b in e.get("spans", []):
                ts, te = len(self.tok.encode(e["text"][:a])), len(self.tok.encode(e["text"][:b]))
                ts, te = min(ts, self.seq), min(te, self.seq)
                if te - ts >= 2:
                    spans.append((ts, te))
            out.append(spans)
        return out


def motif_pairs(token_spans, lengths, has_schema, rng, per_example=32):
    """Eq. 4.18 pairs (b, j, k) with k < j: positives inside one motif span, negatives random
    pairs that do not lie inside a common span. Only examples with an active schema count."""
    pos, neg = [], []
    for b, (spans, n) in enumerate(zip(token_spans, lengths)):
        if not spans or not has_schema[b] or n < 3:
            continue
        cand = [(j, k) for ts, te in spans for j in range(ts + 1, te) for k in range(ts, j)]
        chosen = rng.sample(cand, min(per_example, len(cand)))
        pos += [(b, j, k) for j, k in chosen]
        inside = lambda j, k: any(ts <= k < j < te for ts, te in spans)  # noqa: E731
        tries = 0
        while sum(1 for p in neg if p[0] == b) < len(chosen) and tries < 50 * per_example:
            tries += 1
            j = rng.randrange(1, n)
            k = rng.randrange(0, j)
            if not inside(j, k):
                neg.append((b, j, k))
    return pos, neg


def cultural_loss(model, csam, ids, weights, pos, neg):
    """L_cultural (Eq. 4.18) on the heritage matrices H (heads 0-3) of every CSAM layer,
    averaged over layers, heads and pairs."""
    t = ids.shape[1]
    positions = pos_repr(model, ids)
    positions = positions[:t] if positions.dim() == 2 else positions
    p = torch.tensor(pos, device=ids.device)
    q = torch.tensor(neg, device=ids.device)
    terms = []
    for layer in csam.settings.layers:
        h = csam.group_matrices(layer, positions, weights)[:, 0]               # [B, 4, T, T]
        hp = h[p[:, 0], :, p[:, 1], p[:, 2]]                                   # [N, 4]
        hn = h[q[:, 0], :, q[:, 1], q[:, 2]]
        terms.append(-(F.logsigmoid(hp).mean() + F.logsigmoid(-hn).mean()))
    return torch.stack(terms).mean()


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
CSAM_POS = {"mode": "position"}     # set from --csam-pos (D7)


def pos_repr(model, ids):
    """pos_j for Eq. 4.13: the positional-encoding table (literal reading), or the
    position-encoded input embedding of each token, wte(x_j) + wpe(j) (D7)."""
    if CSAM_POS["mode"] == "position":
        return model.wpe.weight
    return model.wte(ids) + model.wpe.weight[: ids.shape[1]][None]


def adapted_forward(model, csam, dal, ids, labels, mask, dialect, schema, has_schema, use_adapters=True):
    if not use_adapters:
        return model(ids, labels=labels), None, None
    context = (model.wte(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    weights, _ = csam.activation(context)
    weights = torch.where(has_schema[:, None], schema, weights)            # teacher forcing for gold labels
    out = model(ids, labels=labels,
                bias_fn=csam.bias_fn(pos_repr(model, ids), weights) if USE["csam"] else None,
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
def csam_diagnostics(model, csam, batcher, held, n=32):
    """How much can CSAM change attention at all?

    Softmax ignores any bias that is constant along the key axis, so only the part of
    the CSAM bias that varies across keys k (for a fixed query j, over k <= j) can move
    attention. Reports that key-varying part (mean |b - row mean|) for heritage heads
    0-3 and rhetorical heads 4-7, plus how often Eq. 4.20 activates any schema without
    teacher forcing.
    """
    sample = held[:n]
    ids, labels, mask, dialect, schema, has_schema = batcher(sample)
    context = (model.wte(ids) * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    w_free, active = csam.activation(context)
    w = torch.where(has_schema[:, None], schema, w_free)
    t = ids.shape[1]
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=ids.device))
    per_layer = []
    for layer in csam.settings.layers:
        pr = pos_repr(model, ids)
        b = csam.bias(layer, pr[:t] if pr.dim() == 2 else pr, w)              # [B, 12, T, T]
        if b is None:
            continue
        bm = b.masked_fill(~causal, 0)
        cnt = causal.sum(-1).clamp_min(1)
        row_mean = bm.sum(-1, keepdim=True) / cnt[:, None]
        dev = ((b - row_mean).abs().masked_fill(~causal, 0).sum(-1) / cnt)      # [B, 12, T]
        per_layer.append({"layer": layer,
                          "heritage_key_varying": round(dev[:, 0:4].mean().item(), 6),
                          "rhetorical_key_varying": round(dev[:, 4:8].mean().item(), 6),
                          "max_abs_bias": round(b.abs().max().item(), 6)})
    return {"examples": len(sample),
            "with_gold_schema": int(has_schema.sum().item()),
            "free_activation_rate": round(active.any(-1).float().mean().item(), 4),
            "mean_active_schemas_free": round(active.sum(-1).float().mean().item(), 3),
            "per_layer": per_layer}


_ODD = set(chr(c) for c in range(0x0600, 0x0620)) - {"\u060c", "\u061b", "\u061f"}


def banned_token_ids(tok):
    """Vocabulary items containing Arabic sign/format characters (e.g. U+0605, U+060A,
    U+0613) that leak into samples as noise; the comma, semicolon and question mark stay."""
    if not hasattr(tok, "tok"):
        return []
    if not hasattr(tok, "_banned"):
        tok._banned = [i for i in range(len(tok.tok)) if set(tok.tok.decode([i])) & _ODD]
    return tok._banned


@torch.no_grad()
def generate(model, csam, dal, tok, prefix, dialect, schema_vec, max_new, temperature, top_p, device,
             repetition_penalty=1.3, no_repeat_ngram=3, min_new=0):
    ids = torch.tensor([tok.encode(clean(prefix))], device=device)
    d = torch.tensor([DIALECTS.index(dialect)], device=device)
    w = schema_vec[None].to(device)
    for step in range(max_new):
        ctx = ids[:, -model.spec.n_positions:]
        logits = model(ctx, bias_fn=csam.bias_fn(pos_repr(model, ctx), w) if USE["csam"] else None,
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
        bad = banned_token_ids(tok)
        if bad:
            logits[bad] = float("-inf")
        if step < min_new and tok.eos_id is not None:
            logits[tok.eos_id] = float("-inf")
        logits = logits / temperature
        tail = hist[-6:]
        before = tok.decode(tail)
        for _ in range(20):
            probs = torch.softmax(logits, -1)
            sorted_p, order = probs.sort(descending=True)
            keep = sorted_p.cumsum(0) - sorted_p < top_p
            choice = order[keep][torch.multinomial(sorted_p[keep] / sorted_p[keep].sum(), 1)]
            # byte-level BPE: a token can complete a byte sequence into an Arabic sign/format
            # character (e.g. U+0605) that no single vocabulary item contains; resample then
            added = tok.decode(tail + [choice.item()])[len(before):]
            if not (set(added) & _ODD):
                break
            logits[choice] = float("-inf")
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
    ap.add_argument("--lambda-schema", type=float, default=0.05, help="§4.3.5")
    ap.add_argument("--lambda-align", type=float, default=0.1, help="§4.3.5; 0 disables L_cultural")
    ap.add_argument("--csam-init-std", type=float, default=0.02,
                    help="init std of the Eq. 4.13 vectors (option B; 0.02 = previous runs)")
    ap.add_argument("--csam-lr-mult", type=float, default=1.0, help="CSAM learning rate = lr x this (option B)")
    ap.add_argument("--csam-pos", choices=["position", "token_position"], default="position",
                    help="pos_j in Eq. 4.13: positional encoding only, or position-encoded token embedding (D7)")
    ap.add_argument("--iadd-per-region", type=int, default=8000)
    ap.add_argument("--iadd-pack", type=int, default=6)
    ap.add_argument("--parallel-repeat", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500, help="held-out check; the best adapters are kept")
    ap.add_argument("--mix", default="parallel=0.4,iadd=0.4,wikisource=0.2",
                    help="sampling weight per training source; 'uniform' samples examples uniformly")
    ap.add_argument("--min-new-tokens", type=int, default=120)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-csam", action="store_true", help="ablation: train and evaluate without CSAM bias")
    ap.add_argument("--no-dal", action="store_true", help="ablation: train and evaluate without DAL adapters")
    args = ap.parse_args()
    USE["csam"], USE["dal"] = not args.no_csam, not args.no_dal
    CSAM_POS["mode"] = args.csam_pos
    if args.lambda_align and not (ROOT / "data/stories/parallel/motif_spans.jsonl").exists():
        raise SystemExit("L_cultural needs data/stories/parallel/motif_spans.jsonl; "
                         "run scripts/annotate_motif_spans.py or pass --lambda-align 0")
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
    if args.csam_init_std != 0.02:
        with torch.no_grad():
            for prm in (csam.v_pos_j, csam.v_pos_k, csam.v_schema):
                prm.normal_(0.0, args.csam_init_std)
    params = list(csam.parameters()) + list(dal.parameters())
    opt = torch.optim.AdamW([{"params": list(csam.parameters()), "lr": args.lr * args.csam_lr_mult},
                             {"params": list(dal.parameters()), "lr": args.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    batcher = Batcher(tok, len(schema_ids), args.seq, device)
    counts = {}
    for e in train:
        counts[f"{e['source']}/{e['dialect']}"] = counts.get(f"{e['source']}/{e['dialect']}", 0) + 1
    print("train examples:", json.dumps(counts, ensure_ascii=False), "| held-out:", len(held), flush=True)

    base_loss = heldout_loss(model, csam, dal, batcher, held, args.batch, use_adapters=False)
    print("held-out loss, backbone only:", base_loss, flush=True)
    diag_start = csam_diagnostics(model, csam, batcher, held)
    print("CSAM diagnostics at start:", json.dumps(diag_start), flush=True)

    if args.smoke:
        args.eval_every = 10
    missing = [d for d in DIALECTS if not any(e["dialect"] == d and e["source"] == "iadd" for e in train)
               ] if not args.smoke and args.iadd_per_region > 0 else []
    if [d for d in missing if d != "msa"]:
        raise SystemExit(f"no IADD training text for {missing}; check the IADD filters")

    def mean_loss(res):
        vals = [v["loss"] for k, v in res.items() if "/" not in k]
        return sum(vals) / len(vals)

    import copy
    best = {"step": 0, "mean": float("inf"), "state": None}
    by_source: dict[str, list[dict]] = {}
    for e in train:
        by_source.setdefault(e["source"], []).append(e)
    if args.mix == "uniform":
        mix = None
    else:
        mix = {k: float(v) for k, v in (kv.split("=") for kv in args.mix.split(","))}
        mix = {k: w for k, w in mix.items() if k in by_source}
    print("sampling mix:", mix or "uniform", flush=True)

    def draw(n):
        if mix is None:
            return rng.sample(train, n)
        sources = rng.choices(list(mix), weights=list(mix.values()), k=n)
        return [rng.choice(by_source[s]) for s in sources]

    rng = random.Random(args.seed)
    pair_rng = random.Random(args.seed + 1)   # separate stream: ablation arms see the same batches
    log, start = [], time.time()
    for step in range(1, args.steps + 1):
        batch = draw(args.batch)
        ids, labels, mask, dialect, schema, has_schema = batcher(batch)
        out, context, weights = adapted_forward(model, csam, dal, ids, labels, mask, dialect, schema, has_schema)
        loss = out["loss"]
        c_loss = torch.tensor(0.0, device=device)
        if args.lambda_align and USE["csam"]:
            pos, neg = motif_pairs(batcher.token_spans(batch), mask.sum(1).long().tolist(),
                                   has_schema.tolist(), pair_rng)
            if pos and neg:
                c_loss = cultural_loss(model, csam, ids, weights, pos, neg)
                loss = loss + args.lambda_align * c_loss
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
                   "cultural_loss": round(c_loss.item(), 4),
                   "lr": sched.get_last_lr()[0], "seconds": round(time.time() - start)}
            log.append(rec)
            print(rec, flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            csam.eval(); dal.eval()
            res = heldout_loss(model, csam, dal, batcher, held, args.batch, use_adapters=True)
            csam.train(); dal.train()
            m = mean_loss(res)
            log.append({"step": step, "heldout_mean": round(m, 4)})
            print({"step": step, "heldout_mean": round(m, 4)}, flush=True)
            if m < best["mean"]:
                best = {"step": step, "mean": m,
                        "state": (copy.deepcopy(csam.state_dict()), copy.deepcopy(dal.state_dict()))}

    if best["state"] is not None:
        csam.load_state_dict(best["state"][0])
        dal.load_state_dict(best["state"][1])
        print(f"restored best adapters from step {best['step']} (held-out mean {best['mean']:.4f})", flush=True)
    csam.eval()
    dal.eval()
    adapted_loss = heldout_loss(model, csam, dal, batcher, held, args.batch, use_adapters=True)
    print("held-out loss, with CSAM+DAL:", adapted_loss, flush=True)
    diag_end = csam_diagnostics(model, csam, batcher, held)
    print("CSAM diagnostics at end:", json.dumps(diag_end), flush=True)

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
                        max_new=60 if args.smoke else 220, temperature=0.8, top_p=0.9, device=device,
                        min_new=0 if args.smoke else args.min_new_tokens)
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
        "objective": ("L_LM + lambda_align * L_cultural + lambda_schema * L_schema (Eq. 4.16); "
                      "L_cultural on rule-based motif spans" if args.lambda_align and USE["csam"]
                      else "L_LM + lambda_schema * L_schema; L_cultural disabled"),
        "components": dict(USE),
        "decoding": {"temperature": 0.8, "top_p": 0.9, "repetition_penalty": 1.3, "no_repeat_ngram": 3,
                     "min_new_tokens": args.min_new_tokens},
        "sampling_mix": mix or "uniform",
        "train_counts": counts, "heldout_size": len(held),
        "heldout_loss_backbone_only": base_loss, "heldout_loss_with_adapters": adapted_loss,
        "best_step": best["step"],
        "csam_diagnostics": {"start": diag_start, "end": diag_end},
        "generation_banned_tokens": len(banned_token_ids(tok)),
        "odd_chars_in_samples": sum(ch in _ODD for smp in samples for ch in smp["text"]),
        "log": log, "samples": samples,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", sorted(p.name for p in out_dir.iterdir()), flush=True)


if __name__ == "__main__":
    main()
