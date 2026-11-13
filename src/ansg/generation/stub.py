"""Create a tiny, randomly initialised SYNTHETIC causal-LM directory for offline tests.

The directory has the same layout as a real Hugging Face checkpoint (``config.json``,
``model.safetensors``, ``tokenizer.json`` ...) plus an ``ansg_backbone.json`` that
declares ``"synthetic": true``, so every generation made with it is labelled
``synthetic_stub`` and its text is meaningless. Nothing is downloaded: the
byte-level BPE tokenizer is trained locally on a few synthetic Arabic sentences
written for this purpose, and the GPT-2-style weights are random.

Usage:  python -m ansg.generation.stub --output runs/stub_backbone
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import BACKBONE_PROVENANCE_FILE, digest_of, file_inventory

SYNTHETIC_SENTENCES = (
    "كان يا ما كان في قديم الزمان تاجر يسافر بين المدن.",
    "قالت الجدة: اسمعوا الحكاية حتى آخرها.",
    "سار الفتى في الصحراء ليلة كاملة يبحث عن البئر.",
    "وفي الصباح عاد إلى بيته ومعه هدية صغيرة.",
    "هذه جمل تركيبية للاختبار فقط وليست من أي مدونة.",
)
EOS = "<|endoftext|>"


def make_stub_backbone(output: str | Path, *, seed: int = 0, hidden_size: int = 48, layers: int = 2,
                       heads: int = 4, context: int = 128) -> dict:
    """Write the stub directory (must not exist or be empty) and return its provenance."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=320, special_tokens=[EOS], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(SYNTHETIC_SENTENCES, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, bos_token=EOS, unk_token=EOS)
    fast.save_pretrained(out)
    torch.manual_seed(seed)
    config = GPT2Config(vocab_size=len(fast), n_positions=context, n_embd=hidden_size, n_layer=layers,
                        n_head=heads, bos_token_id=fast.eos_token_id, eos_token_id=fast.eos_token_id)
    GPT2LMHeadModel(config).save_pretrained(out, safe_serialization=True)
    provenance = {
        "name": "ansg-synthetic-stub",
        "source": "generated locally by ansg.generation.stub (no download)",
        "license": "not applicable: synthetic random test weights",
        "synthetic": True,
        "language": "ar",
        "seed": seed,
        "directory_digest": digest_of(file_inventory(out)),
    }
    (out / BACKBONE_PROVENANCE_FILE).write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Create a tiny SYNTHETIC backbone for offline tests")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prov = make_stub_backbone(args.output, seed=args.seed)
    print(json.dumps({"output": args.output, **prov}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
