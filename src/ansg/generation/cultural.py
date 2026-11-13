"""Train and save ANSG cultural modules (CSAM + DAL) on top of a frozen pretrained backbone.

Requires the ``model`` and ``pretrained`` extras. The objective is next-token
cross-entropy on example texts (Ch. 5 stage-1 supervised training); the
adversarial and PPO stages and the CRAFT reward are NOT implemented here.

The checkpoint holds only the cultural-module tensors; the backbone stays a
separately hashed directory. Every checkpoint is written together with a
``<checkpoint>.manifest.json`` (``ansg.generation.artifacts``) recording the
backbone digest, tokenizer digest, schema vocabulary hash, training-data hashes,
optimiser steps and the measured loss of each step. ``data_kind`` is declared by
the person who runs the training; the manifest records it, it cannot verify it.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

from .. import __version__
from ..utils.provenance import git_revision, sha256_file
from .artifacts import CULTURAL_FORMAT, CULTURAL_FORMAT_VERSION, cultural_manifest_path, validate_cultural_manifest
from .runtime import CulturalSetup, GenerationError, LoadedBackbone, cultural_state_dict


def load_example_rows(path: str | Path, schema_count: int) -> list[dict]:
    """Rows written by ``ansg-build-examples`` (text, schema_ids, dialect_id); unknowns stay unknown."""
    from ..data.training_dataset import _dialect_id, _schema_ids
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise GenerationError(f"{path}:{line_no}: missing text")
            ids = _schema_ids(row, line_no, None)
            if any(i >= schema_count for i in ids):
                raise GenerationError(f"{path}:{line_no}: schema id outside the vocabulary ({schema_count})")
            rows.append({"text": text, "schema_ids": ids, "dialect_id": _dialect_id(row, line_no)})
    if not rows:
        raise GenerationError(f"{path}: no examples")
    return rows


def _batch(rows: Sequence[dict], tokenizer, max_length: int, device) -> dict:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise GenerationError("tokenizer has neither a pad nor an eos token")
    enc = [tokenizer(r["text"], add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
           for r in rows]
    if any(len(e) < 2 for e in enc):
        raise GenerationError("every training text needs at least two tokens")
    t = max(map(len, enc))
    s = max(1, max(len(r["schema_ids"]) for r in rows))
    input_ids = torch.full((len(rows), t), pad, dtype=torch.long)
    labels = torch.full((len(rows), t), -100, dtype=torch.long)
    attention = torch.zeros((len(rows), t), dtype=torch.long)
    schema_ids = torch.zeros((len(rows), s), dtype=torch.long)
    for i, (e, r) in enumerate(zip(enc, rows)):
        input_ids[i, :len(e)] = torch.tensor(e)
        labels[i, :len(e)] = torch.tensor(e)
        attention[i, :len(e)] = 1
        schema_ids[i, :len(r["schema_ids"])] = torch.tensor(r["schema_ids"], dtype=torch.long)
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention, "schema_ids": schema_ids,
             "schema_mask": schema_ids.ne(0), "dialect_ids": torch.tensor([r["dialect_id"] for r in rows])}
    return {k: v.to(device) for k, v in batch.items()}


def train_cultural_modules(cultural: CulturalSetup, backbone: LoadedBackbone, rows: Sequence[dict], *,
                           steps: int, lr: float = 1e-4, batch_size: int = 4, max_length: int = 256,
                           seed: int = 42, max_grad_norm: float = 1.0) -> dict:
    """Optimise CSAM/DAL with the backbone frozen; returns the measured training record."""
    if steps < 1 or batch_size < 1 or max_length < 2 or lr <= 0:
        raise GenerationError("steps, batch_size >= 1, max_length >= 2 and lr > 0 are required")
    model = cultural.model
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimiser = torch.optim.AdamW(params, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    max_pos = backbone.info["config"].get("max_positions")
    if max_pos:
        max_length = min(max_length, int(max_pos))
    model.train()
    model.backbone.eval()  # frozen backbone: no dropout
    losses = []
    started = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(len(rows), (min(batch_size, len(rows)),), generator=generator).tolist()
        out = model(**_batch([rows[i] for i in idx], backbone.tokenizer, max_length, device))
        optimiser.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimiser.step()
        loss = float(out["loss"].detach())
        if loss != loss or loss in (float("inf"), float("-inf")):
            raise GenerationError("non-finite training loss")
        losses.append(round(loss, 6))
    model.eval()
    return {"objective": "next-token cross-entropy, backbone frozen (Ch. 5 stage 1 only)",
            "optimizer": "AdamW", "lr": lr, "batch_size": batch_size, "max_length": max_length, "seed": seed,
            "optimizer_steps": steps, "loss_per_step": losses, "seconds": round(time.perf_counter() - started, 3),
            "trainable_parameters": int(sum(p.numel() for p in params))}


def save_cultural_modules(cultural: CulturalSetup, backbone: LoadedBackbone, destination: str | Path, *,
                          training: dict[str, Any] | None, data_kind: str | None = None,
                          data_files: dict[str, str | Path] | None = None) -> dict:
    """Write ``destination`` (tensors) and its manifest; refuses to overwrite existing files.

    ``training=None`` saves freshly initialised (untrained) modules.
    """
    dest = Path(destination)
    manifest_path = cultural_manifest_path(dest)
    for p in (dest, manifest_path):
        if p.exists():
            raise GenerationError(f"refusing to overwrite {p}")
    if training is None:
        block = {"status": "untrained", "optimizer_steps": 0, "data_sha256": {}, "data_kind": None}
    else:
        if data_kind not in ("synthetic", "real"):
            raise GenerationError("data_kind must be declared as 'synthetic' or 'real'")
        if not data_files:
            raise GenerationError("trained modules must record the training-data files")
        block = {"status": "trained", "data_kind": data_kind,
                 "data_sha256": {name: sha256_file(path) for name, path in sorted(data_files.items())},
                 **training}
    dest.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu").contiguous() for k, v in cultural_state_dict(cultural.model).items()}
    torch.save(state, dest)
    import transformers
    manifest = {
        "format": CULTURAL_FORMAT,
        "format_version": CULTURAL_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256_file(dest),
        "architecture": cultural.architecture,
        "backbone": {"directory_digest": backbone.info["directory_digest"], "path_at_training": backbone.info["path"],
                     "declared_provenance": backbone.info.get("provenance"), "synthetic": backbone.synthetic},
        "tokenizer": {"digest": backbone.info["tokenizer_digest"], "size": len(backbone.tokenizer)},
        "schema_vocabulary": {"sha256": cultural.schema_vocabulary_sha256,
                              "source_sha256": cultural.schema_vocabulary.source_sha256,
                              "schema_count": cultural.schema_vocabulary.schema_count},
        "init_seed": cultural.init_seed,
        "training": block,
        "git": git_revision(Path(__file__).resolve().parents[3]),
        "software": {"ansg": __version__, "python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__},
        "notice": "Cultural-module weights produced by this run. Not an original dissertation checkpoint.",
    }
    validate_cultural_manifest(manifest, str(manifest_path))
    with manifest_path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest
