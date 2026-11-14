"""Torch-free inspection of model artefacts: hashes, provenance sidecars, manifests.

Two sidecar formats make loaded weights traceable (``docs/GENERATION_READINESS.md``):

* ``ansg_backbone.json`` inside a pretrained-backbone directory: who published the
  checkpoint, under which licence, and the expected directory digest.
* ``<checkpoint>.manifest.json`` next to a cultural-module checkpoint: the backbone,
  tokenizer and schema vocabulary it was trained against and its training record.

No network access, no model download, no torch import.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..utils.provenance import sha256_file

BACKBONE_PROVENANCE_FILE = "ansg_backbone.json"
CULTURAL_FORMAT = "ansg-cultural-modules"
CULTURAL_FORMAT_VERSION = 2
"""2 = bridge v2 of ``ansg.pretrained`` (identity at initialisation, ``cultural_gate``, no extra
LayerNorm). Version-1 checkpoints were produced by the old bridge and are refused."""
CULTURAL_MANIFEST_SUFFIX = ".manifest.json"

WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin", "*.safetensors.index.json", "pytorch_model*.bin.index.json")
TOKENIZER_PATTERNS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "vocab.txt",
                      "special_tokens_map.json", "added_tokens.json", "*.model")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def file_inventory(directory: str | Path) -> dict[str, str]:
    """``{relative posix path: sha256}`` for every regular file, excluding hidden
    entries and the provenance sidecar itself (which records the digest)."""
    root = Path(directory)
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts) or not path.is_file():
            continue
        if rel.as_posix() == BACKBONE_PROVENANCE_FILE:
            continue
        inventory[rel.as_posix()] = sha256_file(path)
    return inventory


def digest_of(inventory: dict[str, str]) -> str:
    """Order-independent digest of a file inventory (``path<TAB>sha256`` lines)."""
    lines = "".join(f"{k}\t{v}\n" for k, v in sorted(inventory.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_backbone_provenance(directory: str | Path) -> dict | None:
    """Validated ``ansg_backbone.json`` or ``None`` when absent. Raises on malformed content."""
    path = Path(directory) / BACKBONE_PROVENANCE_FILE
    if not path.is_file():
        return None
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: must be a JSON object")
    for key in ("name", "source", "license"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"{path}: field {key!r} must be a non-empty string")
    if not isinstance(data.get("synthetic"), bool):
        raise ValueError(f"{path}: field 'synthetic' must be true or false")
    if "directory_digest" in data and not is_sha256(data["directory_digest"]):
        raise ValueError(f"{path}: 'directory_digest' must be a lowercase 64-hex SHA-256")
    return data


def tokenizer_json_size(directory: str | Path) -> int | None:
    """Vocabulary size from ``tokenizer.json`` (base vocab + added tokens), or None."""
    path = Path(directory) / "tokenizer.json"
    if not path.is_file():
        return None
    data = read_json(path)
    vocab = (data.get("model") or {}).get("vocab")
    if isinstance(vocab, dict):
        ids = set(vocab.values())
    elif isinstance(vocab, list):  # Unigram: list of [piece, score]
        ids = set(range(len(vocab)))
    else:
        return None
    ids |= {t["id"] for t in data.get("added_tokens") or [] if isinstance(t, dict) and "id" in t}
    return max(ids) + 1 if ids else 0


def inspect_backbone(directory: str | Path) -> dict:
    """Describe a local Hugging Face-format causal-LM directory without loading it."""
    root = Path(directory)
    info: dict[str, Any] = {"path": str(root), "exists": root.is_dir(), "problems": []}
    if not root.is_dir():
        info["problems"].append("backbone directory does not exist")
        return info
    inventory = file_inventory(root)
    info["files"] = inventory
    info["directory_digest"] = digest_of(inventory)
    weights = sorted(k for k in inventory if _matches(Path(k).name, WEIGHT_PATTERNS))
    tok = {k: v for k, v in inventory.items() if _matches(Path(k).name, TOKENIZER_PATTERNS)}
    info["weight_files"] = weights
    info["tokenizer_files"] = sorted(tok)
    info["tokenizer_digest"] = digest_of(tok) if tok else None
    if "config.json" not in inventory:
        info["problems"].append("config.json missing")
        config = {}
    else:
        config = read_json(root / "config.json")
    if not weights:
        info["problems"].append("no weight files (*.safetensors / pytorch_model*.bin)")
    if not tok:
        info["problems"].append("no tokenizer files")
    archs = config.get("architectures") or []
    info["config"] = {
        "model_type": config.get("model_type"),
        "architectures": archs,
        "vocab_size": config.get("vocab_size"),
        "hidden_size": config.get("n_embd", config.get("hidden_size")),
        "max_positions": config.get("n_positions", config.get("max_position_embeddings")),
    }
    info["causal_lm"] = any(a.endswith(("LMHeadModel", "ForCausalLM")) for a in archs)
    info["tokenizer_size"] = tokenizer_json_size(root)
    try:
        info["provenance"] = read_backbone_provenance(root)
    except ValueError as exc:
        info["provenance"] = None
        info["problems"].append(str(exc))
    return info


def expected_backbone_digest(info: dict, cli_digest: str | None) -> tuple[str | None, str]:
    """Expected digest from the command line (preferred) or the provenance sidecar."""
    if cli_digest:
        if not is_sha256(cli_digest):
            raise ValueError("--backbone-sha256 must be a lowercase 64-hex SHA-256 directory digest")
        return cli_digest, "command line"
    declared = (info.get("provenance") or {}).get("directory_digest")
    return (declared, BACKBONE_PROVENANCE_FILE) if declared else (None, "none supplied")


def cultural_manifest_path(checkpoint: str | Path) -> Path:
    return Path(str(checkpoint) + CULTURAL_MANIFEST_SUFFIX)


def validate_cultural_manifest(data: Any, where: str = "manifest") -> dict:
    """Structural validation of a cultural-module manifest. Raises ``ValueError``."""
    def need(cond: bool, msg: str) -> None:
        if not cond:
            raise ValueError(f"{where}: {msg}")
    need(isinstance(data, dict), "must be a JSON object")
    need(data.get("format") == CULTURAL_FORMAT, f"format must be {CULTURAL_FORMAT!r}")
    need(data.get("format_version") == CULTURAL_FORMAT_VERSION, f"format_version must be {CULTURAL_FORMAT_VERSION}")
    need(is_sha256(data.get("checkpoint_sha256")), "checkpoint_sha256 must be a SHA-256")
    arch = data.get("architecture")
    need(isinstance(arch, dict) and all(isinstance(arch.get(k), int) and arch[k] > 0 for k in
                                        ("schema_count", "schema_dim", "cultural_heads", "lora_rank")),
         "architecture needs positive integers schema_count, schema_dim, cultural_heads, lora_rank")
    need(isinstance(arch.get("lora_alpha"), (int, float)), "architecture.lora_alpha must be numeric")
    backbone, tokenizer, vocab = data.get("backbone"), data.get("tokenizer"), data.get("schema_vocabulary")
    need(isinstance(backbone, dict) and is_sha256(backbone.get("directory_digest")), "backbone.directory_digest missing")
    need(isinstance(tokenizer, dict) and is_sha256(tokenizer.get("digest")), "tokenizer.digest missing")
    need(isinstance(vocab, dict) and is_sha256(vocab.get("sha256")), "schema_vocabulary.sha256 missing")
    training = data.get("training")
    need(isinstance(training, dict), "training block missing")
    need(training.get("status") in ("untrained", "trained"), "training.status must be 'untrained' or 'trained'")
    steps = training.get("optimizer_steps")
    need(isinstance(steps, int) and not isinstance(steps, bool) and steps >= 0, "training.optimizer_steps must be an integer >= 0")
    need(isinstance(training.get("data_sha256"), dict) and all(is_sha256(v) for v in training["data_sha256"].values()),
         "training.data_sha256 must map names to SHA-256 digests")
    if training["status"] == "trained":
        need(steps > 0, "a trained checkpoint needs optimizer_steps > 0")
        need(bool(training["data_sha256"]), "a trained checkpoint must record its training-data hashes")
        need(training.get("data_kind") in ("synthetic", "real"), "training.data_kind must be 'synthetic' or 'real'")
    else:
        need(steps == 0, "an untrained checkpoint cannot record optimizer steps")
    return data


def read_cultural_manifest(checkpoint: str | Path) -> dict:
    path = cultural_manifest_path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"cultural-module manifest not found: {path} (a checkpoint without a manifest "
                                "has no provenance and is refused)")
    return validate_cultural_manifest(read_json(path), str(path))


def cultural_compatibility_problems(manifest: dict, *, checkpoint_sha256: str, backbone_digest: str | None,
                                    tokenizer_digest: str | None, schema_vocabulary_sha256: str | None) -> list[str]:
    """Hash-level compatibility of a cultural checkpoint with the artefacts it is combined with."""
    problems = []
    if manifest["checkpoint_sha256"] != checkpoint_sha256:
        problems.append("cultural checkpoint SHA-256 differs from its manifest")
    if backbone_digest is not None and manifest["backbone"]["directory_digest"] != backbone_digest:
        problems.append("cultural modules were trained against a different backbone (directory digest differs)")
    if tokenizer_digest is not None and manifest["tokenizer"]["digest"] != tokenizer_digest:
        problems.append("cultural modules were trained with a different tokenizer (digest differs)")
    if schema_vocabulary_sha256 is not None and manifest["schema_vocabulary"]["sha256"] != schema_vocabulary_sha256:
        problems.append("schema vocabulary differs from the one the cultural modules were trained with")
    return problems
