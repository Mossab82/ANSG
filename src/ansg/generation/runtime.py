"""Arabic story generation with an explicitly supplied pretrained causal LM (Issue #15).

Requires the ``model`` and ``pretrained`` extras (PyTorch + transformers).

Guarantees
----------
* **No downloads.** The backbone must be a local directory; ``transformers`` is
  called with ``local_files_only=True`` and ``trust_remote_code=False``.
* **Hash-validated loading.** The backbone directory digest must match an expected
  digest (``--backbone-sha256`` or ``ansg_backbone.json``) unless the caller opts
  out explicitly, in which case the record says ``hash_verified: false``.
  Cultural-module checkpoints are always hash-checked against their manifest and
  against the backbone/tokenizer/schema vocabulary they were trained with.
* **Reproducible sampling.** Each sample draws from its own ``torch.Generator``
  seeded with ``seed + index`` on the CPU, so token choices do not depend on
  global RNG state. Bitwise repeatability holds for the same hardware and
  software stack; on GPU pass ``deterministic=True`` as well.
* **Evidence-derived labels** (``ansg.generation.labels``): backbone-only,
  untrained-ANSG and synthetic-stub generations are labelled as such in the
  record and on stderr; the caller cannot override the label.

Dissertation cross-reference: backbone choice C-MODEL-1 (Ch. 4 §4.1.6, Table 5.1,
App. A.3); CSAM Eq. 4.9 / App. A.25 and DAL Eq. 4.24 are applied post-backbone
through ``ansg.pretrained.PretrainedANSG`` (not the in-layer injection of Eq. 4.9,
see that module's docstring); Ch. 6 metrics are NOT computed here.
"""

from __future__ import annotations

import os
import platform
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

from .. import __version__
from ..utils.provenance import git_revision, sha256_file, sha256_text
from . import labels as L
from .artifacts import (cultural_compatibility_problems, expected_backbone_digest, inspect_backbone,
                        read_cultural_manifest)

from .labels import RECORD_SCHEMA  # noqa: E402  (re-exported)
DEFAULT_CULTURAL_ARCHITECTURE = {"schema_dim": 256, "cultural_heads": 12, "lora_rank": 8, "lora_alpha": 16.0}
"""Defaults of ``PretrainedANSG`` (App. A.3 values where specified; see docs/PARAMETER_AUDIT.md)."""


class GenerationError(RuntimeError):
    """Invalid request, missing artefact or failed verification (CLI exit status 2)."""


# --------------------------------------------------------------------------- sampling
@dataclass(frozen=True)
class SamplingConfig:
    """Defaults follow the dissertation's generation settings, Ch. 6 §6.1.2: temperature 0.7,
    top-p 0.95, at most 512 tokens, seed 42 (Table A.6 lists top-p 0.9 for GPT-4: C-EVAL-7)."""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0
    seed: int = 42
    num_samples: int = 1
    stop_at_eos: bool = True

    def validate(self) -> "SamplingConfig":
        if not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise GenerationError("max_new_tokens must be a positive integer")
        if not (self.temperature >= 0.0) or self.temperature > 10:
            raise GenerationError("temperature must be in [0, 10] (0 = greedy decoding)")
        if not (0.0 < self.top_p <= 1.0):
            raise GenerationError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise GenerationError("top_k must be >= 1")
        if not (self.repetition_penalty >= 1.0):
            raise GenerationError("repetition_penalty must be >= 1.0")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise GenerationError("seed must be a non-negative integer")
        if not isinstance(self.num_samples, int) or not 1 <= self.num_samples <= 64:
            raise GenerationError("num_samples must be between 1 and 64")
        return self

    @property
    def decoding(self) -> str:
        return "greedy" if self.temperature == 0 else "sampling"


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig, previous: list[int]) -> torch.Tensor:
    """Repetition penalty, temperature, top-k and nucleus (top-p) filtering of a [V] logit vector."""
    logits = logits.float().clone()
    if cfg.repetition_penalty != 1.0 and previous:
        idx = torch.tensor(sorted(set(previous)), dtype=torch.long, device=logits.device)
        chosen = logits[idx]
        logits[idx] = torch.where(chosen > 0, chosen / cfg.repetition_penalty, chosen * cfg.repetition_penalty)
    if cfg.temperature == 0:
        return logits
    logits = logits / cfg.temperature
    if cfg.top_k is not None and cfg.top_k < logits.numel():
        kth = torch.topk(logits, cfg.top_k).values[-1]
        logits = logits.masked_fill(logits < kth, -torch.inf)
    if cfg.top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # Drop a token once the mass *before* it already exceeds top_p (the top token always stays).
        drop = (torch.cumsum(probs, dim=-1) - probs) > cfg.top_p
        logits = logits.masked_fill(torch.zeros_like(drop).scatter(0, order, drop), -torch.inf)
    return logits


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator, previous: list[int]) -> int:
    filtered = filter_logits(logits, cfg, previous)
    if cfg.temperature == 0:
        return int(torch.argmax(filtered).item())
    probs = torch.softmax(filtered, dim=-1).cpu()
    return int(torch.multinomial(probs, 1, generator=generator).item())


StepFn = Callable[[torch.Tensor, Any], "tuple[torch.Tensor, Any]"]


@torch.no_grad()
def generate_ids(step: StepFn, prompt_ids: list[int], cfg: SamplingConfig, *, seed: int, device: torch.device,
                 eos_token_id: int | None) -> tuple[list[int], str]:
    """Incremental decoding with a KV cache. ``step(input_ids[1,T], past) -> (logits[V], past)``."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = list(prompt_ids)
    new: list[int] = []
    inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past = None
    for _ in range(cfg.max_new_tokens):
        logits, past = step(inp, past)
        token = sample_next(logits, cfg, generator, history)
        new.append(token)
        history.append(token)
        if cfg.stop_at_eos and eos_token_id is not None and token == eos_token_id:
            return new, "eos"
        inp = torch.tensor([[token]], dtype=torch.long, device=device)
    return new, "length"


# --------------------------------------------------------------------------- environment
def resolve_device(spec: str = "auto") -> torch.device:
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        device = torch.device(spec)
    except RuntimeError as exc:
        raise GenerationError(f"unsupported device {spec!r}") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise GenerationError(f"device {spec!r} requested but CUDA is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise GenerationError(f"device {spec!r} requested but only {torch.cuda.device_count()} GPU(s) exist")
    elif device.type == "mps" and not torch.backends.mps.is_available():
        raise GenerationError("device 'mps' requested but MPS is not available")
    elif device.type not in ("cpu", "cuda", "mps"):
        raise GenerationError(f"unsupported device {spec!r}")
    return device


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name not in _DTYPES:
        raise GenerationError(f"dtype must be one of {sorted(_DTYPES)}")
    if name == "float16" and device.type == "cpu":
        raise GenerationError("float16 is not supported for CPU generation; use float32 or bfloat16")
    return _DTYPES[name]


def prepare_prompt(text: str) -> tuple[str, dict]:
    """NFC-normalise an Arabic prompt and describe it (no other rewriting)."""
    if not isinstance(text, str):
        raise GenerationError("prompt must be text")
    prompt = unicodedata.normalize("NFC", text).strip()
    if not prompt:
        raise GenerationError("prompt is empty")
    letters = [c for c in prompt if c.isalpha()]
    arabic = [c for c in letters if "ARABIC" in unicodedata.name(c, "")]
    info = {"text": prompt, "sha256": sha256_text(prompt), "characters": len(prompt),
            "arabic_letter_ratio": round(len(arabic) / len(letters), 4) if letters else 0.0,
            "normalisation": "Unicode NFC + strip of surrounding whitespace"}
    return prompt, info


# --------------------------------------------------------------------------- loading
@dataclass
class LoadedBackbone:
    model: Any
    tokenizer: Any
    info: dict
    hash_verified: bool
    expected_digest_source: str
    synthetic: bool
    load_seconds: float

    def describe(self) -> dict:
        cfg = self.info["config"]
        provenance = self.info.get("provenance")
        return {
            "path": self.info["path"],
            "directory_digest": self.info["directory_digest"],
            "hash_verified": self.hash_verified,
            "expected_digest_source": self.expected_digest_source,
            "files": self.info["files"],
            "declared_provenance": provenance,
            "provenance_declared": provenance is not None,
            "synthetic": self.synthetic,
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "hidden_size": cfg.get("hidden_size"),
            "max_positions": cfg.get("max_positions"),
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
        }


def load_backbone(path: str | Path, *, expected_digest: str | None = None, allow_unverified: bool = False,
                  device: torch.device, dtype: torch.dtype) -> LoadedBackbone:
    """Load a local causal LM + tokenizer after verifying its directory digest."""
    started = time.perf_counter()
    root = Path(path)
    if not root.is_dir():
        raise GenerationError(f"backbone must be an existing local directory: {path!s} "
                              "(automatic downloads are disabled; fetch and verify the checkpoint yourself)")
    info = inspect_backbone(root)
    if info["problems"]:
        raise GenerationError("backbone directory rejected: " + "; ".join(info["problems"]))
    if not info["causal_lm"]:
        raise GenerationError(f"config.json architectures {info['config']['architectures']} is not a causal LM")
    try:
        expected, source = expected_backbone_digest(info, expected_digest)
    except ValueError as exc:
        raise GenerationError(str(exc)) from exc
    if expected is None and not allow_unverified:
        raise GenerationError(
            "no expected backbone digest: pass --backbone-sha256 or add 'directory_digest' to "
            f"{root / 'ansg_backbone.json'} (computed digest: {info['directory_digest']}); "
            "or pass --allow-unverified-backbone to record the run as hash-unverified")
    if expected is not None and expected != info["directory_digest"]:
        raise GenerationError(f"backbone digest mismatch: expected {expected} ({source}), "
                              f"computed {info['directory_digest']}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise GenerationError("install the 'pretrained' extra: pip install -e '.[model,pretrained]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    model.to(device=device, dtype=dtype).eval()
    rows = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > rows:
        raise GenerationError(f"tokenizer has {len(tokenizer)} tokens but the embedding matrix only {rows} rows")
    info["tokenizer_alignment"] = {"tokenizer_size": len(tokenizer), "embedding_rows": rows, "aligned": True}
    provenance = info.get("provenance")
    return LoadedBackbone(model=model, tokenizer=tokenizer, info=info, hash_verified=expected is not None,
                          expected_digest_source=source, synthetic=bool(provenance and provenance["synthetic"]),
                          load_seconds=time.perf_counter() - started)


@dataclass
class CulturalSetup:
    """CSAM/DAL modules on top of the backbone, either loaded or freshly initialised."""
    model: Any
    manifest: dict | None
    checkpoint: dict | None
    schema_vocabulary: Any
    schema_vocabulary_sha256: str
    architecture: dict
    init_seed: int | None


def _load_vocabulary(path: str | Path):
    from ..data.training_examples import SchemaVocabulary
    from .artifacts import read_json
    p = Path(path)
    if not p.is_file():
        raise GenerationError(f"schema vocabulary not found: {p}")
    return SchemaVocabulary.from_dict(read_json(p)), sha256_file(p)


def cultural_state_dict(model) -> dict:
    """Parameters of everything except the (frozen, separately hashed) backbone."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}


def build_cultural(backbone: LoadedBackbone, *, schema_vocabulary: str | Path, checkpoint: str | Path | None,
                   init_seed: int = 0, architecture: dict | None = None) -> CulturalSetup:
    """Wrap the backbone in ``PretrainedANSG``; load trained weights or initialise them (untrained)."""
    from ..pretrained import PretrainedANSG
    vocab, vocab_sha = _load_vocabulary(schema_vocabulary)
    manifest = ckpt_info = None
    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise GenerationError(f"cultural checkpoint not found: {ckpt}")
        try:
            manifest = read_cultural_manifest(ckpt)
        except (FileNotFoundError, ValueError) as exc:
            raise GenerationError(str(exc)) from exc
        ckpt_sha = sha256_file(ckpt)
        problems = cultural_compatibility_problems(
            manifest, checkpoint_sha256=ckpt_sha, backbone_digest=backbone.info["directory_digest"],
            tokenizer_digest=backbone.info["tokenizer_digest"], schema_vocabulary_sha256=vocab_sha)
        if manifest["architecture"]["schema_count"] != vocab.schema_count:
            problems.append("schema_count of the checkpoint differs from the schema vocabulary")
        if problems:
            raise GenerationError("cultural checkpoint rejected: " + "; ".join(problems))
        arch = dict(manifest["architecture"])
        ckpt_info = {"path": str(ckpt), "sha256": ckpt_sha}
    else:
        arch = {**DEFAULT_CULTURAL_ARCHITECTURE, **(architecture or {}), "schema_count": vocab.schema_count}
    torch.manual_seed(init_seed)
    try:
        model = PretrainedANSG(backbone.model, **arch)
    except ValueError as exc:
        raise GenerationError(f"cultural architecture incompatible with backbone: {exc}") from exc
    param = next(backbone.model.parameters())
    model.to(device=param.device, dtype=param.dtype)
    if checkpoint is not None:
        try:
            state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - surfaced as a verification failure
            raise GenerationError(f"cannot read cultural checkpoint safely: {exc}") from exc
        expected = set(cultural_state_dict(model))
        if not isinstance(state, dict) or set(state) != expected:
            raise GenerationError("cultural checkpoint keys do not match the ANSG cultural modules "
                                  f"(missing {sorted(expected - set(state))[:5]}, "
                                  f"unexpected {sorted(set(state) - expected)[:5]})")
        model.load_state_dict(state, strict=False)
    model.eval()
    return CulturalSetup(model=model, manifest=manifest, checkpoint=ckpt_info, schema_vocabulary=vocab,
                         schema_vocabulary_sha256=vocab_sha, architecture=arch,
                         init_seed=None if checkpoint is not None else init_seed)


# --------------------------------------------------------------------------- request/record
@dataclass
class GenerationRequest:
    prompt: str
    backbone: str
    backbone_sha256: str | None = None
    allow_unverified_backbone: bool = False
    device: str = "auto"
    dtype: str = "float32"
    deterministic: bool = False
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    cultural_checkpoint: str | None = None
    untrained_cultural_modules: bool = False
    schema_vocabulary: str | None = None
    schema_nodes: tuple[str, ...] = ()
    dialect: str | None = None
    chat_template: bool = False          # wrap the prompt with the tokenizer's chat template (instruct models)
    system_prompt: str | None = None
    plan: dict | None = None             # provenance block of a story plan the prompt was built from


def _software() -> dict:
    import transformers
    return {"ansg": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__}


def run_generation(req: GenerationRequest) -> dict:
    """Load, verify, generate and return a JSON-serialisable generation record."""
    cfg = req.sampling.validate()
    if req.cultural_checkpoint and req.untrained_cultural_modules:
        raise GenerationError("choose either a cultural checkpoint or --untrained-cultural-modules, not both")
    use_cultural = bool(req.cultural_checkpoint or req.untrained_cultural_modules)
    if use_cultural and not req.schema_vocabulary:
        raise GenerationError("ANSG cultural modules need --schema-vocabulary")
    if not use_cultural and (req.schema_nodes or req.dialect):
        raise GenerationError("schema nodes / dialect need ANSG cultural modules; a backbone-only run "
                              "cannot apply cultural or dialectal conditioning")
    prompt, prompt_info = prepare_prompt(req.prompt)
    device = resolve_device(req.device)
    dtype = resolve_dtype(req.dtype, device)
    if req.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    backbone = load_backbone(req.backbone, expected_digest=req.backbone_sha256,
                             allow_unverified=req.allow_unverified_backbone, device=device, dtype=dtype)
    tok = backbone.tokenizer
    model_input = prompt
    if req.chat_template:
        if not getattr(tok, "chat_template", None):
            raise GenerationError("--chat-template requested but the tokenizer defines no chat template")
        messages = ([{"role": "system", "content": req.system_prompt}] if req.system_prompt else []) + \
            [{"role": "user", "content": prompt}]
        model_input = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(model_input, add_special_tokens=False)["input_ids"]
        prompt_info.update(chat_template=True, system_prompt=req.system_prompt,
                           model_input_sha256=sha256_text(model_input))
    elif req.system_prompt:
        raise GenerationError("--system-prompt requires --chat-template")
    else:
        prompt_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        prompt_info.update(chat_template=False)
    if not prompt_ids:
        raise GenerationError("prompt produced no tokens")
    roundtrip = tok.decode(prompt_ids, skip_special_tokens=not req.chat_template, clean_up_tokenization_spaces=False)
    prompt_info.update(n_tokens=len(prompt_ids), tokenizer_roundtrip_exact=(roundtrip == model_input))
    max_pos = backbone.info["config"].get("max_positions")
    if max_pos and len(prompt_ids) + cfg.max_new_tokens > max_pos:
        raise GenerationError(f"prompt ({len(prompt_ids)} tokens) + max_new_tokens ({cfg.max_new_tokens}) "
                              f"exceeds the backbone context of {max_pos} positions")

    cultural = None
    cultural_block = None
    dialect_id = None
    if use_cultural:
        cultural = build_cultural(backbone, schema_vocabulary=req.schema_vocabulary,
                                  checkpoint=req.cultural_checkpoint, init_seed=cfg.seed)
        try:
            schema_ids = cultural.schema_vocabulary.encode(list(req.schema_nodes))
        except KeyError as exc:
            raise GenerationError(f"schema node not in the schema vocabulary: {exc}") from exc
        dialect_code = None
        if req.dialect is not None:
            from ..data.training_examples import dialect_code as _dialect_code
            try:
                dialect_code, dialect_id = _dialect_code(req.dialect)
            except ValueError as exc:
                raise GenerationError(str(exc)) from exc
        ids = torch.tensor([schema_ids or [0]], dtype=torch.long, device=device)
        mask = ids.ne(0) if schema_ids else torch.zeros_like(ids, dtype=torch.bool)
        dialects = torch.tensor([dialect_id], dtype=torch.long, device=device) if dialect_id is not None else None

        def step(inp, past):
            out = cultural.model(input_ids=inp, schema_ids=ids, dialect_ids=dialects, schema_mask=mask,
                                 past_key_values=past, use_cache=True)
            return out["logits"][0, -1], out["past_key_values"]

        cultural_block = {
            "source": "checkpoint" if cultural.checkpoint else "freshly initialised (untrained)",
            "checkpoint": cultural.checkpoint,
            "manifest": cultural.manifest,
            "init_seed": cultural.init_seed,
            "architecture": cultural.architecture,
            "schema_vocabulary": {"path": str(req.schema_vocabulary), "sha256": cultural.schema_vocabulary_sha256,
                                  "source_sha256": cultural.schema_vocabulary.source_sha256,
                                  "schema_count": cultural.schema_vocabulary.schema_count},
            "schema_node_ids": list(req.schema_nodes),
            "schema_ids": schema_ids,
            "dialect": dialect_code,
            "dialect_id": dialect_id,
            "integration": "post-backbone CSAM + DAL (ansg.pretrained.PretrainedANSG), not in-layer Eq. 4.9",
        }
    else:
        def step(inp, past):
            out = backbone.model(input_ids=inp, past_key_values=past, use_cache=True)
            return out.logits[0, -1], out.past_key_values

    training = cultural.manifest["training"] if cultural and cultural.manifest else None
    mode = L.classify_mode(backbone_synthetic=backbone.synthetic, cultural_training=training,
                           cultural_loaded=cultural is not None)
    label = L.label_block(mode, dialect_requested=dialect_id is not None)
    if not backbone.hash_verified:
        label["warnings"].append("backbone loaded WITHOUT an expected digest (hash_verified = false)")
    if backbone.info.get("provenance") is None:
        label["warnings"].append("backbone has no ansg_backbone.json: licence and source are undeclared")
    if not prompt_info["tokenizer_roundtrip_exact"]:
        label["warnings"].append("tokenizer does not reproduce the prompt exactly (lossy encoding)")

    eos = tok.eos_token_id
    samples = []
    started = time.perf_counter()
    for i in range(cfg.num_samples):
        seed = cfg.seed + i
        new_ids, finish = generate_ids(step, prompt_ids, cfg, seed=seed, device=device, eos_token_id=eos)
        continuation = tok.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text = tok.decode(prompt_ids + new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        samples.append({"index": i, "seed": seed, "text": text, "continuation": continuation,
                        "new_tokens": len(new_ids), "finish_reason": finish,
                        "token_ids_sha256": sha256_text(",".join(map(str, new_ids)))})
    gen_seconds = time.perf_counter() - started
    L.validate_label(label, backbone_synthetic=backbone.synthetic, cultural_loaded=cultural is not None,
                     cultural_training=training)
    return {
        "schema": RECORD_SCHEMA,
        "label": label,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(Path(__file__).resolve().parents[3]),
        "software": _software(),
        "runtime": {"device": str(device), "dtype": req.dtype,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "repeatability": "bitwise for identical seed, hardware and software stack; "
                                     "GPU kernels may differ without deterministic mode"},
        "generation_config": {**asdict(cfg), "decoding": cfg.decoding},
        "backbone": backbone.describe(),
        "tokenizer": {"class": type(tok).__name__, "size": len(tok), "digest": backbone.info["tokenizer_digest"],
                      "files": backbone.info["tokenizer_files"], **backbone.info["tokenizer_alignment"],
                      "eos_token_id": eos},
        "cultural_modules": cultural_block,
        "prompt": prompt_info,
        "plan": req.plan,
        "samples": samples,
        "timing_seconds": {"load": round(backbone.load_seconds, 3), "generate": round(gen_seconds, 3)},
        "notice": "Generated text only. No CRAFT/CMPI metric is computed and nothing here reproduces "
                  "a result reported in the dissertation.",
    }
