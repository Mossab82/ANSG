"""GPT-2 decoder that accepts an additive per-head attention bias (dissertation Eq. 4.9).

The dissertation's CSAM adds a cultural bias to the pre-softmax attention
logits of the backbone:

    Head_i = softmax( Q_i K_iᵀ / √d_k + B_i ) V_i,     B_i = βH_i + γR_i     (Eq. 4.9)

Hugging Face's GPT-2 does not expose per-head pre-softmax logits in a stable
way across versions, so this module re-implements the GPT-2 forward pass
(identical arithmetic, including the ``gelu_new`` activation and tied output
embedding) and loads the weights of any Hugging Face GPT-2 checkpoint, such as
``aubmindlab/aragpt2-base`` (12 layers, 768 hidden, 12 heads). With zero bias
the logits equal those of ``transformers.GPT2LMHeadModel``; the unit test
checks this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch
from torch import nn
from torch.nn import functional as F

# bias_fn(layer_index, seq_len) -> tensor broadcastable to [batch, heads, seq, seq], or None
BiasFn = Callable[[int, int], "torch.Tensor | None"]
# delta_fn(layer_index, x) -> extra output of the attention output projection (same shape as its output), or None
DeltaFn = Callable[[int, "torch.Tensor"], "torch.Tensor | None"]


@dataclass(frozen=True)
class GPT2Spec:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float = 1e-5
    activation: str = "gelu_new"

    @classmethod
    def from_hf_config(cls, config: Mapping) -> "GPT2Spec":
        get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
        act = get("activation_function", "gelu_new")
        if act not in ACTIVATIONS:
            raise ValueError(f"unsupported activation {act!r}; expected one of {sorted(ACTIVATIONS)}")
        return cls(
            vocab_size=get("vocab_size"),
            n_positions=get("n_positions"),
            n_embd=get("n_embd"),
            n_layer=get("n_layer"),
            n_head=get("n_head"),
            layer_norm_epsilon=get("layer_norm_epsilon", 1e-5),
            activation=act,
        )


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


ACTIVATIONS = {"gelu_new": gelu_new, "gelu": F.gelu, "relu": F.relu}


class BiasedAttention(nn.Module):
    def __init__(self, spec: GPT2Spec):
        super().__init__()
        self.n_head = spec.n_head
        self.head_dim = spec.n_embd // spec.n_head
        self.c_attn = nn.Linear(spec.n_embd, 3 * spec.n_embd)
        self.c_proj = nn.Linear(spec.n_embd, spec.n_embd)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None, return_probs: bool = False,
                delta: Callable[[torch.Tensor], torch.Tensor | None] | None = None):
        b, t, d = x.shape
        q, k, v = self.c_attn(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for z in (q, k, v))
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        logits = logits.masked_fill(~causal, torch.finfo(logits.dtype).min)
        probs = torch.softmax(logits, dim=-1)
        z = (probs @ v).transpose(1, 2).contiguous().view(b, t, d)
        out = self.c_proj(z)
        if delta is not None:
            extra = delta(z)
            if extra is not None:
                out = out + extra
        return (out, probs) if return_probs else (out, None)


class Block(nn.Module):
    def __init__(self, spec: GPT2Spec):
        super().__init__()
        self.ln_1 = nn.LayerNorm(spec.n_embd, eps=spec.layer_norm_epsilon)
        self.attn = BiasedAttention(spec)
        self.ln_2 = nn.LayerNorm(spec.n_embd, eps=spec.layer_norm_epsilon)
        self.c_fc = nn.Linear(spec.n_embd, 4 * spec.n_embd)
        self.c_proj = nn.Linear(4 * spec.n_embd, spec.n_embd)
        self.act = ACTIVATIONS[spec.activation]

    def forward(self, x, bias=None, return_probs=False, delta=None):
        a, probs = self.attn(self.ln_1(x), bias, return_probs, delta)
        x = x + a
        x = x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))
        return x, probs


class GPT2WithAttentionBias(nn.Module):
    """GPT-2 language model whose attention logits accept an additive bias per layer and head."""

    def __init__(self, spec: GPT2Spec):
        super().__init__()
        self.spec = spec
        self.wte = nn.Embedding(spec.vocab_size, spec.n_embd)
        self.wpe = nn.Embedding(spec.n_positions, spec.n_embd)
        self.h = nn.ModuleList(Block(spec) for _ in range(spec.n_layer))
        self.ln_f = nn.LayerNorm(spec.n_embd, eps=spec.layer_norm_epsilon)

    def hidden_states(self, input_ids: torch.Tensor, bias_fn: BiasFn | None = None, return_probs: bool = False,
                      delta_fn: DeltaFn | None = None):
        t = input_ids.shape[1]
        if t > self.spec.n_positions:
            raise ValueError(f"sequence length {t} exceeds n_positions {self.spec.n_positions}")
        pos = torch.arange(t, device=input_ids.device)
        x = self.wte(input_ids) + self.wpe(pos)[None]
        all_probs = []
        for i, block in enumerate(self.h):
            bias = bias_fn(i, t) if bias_fn is not None else None
            delta = (lambda z, i=i: delta_fn(i, z)) if delta_fn is not None else None
            x, probs = block(x, bias, return_probs, delta)
            all_probs.append(probs)
        return self.ln_f(x), (all_probs if return_probs else None)

    def forward(self, input_ids: torch.Tensor, bias_fn: BiasFn | None = None, labels: torch.Tensor | None = None,
                return_probs: bool = False, delta_fn: DeltaFn | None = None):
        hidden, probs = self.hidden_states(input_ids, bias_fn, return_probs, delta_fn)
        logits = hidden @ self.wte.weight.T  # tied output embedding, as in GPT-2
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                   ignore_index=-100)
        return {"logits": logits, "loss": loss, "hidden": hidden, "attn_probs": probs}

    # --- weight loading -------------------------------------------------
    @classmethod
    def from_hf_state_dict(cls, spec: GPT2Spec, state_dict: Mapping[str, torch.Tensor]) -> "GPT2WithAttentionBias":
        model = cls(spec)
        model.load_hf_state_dict(state_dict)
        return model

    def load_hf_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Load weights saved by ``transformers`` GPT2Model / GPT2LMHeadModel.

        GPT-2 stores attention and MLP projections as ``Conv1D`` with weight
        shape [in, out]; ``nn.Linear`` expects [out, in], so they are transposed.
        """
        sd = {k.removeprefix("transformer."): v for k, v in state_dict.items()}
        own = {}
        own["wte.weight"] = sd["wte.weight"]
        own["wpe.weight"] = sd["wpe.weight"]
        own["ln_f.weight"], own["ln_f.bias"] = sd["ln_f.weight"], sd["ln_f.bias"]
        for i in range(self.spec.n_layer):
            p = f"h.{i}."
            for ln in ("ln_1", "ln_2"):
                own[p + ln + ".weight"] = sd[p + ln + ".weight"]
                own[p + ln + ".bias"] = sd[p + ln + ".bias"]
            for src, dst in (("attn.c_attn", "attn.c_attn"), ("attn.c_proj", "attn.c_proj"),
                             ("mlp.c_fc", "c_fc"), ("mlp.c_proj", "c_proj")):
                own[p + dst + ".weight"] = sd[p + src + ".weight"].T.contiguous()
                own[p + dst + ".bias"] = sd[p + src + ".bias"]
        self.load_state_dict(own, strict=True)
