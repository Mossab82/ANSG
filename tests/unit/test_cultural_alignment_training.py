"""L_cultural wiring in scripts/train_phase1.py (Eq. 4.18 on rule-based motif spans)."""
import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import train_phase1 as tp  # noqa: E402
from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias  # noqa: E402
from ansg.csam_bias import CSAMSettings, CulturalSchemaBias  # noqa: E402


def test_motif_pairs_respect_spans_and_schema():
    rng = random.Random(0)
    spans = [[(2, 6)], [(1, 4)], []]
    pos, neg = tp.motif_pairs(spans, [10, 10, 10], [True, False, True], rng, per_example=5)
    assert pos and all(b == 0 for b, _, _ in pos)              # example 1 has no schema, 2 no spans
    assert all(2 <= k < j < 6 for _, j, k in pos)
    assert all(not (2 <= k < j < 6) and k < j for _, j, k in neg)
    assert len(neg) == len(pos)


def test_cultural_loss_trains_eq_4_13_parameters():
    torch.manual_seed(0)
    spec = GPT2Spec(vocab_size=50, n_positions=32, n_embd=24, n_layer=2, n_head=12)
    model = GPT2WithAttentionBias(spec)
    csam = CulturalSchemaBias(CSAMSettings(n_head=12, d_model=24, graph_dim=8, layers=(0, 1)),
                              torch.randn(3, 8))
    ids = torch.randint(0, 50, (2, 12))
    weights = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
    pos, neg = tp.motif_pairs([[(2, 7)], [(0, 5)]], [12, 12], [True, True], random.Random(1))
    loss = tp.cultural_loss(model, csam, ids, weights, pos, neg)
    loss.backward()
    assert torch.isfinite(loss)
    assert csam.v_pos_k.grad is not None and csam.v_pos_k.grad.abs().sum() > 0


class _ByteTok:
    """Byte-level tokenizer: every byte is a token, so two tokens can form U+0605."""
    eos_id = None
    pad_id = 0

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="ignore")


class _FakeLM:
    """After byte 0xD8 it prefers 0x85 (-> U+0605) over 0xA7 (-> ا); otherwise it emits 0xD8."""
    class spec:
        n_positions = 512

    def __call__(self, ctx, **_):
        logits = torch.full((1, ctx.shape[1], 256), -20.0)
        if ctx[0, -1].item() == 0xD8:
            logits[0, -1, 0x85], logits[0, -1, 0xA7] = 5.0, 3.0
        else:
            logits[0, -1, 0xD8] = 5.0
        return {"logits": logits}


@pytest.mark.parametrize("with_filter", [True, False])
def test_generation_resamples_tokens_that_form_arabic_sign_characters(monkeypatch, with_filter):
    monkeypatch.setitem(tp.USE, "csam", False)
    monkeypatch.setitem(tp.USE, "dal", False)
    odd = set(tp._ODD)
    if not with_filter:
        monkeypatch.setattr(tp, "_ODD", set())
    torch.manual_seed(0)
    text = tp.generate(_FakeLM(), None, None, _ByteTok(), "كان", "msa", torch.zeros(2), max_new=40,
                       temperature=1.0, top_p=0.999, device="cpu", repetition_penalty=1.0,
                       no_repeat_ngram=0, min_new=0)
    n_odd = sum(ch in odd for ch in text)
    assert (n_odd == 0) if with_filter else (n_odd > 0)
