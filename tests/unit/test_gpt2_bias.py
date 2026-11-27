import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=97, n_positions=32, n_embd=48, n_layer=3, n_head=12)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    ours = GPT2WithAttentionBias.from_hf_state_dict(GPT2Spec.from_hf_config(cfg.to_dict()), hf.state_dict()).eval()
    return hf, ours


def test_zero_bias_matches_transformers(pair):
    hf, ours = pair
    ids = torch.randint(0, 97, (2, 17))
    with torch.no_grad():
        ref = hf(ids).logits
        out = ours(ids)["logits"]
        out_zero = ours(ids, bias_fn=lambda i, t: torch.zeros(1, 12, t, t))["logits"]
    assert torch.allclose(ref, out, atol=1e-5)
    assert torch.allclose(ref, out_zero, atol=1e-5)


def test_loss_matches_transformers(pair):
    hf, ours = pair
    ids = torch.randint(0, 97, (2, 11))
    with torch.no_grad():
        assert math.isclose(hf(ids, labels=ids).loss.item(), ours(ids, labels=ids)["loss"].item(), rel_tol=1e-5)


def test_positive_bias_scales_pre_normalisation_score_by_exp(pair):
    """Eq. 4.14: adding b to one logit multiplies its unnormalised weight by e^b."""
    _, ours = pair
    ids = torch.randint(0, 97, (1, 8))
    b = 0.412
    bias = torch.zeros(1, 12, 8, 8)
    bias[0, 3, 7, 2] = b  # head 3, query position 7 attends to key 2
    with torch.no_grad():
        base = ours(ids, return_probs=True)["attn_probs"][0][0, 3, 7]
        biased = ours(ids, bias_fn=lambda i, t: bias if i == 0 else None, return_probs=True)["attn_probs"][0][0, 3, 7]
    odds_ratio = (biased[2] / biased[0]) / (base[2] / base[0])
    assert torch.isclose(odds_ratio, torch.tensor(math.exp(b)), rtol=1e-4)
    assert biased[2] > base[2]


def test_bias_never_unmasks_future_positions(pair):
    _, ours = pair
    ids = torch.randint(0, 97, (1, 6))
    bias = torch.full((1, 12, 6, 6), 50.0)
    with torch.no_grad():
        probs = ours(ids, bias_fn=lambda i, t: bias, return_probs=True)["attn_probs"][0]
    assert torch.all(probs[0, :, 0, 1:] == 0)
