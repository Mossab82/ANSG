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
