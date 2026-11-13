"""Regression tests for schema padding and empty activation (synthetic only).

A model must not change an example's predictions solely because the
other batch member has more cultural schema IDs.
"""
import pytest
torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytestmark = pytest.mark.model

from ansg.config import ModelConfig
from ansg.data import collate
from ansg.model import ANSG
from ansg.csam import PairwiseCulturalBias, CulturalSchemaAttention


def _model():
    torch.manual_seed(17)
    model = ANSG(ModelConfig()).eval()
    return model


def test_collate_exposes_padding_mask_and_empty_schema():
    rows = [
        {"tokens": [3, 4, 5], "schema_ids": [2], "dialect_id": 0},
        {"tokens": [7, 8, 9], "schema_ids": [3, 4, 5], "dialect_id": 2},
        {"tokens": [10, 11, 12], "schema_ids": [], "dialect_id": 4},
    ]
    batch = collate(rows)
    assert batch["schema_ids"].tolist() == [[2, 0, 0], [3, 4, 5], [0, 0, 0]]
    assert batch["schema_mask"].tolist() == [
        [True, False, False], [True, True, True], [False, False, False]
    ]


def test_same_example_independent_of_batchmate_schema_count():
    model = _model()
    ids = torch.tensor([[20, 21, 22, 23]])
    single = model(ids, torch.tensor([[3]]), torch.tensor([1]))["logits"]
    paired = model(
        torch.cat([ids, ids], dim=0),
        torch.tensor([[3, 0, 0, 0], [5, 6, 7, 8]]),
        torch.tensor([1, 2]),
    )["logits"][0:1]
    torch.testing.assert_close(single, paired, rtol=1e-5, atol=1e-5)


def test_padded_columns_do_not_affect_pairwise_bias_or_csam():
    cfg = ModelConfig()
    pb = PairwiseCulturalBias(cfg).eval()
    csam = CulturalSchemaAttention(cfg).eval()
    ids = torch.tensor([[1, 2]])
    padded = torch.tensor([[1, 2, 0, 0]])
    x = torch.randn(1, 5, cfg.hidden_size)
    torch.testing.assert_close(pb(ids, 5), pb(padded, 5), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(csam(x, ids), csam(x, padded), rtol=1e-5, atol=1e-5)


def test_no_schema_is_finite_and_keeps_csam_input():
    cfg = ModelConfig()
    model = _model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    schemas = torch.tensor([[0, 0], [7, 0]])
    out = model(ids, schemas, torch.tensor([0, 1]), labels=ids)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["loss"])
    x = torch.randn(2, 4, cfg.hidden_size)
    output = model.csam(x, schemas)
    torch.testing.assert_close(output[0], x[0], rtol=0, atol=0)
    bias = model.pairwise_cultural_bias(schemas, 4)
    torch.testing.assert_close(bias[0], torch.zeros_like(bias[0]), rtol=0, atol=0)


def test_mask_shape_and_pad_id_errors():
    csam = CulturalSchemaAttention(ModelConfig())
    x = torch.randn(1, 3, 96)
    with pytest.raises(ValueError, match="shape mismatch"):
        csam(x, torch.tensor([[1, 0]]), torch.tensor([[True]]))
    with pytest.raises(ValueError, match="Padding schema ID 0"):
        csam(x, torch.tensor([[1, 0]]), torch.tensor([[True, True]]))


def test_schema_padding_does_not_break_backprop():
    model = _model()
    ids = torch.tensor([[10, 12, 14], [11, 13, 15]])
    schemas = torch.tensor([[2, 0], [3, 4]])
    out = model(ids, schemas, labels=ids)
    out["loss"].backward()
    assert model.csam.value_proj.weight.grad is not None
    assert torch.isfinite(model.csam.value_proj.weight.grad).all()
