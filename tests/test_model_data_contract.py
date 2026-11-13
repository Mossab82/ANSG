"""Model side of the data contract (Issue #11). Synthetic data only.

* NarrativeDataset keeps missing dialect/schema labels unknown (no MSA default,
  no real schema node invented).
* DAL: dialect id -1 = unknown -> soft gate for that row; known ids unchanged.
* Backward-compatible lazy exports of the relocated model-branch modules.
"""
import json
import warnings

import pytest

torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytestmark = pytest.mark.model

from ansg.config import DIALECTS, ModelConfig  # noqa: E402
from ansg.dal import DialectalAdaptation  # noqa: E402
from ansg.data.training_dataset import UNKNOWN_DIALECT_ID, NarrativeDataset, collate  # noqa: E402
from ansg.model import ANSG  # noqa: E402


def _write(tmp_path, rows):
    p = tmp_path / "rows.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return p


def test_dataset_keeps_missing_labels_unknown(tmp_path):
    ds = NarrativeDataset(_write(tmp_path, [
        {"text": "نص تجريبي أول"},                                   # nothing given
        {"text": "نص تجريبي ثان", "dialect_id": None, "schema_ids": []},
        {"text": "نص تجريبي ثالث", "dialect_id": 3, "schema_ids": [2, 5]},
    ]))
    assert [r["dialect_id"] for r in ds.rows] == [UNKNOWN_DIALECT_ID, UNKNOWN_DIALECT_ID, 3]
    assert [r["schema_ids"] for r in ds.rows] == [[], [], [2, 5]]
    batch = collate(ds.rows)
    assert batch["dialect_ids"].tolist() == [-1, -1, 3]
    assert batch["schema_mask"].tolist() == [[False, False], [False, False], [True, True]]


@pytest.mark.parametrize("bad", [5, -2, "msa", True, 1.0])
def test_dataset_rejects_invalid_dialect(tmp_path, bad):
    with pytest.raises(ValueError, match="dialect_id"):
        NarrativeDataset(_write(tmp_path, [{"text": "نص تجريبي", "dialect_id": bad}]))


@pytest.mark.parametrize("bad", ["1", [1, -3], [1.5], [True]])
def test_dataset_rejects_invalid_schema_ids(tmp_path, bad):
    with pytest.raises(ValueError, match="schema_ids"):
        NarrativeDataset(_write(tmp_path, [{"text": "نص تجريبي", "schema_ids": bad}]))


def _dal():
    torch.manual_seed(0)
    dal = DialectalAdaptation(hidden_size=16, schema_dim=8, rank=4)
    with torch.no_grad():
        dal.up.normal_(std=0.05)  # non-zero adapters so dialect choice changes the output
    return dal


def test_dal_unknown_rows_use_soft_gate_known_rows_unchanged():
    dal = _dal()
    x, s = torch.randn(2, 5, 16), torch.randn(2, 8)
    mixed = dal(x, s, torch.tensor([-1, 2]))
    soft = dal(x, s, None)
    explicit = dal(x, s, torch.tensor([0, 2]))
    torch.testing.assert_close(mixed[0], soft[0])        # unknown -> soft gate
    torch.testing.assert_close(mixed[1], explicit[1])    # known -> one-hot, as before
    assert not torch.allclose(mixed[0], explicit[0])     # unknown is NOT treated as MSA


def test_dal_gate_learns_from_unknown_rows_only():
    dal = _dal()
    x, s = torch.randn(2, 5, 16), torch.randn(2, 8)
    dal(x, s, torch.tensor([1, 2])).sum().backward()
    assert dal.gate[0].weight.grad is None or dal.gate[0].weight.grad.abs().sum() == 0
    dal.zero_grad()
    dal(x, s, torch.tensor([-1, 2])).sum().backward()
    assert dal.gate[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("bad", [[-2, 0], [0, len(DIALECTS)]])
def test_dal_rejects_out_of_range_ids(bad):
    with pytest.raises(ValueError, match="dialect_ids"):
        _dal()(torch.randn(2, 3, 16), torch.randn(2, 8), torch.tensor(bad))


def test_model_runs_with_unknown_dialect_and_empty_schema_rows(tmp_path):
    torch.manual_seed(0)
    model = ANSG(ModelConfig(context_length=32, schema_count=8))
    ds = NarrativeDataset(_write(tmp_path, [{"text": "كان يا ما كان"},
                                            {"text": "في قديم الزمان", "dialect_id": 1, "schema_ids": [3]}]),
                          context_length=32)
    out = model(**collate(ds.rows))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


def test_backward_compatible_lazy_exports():
    import ansg.data
    import ansg.evaluation
    import ansg.knowledge
    from ansg.data import training_dataset
    from ansg.evaluation import basic_metrics
    from ansg.knowledge import lexical_baseline
    assert ansg.data.NarrativeDataset is training_dataset.NarrativeDataset
    assert ansg.data.collate is training_dataset.collate
    assert ansg.evaluation.weighted_cmpi is basic_metrics.weighted_cmpi
    assert ansg.knowledge.Motif is lexical_baseline.Motif
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls = ansg.knowledge.CulturalKnowledgeGraph
    assert cls is lexical_baseline.CulturalKnowledgeGraph
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    with pytest.raises(AttributeError):
        ansg.data.NoSuchName  # noqa: B018
