import pytest

from ansg.evaluation.craft import (
    AggregationWeights, CraftDimension, CraftRegistry, EvaluationSample, MetricNotSpecifiedError,
    aggregate,
)
from ansg.utils.config import load_yaml_config


def test_shipped_config_declares_no_metrics(tmp_path):
    cfg = load_yaml_config("configs/evaluation/craft.yaml")
    assert cfg["metrics"] == []
    assert {CraftDimension(d) for d in cfg["dimensions"]} == set(CraftDimension)


def test_unimplemented_metric_refuses_to_score():
    reg = CraftRegistry.from_config({"metrics": [{"name": "m", "dimension": "dialectal_fidelity"}]})
    assert reg.status_report() == {"m": {"dimension": "dialectal_fidelity", "status": "unverified",
                                         "thesis_reference": None}}
    with pytest.raises(MetricNotSpecifiedError):
        reg.get("m")([EvaluationSample("s", "نص")])


def test_aggregation_requires_verified_weights():
    with pytest.raises(MetricNotSpecifiedError):
        aggregate({"a": 1.0}, None)
    with pytest.raises(MetricNotSpecifiedError):
        aggregate({"a": 1.0}, AggregationWeights({"a": 1.0}, verified=True))
    w = AggregationWeights({"a": 0.25, "b": 0.75}, verified=True, thesis_reference="test-only")
    assert aggregate({"a": 1.0, "b": 0.0}, w) == 0.25
