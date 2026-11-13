"""weighted_cmpi stays an explicitly labelled surrogate (Issue #11, item 7). Torch-free."""
import pytest

from ansg.evaluation import weighted_cmpi
from ansg.evaluation.craft import CraftRegistry, EvaluationSample, MetricNotSpecifiedError
from ansg.utils.config import load_yaml_config


def test_weighted_cmpi_is_flagged_as_surrogate():
    assert weighted_cmpi.is_dissertation_cmpi is False
    assert weighted_cmpi.metric_status == "surrogate"
    assert "SURROGATE" in weighted_cmpi.__doc__ and "NOT the dissertation's CMPI" in weighted_cmpi.__doc__
    assert weighted_cmpi(1, 1, 1, 1) == pytest.approx(1.0)


def test_craft_registry_still_refuses_cmpi():
    reg = CraftRegistry.from_config(load_yaml_config("configs/evaluation/craft.yaml"))
    assert reg.status_report()["CMPI"]["status"] == "incomplete"
    with pytest.raises(MetricNotSpecifiedError):
        reg.get("CMPI")([EvaluationSample("synthetic", "نص")])
