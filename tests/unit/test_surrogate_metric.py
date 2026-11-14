"""weighted_cmpi applies the dissertation's CMPI equation; dimension scorers stay unspecified. Torch-free."""
import pytest

from ansg.evaluation import weighted_cmpi
from ansg.evaluation.cmpi import ansg_cmpi
from ansg.evaluation.craft import CraftRegistry, EvaluationSample, MetricNotSpecifiedError
from ansg.utils.config import load_yaml_config


def test_weighted_cmpi_is_the_thesis_equation_with_unspecified_scorers():
    assert weighted_cmpi.is_dissertation_cmpi_equation is True
    assert weighted_cmpi.dimension_scorers_specified is False
    assert weighted_cmpi.is_dissertation_cmpi is False          # a computed value is never a thesis result
    assert weighted_cmpi.metric_status == "thesis_equation_dimension_scorers_unspecified"
    assert "§5.1" in weighted_cmpi.thesis_reference and "p.131" in weighted_cmpi.thesis_reference
    assert weighted_cmpi(1, 1, 1, 1) == pytest.approx(1.0)
    assert weighted_cmpi(0.9, 0.8, 0.7, 0.6) == pytest.approx(ansg_cmpi(0.9, 0.8, 0.7, 0.6))


def test_craft_registry_still_refuses_to_score_text():
    reg = CraftRegistry.from_config(load_yaml_config("configs/evaluation/craft.yaml"))
    assert reg.status_report()["CMPI"]["status"] == "incomplete"
    with pytest.raises(MetricNotSpecifiedError):
        reg.get("CMPI")([EvaluationSample("synthetic", "نص")])
