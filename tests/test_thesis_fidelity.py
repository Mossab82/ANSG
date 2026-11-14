"""Fidelity pins for thesis items whose numbers are fully specified (docs/THESIS_TO_CODE_TRACEABILITY.md).

Values are transcribed from the dissertation; these tests check arithmetic and
configuration only, never reproduction of reported experimental results.
"""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytestmark = pytest.mark.model

from ansg.config import DIALECTS, ModelConfig  # noqa: E402
from ansg.training import cultural_reward  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_eq_4_30_worked_example():
    # Eq. 4.30 (§4.5.4, p. 119): R_total = 0.45(0.867) + 0.35(0.784) + 0.20(0.950) = 0.854 (as printed)
    value = cultural_reward(0.867, 0.784, 0.950)
    assert value == pytest.approx(0.85455)
    assert abs(value - 0.854) < 1e-3          # the thesis truncates; exact value 0.85455


def test_eq_4_29_weights_are_the_default_and_validated():
    assert cultural_reward(1.0, 0.0, 0.0) == pytest.approx(0.45)
    assert cultural_reward(0.0, 1.0, 0.0) == pytest.approx(0.35)
    assert cultural_reward(0.0, 0.0, 1.0) == pytest.approx(0.20)
    with pytest.raises(ValueError):
        cultural_reward(1, 1, 1, weights=(0.5, 0.5, 0.5))


def test_appendix_a3_csam_dal_coefficients_are_the_defaults():
    a3 = json.loads((ROOT / "configs" / "thesis_appendix_a3_reported.json").read_text())
    cfg = ModelConfig()
    assert (cfg.alpha_base, cfg.beta_heritage, cfg.gamma_rhetorical) == \
        (a3["alpha_base"], a3["beta_heritage"], a3["gamma_rhetorical"])
    assert (cfg.lora_rank, cfg.lora_alpha, cfg.cultural_heads) == (a3["lora_rank"], a3["lora_alpha"],
                                                                   a3["cultural_heads"])
    assert list(DIALECTS) == a3["dialects"]
    # documented conflict, not resolved here: A.3 reward weights differ from Eq. 4.29
    assert a3["reward_weights"] != {"cultural": 0.45, "linguistic": 0.35, "coherence": 0.20}


def test_eq_4_14_bias_is_multiplicative_reweighting():
    # Eq. 4.14 (§4.3.3, p. 102): softmax(z + b) = e^b e^z / sum(e^b e^z)
    torch.manual_seed(0)
    z, b = torch.randn(7), torch.randn(7)
    lhs = torch.softmax(z + b, -1)
    rhs = torch.exp(b) * torch.exp(z) / (torch.exp(b) * torch.exp(z)).sum()
    torch.testing.assert_close(lhs, rhs)
