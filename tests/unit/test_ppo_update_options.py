"""ppo_update: thesis PPO options are opt-in; defaults keep the original behaviour."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from ansg.training import ppo_update  # noqa: E402


class TinyPolicy(nn.Module):
    """Minimal stand-in exposing the ANSG forward contract (logits, values)."""

    def __init__(self, vocab=11, hidden=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)
        self.value = nn.Linear(hidden, 1)

    def forward(self, ids, schema_ids=None, dialect_ids=None):
        h = self.emb(ids)
        return {"logits": self.head(h), "values": self.value(h).squeeze(-1)}


def _rollout(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 11, (2, 5), generator=g)
    actions = torch.randint(0, 11, (2, 5), generator=g)
    with torch.no_grad():
        old = torch.log_softmax(model(ids)["logits"], -1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    adv = torch.randn(2, 5, generator=g)
    ret = torch.randn(2, 5, generator=g)
    return ids, actions, old, adv, ret


def _pair():
    torch.manual_seed(0)
    a = TinyPolicy()
    b = TinyPolicy()
    b.load_state_dict(a.state_dict())
    return a, b


def test_defaults_equal_explicit_original_settings():
    a, b = _pair()
    ids, actions, old, adv, ret = _rollout(a)
    ra = ppo_update(a, old, ids, None, actions, adv, ret, torch.optim.SGD(a.parameters(), lr=0.1))
    rb = ppo_update(b, old, ids, None, actions, adv, ret, torch.optim.SGD(b.parameters(), lr=0.1),
                    max_grad_norm=1.0, entropy_coef=0.0)
    assert ra == rb and "entropy" not in ra
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_entropy_bonus_is_subtracted_and_reported():
    a, b = _pair()
    ids, actions, old, adv, ret = _rollout(a)
    base = ppo_update(a, old, ids, None, actions, adv, ret, torch.optim.SGD(a.parameters(), lr=0.0))
    with_ent = ppo_update(b, old, ids, None, actions, adv, ret, torch.optim.SGD(b.parameters(), lr=0.0), entropy_coef=0.01)
    assert with_ent["entropy"] > 0
    assert with_ent["loss"] == pytest.approx(base["loss"] - 0.01 * with_ent["entropy"], rel=1e-6)


def test_grad_norm_option_changes_the_update():
    a, b = _pair()
    ids, actions, old, adv, ret = _rollout(a)
    ppo_update(a, old, ids, None, actions, adv * 100, ret, torch.optim.SGD(a.parameters(), lr=0.1), max_grad_norm=1.0)
    ppo_update(b, old, ids, None, actions, adv * 100, ret, torch.optim.SGD(b.parameters(), lr=0.1), max_grad_norm=0.5)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
