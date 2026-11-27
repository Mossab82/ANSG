import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias  # noqa: E402
from ansg.dal_lora import DIALECTS, DialectalAdaptation  # noqa: E402

D, T = 48, 6


def test_adapter_count_matches_dissertation_4_4_1():
    dal = DialectalAdaptation(n_layer=12, d_model=768, rank=8)
    assert dal.adapter_parameter_count() == 737_280


@pytest.fixture()
def setup():
    torch.manual_seed(2)
    cfg = transformers.GPT2Config(vocab_size=40, n_positions=16, n_embd=D, n_layer=2, n_head=12)
    hf = transformers.GPT2LMHeadModel(cfg)
    model = GPT2WithAttentionBias.from_hf_state_dict(GPT2Spec.from_hf_config(cfg.to_dict()), hf.state_dict())
    for p in model.parameters():
        p.requires_grad_(False)
    dal = DialectalAdaptation(n_layer=2, d_model=D, formal_schema_index=[0])
    schemas = torch.randn(3, D)
    return model, dal, schemas


def test_zero_init_keeps_backbone_output(setup):
    model, dal, schemas = setup
    ids = torch.randint(0, 40, (2, T))
    fn = dal.delta_fn(torch.tensor([1, 3]), schemas, torch.zeros(2, 3))
    with torch.no_grad():
        assert torch.allclose(model(ids)["logits"], model(ids, delta_fn=fn)["logits"], atol=1e-6)


def test_training_updates_only_adapters(setup):
    model, dal, schemas = setup
    ids = torch.randint(0, 40, (2, T))
    out = model(ids, labels=ids, delta_fn=dal.delta_fn(torch.tensor([1, 2]), schemas, torch.zeros(2, 3)))
    out["loss"].backward()
    assert dal.B.grad is not None and dal.B.grad[:, [1, 2]].abs().sum() > 0
    assert dal.B.grad[:, [0, 3, 4]].abs().sum() == 0      # unused dialects untouched
    assert all(p.grad is None for p in model.parameters())


def test_formal_schema_blocks_colloquial_outside_dialogue(setup):
    _, dal, _ = setup
    egy, msa = DIALECTS.index("egyptian"), DIALECTS.index("msa")
    weights = torch.tensor([[0.9, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.8, 0.0]])
    dialogue = torch.tensor([[0, 0, 1, 1, 0, 0]] * 3)
    ok = dal.compatible(torch.tensor([egy, msa, egy]), weights, T, dialogue)
    assert ok[0].tolist() == [0, 0, 1, 1, 0, 0]   # wisdom active: dialect only inside dialogue
    assert ok[1].tolist() == [1] * T              # MSA always allowed
    assert ok[2].tolist() == [1] * T              # no formal schema active


def test_gate_scales_adapter_output(setup):
    model, dal, schemas = setup
    with torch.no_grad():
        dal.B.normal_()
    ids = torch.randint(0, 40, (1, T))
    weights = torch.tensor([[0.9, 0.0, 0.0]])
    egy = torch.tensor([DIALECTS.index("egyptian")])
    with torch.no_grad():
        base = model(ids)["logits"]
        blocked = model(ids, delta_fn=dal.delta_fn(egy, schemas, weights))["logits"]
        open_ = model(ids, delta_fn=dal.delta_fn(egy, schemas, torch.zeros(1, 3)))["logits"]
    assert torch.allclose(base, blocked, atol=1e-6)
    assert not torch.allclose(base, open_, atol=1e-4)
