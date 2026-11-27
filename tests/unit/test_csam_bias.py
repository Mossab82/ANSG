import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from ansg.backbone.gpt2_bias import GPT2Spec, GPT2WithAttentionBias  # noqa: E402
from ansg.csam_bias import (  # noqa: E402
    CSAMSettings,
    CulturalSchemaBias,
    csam_objective,
    cultural_alignment_loss,
    schema_prediction_loss,
)

D, S, T = 48, 5, 7


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(1)
    cfg = transformers.GPT2Config(vocab_size=50, n_positions=16, n_embd=D, n_layer=2, n_head=12)
    hf = transformers.GPT2LMHeadModel(cfg).eval()
    model = GPT2WithAttentionBias.from_hf_state_dict(GPT2Spec.from_hf_config(cfg.to_dict()), hf.state_dict())
    csam = CulturalSchemaBias(CSAMSettings(d_model=D, graph_dim=10, layers=(0, 1)), torch.randn(S, 10))
    return model, csam


def test_naive_eq_4_13_equals_outer_sum(setup):
    _, csam = setup
    pos = torch.randn(T, D)
    fast = csam.relevance_logits(0, pos)
    c = csam.schema_embeddings()
    g, i, s, j, k = 1, 2, 3, 4, 6
    v = torch.cat([csam.v_pos_j[0, g, i, s], csam.v_pos_k[0, g, i, s], csam.v_schema[0, g, i, s]])
    naive = v @ torch.cat([pos[j], pos[k], c[s]]) + csam.v_const[0, g, i, s]
    assert torch.isclose(fast[g, i, s, j, k], naive, atol=1e-5)


def test_bias_layout_follows_eq_4_9(setup):
    _, csam = setup
    w = torch.zeros(2, S)
    w[0, 1] = 0.9
    bias = csam.bias(0, torch.randn(T, D), w)
    assert bias.shape == (2, 12, T, T)
    assert torch.all(bias[:, 8:] == 0)                 # dialectal heads: no bias
    assert torch.all(bias[1] == 0)                     # no active schema -> no bias
    assert torch.all(bias[0, :8] > 0)                  # sigmoid relevance is positive
    hr = csam.group_matrices(0, torch.randn(T, D), w)
    assert hr.shape == (2, 2, 4, T, T)


def test_inactive_schemas_leave_model_unchanged(setup):
    model, csam = setup
    ids = torch.randint(0, 50, (1, T))
    weights = torch.zeros(1, S)
    with torch.no_grad():
        base = model(ids)["logits"]
        biased = model(ids, bias_fn=csam.bias_fn(model.wpe.weight, weights))["logits"]
    assert torch.allclose(base, biased, atol=1e-6)


def test_activation_threshold_eq_4_20(setup):
    _, csam = setup
    c = csam.schema_embeddings().detach()
    weights, active = csam.activation(c[:2] + 1e-3)    # contexts almost equal to schemas 0 and 1
    assert active[0, 0] and active[1, 1]
    assert torch.all(weights[~active] == 0)


def test_objective_backpropagates_to_csam_parameters(setup):
    model, csam = setup
    ids = torch.randint(0, 50, (2, T))
    context = model.wte(ids).mean(dim=1)
    weights, _ = csam.activation(context)
    weights = weights + 0.5                            # force activation for the gradient check
    out = model(ids, bias_fn=csam.bias_fn(model.wpe.weight, weights), labels=ids)
    h = csam.group_matrices(0, model.wpe.weight[:T], weights)[:, 0, 0]
    align = cultural_alignment_loss(h, torch.tensor([[0, 3, 1]]), torch.tensor([[1, 5, 0]]))
    schema = schema_prediction_loss(csam, context, torch.tensor([[1, 0, 0, 1, 0], [0, 1, 0, 0, 0]]))
    loss = csam_objective(out["loss"], align, schema, 0.1, 0.1)
    loss.backward()
    assert csam.v_pos_j.grad is not None and csam.v_pos_j.grad.abs().sum() > 0
    assert csam.w_proj.weight.grad is not None
