import json
import pytest
import torch
from ansg.config import ModelConfig
from ansg.model import ANSG,count_parameters
from ansg.csam import CulturalSchemaAttention
from ansg.dal import DialectalAdaptation
from ansg.knowledge import CulturalKnowledgeGraph
from ansg.training import NarrativeDiscriminator,discriminator_step,generator_adversarial_step,ppo_update,cultural_reward
from ansg.evaluation import weighted_cmpi
from ansg.data import NarrativeDataset,collate

@pytest.fixture
def cfg(): return ModelConfig().validate()

def test_invalid_appendix_heads():
    with pytest.raises(ValueError,match='1280/12'):
        ModelConfig(hidden_size=1280,transformer_heads=20,cultural_heads=12).validate()

def test_csam_shapes_and_gradient(cfg):
    layer=CulturalSchemaAttention(cfg)
    x=torch.randn(2,7,cfg.hidden_size,requires_grad=True)
    y=layer(x,torch.tensor([[1,2],[3,4]]))
    assert y.shape==x.shape
    y.square().mean().backward()
    assert x.grad is not None and layer.key_proj.weight.grad is not None

def test_dal_dialects(cfg):
    dal=DialectalAdaptation(cfg.hidden_size,cfg.schema_dim,8)
    x=torch.randn(2,6,cfg.hidden_size)
    s=torch.randn(2,cfg.schema_dim)
    assert dal(x,s).shape==x.shape
    assert dal(x,s,torch.tensor([0,4])).shape==x.shape
    assert sum(p.numel() for p in dal.parameters())>0

def test_model_loss_and_generate(cfg):
    model=ANSG(cfg)
    ids=torch.randint(0,cfg.vocab_size,(2,12))
    schemas=torch.tensor([[1,2],[3,4]])
    out=model(ids,schemas,torch.tensor([0,3]),labels=ids)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert model.csam.query_proj.weight.grad is not None
    assert model.dal.up.grad is not None
    counts=count_parameters(model)
    assert counts['total']>=counts['csam']+counts['dal']
    assert model.generate(ids[:,:4],schemas,max_new_tokens=3).shape==(2,7)

def test_knowledge_activation():
    from pathlib import Path
    ckg=CulturalKnowledgeGraph.from_json(Path(__file__).resolve().parents[1]/'examples/ckg_demo.json')
    assert ckg.activate('تحدث الناس عن الصدق والتعاون',top_k=2)

def test_reward_and_cmpi():
    assert cultural_reward(.8,.8,.8)==pytest.approx(.8)
    assert weighted_cmpi(.7,.7,.7,.7)==pytest.approx(.7)

def test_data_loader():
    from pathlib import Path
    dataset=NarrativeDataset(Path(__file__).resolve().parents[1]/'examples/demo.jsonl')
    batch=collate([dataset[0],dataset[1]])
    assert batch['input_ids'].shape==batch['labels'].shape
    assert batch['schema_ids'].ndim==2

def test_adversarial_and_ppo(cfg):
    model=ANSG(cfg)
    disc=NarrativeDiscriminator(cfg.vocab_size,16)
    prompt=torch.randint(0,256,(2,6))
    schemas=torch.tensor([[1],[2]])
    samples=model.generate(prompt,schemas,max_new_tokens=2)
    assert isinstance(discriminator_step(disc,prompt,samples[:,:6],torch.optim.Adam(disc.parameters(),lr=.001)),float)
    result=generator_adversarial_step(model,disc,prompt,schemas,torch.optim.Adam(model.parameters(),lr=.0001),new_tokens=2)
    assert torch.isfinite(torch.tensor(result))
    out=model(prompt,schemas)
    actions=torch.randint(0,256,prompt.shape)
    old=torch.nn.functional.log_softmax(out['logits'],dim=-1).gather(-1,actions.unsqueeze(-1)).squeeze(-1).detach()
    losses=ppo_update(model,old,prompt,schemas,actions,torch.ones_like(old),torch.zeros_like(old),torch.optim.Adam(model.parameters(),lr=.0001))
    assert torch.isfinite(torch.tensor(losses['loss']))

def test_pairwise_bias_causal_head_shape(cfg):
    from ansg.csam import PairwiseCulturalBias
    pb=PairwiseCulturalBias(cfg)
    bias=pb(torch.tensor([[1,2],[3,4]]),7)
    assert bias.shape == (2,cfg.transformer_heads,7,7)
    bias.sum().backward()
    assert pb.schema_embeddings.weight.grad is not None
