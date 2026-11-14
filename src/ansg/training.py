"""Training steps for supervised, adversarial sequence updates and clipped PPO.

Reward functions must be separately validated against thesis CRAFT definitions.
"""
import torch
from torch import nn
from torch.nn import functional as F

class NarrativeDiscriminator(nn.Module):
    def __init__(self,vocab_size,hidden_size):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,hidden_size)
        self.net=nn.Sequential(nn.Linear(hidden_size,hidden_size),nn.GELU(),nn.Linear(hidden_size,1))
    def forward(self,ids):
        return self.net(self.embedding(ids).mean(dim=1)).squeeze(-1)

def supervised_step(model,batch,optimizer,max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out=model(**batch)
    out['loss'].backward()
    nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
    optimizer.step()
    return float(out['loss'].detach())

def discriminator_step(discriminator,real_ids,fake_ids,optimizer):
    discriminator.train()
    optimizer.zero_grad(set_to_none=True)
    real_logits=discriminator(real_ids)
    fake_logits=discriminator(fake_ids.detach())
    loss=F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))+F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits))
    loss.backward()
    optimizer.step()
    return float(loss.detach())

def generator_adversarial_step(model,discriminator,prompt_ids,schema_ids,optimizer,dialect_ids=None,new_tokens=8):
    """Score-function policy gradient; discriminator is a frozen reward source here."""
    model.train()
    with torch.no_grad():
        samples=model.generate(prompt_ids,schema_ids,dialect_ids,max_new_tokens=new_tokens)
        scores=torch.sigmoid(discriminator(samples))
    # Compute conditional likelihood only on newly sampled continuation.
    out=model(samples[:,:-1],schema_ids,dialect_ids)
    log_probs=F.log_softmax(out['logits'][:,-new_tokens:,:],dim=-1)
    chosen=log_probs.gather(-1,samples[:,-new_tokens:].unsqueeze(-1)).squeeze(-1)
    loss=-(chosen.sum(dim=1)*scores.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach())

def ppo_update(model,old_logprobs,ids,schema_ids,actions,advantages,returns,optimizer,dialect_ids=None,clip_eps=0.2,value_coef=0.5,action_mask=None,max_grad_norm=1.0,entropy_coef=0.0):
    """Clipped PPO on a fixed rollout; caller must supply same states/actions as rollout.

    ids: [B,T] states; actions: [B,T] actual next tokens; mask per-token
    training should be applied by the caller to exclude prompt/padding positions.

    Defaults keep this function's original behaviour (grad-norm clip 1.0, no
    entropy bonus). The dissertation's §5.6.3 / Table 5.9 values (clip 0.2, value
    coefficient 0.5, entropy coefficient 0.01, grad-norm 0.5) are one documented
    variant among several; obtain them explicitly with
    ``ansg.reward_spec.load_reward_spec().ppo_update_kwargs(<variant id>)``.
    The KL penalty (coefficient 0.01) is not implemented: the thesis does not
    state whether it is applied in the reward or in the loss.
    """
    model.train()
    out=model(ids,schema_ids,dialect_ids)
    logits=out['logits']
    current=F.log_softmax(logits,dim=-1).gather(-1,actions.unsqueeze(-1)).squeeze(-1)
    if current.shape!=old_logprobs.shape or advantages.shape!=current.shape or returns.shape!=current.shape:
        raise ValueError('PPO tensors must match [batch,seq_len]')
    ratio=(current-old_logprobs.detach()).exp()
    surrogate=torch.minimum(ratio*advantages,torch.clamp(ratio,1-clip_eps,1+clip_eps)*advantages)
    if action_mask is None:
        policy_loss=-surrogate.mean()
        value_loss=F.mse_loss(out['values'],returns)
    else:
        if action_mask.shape!=current.shape or action_mask.sum()<=0:
            raise ValueError('PPO action_mask must match actions and select at least one token')
        mask=action_mask.to(current.dtype)
        policy_loss=-(surrogate*mask).sum()/mask.sum()
        value_loss=(((out['values']-returns)**2)*mask).sum()/mask.sum()
    loss=policy_loss+value_coef*value_loss
    result={}
    if entropy_coef:
        logp=F.log_softmax(logits,dim=-1)
        token_entropy=-(logp.exp()*logp).sum(-1)
        entropy=token_entropy.mean() if action_mask is None else (token_entropy*action_mask.to(token_entropy.dtype)).sum()/action_mask.sum()
        loss=loss-entropy_coef*entropy
        result['entropy']=float(entropy.detach())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
    optimizer.step()
    return {'loss':float(loss.detach()),'policy_loss':float(policy_loss.detach()),'value_loss':float(value_loss.detach()),**result}

def cultural_reward(cultural,linguistic,coherence,weights=(0.45,0.35,0.20)):
    if len(weights)!=3 or abs(sum(weights)-1.0)>1e-6:
        raise ValueError('Chapter 5 reward uses 3 normalized weights')
    return weights[0]*cultural+weights[1]*linguistic+weights[2]*coherence
