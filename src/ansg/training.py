"""Training steps for supervised, adversarial sequence updates and clipped PPO.

Reward functions must be separately validated against thesis CRAFT definitions.
"""
import random
import torch
from torch import nn
from torch.nn import functional as F

class NarrativeDiscriminator(nn.Module):
    """Three-convolution text discriminator with an exposed feature vector.

    The Chapter 5 text specifies a 3-layer CNN and feature matching, but omits
    kernel widths, channels, and the feature-matching coefficient. Those
    details are explicit engineering parameters, not thesis-reported facts.
    """
    def __init__(self,vocab_size,hidden_size,channels=None,kernel_size=3):
        super().__init__()
        channels = channels or hidden_size
        self.embedding=nn.Embedding(vocab_size,hidden_size)
        padding=kernel_size//2
        self.conv1=nn.Conv1d(hidden_size,channels,kernel_size,padding=padding)
        self.conv2=nn.Conv1d(channels,channels,kernel_size,padding=padding)
        self.conv3=nn.Conv1d(channels,channels,kernel_size,padding=padding)
        self.activation=nn.GELU()
        self.classifier=nn.Sequential(nn.Linear(channels*2,channels),nn.GELU(),nn.Linear(channels,1))

    def extract_features(self,ids):
        x=self.embedding(ids).transpose(1,2)
        x=self.activation(self.conv1(x))
        x=self.activation(self.conv2(x))
        x=self.activation(self.conv3(x))
        # Mean + max pooling makes the feature vector insensitive to padding
        # length changes in the fixed-width smoke dataset.
        return torch.cat([x.mean(dim=-1),x.amax(dim=-1)],dim=-1)

    def forward(self,ids,return_features=False):
        features=self.extract_features(ids)
        logits=self.classifier(features).squeeze(-1)
        return (logits,features) if return_features else logits


class FakeReplayBuffer:
    """FIFO buffer for detached generated token sequences (Chapter 5: 10,000)."""
    def __init__(self,capacity=10000):
        if capacity<1:
            raise ValueError('replay buffer capacity must be positive')
        self.capacity=int(capacity)
        self.storage=[]
        self.next_index=0

    def __len__(self):
        return len(self.storage)

    @torch.no_grad()
    def push(self,ids):
        for row in ids.detach().to('cpu'):
            value=row.clone()
            if len(self.storage)<self.capacity:
                self.storage.append(value)
            else:
                self.storage[self.next_index]=value
                self.next_index=(self.next_index+1)%self.capacity

    def sample(self,count,device):
        if count<1 or not self.storage:
            return None
        rows=random.sample(self.storage,min(int(count),len(self.storage)))
        lengths={int(row.numel()) for row in rows}
        if len(lengths)!=1:
            raise ValueError('replay samples must have the same token length')
        return torch.stack(rows).to(device)

    def state_dict(self):
        return {'capacity':self.capacity,'next_index':self.next_index,
                'storage':[row.clone() for row in self.storage]}

    def load_state_dict(self,state):
        capacity=int(state['capacity'])
        if capacity<1 or len(state['storage'])>capacity:
            raise ValueError('invalid replay buffer state')
        self.capacity=capacity
        self.next_index=int(state['next_index'])%capacity
        self.storage=[row.detach().to('cpu').clone() for row in state['storage']]

def supervised_step(model,batch,optimizer,max_grad_norm=1.0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out=model(**batch)
    out['loss'].backward()
    nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
    optimizer.step()
    return float(out['loss'].detach())

def discriminator_step(discriminator,real_ids,fake_ids,optimizer,replay_buffer=None,
                       replay_batch_size=None):
    discriminator.train()
    fake_for_loss=fake_ids.detach()
    replay_count=len(fake_ids) if replay_batch_size is None else int(replay_batch_size)
    replayed=replay_buffer.sample(replay_count,fake_ids.device) if replay_buffer is not None else None
    if replayed is not None:
        fake_for_loss=torch.cat([fake_for_loss,replayed],dim=0)
    if replay_buffer is not None:
        replay_buffer.push(fake_ids)
    optimizer.zero_grad(set_to_none=True)
    real_logits=discriminator(real_ids)
    fake_logits=discriminator(fake_for_loss)
    loss=F.binary_cross_entropy_with_logits(real_logits,torch.ones_like(real_logits))+F.binary_cross_entropy_with_logits(fake_logits,torch.zeros_like(fake_logits))
    loss.backward()
    optimizer.step()
    return float(loss.detach())

def generator_adversarial_step(model,discriminator,prompt_ids,schema_ids,optimizer,dialect_ids=None,
                               new_tokens=8,real_ids=None,feature_matching_weight=0.0,
                               return_metrics=False):
    """Score-function generator update with adversarial and feature-match rewards.

    Discrete generated tokens do not provide a direct gradient through the CNN;
    the discriminator and feature-alignment scores therefore act as rewards in
    the policy-gradient update. ``feature_matching_weight`` is an engineering
    parameter because the thesis does not specify its coefficient.
    """
    if not 0.0<=feature_matching_weight<=1.0:
        raise ValueError('feature_matching_weight must be between 0 and 1')
    model.train()
    with torch.no_grad():
        samples=model.generate(prompt_ids,schema_ids,dialect_ids,max_new_tokens=new_tokens)
        if real_ids is not None:
            real_logits,real_features=discriminator(real_ids,return_features=True)
            fake_logits,fake_features=discriminator(samples,return_features=True)
            target_features=real_features.mean(dim=0,keepdim=True)
            distances=(fake_features-target_features).pow(2).mean(dim=-1)
            scale=real_features.var(dim=0,unbiased=False).mean().sqrt().clamp_min(1e-6)
            # Keep this diagnostic available for the zero-weight ablation arm;
            # the score contributes to the reward only through its coefficient.
            feature_scores=torch.exp(-distances/scale).clamp(0,1)
        else:
            fake_logits=discriminator(samples)
            feature_scores=torch.zeros_like(fake_logits)
        adversarial_scores=torch.sigmoid(fake_logits)
        scores=(1-feature_matching_weight)*adversarial_scores+feature_matching_weight*feature_scores
        # Fixed baseline avoids a zero policy gradient for batch size one.
        advantages=scores-0.5
    # Compute conditional likelihood only on newly sampled continuation.
    out=model(samples[:,:-1],schema_ids,dialect_ids)
    log_probs=F.log_softmax(out['logits'][:,-new_tokens:,:],dim=-1)
    chosen=log_probs.gather(-1,samples[:,-new_tokens:].unsqueeze(-1)).squeeze(-1)
    loss=-(chosen.sum(dim=1)*advantages.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    metrics={'loss':float(loss.detach()),'adversarial_score':float(adversarial_scores.mean()),
             'feature_match_score':float(feature_scores.mean()),'reward':float(scores.mean())}
    return metrics if return_metrics else metrics['loss']

def ppo_update(model,old_logprobs,ids,schema_ids,actions,advantages,returns,optimizer,dialect_ids=None,clip_eps=0.2,value_coef=0.5,action_mask=None):
    """Clipped PPO on a fixed rollout; caller must supply same states/actions as rollout.

    ids: [B,T] states; actions: [B,T] actual next tokens; mask per-token
    training should be applied by the caller to exclude prompt/padding positions.
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
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step()
    return {'loss':float(loss.detach()),'policy_loss':float(policy_loss.detach()),'value_loss':float(value_loss.detach())}

def cultural_reward(cultural,linguistic,coherence,weights=(0.45,0.35,0.20)):
    if len(weights)!=3 or abs(sum(weights)-1.0)>1e-6:
        raise ValueError('Chapter 5 reward uses 3 normalized weights')
    return weights[0]*cultural+weights[1]*linguistic+weights[2]*coherence
