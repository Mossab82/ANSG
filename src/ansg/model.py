"""End-to-end decoder-only research model integrating CSAM and DAL.

This is a trainable PyTorch implementation; it is NOT an imported GPT-2/AraGPT2
checkpoint. Matching a published backbone requires separately verified config,
tokenizer and checkpoint mapping.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .csam import CulturalSchemaAttention, PairwiseCulturalBias
from .dal import DialectalAdaptation

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h = config.transformer_heads
        self.hd = config.hidden_size // self.h
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = config.dropout

    def forward(self, x, bias=None):
        b, t, d = x.shape
        q,k,v = self.qkv(x).split(d, dim=-1)
        q = q.view(b,t,self.h,self.hd).transpose(1,2)
        k = k.view(b,t,self.h,self.hd).transpose(1,2)
        v = v.view(b,t,self.h,self.hd).transpose(1,2)
        if bias is None:
            z = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=self.dropout if self.training else 0.0)
        else:
            logits = (q @ k.transpose(-1,-2)) / math.sqrt(self.hd)
            causal = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
            logits = (logits + bias).masked_fill(~causal, -torch.inf)
            probs = F.dropout(torch.softmax(logits,dim=-1),p=self.dropout,training=self.training)
            z = probs @ v
        return self.proj(z.transpose(1,2).contiguous().view(b,t,d))

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = CausalAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(nn.Linear(config.hidden_size,4*config.hidden_size), nn.GELU(), nn.Linear(4*config.hidden_size,config.hidden_size))

    def forward(self,x,bias=None):
        x = x + self.attn(self.ln1(x),bias)
        return x + self.mlp(self.ln2(x))

class ANSG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.validate()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_length, config.hidden_size)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.layers)])
        self.csam = CulturalSchemaAttention(config)
        self.pairwise_cultural_bias = PairwiseCulturalBias(config)
        self.dal = DialectalAdaptation(config.hidden_size,config.schema_dim,config.lora_rank,config.lora_alpha)
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.value_head = nn.Linear(config.hidden_size,1)

    def forward(self, input_ids, schema_ids, dialect_ids=None, labels=None, schema_mask=None,\n                dialect_compatibility_mask=None):
        b,t = input_ids.shape
        if t > self.config.context_length:
            raise ValueError('Sequence exceeds configured context length')
        if t < 1:
            raise ValueError('Sequence cannot be empty')
        if schema_mask is None:
            schema_mask = schema_ids.ne(0)
        if schema_mask.shape != schema_ids.shape:
            raise ValueError('schema_mask shape mismatch')
        schema_mask = schema_mask.bool()
        if (schema_mask & schema_ids.eq(0)).any():
            raise ValueError('Padding schema ID 0 cannot be active')
        positions = torch.arange(t,device=input_ids.device)
        x = self.token_embedding(input_ids)+self.position_embedding(positions)[None]
        pairwise_bias = self.pairwise_cultural_bias(schema_ids,t,schema_mask=schema_mask)
        for block in self.blocks:
            x = block(x,pairwise_bias)
        x = self.csam(x,schema_ids,schema_mask=schema_mask)
        embeddings = self.csam.schema_embeddings(schema_ids)
        masked_embeddings = embeddings * schema_mask.unsqueeze(-1).to(embeddings.dtype)
        summary = masked_embeddings.sum(dim=1) / schema_mask.sum(dim=1,keepdim=True).clamp_min(1)
        x = self.dal(x,summary,dialect_ids,compatibility_mask=dialect_compatibility_mask)
        x = self.final_ln(x)
        logits = self.output(x) / math.sqrt(self.config.hidden_size)
        values = self.value_head(x).squeeze(-1)
        result = {'logits':logits, 'values':values}
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError('labels and inputs must have identical shape')
            result['loss'] = F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]), labels[:,1:].reshape(-1),ignore_index=-100)
        return result

    @torch.no_grad()
    def generate(self, input_ids, schema_ids, dialect_ids=None, max_new_tokens=32, temperature=1.0, top_k=None, schema_mask=None,\n                 dialect_compatibility_mask=None):
        if temperature <= 0:
            raise ValueError('temperature must be > 0')
        was_training=self.training
        self.eval()
        try:
            tokens = input_ids
            for _ in range(max_new_tokens):
                recent = tokens[:,-self.config.context_length:]
                logits = self(recent,schema_ids,dialect_ids,schema_mask=schema_mask,\n                             dialect_compatibility_mask=dialect_compatibility_mask)['logits'][:,-1,:]/temperature
                if top_k is not None:
                    threshold=torch.topk(logits,min(top_k,logits.shape[-1]),dim=-1).values[:,-1,None]
                    logits=logits.masked_fill(logits<threshold,-torch.inf)
                next_id=torch.multinomial(torch.softmax(logits,dim=-1),1)
                tokens=torch.cat([tokens,next_id],dim=1)
            return tokens
        finally:
            self.train(was_training)

def count_parameters(model):
    # Deduplicate tied embeddings by storage identity
    all_parameters=list(model.parameters())
    return {'total':sum(p.numel() for p in all_parameters),
            'trainable':sum(p.numel() for p in all_parameters if p.requires_grad),
            'csam':sum(p.numel() for p in model.csam.parameters())+sum(p.numel() for p in model.pairwise_cultural_bias.parameters()),
            'dal':sum(p.numel() for p in model.dal.parameters())}
