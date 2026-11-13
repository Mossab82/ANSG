"""Cultural Schema Attention Mechanism.

Combines pre-softmax schema-dependent bias (thesis Eq.4.9) and an explicit
schema cross-attention path (Appendix A.25). Exactly 12 heads split 4/4/4
for the 1280-d appendix config would require an explicit projection to a
head-divisible internal dimension; this implementation rejects it instead.
"""
import math
import torch
from torch import nn

class CulturalSchemaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        d, c, h = config.hidden_size, config.schema_dim, config.cultural_heads
        self.num_heads = h
        self.head_dim = d // h
        self.schema_embeddings = nn.Embedding(config.schema_count, c)
        self.query_proj = nn.Linear(d, d)
        self.key_proj = nn.Linear(c, d)
        self.value_proj = nn.Linear(c, d)
        self.output_proj = nn.Linear(d, d)
        self.heritage_bias = nn.Linear(c, h, bias=False)
        self.rhetorical_bias = nn.Linear(c, h, bias=False)
        self.alpha = config.alpha_base
        self.beta = config.beta_heritage
        self.gamma = config.gamma_rhetorical

    def forward(self, hidden_states, schema_ids, schema_mask=None):
        """[B,T,D] + [B,S] -> [B,T,D], preserving sequence length."""
        batch, length, dim = hidden_states.shape
        if schema_ids.ndim != 2 or schema_ids.shape[0] != batch:
            raise ValueError("schema_ids must have shape [batch, schema_count_active]")
        if schema_ids.shape[1] < 1:
            raise ValueError("Provide at least one active schema (the no-schema ID can be 0)")
        if schema_mask is None:
            schema_mask = schema_ids.ne(0)
        if schema_mask.shape != schema_ids.shape:
            raise ValueError("schema_mask shape mismatch")
        schema_mask = schema_mask.bool()
        if (schema_mask & schema_ids.eq(0)).any():
            raise ValueError("Padding schema ID 0 cannot be active")
        schemas = self.schema_embeddings(schema_ids)
        n = schemas.shape[1]
        q = self.query_proj(hidden_states).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(schemas).view(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(schemas).view(batch, n, self.num_heads, self.head_dim).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # These per-schema projections supply *learned logit biases*, not learned NxN tables.
        hb = self.heritage_bias(schemas).permute(0, 2, 1).unsqueeze(2)
        rb = self.rhetorical_bias(schemas).permute(0, 2, 1).unsqueeze(2)
        logits = logits + self.beta * hb + self.gamma * rb
        # A row may have no active cultural schema. Mask its logits safely,
        # then explicitly zero its attention rather than softmax over PAD.
        active = schema_mask.any(dim=1)
        logits = logits.masked_fill(~schema_mask[:, None, None, :], -torch.inf)
        logits = torch.where(active[:, None, None, None], logits, torch.zeros_like(logits))
        attn = torch.softmax(logits, dim=-1) * schema_mask[:, None, None, :].to(logits.dtype)
        contextualized = attn @ v  # [B,H,T,head_dim]
        group = self.num_heads // 3
        heritage = contextualized[:, :group].transpose(1, 2).reshape(batch, length, group * self.head_dim)
        rhetorical = contextualized[:, group:2*group].transpose(1, 2).reshape(batch, length, group * self.head_dim)
        dialectal = contextualized[:, 2*group:].transpose(1, 2).reshape(batch, length, group * self.head_dim)
        # Project the complete 3-group context back to hidden dimension.
        whole = torch.cat([self.beta * heritage, self.gamma * rhetorical, dialectal], dim=-1)
        output = self.alpha * hidden_states + self.output_proj(whole)
        return torch.where(active[:, None, None], output, hidden_states)

class PairwiseCulturalBias(nn.Module):
    """Implements Eq.4.12/4.13-like schema-conditioned *token pair* bias.

    Generates 12 cultural-head score planes and projects them onto the number
    of underlying transformer attention heads. This 12->4 mapping is an
    explicitly new architectural decision for runnable configurations.
    """
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.context_length = config.context_length
        self.pos_left = nn.Embedding(config.context_length,config.schema_dim)
        self.pos_right = nn.Embedding(config.context_length,config.schema_dim)
        self.schema_embeddings = nn.Embedding(config.schema_count,config.schema_dim)
        self.schema_importance = nn.Linear(config.schema_dim,1)
        self.relevance_h = nn.Linear(config.schema_dim * 3,config.cultural_heads)
        self.relevance_r = nn.Linear(config.schema_dim * 3,config.cultural_heads)
        self.project_to_transformer_heads = nn.Linear(config.cultural_heads,config.transformer_heads,bias=False)
        self.beta = config.beta_heritage
        self.gamma = config.gamma_rhetorical

    def forward(self, schema_ids, seq_len, schema_mask=None):
        if seq_len > self.context_length:
            raise ValueError('Pairwise bias exceeds configured max sequence length')
        b, s = schema_ids.shape
        if s < 1:
            raise ValueError("schema_ids must have at least one PAD or real ID")
        if schema_mask is None:
            schema_mask = schema_ids.ne(0)
        if schema_mask.shape != schema_ids.shape:
            raise ValueError("schema_mask shape mismatch")
        schema_mask = schema_mask.bool()
        if (schema_mask & schema_ids.eq(0)).any():
            raise ValueError("Padding schema ID 0 cannot be active")
        positions=torch.arange(seq_len,device=schema_ids.device)
        left=self.pos_left(positions)[:,None,:].expand(seq_len,seq_len,-1)
        right=self.pos_right(positions)[None,:,:].expand(seq_len,seq_len,-1)
        motifs=self.schema_embeddings(schema_ids)
        importance = self.schema_importance(motifs).squeeze(-1)
        active = schema_mask.any(dim=1, keepdim=True)
        masked = importance.masked_fill(~schema_mask, -torch.inf)
        masked = torch.where(active, masked, torch.zeros_like(masked))
        weights = torch.softmax(masked, dim=-1) * schema_mask.to(importance.dtype)
        total=None
        # Incremental evaluation avoids BxTxTxSxD allocation.
        for i in range(s):
            schema=motifs[:,i,None,None,:].expand(b,seq_len,seq_len,-1)
            features=torch.cat([left[None].expand(b,-1,-1,-1),right[None].expand(b,-1,-1,-1),schema],dim=-1)
            contribution=(self.beta*torch.sigmoid(self.relevance_h(features))+
                          self.gamma*torch.sigmoid(self.relevance_r(features)))
            term=weights[:,i,None,None,None]*contribution
            total=term if total is None else total+term
        return self.project_to_transformer_heads(total).permute(0,3,1,2)
