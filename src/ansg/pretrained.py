"""Optional transfer-learning bridge to an actual Hugging Face causal LM.

This bridge is *post-backbone* cultural conditioning plus DAL, NOT the
pre-softmax modification of internal GPT attention in thesis Eq. 4.9.
For faithful injection into pretrained layers, transformer-specific attention
patching and shape validation are still necessary.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .csam import CulturalSchemaAttention
from .dal import DialectalAdaptation

class PretrainedANSG(nn.Module):
    def __init__(self, backbone, schema_count=1847, schema_dim=256, cultural_heads=12,
                 lora_rank=8, lora_alpha=16, cultural_width=None):
        super().__init__()
        d=int(backbone.config.n_embd if hasattr(backbone.config,'n_embd') else backbone.config.hidden_size)
        heads=int(backbone.config.n_head if hasattr(backbone.config,'n_head') else backbone.config.num_attention_heads)
        # Cultural internal width can differ from backbone width when 1280/12 is non-integral.
        internal=cultural_width or ((d+cultural_heads-1)//cultural_heads)*cultural_heads
        cfg=ModelConfig(hidden_size=internal,transformer_heads=cultural_heads,
                        cultural_heads=cultural_heads,schema_count=schema_count,
                        schema_dim=schema_dim,lora_rank=lora_rank,lora_alpha=lora_alpha)
        cfg.validate()
        self.backbone=backbone
        self.to_cultural=nn.Linear(d,internal) if internal!=d else nn.Identity()
        self.csam=CulturalSchemaAttention(cfg)
        self.from_cultural=nn.Linear(internal,d) if internal!=d else nn.Identity()
        self.dal=DialectalAdaptation(d,schema_dim,lora_rank,lora_alpha)
        self.final_ln=nn.LayerNorm(d)

    @classmethod
    def from_pretrained(cls,model_name='aubmindlab/aragpt2-large',**kwargs):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError('Install transformers for pretrained integration') from exc
        return cls(AutoModelForCausalLM.from_pretrained(model_name),**kwargs)

    def forward(self,input_ids,schema_ids,dialect_ids=None,labels=None,attention_mask=None,schema_mask=None):
        out=self.backbone(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        original=out.hidden_states[-1]
        if schema_mask is None:
            schema_mask = schema_ids.ne(0)
        if schema_mask.shape != schema_ids.shape:
            raise ValueError('schema_mask shape mismatch')
        schema_mask = schema_mask.bool()
        adapted=self.from_cultural(self.csam(self.to_cultural(original),schema_ids,schema_mask=schema_mask))
        schemas=self.csam.schema_embeddings(schema_ids)
        summary=(schemas * schema_mask.unsqueeze(-1).to(schemas.dtype)).sum(dim=1) / schema_mask.sum(dim=1,keepdim=True).clamp_min(1)
        states=self.final_ln(self.dal(original+adapted,summary,dialect_ids))
        logits=self.backbone.get_output_embeddings()(states)
        result={'logits':logits}
        if labels is not None:
            result['loss']=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1),ignore_index=-100)
        return result
