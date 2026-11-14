"""Optional transfer-learning bridge to an actual Hugging Face causal LM.

This bridge is *post-backbone* cultural conditioning plus DAL, NOT the
pre-softmax modification of internal GPT attention in thesis Eq. 4.9.
For faithful injection into pretrained layers, transformer-specific attention
patching and shape validation are still necessary.

Bridge v2 (scientific review, 2026-11): identity at initialisation
--------------------------------------------------------------------
The first bridge computed ``final_ln(dal(h + from_cultural(csam(to_cultural(h)))))``
and projected that with the language-model head. With untrained modules this
was NOT the backbone: ``h`` is already the backbone's final-normalised state, so
the sum doubled it, CSAM (``0.370·h + proj``) and the random ``to/from_cultural``
projections added noise, and a fresh LayerNorm replaced the backbone's learned
final-norm gain. Any attached-but-untrained module therefore degraded the
pretrained Arabic distribution, and "ANSG vs backbone" ablations compared a
damaged backbone with an intact one.

v2 adds only an *increment* to the exact input of the backbone's output head::

    z      = to_cultural(h)
    c      = from_cultural(csam(z) - z)          # what CSAM changes, nothing else
    s      = dal(h + g · c)                      # g: learned scalar gate, initialised to 0
    logits = backbone_logits + W_out · (s - h)   # W_out: output-embedding matrix

``g = 0`` and DAL's zero-initialised up-projection make the untrained bridge
reproduce the backbone's logits exactly (tested), so every measured difference
is attributable to trained cultural parameters. CSAM's own equations are
unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .csam import CulturalSchemaAttention
from .dal import DialectalAdaptation

BRIDGE_VERSION = 2

class PretrainedANSG(nn.Module):
    def __init__(self, backbone, schema_count=1847, schema_dim=256, cultural_heads=12,
                 lora_rank=8, lora_alpha=16, cultural_width=None):
        super().__init__()
        d=int(backbone.config.n_embd if hasattr(backbone.config,'n_embd') else backbone.config.hidden_size)
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
        # ReZero-style gate: 0 at initialisation, so the untrained bridge is the backbone.
        self.cultural_gate=nn.Parameter(torch.zeros(()))

    @classmethod
    def from_pretrained(cls,model_name='aubmindlab/aragpt2-large',**kwargs):
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError('Install transformers for pretrained integration') from exc
        return cls(AutoModelForCausalLM.from_pretrained(model_name),**kwargs)

    def _backbone_states(self,input_ids,attention_mask,extra):
        """Return (input of the output head, backbone logits)."""
        head=self.backbone.get_output_embeddings()
        captured={}
        handle=head.register_forward_pre_hook(lambda _m,args: captured.__setitem__('h',args[0]))
        try:
            out=self.backbone(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True,
                              return_dict=True,**extra)
        finally:
            handle.remove()
        logits=getattr(out,'logits',None)
        if 'h' in captured and logits is not None:
            return captured['h'],logits,out
        # Backbones that do not route through their output embeddings (e.g. test stubs):
        # fall back to the last hidden state and project it with the head.
        states=out.hidden_states[-1]
        return states,head(states),out

    def forward(self,input_ids,schema_ids,dialect_ids=None,labels=None,attention_mask=None,schema_mask=None,
                past_key_values=None,use_cache=False,dialect_compatibility_mask=None):
        # Issue #15: optional backbone KV cache for incremental generation. CSAM (token->schema
        # cross-attention) and DAL act per position, so the newest positions' outputs only need
        # the newest backbone hidden states.
        extra={'past_key_values':past_key_values,'use_cache':True} if use_cache else {}
        original,base_logits,out=self._backbone_states(input_ids,attention_mask,extra)
        if schema_mask is None:
            schema_mask = schema_ids.ne(0)
        if schema_mask.shape != schema_ids.shape:
            raise ValueError('schema_mask shape mismatch')
        schema_mask = schema_mask.bool()
        z=self.to_cultural(original)
        cultural=self.from_cultural(self.csam(z,schema_ids,schema_mask=schema_mask)-z)
        schemas=self.csam.schema_embeddings(schema_ids)
        summary=(schemas * schema_mask.unsqueeze(-1).to(schemas.dtype)).sum(dim=1) / schema_mask.sum(dim=1,keepdim=True).clamp_min(1)
        states=self.dal(original+self.cultural_gate*cultural,summary,dialect_ids,
                        compatibility_mask=dialect_compatibility_mask)
        delta=states-original
        head=self.backbone.get_output_embeddings()
        if isinstance(head,nn.Linear):
            logits=base_logits+F.linear(delta,head.weight)
        else:
            logits=base_logits+(head(states)-head(original))
        result={'logits':logits}
        if use_cache:
            result['past_key_values']=out.past_key_values
        if labels is not None:
            result['loss']=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),labels[:,1:].reshape(-1),ignore_index=-100)
        return result
