"""Validated model specification; does not silently resolve thesis conflicts."""
from dataclasses import dataclass, asdict
import json

DIALECTS = ("msa", "egyptian", "levantine", "gulf", "maghrebi")

@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    context_length: int = 128
    hidden_size: int = 96
    transformer_heads: int = 4
    layers: int = 2
    cultural_heads: int = 12
    schema_count: int = 64
    schema_dim: int = 64
    lora_rank: int = 8
    lora_alpha: float = 16.0
    dropout: float = 0.0
    alpha_base: float = 0.370
    beta_heritage: float = 0.347
    gamma_rhetorical: float = 0.283

    def validate(self):
        if min(self.vocab_size, self.context_length, self.hidden_size, self.layers,
               self.transformer_heads, self.cultural_heads, self.schema_count,
               self.schema_dim, self.lora_rank) <= 0:
            raise ValueError("Positive dimensions are required")
        if self.hidden_size % self.transformer_heads:
            raise ValueError("hidden_size must be divisible by transformer_heads")
        if self.hidden_size % self.cultural_heads:
            raise ValueError("hidden_size must be divisible by cultural_heads; 1280/12 is invalid")
        if self.cultural_heads % 3:
            raise ValueError("cultural_heads must be evenly divisible across 3 cultural groups")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be within [0,1)")
        return self

    @classmethod
    def from_dict(cls, data):
        return cls(**data).validate()

    def save_json(self, path):
        with open(path, 'w', encoding='utf-8') as out:
            json.dump(asdict(self), out, indent=2)
