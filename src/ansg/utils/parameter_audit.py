"""Arithmetic parameter/shape audit of the dissertation's architecture variants.

Pure arithmetic — no model is downloaded or instantiated. Every quantity is
either taken from the thesis (with location) or from documented properties of
public checkpoints (marked ``public``). Where the thesis leaves a dimension
unspecified the function takes it as an explicit argument.

Used by ``scripts/audit_parameters.py`` to generate ``docs/PARAMETER_AUDIT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneShape:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    source: str


# Public configurations of the checkpoints the thesis names (C-MODEL-1).
GPT2_SMALL = BackboneShape("gpt2 (small)", 768, 12, 12, "public; matches Ch.4 §4.4.1/§4.6.1 arithmetic (d=768, 12 layers)")
GPT2_MEDIUM = BackboneShape("gpt2-medium", 1024, 24, 16, "public; named in Ch.4 §4.1.6")
GPT2_LARGE = BackboneShape("gpt2-large", 1280, 36, 20, "public; 774M = Ch.5 Table 5.1 base size")
ARAGPT2_LARGE = BackboneShape("aubmindlab/aragpt2-large", 1280, 36, 20, "hidden/heads from App. A.3 Listing A.2; 36 layers public")
BACKBONES = (GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, ARAGPT2_LARGE)

CSAM_HEADS = 12  # Ch.4 §4.3.6; A.3 num_cultural_heads; A.25 self.num_heads


def head_split(hidden_size: int, num_heads: int) -> tuple[int, int]:
    """(head_dim, remainder) for ``hidden_size // num_heads`` as in A.25 line 22."""
    return hidden_size // num_heads, hidden_size % num_heads


def linear_params(d_in: int, d_out: int, bias: bool = True) -> int:
    return d_in * d_out + (d_out if bias else 0)


def lora_params(d_in: int, d_out: int, rank: int) -> int:
    """B ∈ R^{d_out×r}, A ∈ R^{r×d_in} (Eq. 4.24)."""
    return rank * (d_in + d_out)


def dal_params(hidden: int, layers: int, rank: int = 8, dialects: int = 5,
               matrices_per_layer: int = 1) -> int:
    """DAL adapters. Ch.4 §4.4.1 counts one square adapter per layer
    (2·r·d); Table 5.8 targets q_proj and v_proj (2 matrices per layer)."""
    return lora_params(hidden, hidden, rank) * matrices_per_layer * layers * dialects


def a25_csam_excerpt_params(num_schemas: int = 1847, schema_dim: int = 256,
                            hidden_size: int = 1280) -> dict[str, int]:
    """Parameters of the modules *visible* in App. A.25 Listing A.6 only
    (schema embedding, query/key/value projections). The excerpt calls an
    undefined ``compute_multihead_attention`` and may omit further modules."""
    parts = {
        "schema_embeddings": num_schemas * schema_dim,
        "query_proj": linear_params(hidden_size, hidden_size),
        "key_proj": linear_params(schema_dim, hidden_size),
        "value_proj": linear_params(schema_dim, hidden_size),
    }
    parts["total_visible"] = sum(parts.values())
    return parts


def eq_4_13_generator_params(num_schemas: int, num_heads: int, pos_dim: int, schema_dim: int) -> int:
    """Learned vectors v_{s,i} of Eq. 4.13, one per (schema, head), of size
    dim([pos_j; pos_k; c_s]) = 2·pos_dim + schema_dim. ``pos_dim`` is not stated."""
    return num_schemas * num_heads * (2 * pos_dim + schema_dim)


def sec_4_3_6_pattern_params(num_patterns: int = 847, d_k: int = 64) -> int:
    """§4.3.6: one b_p ∈ R^{d_k} and one weight w_p per pattern."""
    return num_patterns * (d_k + 1)


def bias_matrix_floats(seq_len: int, heads: int = CSAM_HEADS) -> int:
    """§4.6.1: H and R per head are n×n *activations* (24n² floats for 12 heads), not parameters."""
    return heads * seq_len * seq_len * 2
