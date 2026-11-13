"""P0 checks: shapes and parameter counts stated in the dissertation (arithmetic only)."""
import subprocess
import sys

from ansg.utils.parameter_audit import (
    ARAGPT2_LARGE, GPT2_MEDIUM, GPT2_SMALL, a25_csam_excerpt_params, bias_matrix_floats,
    dal_params, head_split,
)


def test_a25_head_split_does_not_divide_for_thesis_backbones():
    assert head_split(ARAGPT2_LARGE.hidden_size, 12) == (106, 8)   # A.3 hidden 1280
    assert head_split(GPT2_MEDIUM.hidden_size, 12) == (85, 4)      # Ch.4 GPT-2 Medium
    assert head_split(GPT2_SMALL.hidden_size, 12) == (64, 0)       # only d = 768 gives d_k = 64


def test_dal_count_reproduces_ch4_only_for_768_x_12():
    assert dal_params(768, 12) == 737_280                          # §4.4.1
    assert dal_params(GPT2_MEDIUM.hidden_size, GPT2_MEDIUM.num_layers) != 737_280
    assert dal_params(768, 12, matrices_per_layer=2) == 1_474_560  # Table 5.8 q_proj+v_proj


def test_a25_excerpt_visible_parameter_total():
    p = a25_csam_excerpt_params()
    assert p == {"schema_embeddings": 472_832, "query_proj": 1_639_680, "key_proj": 328_960,
                 "value_proj": 328_960, "total_visible": 2_770_432}


def test_bias_matrices_are_activation_memory():
    assert bias_matrix_floats(512) == 24 * 512 * 512   # §4.6.1: 24n² floats


def test_a25_coefficients_form_convex_combination():
    assert round(0.370 + 0.347 + 0.283, 3) == 1.0


def test_committed_audit_document_is_current():
    r = subprocess.run([sys.executable, "scripts/audit_parameters.py", "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
