"""Sampling, seeding, device and prompt handling of the generation runtime (Issue #15).

Uses a deterministic fake step function instead of a language model.
"""
import unicodedata

import pytest

torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytestmark = pytest.mark.model

from ansg.generation.runtime import (  # noqa: E402
    GenerationError, SamplingConfig, filter_logits, generate_ids, prepare_prompt, resolve_device, resolve_dtype,
    sample_next,
)

V = 50


def fake_step(inp, past):
    """Logits depend on the last token only; ``past`` counts steps (stands in for a KV cache)."""
    g = torch.Generator().manual_seed(int(inp[0, -1]))
    return torch.randn(V, generator=g) * 3, (past or 0) + 1


def test_sampling_config_validation():
    SamplingConfig().validate()
    for bad in (dict(max_new_tokens=0), dict(temperature=-0.1), dict(top_p=0.0), dict(top_p=1.5),
                dict(top_k=0), dict(repetition_penalty=0.5), dict(seed=-1), dict(num_samples=0)):
        with pytest.raises(GenerationError):
            SamplingConfig(**bad).validate()


def test_top_p_keeps_the_smallest_nucleus():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    kept = torch.isfinite(filter_logits(logits, SamplingConfig(temperature=1.0, top_p=0.7), []))
    assert kept.tolist() == [True, True, False, False]
    kept = torch.isfinite(filter_logits(logits, SamplingConfig(temperature=1.0, top_p=0.1), []))
    assert kept.tolist() == [True, False, False, False]  # the top token always survives
    kept = torch.isfinite(filter_logits(logits, SamplingConfig(temperature=1.0, top_p=1.0, top_k=3), []))
    assert kept.tolist() == [True, True, True, False]


def test_greedy_and_repetition_penalty():
    logits = torch.tensor([1.0, 2.0, 1.9, -1.0])
    cfg = SamplingConfig(temperature=0)
    assert sample_next(logits, cfg, torch.Generator(), []) == 1
    penalised = SamplingConfig(temperature=0, repetition_penalty=2.0)
    assert sample_next(logits, penalised, torch.Generator(), [1]) == 2
    assert filter_logits(logits, penalised, [3])[3] == -2.0  # negative logits are multiplied


def test_generation_is_reproducible_for_a_seed():
    cfg = SamplingConfig(max_new_tokens=30, temperature=1.0, top_p=0.95)
    a = generate_ids(fake_step, [1, 2], cfg, seed=7, device=torch.device("cpu"), eos_token_id=None)
    torch.manual_seed(12345)  # global RNG state must not matter
    b = generate_ids(fake_step, [1, 2], cfg, seed=7, device=torch.device("cpu"), eos_token_id=None)
    c = generate_ids(fake_step, [1, 2], cfg, seed=8, device=torch.device("cpu"), eos_token_id=None)
    assert a == b and a[1] == "length" and len(a[0]) == 30
    assert a[0] != c[0]


def test_generation_stops_at_eos():
    cfg = SamplingConfig(max_new_tokens=5, temperature=0)
    first, _ = generate_ids(fake_step, [3], SamplingConfig(max_new_tokens=1, temperature=0), seed=0,
                            device=torch.device("cpu"), eos_token_id=None)
    ids, reason = generate_ids(fake_step, [3], cfg, seed=0, device=torch.device("cpu"), eos_token_id=first[0])
    assert ids == first and reason == "eos"


def test_device_and_dtype_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "cuda", "mps")
    if not torch.cuda.is_available():
        with pytest.raises(GenerationError, match="CUDA is not available"):
            resolve_device("cuda")
    with pytest.raises(GenerationError):
        resolve_device("tpu")
    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32
    with pytest.raises(GenerationError, match="float16"):
        resolve_dtype("float16", torch.device("cpu"))


def test_prompt_is_nfc_normalised_arabic():
    decomposed = "ا" + "ٔ" + "حمد"   # alef + combining hamza above
    text, info = prepare_prompt("  " + decomposed + "\n")
    assert text == unicodedata.normalize("NFC", decomposed) == "أحمد"
    assert info["arabic_letter_ratio"] == 1.0 and len(info["sha256"]) == 64
    assert prepare_prompt("Once upon a time")[1]["arabic_letter_ratio"] == 0.0
    with pytest.raises(GenerationError):
        prepare_prompt("   ")
