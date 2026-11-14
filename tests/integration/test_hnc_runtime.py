"""ANSG-HNC with the real generation runtime on a SYNTHETIC stub backbone (no download).

Checks the acceptance criteria that need a model: disabling HNC leaves the original
path token-for-token unchanged; HNC never changes model weights; records state
whether HNC was enabled. All generated text is meaningless stub output.
"""
import hashlib
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytest.importorskip("transformers", reason="pretrained extra (transformers) not installed")
pytestmark = pytest.mark.model

from ansg.generation.artifacts import digest_of, file_inventory  # noqa: E402
from ansg.generation.runtime import (GenerationRequest, SamplingConfig, generate_ids, load_backbone,  # noqa: E402
                                     run_generation)
from ansg.generation.stub import make_stub_backbone  # noqa: E402
from ansg.hnc.integration import generate, runtime_generate_fn  # noqa: E402
from ansg.hnc.model import HierarchicalPlan  # noqa: E402
from ansg.hnc.realize import HNCConfig, realise  # noqa: E402

SCEN = Path(__file__).resolve().parents[2] / "research" / "hnc" / "scenarios"
VOLATILE = {"created_utc", "timing_seconds", "hnc"}


@pytest.fixture(scope="module")
def stub(tmp_path_factory):
    root = tmp_path_factory.mktemp("hnc") / "stub"
    make_stub_backbone(root, seed=0, context=4096)
    return root


def request(stub, prompt="كان يا ما كان", max_new=6):
    return GenerationRequest(prompt=prompt, backbone=str(stub), device="cpu",
                             sampling=SamplingConfig(max_new_tokens=max_new, seed=7))


def test_disabled_hnc_reproduces_the_original_path_exactly(stub):
    original = run_generation(request(stub))
    via_hnc = generate(request(stub), HNCConfig(hnc_enabled=False))
    assert via_hnc["hnc"] == {"enabled": False, "variant": "A"}
    assert {k: v for k, v in via_hnc.items() if k not in VOLATILE} == \
        {k: v for k, v in original.items() if k not in VOLATILE}
    assert via_hnc["samples"][0]["token_ids_sha256"] == original["samples"][0]["token_ids_sha256"]
    with pytest.raises(ValueError):
        generate(request(stub), HNCConfig(hnc_enabled=True))


def _state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def test_hnc_does_not_change_model_weights(stub):
    bb = load_backbone(stub, device=torch.device("cpu"), dtype=torch.float32)
    before = _state_sha(bb.model)

    def gen(prompt, i):
        ids = bb.tokenizer(prompt)["input_ids"][-200:]

        def step(inp, past):
            out = bb.model(input_ids=inp, past_key_values=past, use_cache=True)
            return out.logits[0, -1], out.past_key_values
        new, _ = generate_ids(step, ids, SamplingConfig(max_new_tokens=4), seed=i, device=torch.device("cpu"),
                              eos_token_id=None)
        return bb.tokenizer.decode(new)

    plan = HierarchicalPlan.read(SCEN / "kuwait_3level.json")
    r = realise(plan, gen, HNCConfig.for_variant("B", context_chars=50), "مدينة الكويت")
    assert len(r.segments) == 5 and r.to_dict()["hnc"]["enabled"] is True
    assert _state_sha(bb.model) == before
    assert not any(p.grad is not None for p in bb.model.parameters())


def test_hnc_segments_through_the_runtime_record_every_call(stub):
    digest = digest_of(file_inventory(stub))
    records = []
    fn = runtime_generate_fn(request(stub, max_new=4), records)
    plan = HierarchicalPlan.read(SCEN / "aqaba_2level.json")
    r = realise(plan, fn, HNCConfig.for_variant("B", context_chars=50), "العقبة")
    assert len(records) == len(r.segments) == 3
    assert [rec["generation_config"]["seed"] for rec in records] == [7, 8, 9]
    assert all(rec["label"]["mode"] == "synthetic_stub" for rec in records)
    assert digest_of(file_inventory(stub)) == digest      # checkpoint files untouched
