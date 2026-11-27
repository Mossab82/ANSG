import pytest

from ansg.generation.seed_prompt import REGIONS, SeedError, build_story_prompt, load_seed


@pytest.mark.parametrize("region", REGIONS)
def test_prompt_uses_sampled_elements(region):
    prompt = build_story_prompt(region, rng_seed=3)
    assert prompt.region == region
    for key in ("value", "character", "setting"):
        assert prompt.elements[key] in prompt.user
    if prompt.elements["proverb"]:
        assert prompt.elements["proverb"] in prompt.user
    assert prompt.messages()[0]["role"] == "system"


@pytest.mark.parametrize("region", REGIONS)
def test_only_confident_proverbs_are_used(region):
    seed = load_seed(region)
    allowed = {p["text"] for p in seed["proverbs"] if p["confidence"] in {"high", "medium"}}
    for rng_seed in range(20):
        proverb = build_story_prompt(region, rng_seed=rng_seed).elements["proverb"]
        assert proverb is None or proverb in allowed


def test_prompt_is_deterministic_for_a_seed():
    assert build_story_prompt("egyptian", rng_seed=7).user == build_story_prompt("egyptian", rng_seed=7).user
    assert build_story_prompt("egyptian", rng_seed=7).user != build_story_prompt("egyptian", rng_seed=8).user


def test_premise_is_included():
    prompt = build_story_prompt("levantine", premise="راعٍ يجد محفظة ضائعة")
    assert "راعٍ يجد محفظة ضائعة" in prompt.user


def test_unknown_region_is_rejected():
    with pytest.raises(SeedError):
        build_story_prompt("iraqi")
