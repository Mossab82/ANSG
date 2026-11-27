import pytest

from ansg.evaluation.seed_adherence import (
    distinctive_markers,
    lcs_recall,
    normalize,
    seed_adherence,
    tokens,
)
from ansg.generation.seed_prompt import build_story_prompt


def test_normalize_removes_diacritics_and_unifies_letters():
    assert normalize("رُبَّ رَمْيَةٍ مِنْ غَيْرِ رامٍ!") == "رب رميه من غير رام"
    assert normalize("أإآ ى ة") == "ااا ي ه"


def test_lcs_recall_tolerates_insertions_and_conjunction():
    window = tokens("قال الشيخ: وفي الصيف يا ولدي ضيعت اللبن")
    assert lcs_recall("في الصيف ضيعت اللبن", window) == 1.0
    assert lcs_recall("في الصيف ضيعت اللبن", tokens("ضيعت الماء")) == 0.25
    assert lcs_recall("", window) == 0.0


def test_distinctive_markers_are_region_specific():
    markers = distinctive_markers()
    assert "دابا" in markers["maghrebi"]
    assert "دلوقتي" in markers["egyptian"]
    for region, toks in markers.items():
        for other, other_toks in markers.items():
            if region != other:
                assert not toks & other_toks


def _story(prompt, dialogue):
    e = prompt.elements
    return (
        f"{e['opening']}. كان {e['character']} يجلس في {e['setting']}. "
        f"{dialogue} فقال له: {e['proverb']}. {e['closing']}"
    )


def test_full_adherence_scores_high():
    prompt = build_story_prompt("egyptian", rng_seed=1)
    story = _story(prompt, "قال الرجل: عايز إيه دلوقتي؟ مش عارف ليه.")
    result = seed_adherence(story, "egyptian", prompt.elements)
    assert result.proverb == 1.0 and result.opening == 1.0 and result.closing == 1.0
    assert result.dialect == 1.0 and result.target_marker_hits > 0
    assert result.score > 0.9


def test_wrong_dialect_lowers_dialect_component():
    prompt = build_story_prompt("egyptian", rng_seed=1)
    story = _story(prompt, "قال الرجل: شنو بغيتي دابا؟ واخا، بزاف.")
    result = seed_adherence(story, "egyptian", prompt.elements)
    assert result.other_marker_hits > 0
    assert result.dialect < 0.5


def test_missing_elements_score_low():
    prompt = build_story_prompt("levantine", rng_seed=2)
    result = seed_adherence("ذهب الولد إلى المدرسة وعاد إلى البيت.", "levantine", prompt.elements)
    assert result.score < 0.3


def test_msa_penalises_dialect_markers():
    prompt = build_story_prompt("msa", rng_seed=0)
    clean = seed_adherence(_story(prompt, ""), "msa", prompt.elements)
    mixed = seed_adherence(_story(prompt, "قال: عايز إيه دلوقتي؟"), "msa", prompt.elements)
    assert clean.dialect == 1.0
    assert mixed.dialect < 1.0


def test_unrequested_component_is_none_and_weights_renormalise():
    prompt = build_story_prompt("gulf", rng_seed=0)
    elements = dict(prompt.elements, proverb=None)
    result = seed_adherence(_story(prompt, "قال: وش تبي الحين؟"), "gulf", elements)
    assert result.proverb is None
    assert 0.0 <= result.score <= 1.0
