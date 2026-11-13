import pytest

from ansg.evaluation.corpus_metrics import cultural_reference_density

LEX = ["الصبر مفتاح الفرج", "حكمة", "كرم", "إن شاء الله"]
TEXT = "قال الحكيم الصبر مفتاح الفرج ثم ذكر الكرم والحكمة"  # 9 tokens


def test_literal_eq_3_1_ignores_multiword_entries():
    # only single-token entries can match under the literal reading
    assert cultural_reference_density("حكمة كرم باب", LEX, "token_membership") == pytest.approx(2 / 3 * 1000)
    assert cultural_reference_density(TEXT, LEX, "token_membership") == 0.0


def test_multiword_policies_differ_and_are_explicit():
    occ = cultural_reference_density(TEXT, LEX, "marker_occurrences")
    cov = cultural_reference_density(TEXT, LEX, "tokens_covered")
    assert occ == pytest.approx(1 / 9 * 1000)
    assert cov == pytest.approx(3 / 9 * 1000)
    with pytest.raises(ValueError):
        cultural_reference_density(TEXT, LEX, "guess")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cultural_reference_density("", LEX, "token_membership")
