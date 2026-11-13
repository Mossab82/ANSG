import unicodedata

import pytest

from ansg.data.normalization import NormalizationConfig, normalize, simple_tokenize


def test_default_is_non_destructive_for_diacritics_and_orthography():
    raw = "قَالَتْ أَنَّ الْمَلِكَ إِلَى مَدِينَةٍ"
    out = normalize(raw)
    assert out.text == unicodedata.normalize("NFC", raw)
    assert "أ" in out.text and "إ" in out.text and "ة" in out.text and "ى" in out.text


def test_tatweel_and_invisible_controls_removed_by_default():
    out = normalize("حطـــاب‏ فقير")
    assert out.text == "حطاب فقير"
    assert out.operations["tatweel_removed"] == 3
    assert out.operations["invisible_removed"] == 1


def test_presentation_forms_folded():
    out = normalize("ﻻ")  # LAM WITH ALEF ligature
    assert out.text == "لا"
    assert out.offset_map == [0, 0, 1]


def test_opt_in_lossy_operations():
    cfg = NormalizationConfig(strip_diacritics=True, normalize_alef=True,
                              normalize_alef_maqsura=True, normalize_teh_marbuta=True)
    assert normalize("إِلَى مَدِينَةٍ", cfg).text == "الي مدينه"


def test_legacy_profile_strips_diacritics_only():
    assert normalize("قَالَـتْ", NormalizationConfig.legacy()).text == "قالـت"


def test_digits():
    assert normalize("الليلة ١٢١", NormalizationConfig(digits="western")).text == "الليلة 121"
    assert normalize("121", NormalizationConfig(digits="arabic_indic")).text == "١٢١"


def test_whitespace_collapse_and_newline_cap():
    out = normalize("  أ \t ب  \n\n\n\nج  ")
    assert out.text == "أ ب\n\nج"


@pytest.mark.parametrize("raw", ["حطـــاب‏ فقير", "قَالَتْ  \n\n\n شهرزاد", "ﻻ يمل"])
def test_offset_map_projects_spans_back_to_raw(raw):
    out = normalize(raw, NormalizationConfig(strip_diacritics=True))
    assert len(out.offset_map) == len(out.text) + 1
    assert out.offset_map == sorted(out.offset_map)
    for s, e, tok in simple_tokenize(out.text):
        rs, re_ = out.to_raw_span(s, e)
        raw_tok = normalize(raw[rs:re_], NormalizationConfig(strip_diacritics=True)).text
        assert raw_tok == tok


def test_from_raw_span_roundtrip():
    raw = "حطـــاب فقير"
    out = normalize(raw)
    s, e = out.from_raw_span(raw.index("فقير"), len(raw))
    assert out.text[s:e] == "فقير"


def test_unknown_option_rejected():
    with pytest.raises(ValueError):
        NormalizationConfig.from_mapping({"strip_everything": True})
