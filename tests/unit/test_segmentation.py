from ansg.data.segmentation import SegmentationConfig, segment_paragraphs, segment_sentences


def test_paragraphs_with_offsets():
    text = "أ ب.\n\nج د؟"
    paras = segment_paragraphs(text)
    assert [p.text for p in paras] == ["أ ب.", "ج د؟"]
    assert all(text[p.start:p.end] == p.text for p in paras)


def test_sentences_arabic_terminators_and_decimal():
    text = "خرج الحطاب. هل رأى الطائر؟ نعم! وزنه 3.5 رطل؛ ثم عاد"
    sents = [s.text for s in segment_sentences(text)]
    assert sents == ["خرج الحطاب.", "هل رأى الطائر؟", "نعم!", "وزنه 3.5 رطل؛", "ثم عاد"]


def test_arabic_comma_optional():
    text = "قال الملك، ثم سكت."
    assert len(segment_sentences(text)) == 1
    assert len(segment_sentences(text, SegmentationConfig(split_on_arabic_comma=True))) == 2


def test_sentence_offsets_respect_offset_argument():
    text = "أ. ب."
    sents = segment_sentences(text, offset=10)
    assert [(s.start, s.end) for s in sents] == [(10, 12), (13, 15)]
