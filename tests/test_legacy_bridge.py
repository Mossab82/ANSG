from ansg.data.legacy_bridge import normalize_arabic, legacy_sql_row_to_record, NarrativeRecord
import pytest


def test_normalize():
    assert normalize_arabic("سَلامــــ  عليكم", remove_diacritics=True) == "سلام عليكم"


def test_legacy_row_does_not_fake_annotations():
    record = legacy_sql_row_to_record({"Story_ID": 1, "Story_Title": "مثل", "Story_Text": "كان يا ما كان"})
    assert record.story_id == "1" and record.dialect is None
    assert record.motif_ids == () and record.moral_labels == ()


def test_invalid_dialect():
    with pytest.raises(ValueError):
        NarrativeRecord("x", "قصة", dialect="unknown")
