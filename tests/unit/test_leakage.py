"""Leakage audits for frame-tale corpora (synthetic data only)."""
import pytest

from ansg.data.leakage import ngram_overlap, night_split_crossings, records_night_split_audit, story_cycle_groups
from ansg.data.nights import ingest_file
from ansg.data.records import CopyrightStatus, NarrativeMetadata, NarrativeRecord, SourceProvenance

PROV = SourceProvenance(source_id="synth", title="synthetic", copyright_status=CopyrightStatus.PUBLIC_DOMAIN,
                        redistributable=True)


def test_night_level_split_cuts_tales():
    # Shaped like a frame-tale collection: a tale continues over several nights.
    tales = {"tale A": [1, 2], "tale B": [2, 3, 4, 5], "tale C": [5, 6]}
    split = {n: ("train" if n <= 4 else "test") for n in range(1, 7)}
    crossings = night_split_crossings(tales, split)
    assert crossings == {"tale B": {"train": [2, 3, 4], "test": [5]}}
    assert night_split_crossings({"x": [7]}, split) == {}
    assert night_split_crossings({"x": [4, 9]}, split) == {"x": {"train": [4], "unassigned": [9]}}


def test_audit_from_ingested_records(fixtures):
    records, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    tales = [r for r in records if r.narrative.title]
    # the synthetic tale 2 spans nights 11 and 121: a split between them leaks
    report = records_night_split_audit(tales, {1: "train", 2: "train", 11: "train", 121: "test"})
    assert report["tales_crossing_splits"] == 1 and report["verdict"].startswith("LEAKAGE")


def _rec(rid, parent=None):
    return NarrativeRecord(record_id=rid, text="نص", source=PROV,
                           narrative=NarrativeMetadata(parent_story_id=parent))


def test_story_cycle_groups_follow_embedding():
    recs = [_rec("frame"), _rec("inner", "frame"), _rec("inner2", "inner"), _rec("other")]
    assert story_cycle_groups(recs) == {"frame": "frame", "inner": "frame", "inner2": "frame", "other": "other"}
    with pytest.raises(ValueError):
        story_cycle_groups([_rec("a", "b"), _rec("b", "a")])


def test_ngram_overlap_and_formula_masking():
    formula = "بلغني أيها الملك السعيد"
    train = [f"قالت {formula} أن رجلا خرج إلى السوق فاشترى خبزا"]
    test = {"t1": f"قالت {formula} أن امرأة ركبت السفينة نحو الجزيرة البعيدة"}
    raw = ngram_overlap(train, test, n=3)
    masked = ngram_overlap(train, test, n=3, mask_patterns=[formula])
    assert raw.per_document[0]["in_train"] > 0      # shared frame formula inflates overlap
    assert masked.per_document[0]["in_train"] == 0  # ... and only the formula was shared
    assert masked.masked_patterns == 1
    with pytest.raises(ValueError):
        ngram_overlap(train, test, n=0)
