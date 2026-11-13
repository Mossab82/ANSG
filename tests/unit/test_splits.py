import pytest

from ansg.data.records import NarrativeRecord, SourceProvenance
from ansg.data.splits import SplitConfig, assign_splits, check_no_leakage


def recs(n, parent=None):
    out = []
    for i in range(n):
        r = NarrativeRecord(record_id=f"r{i:03d}", text="x", source=SourceProvenance("s", "t"))
        if parent:
            r.narrative.parent_story_id = parent(i)
        out.append(r)
    return out


def test_hash_exact_proportions_and_determinism():
    a = assign_splits(recs(100), SplitConfig())
    b = assign_splits(list(reversed(recs(100))), SplitConfig())
    assert a == b  # independent of input order
    counts = {k: list(a.values()).count(k) for k in set(a.values())}
    assert counts == {"train": 80, "validation": 10, "test": 10}


def test_seed_changes_assignment():
    assert assign_splits(recs(50), SplitConfig(seed=0)) != assign_splits(recs(50), SplitConfig(seed=1))


def test_group_level_no_leakage():
    rs = recs(60, parent=lambda i: f"g{i // 6}")
    assign_splits(rs, SplitConfig(group_by="parent_story_id"))
    assert check_no_leakage(rs, lambda r: r.narrative.parent_story_id) == []


def test_hash_bucket_stable_under_additions():
    small = assign_splits(recs(30), SplitConfig(strategy="hash_bucket"))
    big = assign_splits(recs(60), SplitConfig(strategy="hash_bucket"))
    assert all(big[k] == v for k, v in small.items())


def test_predefined():
    rs = recs(3)
    cfg = SplitConfig(strategy="predefined", predefined={"r000": "test", "r001": "train", "r002": "train"})
    assert assign_splits(rs, cfg) == {"r000": "test", "r001": "train", "r002": "train"}
    with pytest.raises(KeyError):
        assign_splits(recs(4), cfg)


def test_ratio_validation():
    with pytest.raises(ValueError):
        SplitConfig(ratios={"train": 0.5, "test": 0.4})
