#!/usr/bin/env python3
"""Block an ANSG pilot if the collected corpus or its research splits are invalid."""
import hashlib
import json
from pathlib import Path

# Split v2 (scripts/reproduction/derive_pilot_split.py --check keeps these in sync):
# v1 put kalila_dimna-019 in TEST although 98.5% of its 8-grams occur in TRAIN page 007.
BASE = Path("data/corpora/classical_arabic")
TRAIN = {"arabian_nights-001", "arabian_nights-002", "arabian_nights-003", "arabian_nights-004",
         "arabian_nights-005", "kalila_dimna-006", "kalila_dimna-007", "kalila_dimna-008",
         "kalila_dimna-009", "kalila_dimna-011", "kalila_dimna-012", "kalila_dimna-013",
         "kalila_dimna-014", "kalila_dimna-016", "kalila_dimna-019"}
VALIDATION = {"kalila_dimna-010", "kalila_dimna-017", "kalila_dimna-018"}
TEST = {"kalila_dimna-015", "kalila_dimna-020"}
EXCLUDED: set = set()  # split v2: Sinbad page included (0.49% 8-gram overlap); see scripts/reproduction/derive_pilot_split.py

def validate(path=BASE / "stories.jsonl"):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [r["id"] for r in records]
    assert len(ids) == 20 and len(set(ids)) == 20, "Corpus must contain exactly 20 unique source records"
    assert set(ids) == TRAIN | VALIDATION | TEST | EXCLUDED
    assert not (TRAIN & VALIDATION or TRAIN & TEST or VALIDATION & TEST)
    assert len([r for r in records if r["corpus"] == "arabian_nights"]) == 5
    assert len([r for r in records if r["corpus"] == "kalila_dimna"]) == 15
    for r in records:
        assert hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == r["sha256"], r["id"]
        assert len(r["text"].split()) == r["word_count"], r["id"]
        assert r["segment_status"] == "not_story_annotated", r["id"]
    assert sum(r["word_count"] for r in records) == 195197, "Corpus revision changed; reassess split"
    print(json.dumps({
        "status": "text_integrity_verified", "pages": len(ids),
        "train_pages": len(TRAIN), "validation_pages": len(VALIDATION),
        "test_pages": len(TEST), "excluded_pages": len(EXCLUDED),
        "train_words": sum(r["word_count"] for r in records if r["id"] in TRAIN),
        "validation_words": sum(r["word_count"] for r in records if r["id"] in VALIDATION),
        "test_words": sum(r["word_count"] for r in records if r["id"] in TEST),
        "scientific_warning": "Classical text pilot, NOT gold ANHC. Frame formulae may cross groups.",
        "license_warning": "Edition and site terms require review before redistribution and publication."
    }, ensure_ascii=False, indent=2))
    return records

if __name__ == "__main__":
    validate()
