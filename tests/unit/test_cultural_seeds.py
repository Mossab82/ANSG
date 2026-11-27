import json
import re
from pathlib import Path

import pytest

SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "cultural_seeds"
ARABIC = re.compile(r"[؀-ۿ]")
REGIONS = {"msa", "egyptian", "levantine", "gulf", "maghrebi"}
MIN_COUNTS = {
    "values": 5,
    "proverbs": 5,
    "dialect_markers": 10,
    "characters": 3,
    "settings": 3,
    "openings_closings": 2,
}

SEED_FILES = sorted(SEED_DIR.glob("*.json"))


def test_at_least_one_seed_file_exists():
    assert SEED_FILES, "no cultural seed files found"


@pytest.mark.parametrize("path", SEED_FILES, ids=lambda p: p.stem)
def test_seed_file_schema(path):
    seed = json.loads(path.read_text(encoding="utf-8"))
    assert seed["region"] == path.stem
    assert seed["region"] in REGIONS
    assert seed["provenance"]["review_status"] in {"pending_native_review", "native_reviewed"}
    for key, minimum in MIN_COUNTS.items():
        assert len(seed[key]) >= minimum, f"{key} has fewer than {minimum} items"

    for value in seed["values"]:
        assert ARABIC.search(value["concept_ar"]) and value["concept_en"] and value["story_role"]
    for proverb in seed["proverbs"]:
        assert ARABIC.search(proverb["text"])
        assert proverb["confidence"] in {"high", "medium", "low"}
    for marker in seed["dialect_markers"]:
        assert ARABIC.search(marker["msa"]) and ARABIC.search(marker["dialect"])
    for item in seed["openings_closings"]:
        assert item["type"] in {"opening", "closing"}
        assert ARABIC.search(item["text"])
    assert all(ARABIC.search(c) for c in seed["characters"] + seed["settings"])
