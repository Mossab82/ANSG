"""Synthetic end-to-end smoke test across data, knowledge and model layers (Issue #11)."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="model extra (PyTorch) not installed")
pytestmark = [pytest.mark.integration, pytest.mark.model]

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("e2e_smoke", ROOT / "scripts" / "e2e_smoke.py")
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)


def test_end_to_end_synthetic_pipeline(tmp_path):
    report = e2e.run(tmp_path)
    assert report["synthetic"] is True
    names = [s["step"] for s in report["steps"]]
    assert names == ["ingest", "activation_and_vocabulary", "unknowns_stay_unknown", "dataset_and_collate",
                     "forward_backward", "batch_invariance", "generation", "evaluation_separate"]
    assert all(s["ok"] for s in report["steps"])
    saved = json.loads((tmp_path / "e2e_report.json").read_text(encoding="utf-8"))
    assert saved["synthetic"] is True and "no dissertation result" in saved["notice"]
    # The tale without a synthetic trigger has no schema and no dialect; nothing is invented.
    act = report["steps"][1]["examples"]
    assert act[0]["schema_ids"] == [] and act[0]["dialect"] is None
    assert act[1]["dialect"] == "levantine" and act[1]["schema_ids"]
