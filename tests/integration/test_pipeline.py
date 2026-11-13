import json
import shutil

import pytest

from ansg.data.anhc import read_jsonl
from ansg.data.pipeline import MissingSourceError, main, run_pipeline
from ansg.utils.config import load_yaml_config

pytestmark = pytest.mark.integration


def _config(tmp_path, fixtures, **over):
    shutil.copy(fixtures / "nights" / "synthetic_nights.txt", tmp_path / "src.txt")
    cfg = load_yaml_config("configs/data/nights.yaml")
    cfg["sources"][0]["path"] = "src.txt"
    cfg["sources"][0]["provenance"].update(source_id="synth", copyright_status="public_domain",
                                           redistributable=True)
    cfg["output_dir"] = "out"
    cfg["splits"]["ratios"] = {"train": 0.34, "validation": 0.33, "test": 0.33}
    cfg.update(over)
    return cfg


def test_end_to_end(tmp_path, fixtures):
    result = run_pipeline(_config(tmp_path, fixtures), tmp_path)
    out = tmp_path / "out"
    assert result["stats"]["n_records"] == 3
    assert sorted(result["stats"]["split_counts"].values()) == [1, 1, 1]
    recs = [r for s in ("train", "validation", "test") for r in read_jsonl(out / f"{s}.jsonl")]
    assert {r.split for r in recs} == {"train", "validation", "test"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["inputs"] and manifest["outputs"] and manifest["config_sha256"]
    assert manifest["stats"]["annotation_status_counts"]["dia"]["unannotated"] == 3


def test_outputs_are_reproducible(tmp_path, fixtures):
    cfg = _config(tmp_path, fixtures)
    run_pipeline(cfg, tmp_path)
    first = {p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.json*") if p.name != "manifest.json"}
    shutil.rmtree(tmp_path / "out")
    run_pipeline(cfg, tmp_path)
    second = {p.name: p.read_bytes() for p in (tmp_path / "out").glob("*.json*") if p.name != "manifest.json"}
    assert first == second


def test_missing_source_fails_loudly(tmp_path, fixtures):
    cfg = _config(tmp_path, fixtures)
    cfg["sources"][0]["path"] = "does_not_exist.txt"
    with pytest.raises(MissingSourceError):
        run_pipeline(cfg, tmp_path)


def test_cli_with_shipped_config_reports_missing_corpus(capsys):
    assert main(["--config", "configs/data/nights.yaml"]) == 2
    assert "does not bundle corpus texts" in capsys.readouterr().err
