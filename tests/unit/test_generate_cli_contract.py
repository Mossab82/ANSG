"""``ansg-generate`` command-line contract that holds without PyTorch (Issue #15)."""
import sys

import pytest

from ansg.generation import cli


def test_prompt_and_backbone_are_required(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--prompt", "كان يا ما كان"])
    with pytest.raises(SystemExit):
        cli.main(["--backbone", "x"])
    with pytest.raises(SystemExit):
        cli.main(["--backbone", "x", "--prompt", "a", "--prompt-file", "b"])


def test_existing_output_is_never_overwritten(tmp_path, capsys):
    out = tmp_path / "gen.json"
    out.write_text("{}")
    assert cli.main(["--backbone", str(tmp_path), "--prompt", "كان", "--output", str(out)]) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert out.read_text() == "{}"


def test_unreadable_prompt_file(tmp_path, capsys):
    assert cli.main(["--backbone", str(tmp_path), "--prompt-file", str(tmp_path / "missing.txt")]) == 2


def test_missing_model_extra_exits_3(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "ansg.generation.runtime", None)  # simulate a data-only install
    assert cli.main(["--backbone", str(tmp_path), "--prompt", "كان يا ما كان"]) == 3
    assert "pretrained" in capsys.readouterr().err


def test_cli_module_does_not_import_torch():
    import subprocess
    code = "import sys, ansg.generation.cli, ansg.generation.readiness, ansg.generation.stub; " \
           "sys.exit(1 if 'torch' in sys.modules or 'transformers' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
