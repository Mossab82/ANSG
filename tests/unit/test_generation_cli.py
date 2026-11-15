"""CLI contract for carrying story-plan dialects into DAL generation."""

import sys
import types

import pytest

from ansg.generation import cli
from ansg.generation.cli import _resolve_plan_dialect


def test_plan_dialect_is_used_when_cli_dialect_is_omitted():
    plan = {"dal_dialect": "Levantine"}

    assert _resolve_plan_dialect(plan, None) == "Levantine"


def test_equivalent_cli_dialect_alias_is_accepted():
    plan = {"dal_dialect": "Egyptian"}

    assert _resolve_plan_dialect(plan, "egyptian") == "Egyptian"


def test_conflicting_cli_dialect_is_rejected():
    plan = {"dal_dialect": "Levantine"}

    with pytest.raises(ValueError, match="conflicts with story plan dialect"):
        _resolve_plan_dialect(plan, "Gulf")


def test_unknown_cli_dialect_is_rejected():
    plan = {"dal_dialect": "Levantine"}

    with pytest.raises(ValueError, match="invalid story-plan/CLI dialect"):
        _resolve_plan_dialect(plan, "Iraqi")


def test_plan_without_dal_dialect_is_rejected():
    with pytest.raises(ValueError, match="story plan setting has no DAL dialect"):
        _resolve_plan_dialect({}, None)


def _install_cli_fakes(monkeypatch, captured):
    aliases = {
        "MSA": ("msa", 0), "msa": ("msa", 0),
        "Egyptian": ("egyptian", 1), "egyptian": ("egyptian", 1),
        "Levantine": ("levantine", 2), "levantine": ("levantine", 2),
        "Gulf": ("gulf", 3), "gulf": ("gulf", 3),
        "Maghrebi": ("maghrebi", 4), "maghrebi": ("maghrebi", 4),
    }

    def dialect_code(label):
        if label not in aliases:
            raise ValueError(f"unknown dialect label {label!r}")
        return aliases[label]

    training_examples = types.ModuleType("ansg.data.training_examples")
    training_examples.dialect_code = dialect_code
    monkeypatch.setitem(sys.modules, "ansg.data.training_examples", training_examples)

    class GenerationRequest:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class SamplingConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def run_generation(request):
        captured.append(request)
        return {
            "label": {"mode": "test", "banner": "test", "warnings": []},
            "samples": [{
                "index": 0, "seed": 42, "new_tokens": 1,
                "finish_reason": "length", "text": "قصة اختبار",
            }],
        }

    runtime = types.ModuleType("ansg.generation.runtime")
    runtime.GenerationError = type("GenerationError", (Exception,), {})
    runtime.GenerationRequest = GenerationRequest
    runtime.SamplingConfig = SamplingConfig
    runtime.run_generation = run_generation
    monkeypatch.setitem(sys.modules, "ansg.generation.runtime", runtime)
    monkeypatch.setattr(cli, "_utf8_streams", lambda: None)
    monkeypatch.setattr(cli, "_prompt_from_plan", lambda path, settings: (
        "اكتب قصة", {"dal_dialect": "Levantine", "setting": "aqaba"}
    ))


def test_cli_passes_plan_dialect_to_generation_request(monkeypatch):
    captured = []
    _install_cli_fakes(monkeypatch, captured)

    assert cli.main(["--plan", "plan.json", "--backbone", "/local/model"]) == 0

    assert len(captured) == 1
    assert captured[0].dialect == "Levantine"


def test_cli_accepts_matching_dialect_alias_for_plan(monkeypatch):
    captured = []
    _install_cli_fakes(monkeypatch, captured)

    assert cli.main([
        "--plan", "plan.json", "--backbone", "/local/model", "--dialect", "levantine"
    ]) == 0

    assert len(captured) == 1
    assert captured[0].dialect == "Levantine"


def test_cli_rejects_dialect_mismatch_before_generation(monkeypatch, capsys):
    captured = []
    _install_cli_fakes(monkeypatch, captured)

    assert cli.main([
        "--plan", "plan.json", "--backbone", "/local/model", "--dialect", "Gulf"
    ]) == 2

    assert captured == []
    assert "conflicts with story plan dialect" in capsys.readouterr().err
