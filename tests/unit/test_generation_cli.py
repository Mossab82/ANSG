"""CLI contract for carrying story-plan dialects into DAL generation."""

import pytest

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
