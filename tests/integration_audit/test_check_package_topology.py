"""Tests for tools/check_package_topology.py (synthetic file trees only)."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_package_topology", ROOT / "tools" / "check_package_topology.py")
topo = importlib.util.module_from_spec(_spec)
sys.modules["check_package_topology"] = topo
_spec.loader.exec_module(topo)


def make_tree(tmp_path, files):
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# synthetic\n", encoding="utf-8")
    return str(tmp_path)


def kinds(findings):
    return sorted(f.kind for f in findings)


def test_clean_tree_has_no_findings(tmp_path):
    root = make_tree(tmp_path, ["src/ansg/__init__.py", "src/ansg/model.py",
                                "src/ansg/data/__init__.py", "src/ansg/data/records.py",
                                "tests/test_a.py", "tests/unit/test_b.py"])
    assert topo.check(topo.list_files(root)) == []


def test_module_package_collision_reproduces_cross_branch_merge(tmp_path):
    # Mirrors implementation/ansg-core-2026 + claude/relaxed-knuth-q63k61 combined.
    root = make_tree(tmp_path, ["src/ansg/__init__.py",
                                "src/ansg/data.py", "src/ansg/data/__init__.py",
                                "src/ansg/evaluation.py", "src/ansg/evaluation/__init__.py",
                                "src/ansg/knowledge.py", "src/ansg/knowledge/__init__.py"])
    found = topo.check(topo.list_files(root))
    assert kinds(found) == ["module-package"] * 3
    assert {f.paths[0] for f in found} == {"src/ansg/data.py", "src/ansg/evaluation.py", "src/ansg/knowledge.py"}


def test_nested_collision_and_missing_init(tmp_path):
    root = make_tree(tmp_path, ["src/ansg/__init__.py", "src/ansg/models/__init__.py",
                                "src/ansg/models/csam.py", "src/ansg/models/csam/__init__.py",
                                "src/ansg/orphan/helper.py"])
    assert kinds(topo.check(topo.list_files(root))) == ["missing-init", "module-package"]


def test_duplicate_test_basenames_without_init(tmp_path):
    root = make_tree(tmp_path, ["tests/test_core.py", "tests/unit/test_core.py"])
    found = topo.check(topo.list_files(root))
    assert kinds(found) == ["duplicate-test-basename"]
    # Adding __init__.py makes them importable under distinct package names.
    make_tree(tmp_path, ["tests/unit/__init__.py"])
    assert topo.check(topo.list_files(root)) == []


def test_caches_and_egg_info_are_ignored(tmp_path):
    root = make_tree(tmp_path, ["src/ansg/__init__.py", "src/ansg/data.py",
                                "src/ansg/data/__pycache__/x.cpython-311.pyc",
                                "src/ansg.egg-info/PKG-INFO"])
    assert topo.check(topo.list_files(root)) == []


@pytest.mark.parametrize("bad,code", [(False, 0), (True, 1)])
def test_cli_exit_status_and_json(tmp_path, bad, code):
    files = ["src/ansg/__init__.py", "src/ansg/data/__init__.py"] + (["src/ansg/data.py"] if bad else [])
    root = make_tree(tmp_path, files)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_package_topology.py"),
                        "--root", root, "--json"], capture_output=True, text=True)
    assert r.returncode == code
    assert len(json.loads(r.stdout)) == (1 if bad else 0)


def test_this_repository_has_no_topology_findings():
    assert topo.check(topo.list_files(str(ROOT))) == []
