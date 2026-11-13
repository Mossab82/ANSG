#!/usr/bin/env python
"""Detect Python package-topology collisions that ``git merge`` cannot see.

Why this exists (Issue #3, ``docs/INTEGRATION_COMPATIBILITY_AUDIT.md`` §2):
``implementation/ansg-core-2026`` adds the modules ``src/ansg/data.py``,
``evaluation.py`` and ``knowledge.py``; ``claude/relaxed-knuth-q63k61`` adds
packages ``src/ansg/data/``, ``evaluation/`` and ``knowledge/``. The paths
differ, so git merges them without a conflict, but Python resolves
``import ansg.data`` to the *package* and the module becomes unreachable.

Checks (stdlib only, no imports of the audited code):

* ``module-package``  — ``x.py`` next to a package directory ``x/``.
* ``missing-init``     — a directory with ``.py`` files under a package root
  but no ``__init__.py`` (would become an implicit namespace package).
* ``duplicate-test-basename`` — two test files with the same basename in
  directories without ``__init__.py``; pytest's default ``prepend`` import
  mode refuses to collect them.

Usage::

    python tools/check_package_topology.py                 # git-tracked files of this repo
    python tools/check_package_topology.py --root PATH     # any tree (git-tracked if a repo)
    python tools/check_package_topology.py --json

Exit status 1 if any finding is reported, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

IGNORED_PARTS = {"__pycache__", ".git", ".venv", "venv", "build", "dist", ".pytest_cache"}


@dataclass(frozen=True)
class Finding:
    kind: str
    paths: tuple[str, ...]
    message: str


def _relevant(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return not any(p in IGNORED_PARTS or p.endswith(".egg-info") for p in parts)


def list_files(root: str) -> list[str]:
    """Git-tracked files under ``root`` (POSIX, relative); falls back to a walk."""
    try:
        out = subprocess.run(["git", "-C", root, "ls-files", "-z"], capture_output=True, check=True)
        files = [f for f in out.stdout.decode("utf-8").split("\0") if f]
        if files:
            return sorted(f for f in files if _relevant(f))
    except (OSError, subprocess.CalledProcessError):
        pass
    import os

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_PARTS and not d.endswith(".egg-info")]
        for name in filenames:
            rel = os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/")
            found.append(rel)
    return sorted(f for f in found if _relevant(f))


def check(files: list[str], package_roots: tuple[str, ...] = ("src",), test_roots: tuple[str, ...] = ("tests",)) -> list[Finding]:
    paths = [PurePosixPath(f) for f in files]
    fileset = {str(p) for p in paths}
    findings: list[Finding] = []

    # Directories that contain an __init__.py are packages.
    packages = {str(p.parent) for p in paths if p.name == "__init__.py"}

    # module-package collisions anywhere under the package roots.
    for p in paths:
        if p.suffix != ".py" or p.name == "__init__.py":
            continue
        if not any(str(p).startswith(r.rstrip("/") + "/") for r in package_roots):
            continue
        sibling_pkg = str(p.with_suffix(""))
        if sibling_pkg in packages:
            findings.append(Finding(
                "module-package", (str(p), sibling_pkg + "/__init__.py"),
                f"'{p.with_suffix('').name}' is both a module and a package; "
                f"imports resolve to the package and {p} becomes unreachable",
            ))

    # Directories with .py files under a package root but without __init__.py.
    py_dirs = {str(p.parent) for p in paths if p.suffix == ".py"}
    for d in sorted(py_dirs):
        for r in package_roots:
            r = r.rstrip("/")
            if d.startswith(r + "/") and d not in packages:
                findings.append(Finding("missing-init", (d,),
                                        f"{d} has Python files but no __init__.py"))

    # Duplicate test basenames in rootless (no __init__.py) test directories.
    by_name: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        if p.suffix != ".py" or not p.name.startswith("test"):
            continue
        if not any(str(p).startswith(r.rstrip("/") + "/") for r in test_roots):
            continue
        if str(p.parent / "__init__.py") in fileset:
            continue
        by_name[p.name].append(str(p))
    for name, where in sorted(by_name.items()):
        if len(where) > 1:
            findings.append(Finding("duplicate-test-basename", tuple(sorted(where)),
                                    f"pytest (prepend import mode) cannot collect two '{name}' modules"))
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--package-root", action="append", default=None)
    ap.add_argument("--test-root", action="append", default=None)
    args = ap.parse_args(argv)
    findings = check(list_files(args.root), tuple(args.package_root or ["src"]),
                     tuple(args.test_root or ["tests"]))
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2, ensure_ascii=False))
    else:
        for f in findings:
            print(f"[{f.kind}] {f.message}\n    " + "\n    ".join(f.paths))
        print(f"{len(findings)} finding(s) in {args.root}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
