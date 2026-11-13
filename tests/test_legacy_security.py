"""Regression checks for exposed credentials in tracked legacy source files."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEGACY_FILES = ["Main.py", "preprocess.py", "Verb-net.py", "wiki.py"]


def test_legacy_files_have_no_literal_passwords_or_api_tokens():
    literals = re.compile(
        r"(?i)\\b(?:password|api_key|api_token|secret|access_token)\\s*=\\s*(['\"])(?!\\s*\\1)[^'\"]{6,}\\1"
    )
    for path in LEGACY_FILES:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert not literals.search(source), f"Hard-coded credential in {path}"


def test_legacy_access_requires_environment_variables():
    for path in ["Main.py", "preprocess.py", "Verb-net.py"]:
        source = (ROOT / path).read_text(encoding="utf-8")
        assert 'ANSG_MYSQL_PASSWORD' in source
