"""YAML configuration loading and secure environment-variable access.

Rules enforced here:

* Secrets are **never** read from YAML. A config value of the form
  ``${env:NAME}`` is resolved from the process environment at load time, and
  keys that look like credentials must use that indirection.
* ``.env`` files are not loaded implicitly; callers export variables or use a
  secret manager. (Implicit loading makes it too easy to commit a populated
  ``.env``.)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_REF = re.compile(r"^\$\{env:([A-Z_][A-Z0-9_]*)\}$")
_SECRET_KEY = re.compile(r"(password|passwd|secret|token|api[_-]?key|credential)", re.I)


class ConfigError(ValueError):
    """Raised for malformed or insecure configuration."""


def _resolve(node: Any, path: str) -> Any:
    if isinstance(node, Mapping):
        out: dict[str, Any] = {}
        for k, v in node.items():
            key_path = f"{path}.{k}" if path else str(k)
            if _SECRET_KEY.search(str(k)) and isinstance(v, str) and v and not _ENV_REF.match(v):
                raise ConfigError(
                    f"Config key '{key_path}' looks like a credential; use '${{env:NAME}}' "
                    "instead of a literal value."
                )
            out[str(k)] = _resolve(v, key_path)
        return out
    if isinstance(node, list):
        return [_resolve(v, f"{path}[{i}]") for i, v in enumerate(node)]
    if isinstance(node, str):
        m = _ENV_REF.match(node)
        if m:
            return os.environ.get(m.group(1))
    return node


def load_yaml_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a YAML file, resolve ``${env:NAME}`` references, reject literal secrets.

    Relative paths inside the config are left untouched; use
    :func:`resolve_path` with the config's directory to resolve them.
    """
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Top level of {path} must be a mapping")
    return _resolve(raw, "")


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error (value never logged)."""
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"Required environment variable {name} is not set")
    return value


def resolve_path(value: str | os.PathLike[str], base_dir: str | os.PathLike[str]) -> Path:
    """Resolve ``value`` relative to ``base_dir`` unless it is absolute."""
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (Path(base_dir) / p).resolve()
