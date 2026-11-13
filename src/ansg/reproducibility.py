"""Independent experiment provenance and thesis comparison utilities.

No model training or evaluation metrics are implemented here. Never label
user-supplied or synthetic measurements as historical thesis evidence.
Python stdlib only; safe to import without PyTorch or data access.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


VALID_KINDS = frozenset({"synthetic", "real", "not_run"})


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of bytes, streaming large checkpoints."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_disjoint_splits(splits: Mapping[str, Sequence[str]]) -> dict[str, int]:
    """Reject record-ID leakage between partitions; reject duplicate IDs.

    IDs must represent canonical stories/clusters consistently across splits.
    Content-near-duplicate leakage still requires separate corpus-level checks.
    """
    if not splits:
        raise ValueError("At least one split is required")
    global_ids: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split, values in splits.items():
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unrecognized split: {split!r}")
        ids = list(values)
        if not all(isinstance(item, str) and item.strip() for item in ids):
            raise ValueError("All record IDs must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate record ID within {split}")
        for item in ids:
            if item in global_ids:
                raise ValueError(f"Split leakage: record {item!r} in {global_ids[item]} and {split}")
            global_ids[item] = split
        counts[split] = len(ids)
    return counts


def create_manifest(
    *,
    run_id: str,
    git_sha: str,
    configuration: Mapping[str, object],
    seed: int,
    data_files: Mapping[str, str | Path],
    checkpoint_file: str | Path | None = None,
    kind: str = "synthetic",
    timestamp: str | None = None,
) -> dict:
    """Compute provenance from files *actually present* on disk.

    The caller supplies the Git SHA. This utility does not assert that the
    working tree is clean; record that separately in run metadata if needed.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Non-empty run_id is required")
    if not isinstance(git_sha, str) or len(git_sha) != 40 or any(c not in "0123456789abcdefABCDEF" for c in git_sha):
        raise ValueError("git_sha must be a full 40-character hexadecimal commit SHA")
    if kind not in VALID_KINDS or kind == "not_run":
        raise ValueError("A recorded run must be classified as synthetic or real")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(configuration, Mapping):
        raise ValueError("configuration must be a mapping")
    if not data_files:
        raise ValueError("At least one data file is required")
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("timestamp is required")
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "timestamp_utc": timestamp,
        "git_sha": git_sha.lower(),
        "seed": seed,
        "configuration": dict(configuration),
        "data_sha256": {name: sha256_file(path) for name, path in sorted(data_files.items())},
        "checkpoint_sha256": sha256_file(checkpoint_file) if checkpoint_file is not None else None,
    }
    # Fail before writing if config contains non-JSON values or NaNs.
    json.dumps(manifest, allow_nan=False, sort_keys=True)
    return manifest


def save_manifest(manifest: Mapping[str, object], path: str | Path) -> None:
    """Create-only file save: a previously recorded run is never overwritten."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def compare_measurement(
    *, metric: str, thesis_value: float | None, measured_value: float | None,
    unit: str, thesis_reference: str, kind: str, source_run_id: str | None = None,
) -> dict:
    """Compare two numbers only when a genuine measurement is supplied.

    Caller owns the metric computation, dataset, methodology and validity.
    'not_run' can never masquerade as numeric evidence.
    """
    if not metric.strip() or not unit.strip() or not thesis_reference.strip():
        raise ValueError("Metric, unit and thesis_reference are required")
    if kind not in VALID_KINDS:
        raise ValueError("Invalid measurement kind")
    if kind == "not_run":
        if measured_value is not None or source_run_id is not None:
            raise ValueError("A not_run result cannot contain measurements or a run id")
    else:
        if measured_value is None or not source_run_id:
            raise ValueError("Measured result requires a source run id")
    for name, value in (("thesis_value", thesis_value), ("measured_value", measured_value)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"{name} must be finite numeric data")
    delta = None if thesis_value is None or measured_value is None else float(measured_value - thesis_value)
    return {
        "metric": metric,
        "unit": unit,
        "thesis_reference": thesis_reference,
        "thesis_claim": thesis_value,
        "new_measurement": measured_value,
        "difference": delta,
        "measurement_kind": kind,
        "source_run_id": source_run_id,
    }
