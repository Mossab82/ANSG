"""Shared authorization checks for real-data ANSG training."""
import datetime as dt
import hashlib
import json
import os
import re


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_real_data_authorization(manifest_path, examples_path, vocabulary_path):
    """Require approved use, source hashes, and exact derived-file paths and hashes."""
    if not manifest_path:
        raise ValueError("authorization manifest is required when data_kind=real")
    try:
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read authorization manifest: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError("authorization manifest root must be a JSON object")
    if manifest.get("schema") != "ansg.training_authorization/1":
        raise ValueError("authorization manifest schema must be ansg.training_authorization/1")
    if manifest.get("review_status") != "approved" or manifest.get("training_authorized") is not True:
        raise ValueError("real-data training requires explicit approved status and training_authorized=true")
    if not manifest.get("approver") or not isinstance(manifest.get("approved_at"), str):
        raise ValueError("authorization manifest must record approver and approved_at")
    try:
        dt.datetime.fromisoformat(manifest["approved_at"].replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("approved_at must be an ISO-8601 timestamp") from exc

    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("authorization manifest must list source works")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("each source authorization entry must be an object")
        source_path = source.get("path")
        source_hash = source.get("sha256", "")
        if (source.get("training_allowed") is not True or not source.get("license")
                or not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", source_hash)
                or not source_path or not os.path.isfile(source_path)):
            raise ValueError("each source must record an existing file path, license, SHA-256, and training_allowed=true")
        if sha256_file(source_path) != source_hash.lower():
            raise ValueError(f"source SHA-256 does not match: {source_path}")

    derived = manifest.get("derived_files")
    if not isinstance(derived, list):
        raise ValueError("authorization manifest must list derived_files")
    by_role = {}
    for item in derived:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str):
            raise ValueError("each derived file authorization entry must have a role")
        role = item["role"]
        if role in by_role:
            raise ValueError(f"duplicate derived file authorization role: {role}")
        by_role[role] = item

    for role, path in (("examples", examples_path), ("schema_vocabulary", vocabulary_path)):
        item = by_role.get(role)
        if not item or not isinstance(item.get("path"), str):
            raise ValueError(f"authorization manifest is missing the {role} derived file")
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError(f"authorization manifest has a missing or malformed {role} SHA-256")
        expected_path = os.path.realpath(item["path"])
        actual_path = os.path.realpath(path)
        if expected_path != actual_path:
            raise ValueError(f"authorization manifest {role} path does not match the configured input")
        if not os.path.isfile(actual_path):
            raise ValueError(f"authorization manifest {role} file does not exist: {actual_path}")
        if sha256_file(actual_path) != expected_hash.lower():
            raise ValueError(f"authorization manifest {role} SHA-256 does not match file")
    return sha256_file(manifest_path)
