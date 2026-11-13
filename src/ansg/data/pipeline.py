"""Configurable corpus ingestion pipeline (CLI: ``ansg-ingest --config <yaml>``).

Steps: read supplied source files -> normalise (offset-tracked) -> identify
tales and nights -> build ANHC records -> optionally import supplied
annotation files -> validate -> split deterministically -> write JSONL per
split together with a run manifest.

The pipeline fails loudly if a configured source file does not exist; it never
substitutes sample or synthetic text for a missing corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..utils.config import load_yaml_config, resolve_path
from ..utils.provenance import RunManifest
from .anhc import import_annotations, write_jsonl
from .nights import NightsConfig, ingest_file
from .normalization import NormalizationConfig
from .records import AnnotationStatus, LabelInventory, SourceProvenance, validate_record
from .splits import SplitConfig, assign_splits


class MissingSourceError(FileNotFoundError):
    pass


def run_pipeline(config: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    norm_cfg = NormalizationConfig.from_mapping(config.get("normalization"))
    nights_cfg = NightsConfig.from_mapping(config.get("nights"))
    split_section = dict(config.get("splits") or {})
    predefined_file = split_section.pop("predefined_file", None)
    if predefined_file:
        with open(resolve_path(predefined_file, base_dir), encoding="utf-8") as fh:
            split_section["predefined"] = json.load(fh)
        split_section.setdefault("strategy", "predefined")
    split_cfg = SplitConfig.from_mapping(split_section)
    labels = LabelInventory.from_mapping(config.get("label_inventories"))
    out_dir = resolve_path(config.get("output_dir", "data/processed"), base_dir)
    allow_restricted = bool(config.get("allow_restricted_text", False))

    manifest = RunManifest(run_type="corpus_ingestion", config=dict(config))
    sources = config.get("sources") or []
    if not sources:
        raise ValueError("config.sources is empty: supply at least one licensed source file")

    records = []
    reports = []
    for src in sources:
        path = resolve_path(src["path"], base_dir)
        if not path.is_file():
            raise MissingSourceError(
                f"Source file not found: {path}. ANSG does not bundle corpus texts; "
                "place your licensed edition at this path or update the config."
            )
        prov = SourceProvenance.from_mapping(src["provenance"])
        recs, rep = ingest_file(path, prov, nights_cfg, norm_cfg)
        manifest.add_inputs([path])
        records.extend(recs)
        reports.append(asdict(rep))
        manifest.warnings.extend(rep.warnings)

    for imp in config.get("annotation_imports") or []:
        ipath = resolve_path(imp["path"], base_dir)
        counts = import_annotations(records, ipath, imp["layer"], AnnotationStatus(imp["status"]))
        manifest.add_inputs([ipath])
        manifest.stats.setdefault("annotation_imports", []).append({**imp, **counts})

    issues = [i for r in records for i in validate_record(r, labels)]
    errors = [i for i in issues if i.severity == "error"]
    manifest.warnings.extend(f"{i.record_id}: {i.message}" for i in issues if i.severity == "warning")
    if errors:
        msg = "\n".join(f"{i.record_id}: {i.message}" for i in errors[:20])
        raise ValueError(f"{len(errors)} validation error(s):\n{msg}")

    assignment = assign_splits(records, split_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_counts: dict[str, int] = {}
    for name in sorted(set(assignment.values())):
        subset = [r for r in records if r.split == name]
        p = out_dir / f"{name}.jsonl"
        split_counts[name] = write_jsonl(subset, p, allow_restricted_text=allow_restricted)
        manifest.add_output(p)
    with open(out_dir / "split_assignment.json", "w", encoding="utf-8") as fh:
        json.dump(dict(sorted(assignment.items())), fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    manifest.add_output(out_dir / "split_assignment.json")

    manifest.stats.update({
        "n_records": len(records),
        "split_counts": split_counts,
        "ingestion_reports": reports,
        "annotation_status_counts": {
            layer: {s.value: sum(r.annotation_status[layer] == s for r in records) for s in AnnotationStatus}
            for layer in ("dia", "motif", "moral")
        },
    })
    manifest.finish()
    manifest.write(out_dir / "manifest.json")
    return manifest.to_dict()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ANSG corpus ingestion (Phase 1)")
    ap.add_argument("--config", required=True, help="YAML config, e.g. configs/data/nights.yaml")
    args = ap.parse_args(argv)
    cfg_path = Path(args.config).resolve()
    cfg = load_yaml_config(cfg_path)
    # Relative paths resolve against the working directory (run from the repo root)
    # unless the config sets ``paths_relative_to: config``.
    base = cfg_path.parent if cfg.get("paths_relative_to") == "config" else Path.cwd()
    try:
        result = run_pipeline(cfg, base)
    except MissingSourceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: result[k] for k in ("stats", "outputs", "warnings")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
