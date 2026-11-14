#!/usr/bin/env python3
"""Build story-level records for data/corpora/classical_arabic (metadata only; no text).

Inputs (all committed): index.json (pinned revisions, page hashes), story_segments.json
(CI evidence run: boundaries at the pinned revisions), evidence/external/external_evidence.json
(categories, prefix text, night anomalies), data/sources/source_registry.json (rights) and the
derived pilot split (scripts/reproduction/derive_pilot_split.py).

Output: data/corpora/classical_arabic/story_records.json. Every record says it is NOT ANHC.
Dialect is ``null`` with status ``unannotated``: no DIA label is inferred.

    python scripts/reproduction/build_story_records.py [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "corpora" / "classical_arabic"
OUT = BASE / "story_records.json"
RESOURCE = "wikisource_classical_arabic_pages"


def _read(p: Path):
    return json.loads(p.read_text("utf-8"))


def build() -> dict:
    index = {r["id"]: r for r in _read(BASE / "index.json")}
    segs = _read(BASE / "story_segments.json")
    evidence = {p["id"]: p for p in _read(ROOT / "evidence" / "external" / "external_evidence.json")["wikisource"]["pages"]}
    registry = {r["id"]: r for r in _read(ROOT / "data" / "sources" / "source_registry.json")["resources"]}
    spec = importlib.util.spec_from_file_location("dps", ROOT / "scripts" / "reproduction" / "derive_pilot_split.py")
    dps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dps)
    derived = dps.derive()
    page_split = {pid: s for s, ids in derived["split"].items() for pid in ids}
    merged = {}
    for m in derived["group_merges"]:
        a, b = m["pages"]
        merged[b] = merged.get(a, a)
    rights = registry[RESOURCE]
    records, pages = [], []
    for s in segs["segments"]:
        pid = s["source_record_id"]
        page = index[pid]
        ev = evidence.get(pid, {})
        kal = page["corpus"] == "kalila_dimna"
        records.append({
            "story_id": s["segment_id"],
            "title": s["title"],
            "work": "Kalīla wa-Dimna" if kal else "Alf Layla wa-Layla",
            "source": {"site": "ar.wikisource.org", "page_title": page["title"], "page_url": page["url"],
                       "revision_id": page["revision_id"],
                       "permalink": f"https://ar.wikisource.org/w/index.php?oldid={page['revision_id']}",
                       "page_text_sha256": page["sha256"], "categories": ev.get("categories"),
                       "edition": "unidentified Wikisource transcription" + (
                           " (page category 'نصوص مجهولة المصدر' = texts of unknown source)"
                           if "تصنيف:نصوص مجهولة المصدر" in (ev.get("categories") or []) else "")},
            "boundaries": {"block_span": s["block_span"], "evidence": s.get("boundary_evidence"),
                           "text_sha256": s["text_sha256"], "words": s["words"], "nights": s["nights"]},
            "nesting": {"parent_story_id": s["parent_segment_id"], "evidence": s["nesting_evidence"],
                        "candidate_parent": s["candidate_parent"], "ambiguities": s["ambiguities"]},
            "dialect": None, "dialect_status": "unannotated",
            "register": "classical narrative prose (description of the source, not an annotation)",
            "split_group": "alf_layla:frame-cycle" if not kal else f"kalila:{merged.get(pid, pid)}",
            "pilot_split_v2": page_split.get(pid),
            "rights": {"registry_id": RESOURCE, "permission_status": rights["permission_status"],
                       "allowed_uses": rights["allowed_uses"], "site_licence": "CC BY-SA 4.0 (Wikisource site, "
                       "evidence/external/external_evidence.json)", "underlying_text": "pre-modern; edition unknown"},
            "is_anhc": False,
        })
    for pid, page in index.items():
        ev = evidence.get(pid, {})
        pages.append({"source_record_id": pid, "revision_id": page["revision_id"], "words": page["word_count"],
                      "segments": sum(1 for r in records if r["story_id"].startswith(pid + "/")),
                      "unassigned_prefix": ev.get("untitled_prefix"), "night_sequence": ev.get("night_sequence")})
    return {"schema": "ansg.story_records/1",
            "status": "automatic segmentation at pinned revisions; NOT ANHC; human review required",
            "segments_source": {"file": "data/corpora/classical_arabic/story_segments.json",
                                "github_run_id": segs.get("github_run_id")},
            "split_version": derived["version"], "records": records, "pages": pages}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    payload = json.dumps(build(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        ok = OUT.exists() and OUT.read_text("utf-8") == payload
        print("story records are current" if ok else f"{OUT} is stale: rerun build_story_records.py",
              file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    OUT.write_text(payload, encoding="utf-8")
    d = json.loads(payload)
    print(json.dumps({"records": len(d["records"]), "pages": len(d["pages"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
