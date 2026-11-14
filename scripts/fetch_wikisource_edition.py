#!/usr/bin/env python
"""Fetch a multi-page Wikisource edition by DISCOVERING its pages, then segment it into tales.

Network access to the MediaWiki API is required, so this is run by the owner, not in CI.
Nothing is downloaded on import. Outputs (under --output, which belongs in data/private/):

* ``manifest.json``   — site licence info, discovery report (reading order, missing
                        titles, redirects, unlinked sub-pages), per-page id, revision id,
                        permalink, templates, licence-like templates, SHA-256s;
* ``blocks.jsonl``    — ordered heading/text blocks with their page;
* ``structure.json``  — tales, nights, evidence-backed nesting, ambiguities;
* ``records/all.jsonl`` and ``split_audit.json`` — ANHC tale records and a story-cycle split.

Example (Hindawi 2022 edition, part 1):

    python scripts/fetch_wikisource_edition.py \\
        --index-title "ألف ليلة وليلة (مؤسسة هنداوي، 2022)/الجزء الأول" \\
        --source-id wikisource-hindawi-2022-p1 --output data/private/wikisource/hindawi_p1
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--api", default="https://ar.wikisource.org/w/api.php")
    ap.add_argument("--site", default="https://ar.wikisource.org")
    ap.add_argument("--index-title", required=True)
    ap.add_argument("--subpage-prefix", help="default: the index title")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--accept-textual-cues", action="store_true",
                    help="use framing cues (not only heading hierarchy) as nesting evidence")
    ap.add_argument("--user-agent", default="ANSGResearch/0.3 (research; https://github.com/Mossab82/ANSG)")
    args = ap.parse_args(argv)

    from ansg.data.anhc import write_jsonl
    from ansg.data.mediawiki import EditionFetcher, MediaWikiClient, urllib_transport
    from ansg.data.records import CopyrightStatus, SourceProvenance
    from ansg.data.splits import SplitConfig
    from ansg.data.tale_structure import StructureConfig, analyse_structure, cycle_split, tale_records

    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} is not empty", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    fetcher = EditionFetcher(MediaWikiClient(urllib_transport(args.api, args.user_agent)), args.site)
    pages, manifest = fetcher.fetch_edition(args.index_title, args.subpage_prefix)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    blocks = [b for p in pages for b in p.blocks]
    with (out / "blocks.jsonl").open("w", encoding="utf-8") as fh:
        for p in pages:
            for b in p.blocks:
                fh.write(json.dumps({"page": p.title, "revid": p.revid, **asdict(b)}, ensure_ascii=False) + "\n")
    report = analyse_structure(blocks, StructureConfig(accept_textual_cues=args.accept_textual_cues))
    (out / "structure.json").write_text(json.dumps(
        {"tales": [asdict(t) for t in report.tales], "night_blocks": report.night_blocks,
         "ambiguities": report.ambiguities, "unparsed_night_headings": report.unparsed_night_headings},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prov = SourceProvenance(source_id=args.source_id, title=args.index_title, url=args.site,
                            copyright_status=CopyrightStatus.UNKNOWN, redistributable=False)
    records = tale_records(blocks, report, prov)
    write_jsonl(records, out / "records" / "all.jsonl")
    audit = cycle_split(records, SplitConfig(ratios={"train": 0.8, "validation": 0.1, "test": 0.1}, seed=42))
    (out / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pages": len(pages), "missing": len(manifest["discovery"]["missing"]),
                      "rejected": len(manifest["rejected_pages"]), "tales": len(report.tales),
                      "ambiguities": len(report.ambiguities), "story_cycles": audit["cycles"],
                      "leaking_cycles": audit["leaking_cycles"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
