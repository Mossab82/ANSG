#!/usr/bin/env python3
"""Build a local, provenance-preserving inventory for ANHC and the dialect guide.

All external corpus text and derived lexical rows stay under quarantine. This
script does not clear rights, confirm a dialect label, or admit text to training.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANHC = ROOT / "data/corpora/anhc"
QUARANTINE = ANHC / "quarantine"
LAHGA = QUARANTINE / "external/lahga"
OUT = QUARANTINE / "anhc_resource_inventory"
DB = OUT / "anhc_resource_inventory.sqlite"
REVISION = "3bacbff0dede7f42c6f9d08a7ca56b7ca6bcdaa8"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dump_csv(path: Path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dialect_registry = load_json(ROOT / "data/sources/dialect_registry.json")
    narrative_registry = load_json(ROOT / "data/sources/narrative_registry.json")
    merged_manifest = load_json(ANHC / "merged_candidate_manifest.json")

    con = sqlite3.connect(DB)
    con.executescript("""
      PRAGMA foreign_keys=ON;
      DROP TABLE IF EXISTS lexical_examples;
      DROP TABLE IF EXISTS lexical_forms;
      DROP TABLE IF EXISTS story_assets;
      DROP TABLE IF EXISTS resources;
      DROP TABLE IF EXISTS dialects;
      CREATE TABLE dialects(
        dialect_id TEXT PRIMARY KEY, label_ar TEXT NOT NULL, family TEXT,
        granularity TEXT, source_coverage TEXT, narrative_coverage TEXT,
        note TEXT
      );
      CREATE TABLE resources(
        resource_id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT,
        resource_class TEXT, varieties_json TEXT, description TEXT,
        license_claim TEXT, rights_status TEXT, admission_status TEXT,
        suitability TEXT, local_records INTEGER, local_status TEXT,
        limitations_json TEXT, user_supplied INTEGER DEFAULT 0
      );
      CREATE TABLE story_assets(
        asset_id TEXT PRIMARY KEY, source_id TEXT, title TEXT,
        origin_type TEXT, dialect_label TEXT, record_count INTEGER,
        whitespace_words INTEGER, local_path TEXT, rights_status TEXT,
        annotation_status TEXT, split_status TEXT, notes TEXT
      );
      CREATE TABLE lexical_forms(
        entry_id INTEGER PRIMARY KEY, headword TEXT, kind TEXT,
        definition TEXT, form TEXT, dialect TEXT, dialect_name TEXT,
        dialect_group TEXT, language TEXT, meaning TEXT, notes TEXT,
        needs_review INTEGER, example_count INTEGER, word_id INTEGER,
        source_url TEXT, source_revision TEXT, license TEXT,
        evidence_status TEXT, training_status TEXT
      );
      CREATE INDEX lexical_forms_headword_idx ON lexical_forms(headword);
      CREATE INDEX lexical_forms_dialect_idx ON lexical_forms(dialect);
      CREATE INDEX lexical_forms_group_idx ON lexical_forms(dialect_group);
      CREATE TABLE lexical_examples(
        example_id INTEGER PRIMARY KEY, entry_id INTEGER, word_id INTEGER,
        headword TEXT, form TEXT, dialect TEXT, dialect_group TEXT,
        language TEXT, text TEXT, gloss TEXT, needs_review INTEGER,
        source_url TEXT, source_revision TEXT, license TEXT,
        FOREIGN KEY(entry_id) REFERENCES lexical_forms(entry_id)
      );
      CREATE INDEX lexical_examples_form_idx ON lexical_examples(form);
      CREATE INDEX lexical_examples_dialect_idx ON lexical_examples(dialect);
    """)

    # Normalize the source registries, preserving each source's own status.
    resources = {}
    for r in dialect_registry.get("datasets", []):
        rid = r["id"]
        resources[rid] = {
            "resource_id": rid, "title": r.get("title", rid), "url": r.get("url"),
            "resource_class": "dialect_or_language_resource",
            "varieties_json": json.dumps(r.get("variety", []), ensure_ascii=False),
            "description": r.get("data"), "license_claim": r.get("license") or r.get("source_license_claim"),
            "rights_status": r.get("rights_status") or r.get("license_status") or r.get("admission"),
            "admission_status": r.get("status") or r.get("admission"),
            "suitability": "; ".join(r.get("roles", [])), "local_records": None,
            "local_status": (r.get("local_intake") or {}).get("status", "metadata_only"),
            "limitations_json": json.dumps(r.get("limits", []), ensure_ascii=False),
            "user_supplied": int(rid in {"Arabic_VerbNet", "QCRI_Dialectal_Arabic_Resources", "Arabic_Dialect_Identification_GitHub_Project"}),
        }
    for r in narrative_registry.get("sources", []):
        rid = r["id"]
        res = {
            "resource_id": rid, "title": r.get("title", rid), "url": r.get("url"),
            "resource_class": "narrative_source",
            "varieties_json": json.dumps(r.get("variety", []), ensure_ascii=False),
            "description": r.get("material"), "license_claim": r.get("declared_license"),
            "rights_status": r.get("rights_status"), "admission_status": r.get("admission_status"),
            "suitability": "; ".join(r.get("role", [])), "local_records": None,
            "local_status": "not_ingested", "limitations_json": json.dumps(r.get("blockers", []), ensure_ascii=False),
            "user_supplied": int(rid in {"sanaa_ai_stories"}),
        }
        if rid == "tinystories_algerian_darija_synthetic":
            res["local_records"] = 11326
            res["local_status"] = "downloaded_local_quarantine"
        resources[rid] = res
    # User-shared Wojood/ArabicNER link is listed among non-story resources.
    for r in narrative_registry.get("not_story_corpora", []):
        rid = r["id"]
        resources.setdefault(rid, {
            "resource_id": rid, "title": rid, "url": r.get("url"),
            "resource_class": "auxiliary_not_story_corpus", "varieties_json": "[]",
            "description": r.get("reason"), "license_claim": None,
            "rights_status": "review_required", "admission_status": "metadata_only",
            "suitability": r.get("role"), "local_records": None,
            "local_status": "metadata_only", "limitations_json": json.dumps([r.get("reason")], ensure_ascii=False),
            "user_supplied": int(rid == "wojood_arabic_ner"),
        })

    # Local narrative material: counts are candidate rows, not approved stories.
    source_counts, word_counts = Counter(), Counter()
    if (QUARANTINE / "merged_narrative_candidates.jsonl").exists():
        for row in load_jsonl(QUARANTINE / "merged_narrative_candidates.jsonl"):
            src = row.get("source", {})
            sid = src.get("source_id", "unknown")
            source_counts[(sid, src.get("title"))] += 1
            word_counts[sid] += len((row.get("text") or "").split())
    for (sid, title), n in source_counts.items():
        res = resources.get(sid)
        if res:
            res["local_records"] = n
            res["local_status"] = "candidate_rows_in_local_quarantine"
    for r in resources.values():
        con.execute("INSERT INTO resources VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            r.get("resource_id"), r.get("title"), r.get("url"), r.get("resource_class"),
            r.get("varieties_json", "[]"), r.get("description"), r.get("license_claim"),
            r.get("rights_status"), r.get("admission_status"), r.get("suitability"),
            r.get("local_records"), r.get("local_status"), r.get("limitations_json", "[]"),
            r.get("user_supplied", 0)))

    assets = []
    for (sid, title), n in sorted(source_counts.items()):
        dialect = "Algerian Darija (synthetic)" if sid == "tinystories_algerian_darija_synthetic" else "MSA / heritage Arabic"
        local_path = "data/corpora/anhc/quarantine/merged_narrative_candidates.jsonl"
        asset_note = "Counts are candidate rows; parallel editions, inherited story boundaries, and exact unique-token counts are unresolved."
        if sid.startswith("user-pdf-"):
            local_path = "data/corpora/anhc/quarantine/user_books/كليلة ودمنة_31354_Foulabook.com_.pdf; extracted candidate rows remain separately in merged_narrative_candidates.jsonl"
            asset_note = "101-page user-supplied PDF; 15 chapter candidates were extracted. PDF streams have structural warnings; compare OCR/text with the PDF and collate with heritage seed before use."
        assets.append({"asset_id": sid, "source_id": sid, "title": title,
                       "origin_type": "synthetic_adaptation" if "tinystories" in sid else "heritage_or_user_pdf_candidate",
                       "dialect_label": dialect, "record_count": n, "whitespace_words": word_counts[sid],
                       "local_path": local_path,
                       "rights_status": "review_required; not cleared for training or redistribution",
                       "annotation_status": "unannotated; boundary and source review required",
                       "split_status": "unassigned",
                       "notes": asset_note})
    assets.append({"asset_id": "alf_layla_user_pdf", "source_id": "user_pdf", "title": "ألف ليلة وليلة - نسخة أصلية نادرة (PDF supplied by user)",
                   "origin_type": "user_supplied_book", "dialect_label": "MSA / heritage Arabic", "record_count": None,
                   "whitespace_words": None, "local_path": "user-provided Library PDF; OCR and edition review pending",
                   "rights_status": "edition and reuse rights review required", "annotation_status": "not yet normalized into narrative records",
                   "split_status": "unassigned", "notes": "Do not count as incorporated corpus text until OCR, provenance, segmentation, and rights review."})
    assets.append({"asset_id": "kalila_pdf_user_supplied", "source_id": "user_pdf", "title": "كليلة ودمنة (PDF supplied by user)",
                   "origin_type": "user_supplied_book", "dialect_label": "MSA / heritage Arabic", "record_count": 15,
                   "whitespace_words": 32672, "local_path": "data/corpora/anhc/quarantine/user_books/كليلة ودمنة_31354_Foulabook.com_.pdf",
                   "rights_status": "edition and reuse rights review required", "annotation_status": "15 extracted chapter candidates; source-boundary and text collation pending",
                   "split_status": "unassigned", "notes": "The local extraction pipeline produced 15 PDF chapter candidates (32,672 whitespace words); these are already counted in the merged 11,368 candidate rows and must not be added twice."})
    con.executemany("INSERT INTO story_assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [tuple(a[k] for k in ("asset_id","source_id","title","origin_type","dialect_label","record_count","whitespace_words","local_path","rights_status","annotation_status","split_status","notes")) for a in assets])

    # Source-level dialect coverage rows, distinct from actual narrative coverage.
    dialects = [
        ("MSA", "العربية الفصحى", "reference", "general", "heritage/story sources and dictionary headwords", "candidate heritage records; no rights-cleared count", "Baseline reference; not dialect output."),
        ("DZA", "الدارجة الجزائرية", "Maghrebi", "Algeria; region unspecified", "TinyStories; Arabic dialect dictionary", "11,326 synthetic candidates in quarantine", "Source label only; not independently speaker-verified."),
        ("MAR", "الدارجة المغربية", "Maghrebi", "Morocco; mixed dictionary varieties/oral tale study", "Moroccan oral-tale formula evidence; lexical dictionary", "no human-authored dialect story cohort admitted", "Separate study snippets from dictionary candidates."),
        ("EGY", "العامية المصرية", "Egyptian", "Cairene and broader source labels", "dictionary entries; auxiliary short-text resources", "no story corpus admitted", "Lexical evidence does not equal a story cohort."),
        ("LEV", "الشامية", "Levantine", "Palestinian, Jordanian, other source labels", "dictionary; Curras/Currasat candidate; Jordanian speech dataset metadata", "story text not ingested from these external sources", "Do not infer Aqaba coverage from Amman/Salt."),
        ("GULF", "الخليجية", "Gulf", "multiple source labels; country varies", "dictionary and QCRI auxiliary resources", "no story corpus admitted", "Needs country-specific validation."),
        ("YEM", "اليمنية", "Yemeni", "variety-specific dictionary labels", "dictionary entries; Sanaa source discovery", "no story cohort admitted", "Supplementary dialect guide only unless scope is explicitly expanded."),
    ]
    con.executemany("INSERT INTO dialects VALUES (?,?,?,?,?,?,?)", dialects)

    entries_path = LAHGA / "entries.jsonl"
    if entries_path.exists():
        for row in load_jsonl(entries_path):
            con.execute("INSERT OR REPLACE INTO lexical_forms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                row.get("entry_id"), row.get("headword"), row.get("kind"), row.get("definition"),
                row.get("form"), row.get("dialect"), row.get("dialect_name"), row.get("dialect_group"),
                row.get("language"), row.get("meaning"), row.get("notes"), int(bool(row.get("needs_review"))),
                row.get("examples"), row.get("word_id"), row.get("source"), REVISION,
                "CC-BY-SA-4.0", "dictionary_candidate_not_independently_adjudicated",
                "not_admitted_training_rights_policy_review_required"))
    examples_path = LAHGA / "examples.jsonl"
    if examples_path.exists():
        for row in load_jsonl(examples_path):
            con.execute("INSERT OR REPLACE INTO lexical_examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                row.get("example_id"), row.get("entry_id"), row.get("word_id"), row.get("headword"),
                row.get("form"), row.get("dialect"), row.get("dialect_group"), row.get("language"),
                row.get("text"), row.get("gloss"), int(bool(row.get("needs_review"))), row.get("source"),
                REVISION, "CC-BY-SA-4.0"))
    con.commit()

    # Stable, inspectable exports: complete lexical crosswalk plus source/story summaries.
    dump_csv(OUT / "resource_registry.csv",
             ["resource_id","title","url","resource_class","varieties_json","description","license_claim","rights_status","admission_status","suitability","local_records","local_status","limitations_json","user_supplied"],
             [dict(zip(["resource_id","title","url","resource_class","varieties_json","description","license_claim","rights_status","admission_status","suitability","local_records","local_status","limitations_json","user_supplied"], row)) for row in con.execute("SELECT * FROM resources ORDER BY resource_class, resource_id")])
    dump_csv(OUT / "dialect_lexicon_all_forms.csv",
             ["entry_id","headword","form","dialect","dialect_name","dialect_group","language","meaning","definition","needs_review","example_count","source_url","source_revision","license","evidence_status","training_status"],
             [dict(zip(["entry_id","headword","form","dialect","dialect_name","dialect_group","language","meaning","definition","needs_review","example_count","source_url","source_revision","license","evidence_status","training_status"], row)) for row in con.execute("SELECT entry_id,headword,form,dialect,dialect_name,dialect_group,language,meaning,definition,needs_review,example_count,source_url,source_revision,license,evidence_status,training_status FROM lexical_forms ORDER BY headword,dialect")])
    dump_csv(OUT / "dialect_lexicon_examples.csv",
             ["example_id","entry_id","headword","form","dialect","dialect_group","language","text","gloss","needs_review","source_url","source_revision","license"],
             [dict(zip(["example_id","entry_id","headword","form","dialect","dialect_group","language","text","gloss","needs_review","source_url","source_revision","license"], row)) for row in con.execute("SELECT example_id,entry_id,headword,form,dialect,dialect_group,language,text,gloss,needs_review,source_url,source_revision,license FROM lexical_examples ORDER BY example_id")])
    dump_csv(OUT / "story_assets.csv",
             ["asset_id","source_id","title","origin_type","dialect_label","record_count","whitespace_words","local_path","rights_status","annotation_status","split_status","notes"],
             [dict(zip(["asset_id","source_id","title","origin_type","dialect_label","record_count","whitespace_words","local_path","rights_status","annotation_status","split_status","notes"], row)) for row in con.execute("SELECT * FROM story_assets ORDER BY origin_type,title")])

    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["resources","story_assets","lexical_forms","lexical_examples"]}
    counts["lexical_forms_by_group"] = dict(con.execute("SELECT dialect_group,COUNT(*) FROM lexical_forms GROUP BY dialect_group ORDER BY dialect_group"))
    counts["needs_review_forms"] = con.execute("SELECT COUNT(*) FROM lexical_forms WHERE needs_review=1").fetchone()[0]
    counts["dictionary_examples_needs_review"] = con.execute("SELECT COUNT(*) FROM lexical_examples WHERE needs_review=1").fetchone()[0]
    counts["story_candidate_manifest"] = merged_manifest
    counts["not_admitted_training_rows"] = con.execute("SELECT COUNT(*) FROM lexical_forms WHERE training_status!='admitted'").fetchone()[0]
    report = {
        "schema": "ansg.anhc_resource_inventory/1.0",
        "status": "consolidated_local_review_inventory_not_training_corpus",
        "built_from": ["data/sources/dialect_registry.json", "data/sources/narrative_registry.json", "data/corpora/anhc/merged_candidate_manifest.json", "local quarantine candidates", f"Lahga dataset snapshot {REVISION}"],
        "lahga_license": "CC-BY-SA-4.0; preserve attribution and share-alike obligations; training/release compatibility remains under review",
        "counts": counts,
        "training_readiness": "not_ready_for_claimed_final_ANHC_or_multidialect_training; ready for pipeline engineering pilot with quarantined synthetic DZA plus approved non-text resources, while using the lexicon only as review candidates",
        "limitations": ["Lahga entries are lexical suggestions, not story evidence or verified verb paradigms.", "needs_review=false is not equivalent to native-speaker adjudication.", "No externally sourced narrative corpus is cleared merely by inclusion in this inventory.", "Existing 11,368 merged records remain unannotated/unassigned and rights unresolved.", "The user-supplied Alf Layla PDF still needs OCR and source/edition review before incorporation."]
    }
    (OUT / "inventory_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"database": str(DB.relative_to(ROOT)), "summary": str((OUT / "inventory_summary.json").relative_to(ROOT)), "counts": counts}, ensure_ascii=False, indent=2))
    con.close()


if __name__ == "__main__":
    main()
