#!/usr/bin/env python
"""Build the OpenITI heritage source registry for the thesis's classical adab works.

Offline. Reads the CI-collected metadata evidence
(data/sources/openiti/openiti_metadata_evidence.json) and the ANHC direct-build
inventory already in the repository, and writes
data/sources/openiti/openiti_heritage_registry.json.

Every rights, redistribution and admission decision defaults to ``pending``.
Nothing is ingested: the registry holds identifiers, editions, fingerprints and
overlap findings only. ``--check`` fails if the committed registry is stale.

    python scripts/openiti/build_openiti_heritage_registry.py [--check]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/sources/openiti/openiti_metadata_evidence.json"
OUT = ROOT / "data/sources/openiti/openiti_heritage_registry.json"
DIRECT_BUILD = ROOT / "data/corpora/anhc_direct_build"
THESIS_COMBINED_TOKENS = 13_300_000  # §3.2.2, p. 57, for all three families together

WORKS = {
    "maqamat_al_hariri": {"title": "Maqāmāt al-Ḥarīrī", "author": "al-Qāsim b. ʿAlī al-Ḥarīrī (d. 516/1122)",
                          "author_patterns": ["حريري"]},
    "kitab_al_aghani": {"title": "Kitāb al-Aghānī", "author": "Abū l-Faraj al-Iṣfahānī (d. 356/967)",
                        "author_patterns": ["ابو الفرج", "اصفهاني ابو الفرج", "اصبهاني"]},
    "al_jahiz_works": {"title": "Works of al-Jāḥiẓ (thesis does not list which)", "author": "al-Jāḥiẓ (d. 255/868-9)",
                       "author_patterns": ["جاحظ"]},
}

# Work-level overlap with ANHC direct-build records, checked by title and author
# in the direct-build inventory (story_units / adjacent queue). Hindawi
# version-level identity is detected automatically from the OpenITI version id.
CURATED_WORK_OVERLAP = {
    "0255Jahiz.HaninIlaAwtan": ["hindawi-25703937"],
    "0255Jahiz.MahasinWaAddad": ["hindawi-35939520"],
    "0255Jahiz.Rasail": ["hindawi-47090584"],
    "0255Jahiz.TarbicWaTadwir": ["hindawi-47248193"],
    "0255Jahiz.BayanWaTabyin": ["hindawi-71716470"],
    "0255Jahiz.TabsiraBiTijara": ["hindawi-73505194"],
}
# Direct-build records matched by an author pattern that are *not* works of the
# target author (studies about him, homonyms). Listed so the scan stays explicit.
CURATED_NOT_TARGET = {
    "hindawi-35168479": "Khalil Mardam, al-Jāḥiẓ (a modern study, not a work by al-Jāḥiẓ)",
    "hindawi-70252813": "ʿImād al-Dīn al-Iṣfahānī, history of the Seljuqs (not Abū l-Faraj)",
    "hindawi-90909497": "Abū l-Faraj Ibn al-Jawzī, Akhbār al-ḥamqā wa-l-mughaffalīn (not Abū l-Faraj al-Iṣfahānī)",
}

# Unfilled OpenITI .yml template text, which is not information about the work.
YML_PLACEHOLDERS = ("a free running comment here", "permalink", "YYYY", "name of the annotator", "src@keyword",
                    "URIs from Althurayya", "YEAR-MON-DA", "URI of a book from OpenITI", "viaf@id", "all@id",
                    "Kitāb al-Muʾallif", "Risālaŧ al-Muʾallif", "formalized issues, separated with commas")


def yml_info(fields):
    return {k: v for k, v in fields.items()
            if k.split("#")[0] in ("80", "90") and v and not any(ph in v for ph in YML_PLACEHOLDERS)}

ID_PREFIX_RE = re.compile(r"\.([A-Za-z]+?)(\d[\w]*)-([a-z]{3}\d+)$")


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"[ً-ْٰـ]", "", s)
    s = re.sub("[إأآٱ]", "ا", s).replace("ى", "ي").replace("ة", "ه")
    return re.sub(r"\s+", " ", s).strip()


def load_direct_build():
    recs = {}
    for f in sorted(glob.glob(str(DIRECT_BUILD / "story_units" / "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            rid = r.get("source_record_id")
            if rid and rid not in recs:
                recs[rid] = {"title": r.get("title_candidate"), "author": r.get("author_candidate"),
                             "source_text_sha256": r.get("source_text_sha256")}
    queue = DIRECT_BUILD / "release_v1" / "adjacent_source_review_queue.csv"
    if queue.exists():
        for r in csv.DictReader(open(queue, encoding="utf-8-sig")):
            recs.setdefault(r["record_id"], {"title": r["title"], "author": r["author"],
                                             "source_text_sha256": r["source_text_sha256"]})
    split = {}
    for line in open(DIRECT_BUILD / "release_v1" / "split_assignments.jsonl", encoding="utf-8"):
        r = json.loads(line)
        split[r["record_id"]] = r
    return recs, split


def version_source(version_uri):
    m = ID_PREFIX_RE.search(version_uri)
    if not m:
        return {"collection_prefix": None, "source_id": None, "language_version": None}
    return {"collection_prefix": m.group(1), "source_id": m.group(1) + m.group(2), "language_version": m.group(3)}


def release_index(evidence):
    """versionUri -> list of release tags whose metadata CSV lists it."""
    idx = defaultdict(list)
    rr = evidence.get("release_repository", {})
    if not rr.get("ok"):
        return idx, rr.get("error")
    for entry in rr["value"].get("inspected", []):
        mc = entry.get("metadata_csv") or {}
        for row in mc.get("target_rows", []) or []:
            uri = row.get("version_uri") or row.get("versionUri")
            if uri:
                idx[uri].append(entry["tag"])
    return idx, None


def release_text_fps(evidence):
    rr = evidence.get("release_repository", {})
    if not rr.get("ok"):
        return None, {}
    rf = rr["value"].get("release_text_fingerprints") or {}
    return rf.get("tag"), rf.get("versions", {})


def zenodo_summary(evidence):
    z = evidence.get("zenodo", {})
    if not z.get("ok"):
        return {"status": "pending", "reason": f"Zenodo metadata not collected: {z.get('error')}"}
    rel = z["value"]["releases"]
    if not rel:
        return {"status": "pending", "reason": "no OpenITI release found under the concept record"}
    latest = rel[0]
    licences = sorted({r.get("license") for r in rel if r.get("license")})
    md5_check = "pending"
    rr = evidence.get("release_repository", {})
    if rr.get("ok"):
        for e in rr["value"].get("inspected", []):
            mc = e.get("metadata_csv") or {}
            if mc.get("md5") and e["tag"].lstrip("v") == latest.get("version"):
                z_md5 = next((f["checksum"] for f in latest.get("files", []) if f["key"] == mc.get("name")), None)
                md5_check = {"github_tag": e["tag"], "file": mc["name"], "github_md5": mc["md5"], "zenodo_checksum": z_md5,
                             "match": z_md5 == f"md5:{mc['md5']}"}
    return {"status": "collected", "concept_record": z["value"]["query"], "concept_doi": latest.get("conceptdoi"),
            "release_count": len(rel), "metadata_file_identity_github_vs_zenodo": md5_check,
            "release_archives_downloaded": False,
            "latest": {k: latest.get(k) for k in ("recid", "version", "doi", "publication_date", "license", "access_right")},
            "latest_files": latest.get("files"), "licences_across_releases": licences,
            "all_releases": [{k: r.get(k) for k in ("version", "doi", "publication_date", "license")} for r in rel]}


def build(evidence, recs, split):
    zen = zenodo_summary(evidence)
    rel_idx, rel_err = release_index(evidence)
    cat = evidence["kitab_catalogue"]
    books_ev = evidence.get("books", {})
    hindawi_versions = {h["versionUri"]: "hindawi-" + h["hindawi_id"] for h in cat.get("hindawi_derived_versions", [])}

    release_licence = (zen.get("latest") or {}).get("license") if zen["status"] == "collected" else None
    rel_tag, rel_fps = release_text_fps(evidence)
    versions_by_book = defaultdict(list)
    for row in cat["target_rows"]:
        versions_by_book[(row["work_key"], row["book"])].append(row)

    works = []
    for work_key, meta in WORKS.items():
        books = []
        for (wk, book), rows in sorted(versions_by_book.items()):
            if wk != work_key:
                continue
            bev = books_ev.get(book, {})
            bval = bev.get("value", {}) if bev.get("ok") else {}
            fps = bval.get("text_fingerprints", {})
            ymls = bval.get("yml", {})
            vers = []
            for r in sorted(rows, key=lambda r: (r["status"] != "pri", r["versionUri"])):
                uri = r["versionUri"]
                fp_name = next((n for n in fps if n.startswith(uri)), None)
                fp = fps.get(fp_name, {}) if fp_name else {}
                vyml = ymls.get(uri + ".yml", {})
                hid = hindawi_versions.get(uri)
                ed = (r.get("ed_info") or "").strip()
                vers.append({
                    "version_uri": uri,
                    "openiti_status": r.get("status"),
                    **version_source(uri),
                    "edition_info_verbatim": ed or None,
                    "edition_identity": "catalogued" if ed and ed.lower() != "none" else "pending",
                    "uncorrected_ocr": r.get("uncorrected_OCR") == "True",
                    "openiti_tok_length": int(r["tok_length"]) if (r.get("tok_length") or "").isdigit() else None,
                    "tags": [t.strip() for t in (r.get("tags") or "").split("::") if t.strip()],
                    "repository": bval.get("repo"),
                    "repository_commit": bval.get("master_commit"),
                    "file": fp_name,
                    "git_blob_sha": fp.get("git_blob_sha"),
                    "sha256": fp.get("sha256"),
                    "bytes": fp.get("bytes"),
                    "fingerprint_status": "collected" if fp.get("sha256") else "pending",
                    "official_release_text": _release_match(rel_tag, rel_fps.get(uri), fp),
                    "version_yml": yml_info(vyml),
                    "in_official_release_metadata": sorted(rel_idx.get(uri, [])) or ("pending" if rel_err or not rel_idx else []),
                    "direct_build_version_overlap": hid if hid in split else None,
                })
            work_overlap = [{"record_id": rid, **recs.get(rid, {}), "split": split.get(rid, {}).get("split"),
                             "work_group": split.get(rid, {}).get("work_group")}
                            for rid in CURATED_WORK_OVERLAP.get(book, [])]
            book_yml = ymls.get(book + ".yml", {})
            books.append({
                "book_uri": book,
                "repository": bval.get("repo"),
                "repository_commit": bval.get("master_commit"),
                "folder_last_commit": bval.get("folder_last_commit"),
                "collection_status": "collected" if bev.get("ok") else f"pending ({bev.get('error', 'not collected')})",
                "book_yml": yml_info(book_yml),
                "versions": vers,
                "direct_build_work_overlap": work_overlap,
            })
        # author scan of the direct build, so no matching record is silently missed
        scan = []
        for rid, r in sorted(recs.items()):
            if any(p in norm(r.get("author")) for p in map(norm, meta["author_patterns"])):
                mapped = next((b for b, ids in CURATED_WORK_OVERLAP.items() if rid in ids), None)
                scan.append({"record_id": rid, "title": r.get("title"), "author": r.get("author"),
                             "mapped_book_uri": mapped,
                             "not_target_reason": CURATED_NOT_TARGET.get(rid),
                             "status": "mapped" if mapped else ("not_target" if rid in CURATED_NOT_TARGET else "pending")})
        primary_tokens = sum(v["openiti_tok_length"] or 0 for b in books for v in b["versions"] if v["openiti_status"] == "pri")
        works.append({
            "work_key": work_key, **{k: meta[k] for k in ("title", "author")},
            "thesis_reference": "§3.2.2, p. 57 (edition, version and source not stated in the thesis)",
            "openiti_books": books,
            "openiti_primary_version_tokens": primary_tokens,
            "direct_build_author_scan": scan,
            "rights": rights_block(release_licence),
            "admission": {
                "status": "pending",
                "text_ingested": False,
                "blocking": [
                    "owner review of the OpenITI licence terms for ANSG use (training, derivatives, redistribution)",
                    "rights in the underlying printed edition and the third-party digital source are not established",
                    "choose one version per work and record it; do not count primary and secondary versions twice",
                    "deduplicate against ANHC direct-build records before any split assignment",
                ],
            },
        })
    return zen, works


def _release_match(tag, rfp, fp):
    if not tag or not rfp or "sha256" not in rfp:
        return {"release_tag": tag, "status": "pending"}
    return {"release_tag": tag, "path": rfp.get("path"), "sha256": rfp["sha256"],
            "status": "identical_to_fingerprint" if fp.get("sha256") == rfp["sha256"] else "differs_from_master"}


def rights_block(release_licence):
    return {
        "openiti_release_licence_as_published": release_licence or "pending",
        "licence_terms_summary": ("CC BY-NC-SA 4.0: attribution, non-commercial use only, derivatives under the same licence"
                                  if (release_licence or "").lower() == "cc-by-nc-sa-4.0" else "pending"),
        "github_author_repository_licence_file": "absent (no LICENSE file in OpenITI/0275AH, 0375AH, 0525AH master)",
        "underlying_edition_and_digital_source_rights": "pending",
        "permitted_use_for_ansg_training": "pending",
        "redistribution_in_this_repository": "pending",
        "note": ("A release licence covers OpenITI's publication; it does not by itself establish rights in modern "
                 "editorial work or third-party digitisations. Whether model training or released checkpoints count as "
                 "NC-SA derivatives, and whether ANHC's own licence is compatible, is for the owner to decide."),
    }


def registry(evidence_path=EVIDENCE):
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    recs, split = load_direct_build()
    zen, works = build(evidence, recs, split)
    cat = evidence["kitab_catalogue"]
    hind = cat.get("hindawi_derived_versions", [])
    overlap_ids = sorted({"hindawi-" + h["hindawi_id"] for h in hind} & set(split))
    total_primary = sum(w["openiti_primary_version_tokens"] for w in works)
    return {
        "schema": "ansg.openiti_heritage_registry/1",
        "generated_by": "scripts/openiti/build_openiti_heritage_registry.py",
        "evidence": {"path": str(Path(evidence_path).relative_to(ROOT)) if Path(evidence_path).is_relative_to(ROOT) else str(evidence_path),
                     "collected_at_utc": evidence.get("collected_at_utc"), "ci_run": evidence.get("ci_run")},
        "status": "pending",
        "is_anhc": False,
        "reproduces_thesis_data": False,
        "text_committed": False,
        "official_releases": zen,
        "catalogue": {k: cat.get(k) for k in ("repo", "path", "commit", "sha256", "bytes", "rows")},
        "release_repository_metadata": _release_summary(evidence),
        "works": works,
        "token_comparison": {
            "thesis_combined_tokens": THESIS_COMBINED_TOKENS,
            "openiti_primary_version_tokens_all_three": total_primary,
            "ratio": round(total_primary / THESIS_COMBINED_TOKENS, 4),
            "units_note": "OpenITI tok_length is OpenITI's own word-token count; the thesis tokenizer and the direct-build SentencePiece counts are different counters. This is not a claim that 13.3M or 70.4M tokens are reached.",
        },
        "direct_build_overlap_summary": {
            "openiti_hindawi_derived_versions": len(hind),
            "direct_build_records": len(split),
            "direct_build_records_also_in_openiti_as_hindawi_versions": len(overlap_ids),
            "record_ids": overlap_ids,
            "consequence": "Any OpenITI import must exclude or group these versions with their direct-build records to avoid double counting and split leakage.",
        },
    }


def _release_summary(evidence):
    rr = evidence.get("release_repository", {})
    if not rr.get("ok"):
        return {"status": "pending", "reason": rr.get("error")}
    out = []
    for e in rr["value"].get("inspected", []):
        mc = e.get("metadata_csv") or {}
        out.append({"tag": e["tag"], "commit": e.get("commit"),
                    "metadata_files": e.get("metadata_files") or e.get("metadata_folder"),
                    "metadata_csv": {k: mc.get(k) for k in ("name", "sha256", "bytes", "rows")} if mc.get("name") else mc or None,
                    "target_rows_found": len(mc.get("target_rows") or [])})
    return {"status": "collected", "tags_listed": rr["value"].get("tags_listed"), "inspected": out}


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", default=str(EVIDENCE))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = dumps(registry(args.evidence))
    out = Path(args.out)
    if args.check:
        if not out.exists() or out.read_text(encoding="utf-8") != text:
            print(f"{out} is stale; run scripts/openiti/build_openiti_heritage_registry.py", file=sys.stderr)
            return 1
        print(f"{out} is current")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
