#!/usr/bin/env python
"""Collect OpenITI catalogue/release metadata for the thesis's classical adab works.

Metadata-only evidence collector (run in CI, where the network can reach GitHub
and Zenodo). For Maqāmāt al-Ḥarīrī, Kitāb al-Aghānī and the works of al-Jāḥiẓ it
records:

* the official OpenITI releases on Zenodo (versions, DOIs, dates, licence, file
  names/sizes/checksums) **without downloading the release archives**;
* the KITAB catalogue rows (OpenITI_Github_clone_metadata_light.csv) for the
  target works, pinned to the commit of kitab-metadata-automation that was read;
* per book: the OpenITI author repository, its pinned master commit, the files
  in the book folder (git blob SHA, size) and the version ``.yml`` metadata;
* per target version text: SHA-256 and byte size of the file at the pinned
  commit. The text is hashed in memory and discarded; it is never written.

Nothing here ingests text, assigns labels, or changes the corpus.

    GITHUB_TOKEN=... python scripts/openiti/collect_openiti_metadata.py \
        --out data/sources/openiti/openiti_metadata_evidence.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

ZENODO_CONCEPT_RECID = "3082463"  # OpenITI concept record (RELEASE README)
KITAB_REPO = "OpenITI/kitab-metadata-automation"
KITAB_CSV_PATH = "output/OpenITI_Github_clone_metadata_light.csv"
RAW = "https://raw.githubusercontent.com"
API = "https://api.github.com"
UA = "ansg-openiti-metadata-audit"

# Target works named in the thesis, §3.2.2 p. 57 ("Maqāmāt al-Harīrī, Kitāb
# al-Aghānī, and Al-Jāḥiẓ's works contribute 13.3M tokens").
TARGET_BOOKS = {"0516IbnCaliHariri.Maqamat": "maqamat_al_hariri", "0362AbuFarajIsbahani.Aghani": "kitab_al_aghani"}
TARGET_AUTHOR_PREFIXES = {"0255Jahiz.": "al_jahiz_works"}
# Related works recorded for context only (commentaries, abridgements, compilations).
RELATED_PATTERNS = [r"Maqamat.*Hariri|Hariri.*Maqamat", r"Aghani(?!s)", r"Jahiz"]
MAX_TEXT_BYTES = 64 * 1024 * 1024
MAX_METADATA_CSV_BYTES = 200 * 1024 * 1024


def _request(url, *, token=None, accept=None, max_bytes=None):
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    if token and url.startswith(API):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read(max_bytes + 1 if max_bytes else -1)
        if max_bytes and len(data) > max_bytes:
            raise ValueError(f"response exceeds {max_bytes} bytes")
        return resp.status, data


def _json(url, token=None):
    status, data = _request(url, token=token, accept="application/vnd.github+json" if url.startswith(API) else "application/json")
    return json.loads(data.decode("utf-8"))


def _attempt(fn, *args, **kwargs):
    try:
        return {"ok": True, "value": fn(*args, **kwargs)}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:  # recorded, not hidden
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def zenodo_releases():
    """All versions of the OpenITI concept record (tries the documented endpoints in turn)."""
    attempts, data, used = [], None, None
    latest = _attempt(_json, f"https://zenodo.org/api/records/{ZENODO_CONCEPT_RECID}")
    attempts.append({"url": f"/api/records/{ZENODO_CONCEPT_RECID}", "ok": latest["ok"], "error": latest.get("error")})
    candidates = []
    if latest["ok"]:
        rid = latest["value"].get("id")
        candidates += [f"https://zenodo.org/api/records/{rid}/versions", f"https://zenodo.org/api/records/{rid}/versions?size=100"]
    candidates += [
        "https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": f"conceptrecid:{ZENODO_CONCEPT_RECID}", "all_versions": "1", "size": 100}),
        "https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": f"parent.id:{ZENODO_CONCEPT_RECID}", "allversions": "true", "size": 100}),
    ]
    for url in candidates:
        res = _attempt(_json, url)
        attempts.append({"url": url.replace("https://zenodo.org", ""), "ok": res["ok"], "error": res.get("error")})
        if res["ok"] and res["value"].get("hits", {}).get("hits"):
            data, used = res["value"], url
            break
    if data is None:
        if not latest["ok"]:
            raise RuntimeError(f"no Zenodo endpoint answered: {attempts}")
        # Fall back to the latest release only (the concept record resolves to it).
        data, used = {"hits": {"hits": [latest["value"]], "total": None}}, f"https://zenodo.org/api/records/{ZENODO_CONCEPT_RECID}"
    out = []
    for hit in data.get("hits", {}).get("hits", []):
        md = hit.get("metadata", {})
        lic = md.get("license") or {}
        out.append({
            "recid": hit.get("id"),
            "doi": hit.get("doi") or md.get("doi"),
            "conceptdoi": hit.get("conceptdoi"),
            "version": md.get("version"),
            "title": md.get("title"),
            "publication_date": md.get("publication_date"),
            "license": lic.get("id") if isinstance(lic, dict) else lic,
            "access_right": md.get("access_right"),
            "files": _zenodo_files(hit),
            "related_identifiers": md.get("related_identifiers"),
            "description_excerpt": re.sub(r"<[^>]+>", " ", md.get("description") or "")[:8000],
        })
    out.sort(key=lambda r: r.get("publication_date") or "", reverse=True)
    return {"query": f"concept record {ZENODO_CONCEPT_RECID}", "endpoint": used.replace("https://zenodo.org", ""),
            "attempts": attempts, "total": data.get("hits", {}).get("total"), "releases": out}


def release_metadata(target_books, target_prefixes, token, max_tags=3):
    """Official release metadata CSVs from OpenITI/RELEASE tags (metadata folder only)."""
    tags = _json(f"{API}/repos/OpenITI/RELEASE/tags?per_page=100", token)
    out = {"tags_listed": [t["name"] for t in tags], "inspected": []}
    for tag in tags[:max_tags]:
        entry = {"tag": tag["name"], "commit": tag["commit"]["sha"]}
        listing = _attempt(_json, f"{API}/repos/OpenITI/RELEASE/contents/metadata?ref={tag['commit']['sha']}", token)
        if not listing["ok"]:
            entry["metadata_folder"] = listing
            out["inspected"].append(entry)
            continue
        entry["metadata_files"] = [{"name": f["name"], "size": f["size"], "git_blob_sha": f["sha"]} for f in listing["value"]]
        csvs = [f for f in listing["value"] if f["name"].endswith((".csv", ".tsv")) and "merged" not in f["name"] and f["size"] <= MAX_METADATA_CSV_BYTES]
        if csvs:
            f = csvs[0]
            res = _attempt(_request, f"{RAW}/OpenITI/RELEASE/{tag['commit']['sha']}/metadata/{urllib.parse.quote(f['name'])}", max_bytes=MAX_METADATA_CSV_BYTES)
            if res["ok"]:
                data = res["value"][1]
                csv.field_size_limit(10**9)
                rows = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
                key = "version_uri" if rows and "version_uri" in rows[0] else "versionUri"
                hits = [r for r in rows if any(str(r.get(key, "")).startswith(b + ".") for b in target_books)
                        or any(str(r.get(key, "")).startswith(p) for p in target_prefixes)]
                entry["metadata_csv"] = {"name": f["name"], "sha256": hashlib.sha256(data).hexdigest(),
                                         "md5": hashlib.md5(data).hexdigest(), "bytes": len(data),
                                         "rows": len(rows), "columns": list(rows[0].keys()) if rows else [],
                                         "target_rows": hits}
            else:
                entry["metadata_csv"] = res
        out["inspected"].append(entry)
    # Fingerprint target texts as published at the newest tag that has metadata
    # (GitHub copy of the release). Hashed in memory; nothing is written.
    newest = next((e for e in out["inspected"] if (e.get("metadata_csv") or {}).get("target_rows")), None)
    if newest:
        fps = {}
        for row in newest["metadata_csv"]["target_rows"]:
            path = (row.get("local_path") or "").lstrip("./")
            uri = row.get("version_uri") or row.get("versionUri")
            if not path or not uri:
                continue
            res = _attempt(_request, f"{RAW}/OpenITI/RELEASE/{newest['commit']}/{urllib.parse.quote(path)}", max_bytes=MAX_TEXT_BYTES)
            fps[uri] = ({"path": path, "sha256": hashlib.sha256(res["value"][1]).hexdigest(), "bytes": len(res["value"][1]),
                         "text_written_to_disk": False} if res["ok"] else {"path": path, "error": res["error"]})
        out["release_text_fingerprints"] = {"tag": newest["tag"], "commit": newest["commit"], "versions": fps}
    return out


def _zenodo_files(hit):
    files = hit.get("files")
    if isinstance(files, dict):  # InvenioRDM shape: {"entries": {...}} or {"entries": [...]}
        entries = files.get("entries") or {}
        files = list(entries.values()) if isinstance(entries, dict) else entries
    return [{"key": f.get("key"), "size": f.get("size"), "checksum": f.get("checksum")} for f in files or []]


def latest_commit(repo, path, token):
    q = urllib.parse.urlencode({"path": path, "per_page": 1})
    commits = _json(f"{API}/repos/{repo}/commits?{q}", token)
    if not commits:
        return None
    c = commits[0]
    return {"sha": c["sha"], "date": c["commit"]["committer"]["date"], "message": c["commit"]["message"].splitlines()[0][:200]}


def kitab_catalogue(token):
    commit = latest_commit(KITAB_REPO, KITAB_CSV_PATH, token)
    url = f"{RAW}/{KITAB_REPO}/{commit['sha']}/{KITAB_CSV_PATH}"
    _, data = _request(url, max_bytes=MAX_METADATA_CSV_BYTES)
    csv.field_size_limit(10**9)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"))
    return {"repo": KITAB_REPO, "path": KITAB_CSV_PATH, "commit": commit, "url": url,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "rows": len(rows)}, rows


def classify(row):
    book = row.get("book", "")
    if book in TARGET_BOOKS:
        return "target", TARGET_BOOKS[book]
    for prefix, work in TARGET_AUTHOR_PREFIXES.items():
        if row.get("versionUri", "").startswith(prefix):
            return "target", work
    if any(re.search(p, book) for p in RELATED_PATTERNS):
        return "related", None
    return None, None


def repo_and_path(url):
    # https://raw.githubusercontent.com/OpenITI/0525AH/master/data/<author>/<book>/<version>
    m = re.match(r"https://raw\.githubusercontent\.com/OpenITI/([^/]+)/[^/]+/(data/.+)$", url or "")
    return (m.group(1), m.group(2)) if m else (None, None)


def parse_yml_fields(text):
    """Parse OpenITI '00#VERS#URI######: value' style metadata (with indented continuations)."""
    fields, key = {}, None
    for line in text.splitlines():
        m = re.match(r"^(\d\d#[A-Z]+#[A-Z0-9#]+?)#*:\s?(.*)$", line)
        if m:
            key = m.group(1)
            fields[key] = m.group(2).strip()
        elif key and line.startswith("    "):
            fields[key] += " " + line.strip()
    return fields


def book_evidence(repo, book_dir, version_uris, token):
    head = _json(f"{API}/repos/OpenITI/{repo}/branches/master", token)["commit"]["sha"]
    listing = _json(f"{API}/repos/OpenITI/{repo}/contents/{urllib.parse.quote(book_dir)}?ref={head}", token)
    files = [{"name": f["name"], "git_blob_sha": f["sha"], "size": f["size"]} for f in listing if f.get("type") == "file"]
    folder_commit = latest_commit(f"OpenITI/{repo}", book_dir, token)
    ymls = {}
    for f in files:
        if f["name"].endswith(".yml"):
            _, data = _request(f"{RAW}/OpenITI/{repo}/{head}/{book_dir}/{urllib.parse.quote(f['name'])}", max_bytes=1024 * 1024)
            ymls[f["name"]] = parse_yml_fields(data.decode("utf-8", errors="replace"))
    texts = {}
    for f in files:
        if f["name"].endswith(".yml"):
            continue
        if not any(f["name"].startswith(v) for v in version_uris):
            continue
        res = _attempt(_request, f"{RAW}/OpenITI/{repo}/{head}/{book_dir}/{urllib.parse.quote(f['name'])}", max_bytes=MAX_TEXT_BYTES)
        if res["ok"]:
            data = res["value"][1]
            texts[f["name"]] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                                "git_blob_sha": f["git_blob_sha"], "text_written_to_disk": False}
        else:
            texts[f["name"]] = {"error": res["error"]}
    return {"repo": f"OpenITI/{repo}", "master_commit": head, "book_dir": book_dir,
            "folder_last_commit": folder_commit, "files": files, "yml": ymls, "text_fingerprints": texts}


CATALOGUE_FIELDS = ["versionUri", "book", "status", "language", "uncorrected_OCR", "date", "author_ar", "author_lat",
                    "title_ar", "title_lat", "ed_info", "id", "tok_length", "char_length", "url", "tags"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")

    evidence = {
        "schema": "ansg.openiti_metadata_evidence/1",
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "collector": "scripts/openiti/collect_openiti_metadata.py",
        "ci_run": {k: os.environ.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_REF_NAME")},
        "policy": "metadata and fingerprints only; no release archive downloaded; no text written or committed",
        "thesis_reference": "Tesis_Ibrahim §3.2.2, p. 57: Maqāmāt al-Harīrī, Kitāb al-Aghānī and al-Jāḥiẓ's works, 13.3M tokens (edition not stated)",
    }
    evidence["zenodo"] = _attempt(zenodo_releases)

    evidence["release_repository"] = _attempt(release_metadata, list(TARGET_BOOKS), list(TARGET_AUTHOR_PREFIXES), token)

    cat = _attempt(kitab_catalogue, token)
    if not cat["ok"]:
        evidence["kitab_catalogue"] = cat
        _write(args.out, evidence)
        return 1
    cat_meta, rows = cat["value"]
    targets, related = [], []
    for r in rows:
        kind, work = classify(r)
        if kind == "target":
            targets.append({"work_key": work, **{k: r.get(k) for k in CATALOGUE_FIELDS}})
        elif kind == "related":
            related.append({k: r.get(k) for k in ("versionUri", "book", "status", "title_ar", "tok_length", "url")})
    # All catalogue versions derived from Hindawi books: the ANHC direct build already
    # holds Hindawi books (Arabic E-Book Corpus), so these are potential duplicates.
    hindawi = []
    for r in rows:
        m = re.search(r"\.Hindawi0*(\d+)-", r.get("versionUri", ""))
        if m:
            hindawi.append({"versionUri": r["versionUri"], "hindawi_id": m.group(1), "status": r.get("status"),
                            "tok_length": r.get("tok_length")})
    evidence["kitab_catalogue"] = {**cat_meta, "target_rows": targets, "related_rows_not_targets": related,
                                   "hindawi_derived_versions": hindawi}

    books = {}
    for t in targets:
        repo, path = repo_and_path(t["url"])
        if not repo:
            continue
        book_dir = path.rsplit("/", 1)[0]
        books.setdefault((repo, book_dir), set()).add(t["versionUri"])
    evidence["books"] = {}
    for (repo, book_dir), versions in sorted(books.items()):
        evidence["books"][book_dir.rsplit("/", 1)[-1]] = _attempt(book_evidence, repo, book_dir, sorted(versions), token)
    _write(args.out, evidence)
    failed = [k for k, v in evidence["books"].items() if not v["ok"]]
    print(f"targets={len(targets)} related={len(related)} books={len(books)} failed_books={failed} zenodo_ok={evidence['zenodo']['ok']}")
    return 0


def _write(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write("\n")


if __name__ == "__main__":
    sys.exit(main())
