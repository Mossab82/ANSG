#!/usr/bin/env python
"""Collect metadata about where the thesis says the CKG / CST-Onto can be obtained.

Metadata only (run in CI, where these hosts are reachable). It records HTTP
status, resolved URLs and link targets; it downloads no ontology or graph and
writes no third-party text beyond short titles and URLs.

Sources checked (dissertation references):
* Table 9.6, p. 254: CKG "OWL ontology with 1,847 cultural patterns",
  https://github.com/mossab-ansg/ckg-arabic, DOI 10.5281/zenodo.8847378.
* Bibliography / App. B.3.3: CST-Onto paper, K-CAP '25, DOI 10.1145/3731443.3771364.
Plus public searches for "CST-Onto" on Zenodo and GitHub.

    GITHUB_TOKEN=... python scripts/ckg/collect_ckg_source_metadata.py --out evidence/ckg/ckg_source_evidence.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

UA = "ansg-ckg-source-audit"
CKG_ZENODO = "8847378"
CKG_REPO = "mossab-ansg/ckg-arabic"
CST_ONTO_DOI = "10.1145/3731443.3771364"
LINK_HINTS = re.compile(r"\.(owl|ttl|rdf|zip|jsonld)(\?|$)|github\.com|zenodo\.org|supplement|figshare|osf\.io", re.I)


def fetch(url, token=None, max_bytes=4 * 1024 * 1024, accept=None):
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    if token and url.startswith("https://api.github.com"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read(max_bytes)
            return {"url": url, "status": resp.status, "final_url": resp.geturl(), "body": body}
    except urllib.error.HTTPError as exc:
        return {"url": url, "status": exc.code, "final_url": exc.geturl() if hasattr(exc, "geturl") else None, "body": b""}
    except Exception as exc:  # recorded, not hidden
        return {"url": url, "status": None, "error": f"{type(exc).__name__}: {exc}", "body": b""}


def meta(res):
    return {k: v for k, v in res.items() if k != "body"}


def as_json(res):
    try:
        return json.loads(res["body"].decode("utf-8")) if res.get("body") else None
    except ValueError:
        return None


def crossref(doi):
    res = fetch(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}")
    out = meta(res)
    msg = (as_json(res) or {}).get("message") or {}
    if msg:
        out["record"] = {
            "title": (msg.get("title") or [None])[0],
            "container": (msg.get("container-title") or [None])[0],
            "published": msg.get("published", {}).get("date-parts"),
            "publisher": msg.get("publisher"),
            "license": [l.get("URL") for l in msg.get("license") or []],
            "links": [l.get("URL") for l in msg.get("link") or []],
            "relation": msg.get("relation"),
            "resource_primary": (msg.get("resource") or {}).get("primary", {}).get("URL"),
            "references_count": msg.get("references-count"),
            "reference_urls_with_artifact_hints": sorted({r.get("URL") or r.get("DOI") or "" for r in msg.get("reference") or []
                                                        if LINK_HINTS.search(str(r.get("URL") or r.get("unstructured") or ""))})[:50],
        }
    return out


def page_links(url):
    res = fetch(url)
    out = meta(res)
    html = res.get("body", b"").decode("utf-8", errors="replace")
    hrefs = sorted(set(re.findall(r'href="([^"]+)"', html)))
    out["artifact_like_links"] = [h for h in hrefs if LINK_HINTS.search(h)][:100]
    out["bytes"] = len(res.get("body", b""))
    return out


def zenodo_record(recid):
    api = fetch(f"https://zenodo.org/api/records/{recid}")
    doi = fetch(f"https://doi.org/10.5281/zenodo.{recid}")
    return {"api": meta(api), "doi_resolution": meta(doi)}


def zenodo_search(q):
    res = fetch("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": q, "size": 25}))
    out = meta(res)
    hits = ((as_json(res) or {}).get("hits") or {}).get("hits") or []
    out["hits"] = [{"id": h.get("id"), "doi": h.get("doi"), "title": (h.get("metadata") or {}).get("title"),
                    "resource_type": ((h.get("metadata") or {}).get("resource_type") or {}).get("type"),
                    "license": ((h.get("metadata") or {}).get("license") or {}).get("id"),
                    "files": [f.get("key") for f in (h.get("files") or [])][:20]} for h in hits]
    return out


def github_search(q, token):
    res = fetch("https://api.github.com/search/repositories?" + urllib.parse.urlencode({"q": q, "per_page": 25}), token)
    out = meta(res)
    items = (as_json(res) or {}).get("items") or []
    out["items"] = [{"full_name": i.get("full_name"), "html_url": i.get("html_url"), "description": (i.get("description") or "")[:200],
                     "license": (i.get("license") or {}).get("spdx_id"), "pushed_at": i.get("pushed_at")} for i in items]
    return out


ONTOLOGY_LIKE = re.compile(r"\.(owl|ttl|rdf|nt|n3|jsonld|owx|ofn|swrl|graphml|gexf)$|(^|/)[^/]*(ckg|cst[-_]?onto|ontolog)[^/]*$", re.I)


def owner_repositories(owner, token, limit=60):
    """Public repositories of `owner`: licence and ontology/graph-like file paths (no content)."""
    res = fetch(f"https://api.github.com/users/{owner}/repos?per_page=100&type=owner&sort=pushed", token)
    out = meta(res)
    repos = as_json(res) or []
    out["repositories"] = []
    for r in repos[:limit]:
        full = r.get("full_name")
        entry = {"full_name": full, "html_url": r.get("html_url"), "default_branch": r.get("default_branch"),
                 "license": (r.get("license") or {}).get("spdx_id"), "pushed_at": r.get("pushed_at"),
                 "description": (r.get("description") or "")[:200]}
        tree = fetch(f"https://api.github.com/repos/{full}/git/trees/{r.get('default_branch')}?recursive=1", token)
        tj = as_json(tree) or {}
        entry["tree_status"] = tree.get("status")
        entry["tree_truncated"] = tj.get("truncated")
        entry["ontology_like_files"] = [{"path": t.get("path"), "size": t.get("size"), "git_blob_sha": t.get("sha")}
                                        for t in tj.get("tree") or []
                                        if t.get("type") == "blob" and ONTOLOGY_LIKE.search(t.get("path") or "")][:200]
        out["repositories"].append(entry)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN")
    ev = {
        "schema": "ansg.ckg_source_evidence/1",
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "collector": "scripts/ckg/collect_ckg_source_metadata.py",
        "ci_run": {k: os.environ.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_REF_NAME")},
        "policy": "metadata only; no ontology, graph or paper text downloaded or committed",
        "thesis_references": {
            "ckg_artifact": "Table 9.6, p. 254: OWL ontology with 1,847 cultural patterns; github.com/mossab-ansg/ckg-arabic; DOI 10.5281/zenodo.8847378",
            "cst_onto_paper": "Bibliography (Ibrahim, Gervás, Méndez 2025a) and App. B.3.3: K-CAP '25, pp. 174-181, DOI 10.1145/3731443.3771364",
        },
        "ckg_zenodo_record": zenodo_record(CKG_ZENODO),
        "ckg_github_repo": meta(fetch(f"https://api.github.com/repos/{CKG_REPO}", token)),
        "ckg_github_owner": meta(fetch(f"https://api.github.com/users/{CKG_REPO.split('/')[0]}", token)),
        "cst_onto_paper_crossref": crossref(CST_ONTO_DOI),
        "cst_onto_paper_doi_resolution": meta(fetch(f"https://doi.org/{CST_ONTO_DOI}")),
        "cst_onto_paper_landing_page": page_links(f"https://dl.acm.org/doi/{CST_ONTO_DOI}"),
        "zenodo_search": {q: zenodo_search(q) for q in ('"CST-Onto"', '"CSTOnto"', 'title:"Cultural Schema Theory" AND arabic')},
        "github_search": {q: github_search(q, token) for q in ("CST-Onto", "cst-onto ontology", "ckg-arabic")},
        # The AI-Cinema paper (Complexity 2026, p.1) states "All resources are publicly available at
        # https://github.com/Mossab82/AI-Cinema" and builds on CST-Onto; list the owner's public repositories.
        "owner_repositories": owner_repositories("Mossab82", token),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(ev, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) and "status" in v else "see file") for k, v in ev.items()
                      if k.startswith(("ckg_", "cst_onto"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
