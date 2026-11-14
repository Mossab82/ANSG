#!/usr/bin/env python3
"""Collect first-hand evidence about the thesis's external artefacts and the heritage corpus sources.

Runs on a GitHub-hosted runner (CPU, internet); see .github/workflows/reproduction-evidence.yml.
Only metadata and short quoted snippets are recorded, never dataset contents.

1. Thesis artefact identifiers (Table 9.6, p. 254; App. A.20.2, p. 290): the five Zenodo DOIs
   and the github.com/mossab-ansg repositories (existence, title, dates).
2. MADAR (thesis §3.1.5 p. 55; §4.2.4 p. 94; §5.4.3 p. 140): official page and LREC 2018
   paper, so the thesis entry "2.1M / 26 cities" can be checked against primary sources.
3. Arabic Wikisource pages of data/corpora/classical_arabic: at the PINNED revision ids,
   page metadata (templates, categories, licence-like templates), re-extraction with the
   original builder logic (does the frozen text reproduce byte for byte?), and structured
   heading/text blocks for tale segmentation (src/ansg/data/tale_structure.py).

Outputs: evidence/external/external_evidence.json and
data/corpora/classical_arabic/story_segments.json (no text, spans and hashes only).
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
UA = "ANSG-reproduction-audit/0.1 (academic; https://github.com/Mossab82/ANSG)"
WIKI_API = "https://ar.wikisource.org/w/api.php"
ZENODO = ["8847291", "8847342", "8847356", "8847367", "8847378"]
GITHUB_REPOS = ["anhc-test-set", "sahda", "ansg-framework", "ansg-weights", "craft-eval", "ansg-train", "ckg-arabic"]
MADAR_PAGES = ["https://camel.abudhabi.nyu.edu/madar-parallel-corpus/", "https://sites.google.com/nyu.edu/madar/"]
MADAR_PAPER = "https://aclanthology.org/L18-1535.pdf"
KEYWORDS = re.compile(r"(corpus|cities|city|MSA|sentence|licen[cs]e|distribut|modif|internal|research|evaluat|26|25|"
                      r"CORPUS|copyright|commercial)", re.I)


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def get(url: str, accept: str = "*/*", timeout: int = 40) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read()
            return {"url": url, "status": r.status, "final_url": r.geturl(),
                    "content_type": r.headers.get("Content-Type"), "sha256": sha(body), "bytes": len(body),
                    "_body": body}
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "error": str(e)}
    except Exception as e:  # noqa: BLE001 - network failures are evidence too
        return {"url": url, "status": None, "error": f"{type(e).__name__}: {e}"}


def public(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def text_of_html(body: bytes) -> str:
    html = body.decode("utf-8", "replace")
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html)).strip()


def snippets(text: str, limit: int = 40) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p[:400] for p in parts if KEYWORDS.search(p)][:limit]


def zenodo() -> list[dict]:
    out = []
    for rid in ZENODO:
        r = get(f"https://zenodo.org/api/records/{rid}", "application/json")
        rec = public(r)
        if r.get("status") == 200:
            meta = json.loads(r["_body"]).get("metadata", {})
            rec.update(title=meta.get("title"), creators=[c.get("name") for c in meta.get("creators", [])],
                       publication_date=meta.get("publication_date"), resource_type=meta.get("resource_type"),
                       doi=meta.get("doi"), access_right=meta.get("access_right"))
        d = get(f"https://doi.org/10.5281/zenodo.{rid}")
        rec["doi_resolution"] = {k: d.get(k) for k in ("status", "final_url", "error")}
        out.append(rec)
        time.sleep(1)
    return out


def github() -> dict:
    user = public(get("https://api.github.com/users/mossab-ansg", "application/vnd.github+json"))
    repos = [public(get(f"https://api.github.com/repos/mossab-ansg/{r}", "application/vnd.github+json"))
             for r in GITHUB_REPOS]
    return {"user": user, "repos": repos}


def madar() -> dict:
    out = {"pages": [], "paper": None}
    for url in MADAR_PAGES:
        r = get(url, "text/html")
        rec = public(r)
        if r.get("status") == 200:
            rec["snippets"] = snippets(text_of_html(r["_body"]))
        out["pages"].append(rec)
    r = get(MADAR_PAPER, "application/pdf", timeout=60)
    rec = public(r)
    if r.get("status") == 200:
        try:
            from pypdf import PdfReader
            text = " ".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(r["_body"])).pages)
            text = re.sub(r"\s+", " ", text)
            rec["snippets"] = [s for s in snippets(text, 80)
                               if re.search(r"\d", s) and re.search(r"sentence|corpus|cit", s, re.I)][:40]
        except Exception as e:  # noqa: BLE001
            rec["pdf_error"] = f"{type(e).__name__}: {e}"
    out["paper"] = rec
    return out


def wiki(params: dict) -> dict:
    url = WIKI_API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    r = get(url, "application/json", timeout=60)
    if r.get("status") != 200:
        raise RuntimeError(f"wiki API {r.get('status')} {r.get('error')}")
    data = json.loads(r["_body"])
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    time.sleep(1)
    return data


def builder_module():
    spec = importlib.util.spec_from_file_location("bcc", ROOT / "scripts" / "build_classical_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corpus_pages() -> tuple[list[dict], list[dict]]:
    from ansg.data.mediawiki import extract_blocks
    from ansg.data.tale_structure import analyse_structure
    rows = [json.loads(l) for l in (ROOT / "data/corpora/classical_arabic/stories.jsonl").read_text("utf-8").splitlines()
            if l.strip()]
    bcc = builder_module()
    pages, segments = [], []
    for row in rows:
        rec = {"id": row["id"], "title": row["title"], "pinned_revision_id": row["revision_id"],
               "frozen_sha256": row["sha256"]}
        try:
            meta = wiki({"action": "query", "revids": row["revision_id"],
                         "prop": "revisions|categories|templates|info", "rvprop": "ids|timestamp|size|comment",
                         "cllimit": "max", "tllimit": "max"})
            page = meta["query"]["pages"][0]
            rev = page.get("revisions", [{}])[0]
            templates = [t["title"] for t in page.get("templates", [])]
            rec.update(pageid=page.get("pageid"), page_title=page.get("title"), revision_timestamp=rev.get("timestamp"),
                       revision_size=rev.get("size"), latest_revision_id=page.get("lastrevid"),
                       categories=[c["title"] for c in page.get("categories", [])], templates=templates,
                       licence_like_templates=[t for t in templates if re.search(
                           r"PD|ملكية|رخص|license|licence|CC|GFDL|حقوق", t, re.I)])
            parsed = wiki({"action": "parse", "oldid": row["revision_id"], "prop": "text|revid|sections"})["parse"]
            html = parsed["text"]
            # 1) the original builder's extraction applied to the pinned revision
            original_request = bcc.request
            bcc.request = lambda **kw: {"parse": parsed}
            try:
                rebuilt = bcc.extract(row["title"])
            finally:
                bcc.request = original_request
            rec["builder_reextraction_sha256"] = rebuilt["sha256"]
            rec["frozen_text_reproducible_from_pinned_revision"] = rebuilt["sha256"] == row["sha256"]
            # 2) structured blocks (headings keep their level; short titles are not dropped)
            blocks = extract_blocks(html)
            rec["blocks"] = len(blocks)
            rec["headings"] = [{"level": b.level, "text": b.text} for b in blocks if b.kind == "heading"][:400]
            rec["mediawiki_sections"] = [{"level": s.get("level"), "line": s.get("line")}
                                         for s in parsed.get("sections", [])][:400]
            report = analyse_structure(blocks)
            rec["tales_found"] = len(report.tales)
            rec["night_markers"] = len(report.night_blocks)
            rec["ambiguities"] = report.ambiguities[:200]
            for t in report.tales:
                text = "\n\n".join(b.text for b in blocks[t.start_block:t.end_block])
                segments.append({"segment_id": f"{row['id']}/{t.id}", "source_record_id": row["id"],
                                 "source_title": row["title"], "revision_id": row["revision_id"],
                                 "title": t.title, "block_span": [t.start_block, t.end_block],
                                 "heading_level": t.heading_level, "parent_segment_id":
                                 f"{row['id']}/{t.parent}" if t.parent else None,
                                 "nesting_evidence": t.nesting_evidence,
                                 "candidate_parent": f"{row['id']}/{t.candidate_parent}" if t.candidate_parent else None,
                                 "nights": t.nights, "ambiguities": t.ambiguities,
                                 "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 "words": len(text.split())})
        except Exception as e:  # noqa: BLE001 - record and continue; the report shows the failure
            rec["error"] = f"{type(e).__name__}: {e}"
        pages.append(rec)
        print(json.dumps({k: rec.get(k) for k in ("id", "frozen_text_reproducible_from_pinned_revision",
                                                    "blocks", "tales_found", "night_markers",
                                                    "licence_like_templates", "error")}, ensure_ascii=False),
              flush=True)
    return pages, segments


def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    rights = wiki({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo|general"})["query"]
    pages, segments = corpus_pages()
    evidence = {
        "schema": "ansg.external_evidence/1",
        "collected_utc": started,
        "collector": "scripts/reproduction/collect_external_evidence.py",
        "github_run": {"repository": "Mossab82/ANSG", "run_id": __import__("os").environ.get("GITHUB_RUN_ID"),
                       "sha": __import__("os").environ.get("GITHUB_SHA")},
        "thesis_artefacts": {"zenodo": zenodo(), "github": github()},
        "madar": madar(),
        "wikisource": {"site_rights": rights.get("rightsinfo"),
                       "site": {k: rights.get("general", {}).get(k) for k in ("sitename", "base", "generator")},
                       "pages": pages},
    }
    out = ROOT / "evidence" / "external" / "external_evidence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    seg = {"schema": "ansg.story_segments/1", "derived_from": "data/corpora/classical_arabic/stories.jsonl revisions",
           "method": "ansg.data.tale_structure.analyse_structure on rendered HTML of the pinned revision",
           "text_included": False, "collected_utc": started, "github_run_id": evidence["github_run"]["run_id"],
           "status": "automatic segmentation; nesting only from heading hierarchy; requires human review",
           "segments": segments}
    (ROOT / "data/corpora/classical_arabic/story_segments.json").write_text(
        json.dumps(seg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "zenodo": [{k: z.get(k) for k in ("url", "status", "title", "error")} for z in evidence["thesis_artefacts"]["zenodo"]],
        "github_user_status": evidence["thesis_artefacts"]["github"]["user"].get("status"),
        "github_repo_status": {r["url"].rsplit("/", 1)[-1]: r.get("status")
                               for r in evidence["thesis_artefacts"]["github"]["repos"]},
        "madar_pages": [{k: p.get(k) for k in ("url", "status")} for p in evidence["madar"]["pages"]],
        "madar_paper_status": evidence["madar"]["paper"].get("status"),
        "wikisource_pages_reproducible": sum(bool(p.get("frozen_text_reproducible_from_pinned_revision")) for p in pages),
        "wikisource_pages": len(pages), "segments": len(segments),
    }
    print("EVIDENCE SUMMARY " + json.dumps(summary, ensure_ascii=False))
    for p in evidence["madar"]["pages"] + [evidence["madar"]["paper"]]:
        for s in p.get("snippets", [])[:25]:
            print("MADAR SNIPPET | " + p["url"] + " | " + s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
