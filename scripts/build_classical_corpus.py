#!/usr/bin/env python3
"""Build an auditable Arabic heritage corpus from actual Wikisource page titles.

One JSONL record == one source page / narrative CYCLE, not an independent sub-tale.
This script MUST NOT produce a 'gold' ANHC corpus or a research-grade split.
"""
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup

API = "https://ar.wikisource.org/w/api.php"
UA = "ANSG-research-corpus/0.1 (noncommercial academic prototype; https://github.com/Mossab82/ANSG)"
OUT = Path("data/corpora/classical_arabic")
ROOT = "ألف ليلة وليلة/"
NIGHTS = [ROOT + x for x in ("الجزء الأول", "الجزء الثاني", "الجزء الثالث", "الجزء الرابع", "السندباد البحري")]
KALILA = [
    "الناسك وابن عرس", "باب إيلاذ وبلاذ وايراخت", "باب ابن الملك وأصحابه",
    "باب ابن الملك والطائر فنزة", "باب الأسد والثور وهو أول الكتاب",
    "باب الأسد والشغبر الناسك وهو ابن آوى", "باب البوم والغربان",
    "باب الجرذ والسنور", "باب الحمامة المطوقة",
    "باب الحمامة والثعلب ومالك الحزين", "باب السائح والصائغ",
    "باب الفحص عن أمر دمنة", "باب القرد والغيلم",
    "باب اللبوة والإسوار والشغبر", "باب الناسك والضيف",
]
PAGES = [(x, "arabian_nights") for x in NIGHTS] + [
    ("كليلة ودمنة/" + x, "kalila_dimna") for x in KALILA
]
# Fail hard if a source is missing; never silently fill records with invented prose.
MIN_ARABIC = 450
MIN_WORDS = 130

def request(**params):
    params.update({"format": "json", "formatversion": "2"})
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=55) as resp:
                data = json.load(resp)
            if "error" in data:
                raise ValueError(str(data["error"]))
            return data
        except ValueError:
            raise
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"API failed {params.get('page') or params.get('titles')}: {e}") from e
            time.sleep(3 * (attempt + 1))

def extract(title):
    data = request(action="parse", page=title, prop="text|revid", redirects=1)
    info = data["parse"]
    canonical = info["title"]
    fragment = info["text"]
    if isinstance(fragment, dict):
        fragment = fragment["*"]
    soup = BeautifulSoup(fragment, "html.parser")
    root = soup.select_one(".mw-parser-output") or soup
    for node in root.select("script, style, .mw-editsection, .reference, .reflist, .toc, .noprint, .navbox, .mw-collapsible, .ws-noexport, sup.reference"):
        node.decompose()
    # Preserve the relative order of narrative headings and paragraphs.
    content = []
    for node in root.find_all(["h2", "h3", "h4", "p"], recursive=True):
        # Eliminate elements embedded in another of these elements.
        if node.find_parent(["h2", "h3", "h4", "p"]):
            continue
        txt = " ".join(node.get_text(" ", strip=True).split())
        if len(txt) >= 30:
            content.append(txt)
    if not content:
        txt = root.get_text(" ", strip=True)
        content = [" ".join(txt.split())]
    text = "\n\n".join(content).strip()
    arabic = sum("\u0600" <= ch <= "\u06ff" for ch in text)
    words = len(text.split())
    if arabic < MIN_ARABIC or words < MIN_WORDS:
        raise ValueError(f"Insufficient source {canonical!r}: {arabic} Arabic chars, {words} words")
    clean_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    url = "https://ar.wikisource.org/wiki/" + urllib.parse.quote(canonical.replace(" ", "_"), safe="/")
    return dict(title=canonical, url=url, revision_id=info.get("revid"),
                text=text, sha256=clean_hash, word_count=words,
                arabic_characters=arabic, paragraphs=len(content))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, failures, seen_hash = [], [], set()
    for idx, (title, corpus) in enumerate(PAGES, 1):
        try:
            row = extract(title)
            if row["sha256"] in seen_hash:
                print("DUPLICATE", title, flush=True)
                continue
            seen_hash.add(row["sha256"])
            # The whole source page is a single *cycle* for conservatively
            # preventing story-level leakage. Embedded tales are NOT extracted yet.
            row.update(id=f"{corpus}-{idx:03d}", corpus=corpus,
                       language="ar", narrative_unit="source_page_or_cycle",
                       story_parent_id=None, split="unassigned",
                       license_status="review_page_revision_before_training",
                       segment_status="not_story_annotated")
            records.append(row)
            print("OK", corpus, row["title"], row["word_count"], row["sha256"][:12], flush=True)
        except Exception as exc:
            failures.append({"title": title, "reason": f"{type(exc).__name__}: {exc}"})
            print("FAILED", title, failures[-1]["reason"], file=sys.stderr, flush=True)
    # Write partial diagnostics, but NEVER publish an allegedly complete corpus
    # if some nominated pages could not be fetched.
    report = {
        "schema": "ansg.heritage_corpus_build/1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_api": API,
        "requested": len(PAGES), "accepted": len(records),
        "failures": failures,
        "arabian_nights_pages": sum(r["corpus"] == "arabian_nights" for r in records),
        "kalila_dimna_pages": sum(r["corpus"] == "kalila_dimna" for r in records),
        "words": sum(r["word_count"] for r in records),
        "verified_rights_for_model_training": False,
        "story_independent_train_validation_test_split": False,
        "scientific_note": "Source-page/cycle records; NOT nested-story annotations, ANHC or thesis experiments.",
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Corpus rejected due to missing/invalid source(s); inspect build_report.json")
    with (OUT / "stories.jsonl").open("w", encoding="utf-8") as dst:
        for item in records:
            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
    (OUT / "index.json").write_text(json.dumps([
        {k: v for k, v in row.items() if k != "text"} for row in records
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"CORPUS VERIFIED FOR TEXT EXTRACTION ONLY: {len(records)} source pages, {report['words']} words", flush=True)

if __name__ == "__main__":
    main()
