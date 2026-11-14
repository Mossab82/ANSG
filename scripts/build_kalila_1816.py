#!/usr/bin/env python3
"""Fetch the Arabic Wikisource 1816 Kalila wa Dimna edition with pinned revisions."""
import argparse
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

API = "https://ar.wikisource.org/w/api.php"
SITE = "https://ar.wikisource.org"
INDEX = "كليلة ودمنة (1816)"
UA = "ANSGResearch/0.3 (heritage corpus collection; https://github.com/Mossab82/ANSG)"
PARATEXT = {"كليلة ودمنة (1816)/باب عرض الكتاب"}


class PageText(HTMLParser):
    SKIP = {"mw-editsection", "reference", "references", "mw-references-wrap", "noprint",
            "ws-noexport", "toc", "navbox", "ws-header", "headertemplate", "mw-jump-link",
            "printfooter", "catlinks", "metadata"}
    BLOCK = {"p", "div", "li", "dd", "dt", "blockquote", "tr", "table", "ul", "ol",
             "dl", "section", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "center", "br"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.root_depth = None
        self.buffer = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "div" and "mw-parser-output" in classes:
            self.root_depth = len(self.stack)
        in_root = self.root_depth is not None
        skip = tag in {"script", "style"} or bool(classes & self.SKIP) or attrs.get("role") == "navigation"
        if in_root and tag in self.BLOCK:
            self.flush()
        self.stack.append((tag, skip))

    def handle_endtag(self, tag):
        if self.root_depth is not None and tag in self.BLOCK:
            self.flush()
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if self.root_depth is not None and len(self.stack) <= self.root_depth:
            self.root_depth = None

    def handle_data(self, data):
        if self.root_depth is not None and not any(skip for _, skip in self.stack):
            self.buffer.append(data)

    def flush(self):
        value = " ".join("".join(self.buffer).split())
        self.buffer = []
        if value and (not self.blocks or self.blocks[-1] != value):
            self.blocks.append(value)

    def close(self):
        super().close()
        self.flush()


def api(**params):
    params.update(format="json", formatversion=2)
    url = API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=50) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="anhc_direct_build")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    index = api(action="parse", page=INDEX, prop="links|revid")["parse"]
    titles = [link["title"] for link in index.get("links", [])
              if link.get("ns") == 0 and link["title"].startswith(INDEX + "/باب ")]
    records, failures, excluded = [], [], []
    for number, requested_title in enumerate(titles, 1):
        if requested_title in PARATEXT:
            excluded.append({"title": requested_title, "reason": "front matter, not a narrative chapter"})
            continue
        try:
            parsed = api(action="parse", page=requested_title, prop="text|revid", redirects=1)["parse"]
            extractor = PageText()
            extractor.feed(parsed.get("text", ""))
            extractor.close()
            text = "\n\n".join(extractor.blocks).strip()
            arabic_chars = sum("\u0600" <= char <= "\u06ff" for char in text)
            if len(text) < 200 or arabic_chars < 200:
                raise ValueError(f"insufficient Arabic narrative text: {arabic_chars} characters")
            title, revision = parsed["title"], int(parsed["revid"])
            permalink = SITE + "/w/index.php?title=" + urllib.parse.quote(title.replace(" ", "_")) + "&oldid=" + str(revision)
            record = {
                "schema": "ansg.narrative_record/1",
                "record_id": f"kalila-1816-{number:02d}",
                "text": text,
                "source": {
                    "work": "كليلة ودمنة", "edition_title": INDEX, "edition_year": 1816,
                    "edition_city": "Paris", "editor": "Sylvestre de Sacy",
                    "page_title": title, "page_revision_id": revision, "page_url": permalink,
                    "index_title": INDEX, "index_revision_id": index.get("revid"),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "license": "CC BY-SA 4.0",
                    "attribution": "Wikisource contributors; transcribed from the 1816 edition; HTML-to-text extraction and whitespace separation only."
                },
                "narrative": {
                    "chapter_title": title.split("/باب ", 1)[-1],
                    "period": "classical_pre_1800", "language": "ar",
                    "dialect": "unannotated_classical_register",
                    "narrative_unit": "source_chapter; may contain nested tales",
                    "parent_story_id": None
                },
                "annotations": {"DIA": None, "MOTIF": None, "MORAL": None},
                "split": "unassigned",
                "counts": {"whitespace_words": len(text.split()), "arabic_characters": arabic_chars},
                "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
            }
            records.append(record)
            print(f"OK {number}/{len(titles)} words={len(text.split())} revid={revision}", flush=True)
            time.sleep(0.2)
        except Exception as exc:
            failures.append({"title": requested_title, "error": f"{type(exc).__name__}: {exc}"})
            print(f"FAILED {requested_title}: {exc}", flush=True)

    jsonl = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    (out / "kalila_1816.jsonl").write_text(jsonl, encoding="utf-8")
    with (out / "kalila_1816.jsonl.gz").open("wb") as target:
        target.write(gzip.compress(jsonl.encode("utf-8"), compresslevel=9, mtime=0))
    shards = out / "records"
    segments_root = shards / "segments"
    shards.mkdir(exist_ok=True)
    segment_count = 0
    for row in records:
        payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        compressed_shard = shards / f"{row['record_id']}.jsonl.gz"
        compressed_shard.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
        parts = [row["text"][i:i + 3000] for i in range(0, len(row["text"]), 3000)]
        segment_dir = segments_root / row["record_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_paths = []
        for part_number, part_text in enumerate(parts, 1):
            part_name = f"part-{part_number:03d}.jsonl"
            part_row = {
                "schema": "ansg.narrative_text_segment/1", "record_id": row["record_id"],
                "segment_index": part_number, "segment_count": len(parts),
                "source_text_sha256": row["source_text_sha256"], "text": part_text
            }
            (segment_dir / part_name).write_text(
                json.dumps(part_row, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            segment_paths.append(f"segments/{row['record_id']}/{part_name}")
        segment_count += len(parts)
        manifest = {
            "schema": "ansg.narrative_record_manifest/1", "record_id": row["record_id"],
            "source": row["source"], "narrative": row["narrative"],
            "annotations": row["annotations"], "split": row["split"], "counts": row["counts"],
            "source_text_sha256": row["source_text_sha256"],
            "text_storage": {"format": "ordered_utf8_jsonl_segments", "concatenate_field": "text",
                             "segments": segment_paths}
        }
        (shards / f"{row['record_id']}.jsonl").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "schema": "ansg.anhc_direct_build_report/1",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "source_index": {"title": index["title"], "revision_id": index.get("revid")},
        "edition": "Paris, 1816; associated with editor Sylvestre de Sacy",
        "source_license": "CC BY-SA 4.0 (Wikisource page footer; attribution and share-alike apply)",
        "requested_chapters": len(titles), "accepted_chapters": len(records), "excluded_paratext": excluded,
        "record_shards": len(records), "text_segments": segment_count,
        "storage_format": "chapter manifests plus ordered UTF-8 JSONL text segments; concatenate segment text in ascending segment_index to recover the source text exactly",
        "failures": failures,
        "whitespace_words": sum(row["counts"]["whitespace_words"] for row in records),
        "annotations_complete": False, "story_level_segmentation_complete": False,
        "split": "unassigned", "tokenizer": "not selected; whitespace-word counts only",
        "note": "Directly fetched source chapters; not the historical thesis ANHC and not a thesis-metric reproduction."
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("REPORT " + json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
