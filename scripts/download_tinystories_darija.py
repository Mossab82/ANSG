#!/usr/bin/env python3
"""Download a pinned TinyStories Algerian Darija snapshot into local quarantine."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPO = "touati-kamel/TinyStories-Algerian-Darija"
REVISION = "82760dc9de372ff56a531766194f91f78e270520"
DEFAULT_DEST = ROOT / "data/corpora/anhc/quarantine/external/tinystories-algerian-darija"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true"
    req = urllib.request.Request(tree_url, headers={"User-Agent": "ANHC-research-build/1.0"})
    entries = json.load(urllib.request.urlopen(req, timeout=30))
    selected = [
        e for e in entries if e.get("type") == "file" and
        (e.get("path", "").endswith(".parquet") or e.get("path") in {"README.md", "processed_story_ids.json"})
    ]
    if not selected:
        raise SystemExit("No source files found in the pinned repository snapshot")

    def download(entry: dict) -> tuple[str, int]:
        rel = entry["path"]
        dest = args.destination / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == entry["size"]:
            return rel, entry["size"]
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{rel}"
        request = urllib.request.Request(url, headers={"User-Agent": "ANHC-research-build/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read()
        if len(payload) != entry["size"]:
            raise ValueError(f"Size mismatch for {rel}: {len(payload)} != {entry['size']}")
        dest.write_bytes(payload)
        return rel, len(payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = list(pool.map(download, selected))
    print(json.dumps({
        "repository": REPO,
        "revision": REVISION,
        "files": len(results),
        "bytes": sum(size for _, size in results),
        "destination": str(args.destination.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
