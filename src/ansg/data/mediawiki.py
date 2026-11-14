"""Discovery and retrieval of a multi-page edition from a MediaWiki site (e.g. Arabic Wikisource).

Why
---
The heritage pilot assumed one page per night, titled ``.../فلما كانت الليلة N``.
Run 43559807003 showed that this is false for the Hindawi 2022 edition on Arabic
Wikisource: nights 1–2 resolved, then the API answered ``missingtitle`` for night 3.
Page titles must be *discovered*, not constructed.

What this module does (stdlib only, no torch, no network on import)
-------------------------------------------------------------------
1. :meth:`EditionFetcher.discover` parses the rendered **index page** and takes
   the links to its sub-pages *in reading order*; red links are recorded as
   missing; ``list=allpages`` finds sub-pages the index does not link (reported,
   never silently inserted into the reading order).
2. :meth:`MediaWikiClient.resolve` follows normalisation and redirects and
   reports ``missing`` titles instead of failing on the first one.
3. :meth:`EditionFetcher.fetch_page` retrieves the **rendered HTML**
   (``action=parse``, ``redirects=1``). Wikisource pages often transclude their
   text, so ``wikitext`` can be almost empty. It records the page id, revision id
   (a permalink ``oldid``), the templates used (transclusions, licence tags) and
   SHA-256 of both the HTML and the extracted text.
4. :func:`extract_blocks` turns HTML into ordered heading/paragraph blocks,
   skipping editorial chrome (edit links, references, navigation headers, TOC).
5. Site licence information (``meta=siteinfo&siprop=rightsinfo``) is recorded
   together with licence-like page templates. Nothing is inferred beyond that:
   the licence of the underlying edition must still be checked by a person.

The network transport is injectable; tests use canned, SYNTHETIC API responses.
The real fetch is run by the owner (``scripts/fetch_wikisource_edition.py``).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Iterable, Mapping
from urllib.parse import quote, unquote, urlencode

Transport = Callable[[Mapping[str, Any]], dict]

DEFAULT_SKIP_CLASSES = ("mw-editsection", "reference", "references", "mw-references-wrap", "noprint",
                        "ws-noexport", "toc", "navbox", "ws-header", "headertemplate", "mw-jump-link",
                        "printfooter", "catlinks")
_BLOCK_TAGS = {"p", "div", "li", "dd", "dt", "blockquote", "tr", "table", "ul", "ol", "dl", "section",
               "h1", "h2", "h3", "h4", "h5", "h6", "pre", "center", "poem"}
_VOID = {"br", "img", "hr", "meta", "link", "input", "wbr", "source", "col", "area", "base", "embed"}
_LICENCE_HINT = re.compile(r"(PD|public[ _]domain|CC[- ]BY|license|licence|GFDL|ملكية[ _]عامة|رخص)", re.I)


class MediaWikiError(RuntimeError):
    """An API error (``code``/``info``) that retrying will not fix."""

    def __init__(self, code: str, info: str, params: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {info}")
        self.code, self.info, self.params = code, info, dict(params or {})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- HTML -> blocks
@dataclass(frozen=True)
class Block:
    kind: str          # "heading" | "text"
    text: str
    level: int = 0     # heading level 1..6 (0 for text)


class _Extractor(HTMLParser):
    def __init__(self, skip_classes: Iterable[str]):
        super().__init__(convert_charrefs=True)
        self.skip_classes = set(skip_classes)
        self.stack: list[tuple[str, bool]] = []   # (tag, skipping)
        self.blocks: list[Block] = []
        self.buf: list[str] = []
        self.heading: int = 0

    def _skipping(self) -> bool:
        return any(s for _, s in self.stack)

    def _flush(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if text:
            self.blocks.append(Block("heading" if self.heading else "text", text, self.heading))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        classes = set((a.get("class") or "").split())
        skip = tag in ("script", "style", "sup") and (tag != "sup" or "reference" in classes) \
            or bool(classes & self.skip_classes) or a.get("role") == "navigation"
        if tag in _BLOCK_TAGS or tag == "br":
            self._flush()
        if tag in _VOID:
            return
        if re.fullmatch(r"h[1-6]", tag) and not self._skipping() and not skip:
            self.heading = int(tag[1])
        self.stack.append((tag, skip))

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        if tag in _BLOCK_TAGS:
            self._flush()
        if re.fullmatch(r"h[1-6]", tag):
            self.heading = 0
        # pop to the matching tag (tolerates unclosed children)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data: str) -> None:
        if not self._skipping():
            self.buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def extract_blocks(html: str, skip_classes: Iterable[str] = DEFAULT_SKIP_CLASSES) -> list[Block]:
    """Ordered heading/text blocks of rendered MediaWiki HTML, without editorial chrome."""
    p = _Extractor(skip_classes)
    p.feed(html)
    p.close()
    return p.blocks


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        a = dict(attrs)
        href = a.get("href") or ""
        classes = set((a.get("class") or "").split())
        title = a.get("title")
        if href.startswith("/wiki/"):
            title = unquote(href[len("/wiki/"):].split("#")[0]).replace("_", " ")
        elif "redlink=1" in href and title:
            title = re.sub(r"\s*\((?:page does not exist|الصفحة غير موجودة)\)\s*$", "", title)
        else:
            return
        self.links.append({"title": title, "missing": "new" in classes or "redlink=1" in href})


def extract_links(html: str) -> list[dict]:
    """Internal links in document order: ``[{"title", "missing"}]`` (red links marked missing)."""
    p = _LinkExtractor()
    p.feed(html)
    p.close()
    return p.links


# --------------------------------------------------------------------------- client
def urllib_transport(api_url: str, user_agent: str, timeout: float = 35.0, retries: int = 3,
                     pause: float = 1.0) -> Transport:
    """HTTPS GET transport for the real API (used only when called; never at import)."""
    from urllib.request import Request, urlopen

    def call(params: Mapping[str, Any]) -> dict:
        url = api_url + "?" + urlencode({**params, "format": "json", "formatversion": 2})
        last: Exception | None = None
        for attempt in range(retries):
            try:
                with urlopen(Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"}),
                             timeout=timeout) as resp:
                    data = json.load(resp)
                time.sleep(pause)   # be polite to the server
                return data
            except Exception as exc:  # noqa: BLE001 - network errors are retried, API errors are not
                last = exc
                time.sleep(pause * (attempt + 1))
        raise RuntimeError(f"request failed after {retries} attempts: {last}")
    return call


@dataclass
class Resolution:
    requested: str
    title: str | None              # final title after normalisation/redirects; None if missing
    pageid: int | None
    redirected_from: list[str] = field(default_factory=list)
    missing: bool = False


class MediaWikiClient:
    def __init__(self, transport: Transport):
        self.transport = transport

    def call(self, params: Mapping[str, Any]) -> dict:
        data = self.transport(dict(params))
        err = data.get("error")
        if err:
            raise MediaWikiError(str(err.get("code")), str(err.get("info")), params)
        return data

    def resolve(self, titles: Iterable[str]) -> dict[str, Resolution]:
        titles = list(dict.fromkeys(titles))
        out: dict[str, Resolution] = {}
        for i in range(0, len(titles), 50):        # API limit for anonymous clients
            chunk = titles[i:i + 50]
            q = self.call({"action": "query", "titles": "|".join(chunk), "redirects": 1}).get("query", {})
            norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
            redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
            pages = {p["title"]: p for p in q.get("pages", [])}
            for t in chunk:
                cur, chain = norm.get(t, t), []
                seen = {cur}
                while cur in redir:
                    chain.append(cur)
                    cur = redir[cur]
                    if cur in seen:
                        break
                    seen.add(cur)
                page = pages.get(cur, {})
                missing = bool(page.get("missing")) or "pageid" not in page
                out[t] = Resolution(t, None if missing else cur, page.get("pageid"), chain, missing)
        return out

    def allpages(self, prefix: str, namespace: int = 0) -> list[str]:
        params: dict[str, Any] = {"action": "query", "list": "allpages", "apprefix": prefix,
                                  "apnamespace": namespace, "aplimit": "max"}
        titles: list[str] = []
        while True:
            data = self.call(params)
            titles += [p["title"] for p in data.get("query", {}).get("allpages", [])]
            cont = data.get("continue")
            if not cont:
                return titles
            params = {**params, **cont}

    def parse(self, title: str) -> dict:
        return self.call({"action": "parse", "page": title, "prop": "text|revid|templates|displaytitle",
                          "redirects": 1}).get("parse", {})

    def rights(self) -> dict:
        return self.call({"action": "query", "meta": "siteinfo", "siprop": "rightsinfo"}) \
            .get("query", {}).get("rightsinfo", {})


# --------------------------------------------------------------------------- edition
@dataclass
class PageSource:
    order: int
    requested_title: str
    title: str
    pageid: int | None
    revid: int | None
    permalink: str
    redirected_from: list[str]
    templates: list[str]
    licence_templates: list[str]
    html_sha256: str
    text_sha256: str
    blocks: list[Block]

    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)

    def manifest_entry(self) -> dict:
        d = asdict(self)
        d.pop("blocks")
        d["n_blocks"] = len(self.blocks)
        d["headings"] = [{"level": b.level, "text": b.text} for b in self.blocks if b.kind == "heading"]
        return d


@dataclass
class Discovery:
    index_title: str
    ordered: list[str]                          # resolved titles in reading order
    missing: list[dict] = field(default_factory=list)       # red links / missingtitle
    redirects: dict[str, list[str]] = field(default_factory=dict)
    unlinked_subpages: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)


class EditionFetcher:
    def __init__(self, client: MediaWikiClient, site_url: str, skip_classes: Iterable[str] = DEFAULT_SKIP_CLASSES):
        self.client = client
        self.site_url = site_url.rstrip("/")
        self.skip_classes = tuple(skip_classes)

    def discover(self, index_title: str, subpage_prefix: str | None = None) -> Discovery:
        prefix = (subpage_prefix or index_title).rstrip("/") + "/"
        idx = self.client.parse(index_title)
        links = [l for l in extract_links(idx.get("text", "")) if l["title"].startswith(prefix)]
        disc = Discovery(index_title, [])
        candidates = []
        for l in links:
            if l["missing"]:
                disc.missing.append({"title": l["title"], "reason": "red link on the index page"})
            elif l["title"] not in candidates:
                candidates.append(l["title"])
        res = self.client.resolve(candidates)
        for t in candidates:
            r = res[t]
            if r.missing:
                disc.missing.append({"title": t, "reason": "missingtitle"})
                continue
            if r.redirected_from:
                disc.redirects[t] = r.redirected_from + [r.title]
            if r.title in disc.ordered:
                disc.ambiguities.append(f"{t!r} resolves to {r.title!r}, already listed: kept first position")
                continue
            disc.ordered.append(r.title)
        ns_prefix = prefix
        listed = set(disc.ordered) | {t for t in candidates}
        for t in self.client.allpages(ns_prefix):
            if t not in listed:
                disc.unlinked_subpages.append(t)
        if disc.unlinked_subpages:
            disc.ambiguities.append(f"{len(disc.unlinked_subpages)} sub-page(s) exist but are not linked from the "
                                    "index; reading order unknown, so they are NOT included")
        return disc

    def fetch_page(self, title: str, order: int) -> PageSource:
        parsed = self.client.parse(title)
        html = parsed.get("text", "")
        if isinstance(html, dict):
            html = html.get("*", "")
        blocks = extract_blocks(html, self.skip_classes)
        text = "\n\n".join(b.text for b in blocks)
        templates = [t.get("title", "") for t in parsed.get("templates", [])]
        final = parsed.get("title", title)
        revid = parsed.get("revid")
        permalink = (f"{self.site_url}/w/index.php?title={quote(final.replace(' ', '_'))}&oldid={revid}"
                     if revid else f"{self.site_url}/wiki/{quote(final.replace(' ', '_'))}")
        return PageSource(order=order, requested_title=title, title=final, pageid=parsed.get("pageid"),
                          revid=revid, permalink=permalink,
                          redirected_from=[r.get("from") for r in parsed.get("redirects", [])],
                          templates=templates, licence_templates=[t for t in templates if _LICENCE_HINT.search(t)],
                          html_sha256=sha256_text(html), text_sha256=sha256_text(text), blocks=blocks)

    def fetch_edition(self, index_title: str, subpage_prefix: str | None = None,
                      min_arabic_letters: int = 100) -> tuple[list[PageSource], dict]:
        disc = self.discover(index_title, subpage_prefix)
        pages, rejected = [], []
        for i, t in enumerate(disc.ordered):
            try:
                page = self.fetch_page(t, i)
            except MediaWikiError as exc:
                rejected.append({"title": t, "reason": str(exc)})
                continue
            arabic = sum("؀" <= c <= "ۿ" for c in page.text())
            if arabic < min_arabic_letters:
                rejected.append({"title": t, "reason": f"only {arabic} Arabic letters after extraction",
                                 "templates": page.templates})
                continue
            pages.append(page)
        manifest = {
            "schema": "ansg.wikisource_edition/1",
            "site": self.site_url, "index_title": index_title,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "site_rights": self.client.rights(),
            "discovery": asdict(disc),
            "pages": [p.manifest_entry() for p in pages],
            "rejected_pages": rejected,
            "edition_text_sha256": sha256_text("\n\n".join(p.text() for p in pages)),
            "licence_note": "Site licence and page licence templates are recorded as found; the licence of "
                            "the underlying edition must be verified by a person before redistribution.",
        }
        return pages, manifest
