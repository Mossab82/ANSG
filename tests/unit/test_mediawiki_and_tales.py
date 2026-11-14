"""MediaWiki page discovery, rendered-text extraction and tale structure.

Every API response and every Arabic sentence below is SYNTHETIC, written for these
tests in the shape of MediaWiki API output. Nothing is fetched and nothing here is
text from any edition of the Nights.
"""
import pytest

from ansg.data.mediawiki import (
    Block, EditionFetcher, MediaWikiClient, MediaWikiError, extract_blocks, extract_links,
)
from ansg.data.records import SourceProvenance
from ansg.data.splits import SplitConfig
from ansg.data.tale_structure import StructureConfig, analyse_structure, cycle_split, tale_records

ROOT = "كتاب تجريبي"
P1, P2, P3 = f"{ROOT}/الصفحة الأولى", f"{ROOT}/الصفحة الثانية", f"{ROOT}/الصفحة الثالثة"
OLD = f"{ROOT}/اسم قديم"


def href(title):
    from urllib.parse import quote
    return "/wiki/" + quote(title.replace(" ", "_"))


INDEX_HTML = f"""<div class="mw-parser-output">
<div class="ws-header">رأس الصفحة <a href="{href('مؤلف')}">مؤلف</a></div>
<ul><li><a href="{href(P1)}">١</a></li>
<li><a href="{href(OLD)}">٢</a></li>
<li><a class="new" href="/w/index.php?title=x&action=edit&redlink=1" title="{P3} (الصفحة غير موجودة)">٣</a></li>
<li><a href="{href(P1)}">مكرر</a></li></ul></div>"""

PAGE1_HTML = """<div class="mw-parser-output">
<div class="mw-heading mw-heading2"><h2>حكاية التاجر والطائر<span class="mw-editsection">[تحرير]</span></h2></div>
<p>الليلة الأولى</p>
<p>كان تاجر يسافر بين المدن ويحمل بضاعته على ظهر جمله الصبور كل صباح.<sup class="reference">[1]</sup></p>
<p>فقال التاجر لرفاقه: اسمعوا وما حكاية الطائر الذي رأيته؟</p>
<h3>حكاية الطائر الذهبي</h3>
<p>طار الطائر فوق الجبال حتى وصل إلى بستان بعيد فيه شجرة عالية.</p>
<style>.x{color:red}</style><div class="noprint">انتقل إلى الصفحة التالية</div>
</div>"""

PAGE2_HTML = """<div class="mw-parser-output">
<p>الليلة الثانية</p>
<h2>حكاية الصياد والبحر</h2>
<p>ألقى الصياد شبكته في البحر فلم يجد شيئا ثم ألقاها ثانية فوجد صندوقا مغلقا.</p>
<p>فقال الصياد لابنه: أحكي لك ما جرى لي</p>
<p>حكاية الصندوق المغلق</p>
<p>كان في الصندوق مفتاح صغير ورسالة قديمة لا يعرف أحد كاتبها.</p>
</div>"""


class FakeAPI:
    """Canned SYNTHETIC MediaWiki responses keyed on the request parameters."""

    def __init__(self):
        self.calls = []

    def __call__(self, params):
        self.calls.append(params)
        action = params["action"]
        if action == "parse":
            page = params["page"]
            if page == ROOT:
                return {"parse": {"title": ROOT, "pageid": 1, "revid": 10, "text": INDEX_HTML, "templates": []}}
            if page == P1:
                return {"parse": {"title": P1, "pageid": 2, "revid": 20, "text": PAGE1_HTML,
                                  "templates": [{"ns": 10, "title": "قالب:PD-old"}, {"ns": 10, "title": "قالب:رأس"}]}}
            if page == P2:
                return {"parse": {"title": P2, "pageid": 3, "revid": 30, "text": PAGE2_HTML, "templates": [],
                                  "redirects": [{"from": OLD, "to": P2}]}}
            return {"error": {"code": "missingtitle", "info": "The page you specified doesn't exist."}}
        if action == "query" and params.get("list") == "allpages":
            if "apcontinue" not in params:
                return {"query": {"allpages": [{"title": P1}, {"title": P2}]}, "continue": {"apcontinue": "x"}}
            return {"query": {"allpages": [{"title": f"{ROOT}/ملحق"}]}}
        if action == "query" and params.get("meta") == "siteinfo":
            return {"query": {"rightsinfo": {"url": "https://example.invalid/licence", "text": "synthetic licence"}}}
        if action == "query":
            titles = params["titles"].split("|")
            pages = []
            for t in titles:
                target = P2 if t == OLD else t
                pages.append({"title": target, "pageid": 2 if target == P1 else 3} if target in (P1, P2)
                             else {"title": target, "missing": True})
            return {"query": {"redirects": [{"from": OLD, "to": P2}] if OLD in titles else [], "pages": pages}}
        raise AssertionError(params)


@pytest.fixture
def fetcher():
    return EditionFetcher(MediaWikiClient(FakeAPI()), "https://wiki.example.invalid")


def test_blocks_skip_editorial_chrome():
    blocks = extract_blocks(PAGE1_HTML)
    texts = [b.text for b in blocks]
    assert blocks[0] == Block("heading", "حكاية التاجر والطائر", 2)          # edit link removed
    assert not any("[1]" in t or "انتقل" in t or "color" in t for t in texts)
    assert Block("heading", "حكاية الطائر الذهبي", 3) in blocks


def test_links_in_reading_order_with_red_links():
    links = extract_links(INDEX_HTML)
    assert [l["title"] for l in links if not l["missing"]][:3] == ["مؤلف", P1, OLD]
    assert [l for l in links if l["missing"]] == [{"title": P3, "missing": True}]


def test_discovery_follows_redirects_and_reports_gaps(fetcher):
    d = fetcher.discover(ROOT)
    assert d.ordered == [P1, P2]                                   # reading order, duplicate dropped
    assert d.redirects == {OLD: [OLD, P2]}
    assert {m["title"] for m in d.missing} == {P3}                 # red link recorded, not fatal
    assert d.unlinked_subpages == [f"{ROOT}/ملحق"]                 # found via allpages (continued)
    assert any("NOT included" in a for a in d.ambiguities)


def test_edition_manifest_records_provenance(fetcher):
    pages, manifest = fetcher.fetch_edition(ROOT, min_arabic_letters=20)
    assert [p.title for p in pages] == [P1, P2]
    e = manifest["pages"][0]
    assert e["revid"] == 20 and "oldid=20" in e["permalink"] and len(e["text_sha256"]) == 64
    assert e["licence_templates"] == ["قالب:PD-old"]
    assert manifest["site_rights"]["text"] == "synthetic licence"
    assert manifest["pages"][1]["redirected_from"] == [OLD]
    assert len(manifest["edition_text_sha256"]) == 64


def test_missing_page_is_an_error_not_a_guess():
    client = MediaWikiClient(FakeAPI())
    with pytest.raises(MediaWikiError) as exc:
        client.parse(P3)
    assert exc.value.code == "missingtitle"
    assert client.resolve([P3])[P3].missing


def test_short_pages_are_rejected_with_reason(fetcher):
    _, manifest = fetcher.fetch_edition(ROOT, min_arabic_letters=10_000)
    assert len(manifest["rejected_pages"]) == 2 and "Arabic letters" in manifest["rejected_pages"][0]["reason"]


# ---------------------------------------------------------------- tale structure
def blocks():
    return extract_blocks(PAGE1_HTML) + extract_blocks(PAGE2_HTML)


def test_heading_hierarchy_gives_nesting_and_nights():
    rep = analyse_structure(blocks())
    titles = [(t.title, t.parent, t.nesting_evidence) for t in rep.tales]
    assert titles[0] == ("حكاية التاجر والطائر", None, None)
    assert titles[1] == ("حكاية الطائر الذهبي", "tale-0001", "heading_hierarchy")
    assert titles[2] == ("حكاية الصياد والبحر", None, None)     # h2 closes the h2/h3 cycle
    assert rep.tales[0].nights == [1] and rep.tales[2].nights == [2]
    assert sorted(rep.night_blocks.values()) == [1, 2]


def test_textual_cue_is_only_a_candidate_unless_accepted():
    rep = analyse_structure(blocks())
    box = rep.tales[3]
    assert box.title == "حكاية الصندوق المغلق" and box.heading_level is None
    assert box.parent is None and box.candidate_parent == "tale-0003"
    assert any("candidate only" in a for a in rep.ambiguities)
    accepted = analyse_structure(blocks(), StructureConfig(accept_textual_cues=True)).tales[3]
    assert (accepted.parent, accepted.nesting_evidence) == ("tale-0003", "textual_cue")


def test_ambiguous_sibling_or_embedded_is_logged_not_invented():
    b = [Block("heading", "حكاية الأول", 2), Block("text", "نص الحكاية الأولى الطويل هنا."),
         Block("heading", "حكاية الثاني", 2), Block("text", "نص آخر.")]
    rep = analyse_structure(b)
    assert all(t.parent is None for t in rep.tales)
    assert any("no markup or cue decides" in a for a in rep.ambiguities)


def test_records_and_story_cycle_split_keep_cycles_together():
    rep = analyse_structure(blocks(), StructureConfig(accept_textual_cues=True))
    recs = tale_records(blocks(), rep, SourceProvenance(source_id="synth", title="synthetic"))
    assert recs[1].narrative.parent_story_id == "synth:tale-0001"
    assert recs[1].narrative.extra["nesting_evidence"] == "heading_hierarchy"
    audit = cycle_split(recs, SplitConfig(ratios={"train": 0.5, "test": 0.5}, seed=1))
    assert audit["cycles"] == 2 and audit["leaking_cycles"] == []
    split = audit["assignment"]
    assert split["synth:tale-0001"] == split["synth:tale-0002"]
    assert split["synth:tale-0003"] == split["synth:tale-0004"]


def test_unparsed_night_headings_are_reported():
    rep = analyse_structure([Block("text", "الليلة الكذا"), Block("heading", "حكاية ما", 2),
                             Block("text", "نص.")])
    assert rep.unparsed_night_headings == ["الليلة الكذا"]


def test_plural_and_ordinal_tale_titles_are_recognised():
    """Real heading forms found in the pinned Wikisource pages (evidence run 43564395466):
    the frame tale uses the plural 'حكايات', Sinbad's voyages use 'الحكاية الثانية ...'."""
    b = [Block("heading", "حكايات الملك وأخيه", 2), Block("text", "نص الإطار."),
         Block("heading", "حكاية التاجر", 2), Block("text", "نص."),
         Block("heading", "الحكاية الثانية من حكايات المسافر", 2), Block("text", "نص.")]
    rep = analyse_structure(b)
    assert [t.title for t in rep.tales] == ["حكايات الملك وأخيه", "حكاية التاجر", "الحكاية الثانية من حكايات المسافر"]
    assert all(t.parent is None for t in rep.tales)        # equal heading levels: no nesting inferred
