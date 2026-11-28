"""Build a Cultural Knowledge Graph from the regional seed files.

The dissertation's CKG is initialised from CST-Onto (§4.3.2), whose OWL file is
not available. This module builds a small substitute graph, in the
``ansg.knowledge.ckg`` JSON format, from ``data/cultural_seeds``:

* the eight virtue dimensions the dissertation lists for CST-Onto (justice,
  wisdom, courage, temperance, faith, charity, patience, gratitude), plus a
  ninth class, ``social_bond``, added here for relational values such as
  loyalty and neighbourliness that do not fit the eight;
* one ``schema`` node per distinct value concept, attached to a virtue class
  by the explicit table ``VALUE_TO_VIRTUE`` (an editorial choice of this
  implementation, to be reviewed);
* region, proverb, character, setting, opening, closing and dialect-marker
  nodes, linked to their region.

The graph is a stand-in, not CST-Onto, and has far fewer concepts than the 847
the dissertation reports.
"""

from __future__ import annotations

import hashlib
from typing import Any

from ..generation.seed_prompt import REGIONS, load_seed

VIRTUES = ("justice", "wisdom", "courage", "temperance", "faith", "charity", "patience", "gratitude", "social_bond")
VIRTUE_LABELS_AR = {
    "justice": "العدل", "wisdom": "الحكمة", "courage": "الشجاعة", "temperance": "الاعتدال", "faith": "الإيمان",
    "charity": "الإحسان", "patience": "الصبر", "gratitude": "الشكر", "social_bond": "الروابط الاجتماعية",
}

# Editorial mapping of seed value concepts to virtue classes. Every value in the
# seed files must appear here; the unit test fails on an unmapped value.
VALUE_TO_VIRTUE = {
    "الكرم": "charity", "إكرام الضيف": "charity", "البركة": "gratitude",
    "الشهامة": "courage", "النخوة": "courage", "الشجاعة": "courage",
    "الصبر": "patience",
    "الحكمة": "wisdom", "الفطنة": "wisdom", "التدبير": "wisdom",
    "القناعة": "temperance", "التواضع": "temperance",
    "العدل": "justice", "الأمانة": "justice",
    "التوبة وإصلاح الخطأ": "faith",
    "الوفاء": "social_bond", "الوفاء بالعهد": "social_bond", "حفظ العِشرة": "social_bond",
    "صلة الرحم": "social_bond", "حفظ الجيرة": "social_bond", "التعاون": "social_bond",
    "الفزعة والتعاون": "social_bond", "التضامن العائلي": "social_bond", "التكافل بين الجيران": "social_bond",
    "برّ الوالدين": "social_bond", "احترام الكبير": "social_bond", "حفظ ماء الوجه": "social_bond",
    # values introduced by the parallel stories (2026-11-28); same editorial status as above
    "الصدق": "justice", "الحياء": "temperance", "الحلم": "temperance",
    "الإيثار": "charity", "الرحمة": "charity", "العفو": "charity",
    "الشكر": "gratitude",
}
# English labels for values that appear only in the parallel stories (no seed-file entry).
STORY_ONLY_VALUES_EN = {
    "الصدق": "truthfulness", "الحياء": "modesty", "الحلم": "forbearance", "الإيثار": "altruism",
    "الرحمة": "mercy", "العفو": "forgiveness", "الشكر": "gratitude",
}


def _nid(kind: str, text: str) -> str:
    return f"{kind}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def build_seed_ckg(seed_dir=None) -> dict[str, Any]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def node(kind: str, key: str, ar: str, en: str | None = None, **attrs) -> str:
        nid = key if ":" in key else _nid(kind, key)
        if nid not in nodes:
            labels = {"ar": ar}
            if en:
                labels["en"] = en
            nodes[nid] = {"id": nid, "kind": kind, "labels": labels, "attributes": attrs}
        return nid

    def edge(src: str, rel: str, dst: str) -> None:
        edges.append({"source": src, "relation": rel, "target": dst})

    for v in VIRTUES:
        node("virtue", f"virtue:{v}", VIRTUE_LABELS_AR[v], v.replace("_", " "))

    for region in REGIONS:
        seed = load_seed(region, seed_dir)
        r = node("region", f"region:{region}", region, region)
        for value in seed["values"]:
            name = value["concept_ar"]
            if name not in VALUE_TO_VIRTUE:
                raise KeyError(f"value {name!r} ({region}) has no virtue mapping in VALUE_TO_VIRTUE")
            s = node("schema", f"schema:{name}", name, value["concept_en"])
            edge(s, "is_a", f"virtue:{VALUE_TO_VIRTUE[name]}")
            edge(r, "values", s)
        for p in seed["proverbs"]:
            n = node("proverb", p["text"], p["text"], confidence=p["confidence"], meaning=p["meaning_msa"])
            edge(r, "has_proverb", n)
        for kind, field in (("character", "characters"), ("setting", "settings")):
            for text in seed[field]:
                edge(r, f"has_{kind}", node(kind, text, text))
        for oc in seed["openings_closings"]:
            edge(r, f"has_{oc['type']}", node(oc["type"], oc["text"], oc["text"]))
        for m in seed["dialect_markers"]:
            edge(r, "has_marker", node("marker", f"{region}|{m['dialect']}", m["dialect"], msa=m["msa"]))

    # schema nodes for values used by the parallel stories but absent from the seed files
    for name, en in STORY_ONLY_VALUES_EN.items():
        if f"schema:{name}" not in nodes:
            s = node("schema", f"schema:{name}", name, en, source="parallel_stories")
            edge(s, "is_a", f"virtue:{VALUE_TO_VIRTUE[name]}")

    return {
        "metadata": {
            "name": "ANSG seed CKG",
            "source": "data/cultural_seeds (ChatGPT-drafted, pending native review)",
            "note": "Stand-in for CST-Onto; not the dissertation's 847-concept ontology.",
        },
        "ontology_classes": [{"iri": f"virtue:{v}", "label": v} for v in VIRTUES],
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "edges": edges,
    }
