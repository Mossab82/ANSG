import pytest

from ansg.data.nights import NightsConfig, find_formulae
from ansg.data.normalization import NormalizationConfig, normalize
from ansg.data.records import NarrativeRecord, SourceProvenance
from ansg.data.splits import SplitConfig, assign_splits, check_no_leakage, cluster_records_dbscan
from ansg.knowledge.ckg import CosineThresholdActivator, CulturalKnowledgeGraph, load_declared_variants
from ansg.utils.config import load_yaml_config


def test_appendix_a8_profile():
    cfg = NormalizationConfig.from_mapping({"profile": "thesis_appendix_a8"})
    assert cfg.unicode_form == "NFKC" and cfg.normalize_alef and not cfg.strip_diacritics
    out = normalize("إِلَى أَحَدٍ ـ", cfg).text
    assert out.startswith("اِلَى اَحَدٍ")  # alef folded, diacritics kept, tatweel dropped
    over = NormalizationConfig.from_mapping({"profile": "thesis_appendix_a8", "normalize_alef": False})
    assert not over.normalize_alef
    with pytest.raises(ValueError):
        NormalizationConfig.from_mapping({"profile": "nope"})


def test_table_5_4_formulae_folded_matching_maps_offsets():
    text = "كَانَ يَا مَا كَانَ في قديم الزمان، وَاللهُ أَعْلَم."
    cfg = NightsConfig(include_thesis_table_5_4_formulae=True)
    found = find_formulae(text, cfg)
    types = [f["type"] for f in found]
    assert types == ["t5.4:opening", "t5.4:temporal_anchor", "t5.4:epistemic_closure"]
    for f in found:
        assert text[f["start"]:f["end"]] == f["text"]
    assert not find_formulae(text, NightsConfig())  # off unless enabled


def test_cosine_activator_two_thesis_variants(fixtures):
    g = CulturalKnowledgeGraph.from_json(fixtures / "ckg" / "synthetic_ckg.json")
    emb = {"schema:test_hospitality": [1.0, 0.0], "motif:test_magic_ring": [0.8, 0.6],
           "motif:test_transformation": [0.0, 1.0]}
    enc = lambda _text: [1.0, 0.0]  # noqa: E731
    ch4 = CosineThresholdActivator(emb, enc, threshold=0.5, inclusive=False).activate("x", g)
    assert ch4.scores == {"schema:test_hospitality": 1.0, "motif:test_magic_ring": pytest.approx(0.8)}
    a4 = CosineThresholdActivator(emb, enc, threshold=0.8, inclusive=True, max_active=1).activate("x", g)
    assert list(a4.scores) == ["schema:test_hospitality"]
    with pytest.raises(ValueError):
        CosineThresholdActivator({"a": [1.0], "b": [1.0, 0.0]}, enc, 0.5, False)


def test_shipped_ckg_config_keeps_conflicting_declarations():
    cfg = load_yaml_config("configs/knowledge/ckg.yaml")
    variants = load_declared_variants(cfg["declared_variants"])
    assert {"cst_onto_ch3", "cst_onto_table_a16", "ckg_ch4_table_4_8", "ckg_appendix_a"} <= set(variants)
    assert variants["cst_onto_table_a16"].other_counts["total_entities_stated"] == 1847


def _rec(i, text):
    return NarrativeRecord(record_id=f"r{i}", text=text, source=SourceProvenance("s", "t"))


def test_dbscan_groups_parallel_witnesses_into_one_split():
    pytest.importorskip("sklearn")
    a = "كان التاجر يسافر إلى بغداد ويبيع الحرير في السوق الكبير"
    recs = [_rec(0, a), _rec(1, a + " قديما"),  # two witnesses of one tale
            _rec(2, "خرج الصياد إلى البحر فوجد خاتما في بطن سمكة"),
            _rec(3, "تعلم الشاب الحكمة من الشيخ على جبل قاف"),
            _rec(4, "حكم الملك بالعدل بين الناس وأكرم الضيوف")]
    clusters = cluster_records_dbscan(recs)
    assert clusters["r0"] == clusters["r1"] and clusters["r0"].startswith("dbscan:")
    assert clusters["r2"].startswith("noise:")
    assign_splits(recs, SplitConfig(group_by="split_cluster",
                                    ratios={"train": 0.5, "validation": 0.25, "test": 0.25}))
    assert check_no_leakage(recs, lambda r: r.narrative.extra["split_cluster"]) == []
    assert recs[0].split == recs[1].split
