import pytest

from ansg.knowledge.arabic_namenet import ArabicNameNet, TableMapping

ENT = dict(path="synthetic_entities.tsv", format="tsv",
           columns={"id": "entity_id", "name": "name", "type": "type", "variants": "variants"},
           variant_separator="|", attribute_columns=["gender"])
REL = dict(path="synthetic_relations.tsv", format="tsv",
           columns={"source": "source", "relation": "rel", "target": "target"})


@pytest.fixture
def nn(fixtures):
    return ArabicNameNet.load(TableMapping.from_mapping(ENT), TableMapping.from_mapping(REL),
                              base_dir=fixtures / "namenet")


def test_load_entities_and_attributes(nn):
    assert nn.stats() == {"entities": 4, "relations": 3,
                          "entity_types": {"person": 3, "place": 1}, "dangling_relations": 1}
    assert nn.entities["e2"].variants == ["شهرازاد", "شهرزاد"]
    assert nn.entities["e1"].attributes == {"gender": "m"}


def test_lookup_with_folding(nn):
    assert [e.entity_id for e in nn.lookup("شَهْرَزَاد")] == ["e2"]
    assert [e.entity_id for e in nn.lookup("شهرازاد")] == ["e2"]


def test_relations(nn):
    assert [r.target_id for r in nn.related("e1")] == ["e3", "e9"]
    assert nn.dangling_relations()[0].target_id == "e9"


def test_find_in_text_longest_match(nn):
    text = "حكم شاه الزمان سمرقند وكان شهريار أخاه"
    hits = nn.find_in_text(text)
    assert [(h.surface, h.entity_ids) for h in hits] == [
        ("شاه الزمان", ("e3",)), ("سمرقند", ("e4",)), ("شهريار", ("e1",))]
    assert all(text[h.start:h.end] == h.surface for h in hits)


def test_mapping_errors(fixtures):
    bad = TableMapping.from_mapping({**ENT, "columns": {"name": "nom"}})
    with pytest.raises(KeyError):
        ArabicNameNet.load(bad, base_dir=fixtures / "namenet")
    with pytest.raises(ValueError):
        ArabicNameNet.load(TableMapping.from_mapping({**ENT, "columns": {"id": "entity_id"}}),
                           base_dir=fixtures / "namenet")
    with pytest.raises(FileNotFoundError):
        ArabicNameNet.load(TableMapping.from_mapping({**ENT, "path": "nope.tsv"}),
                           base_dir=fixtures / "namenet")
