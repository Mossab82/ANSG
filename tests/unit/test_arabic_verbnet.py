import pytest

from ansg.knowledge.arabic_verbnet import ArabicVerbNet, UnsafeXMLError


@pytest.fixture
def vn(fixtures):
    return ArabicVerbNet.from_directory(fixtures / "verbnet")


def test_classes_members_and_subclasses(vn):
    assert set(vn.classes) == {"test-write-1", "test-write-1-1", "synthetic_class_b#2"}
    top = vn.classes["test-write-1"]
    assert top.subclass_ids == ["test-write-1-1"]
    assert vn.classes["test-write-1-1"].parent_id == "test-write-1"
    m = top.members[0]
    assert (m.verb, m.root, m.deverbals, m.participles) == ("كتب", "ك ت ب", ["كتابة", "كتب"], ["كاتب"])
    assert m.attributes["name"] == "kataba"


def test_roles_restrictions_and_inheritance(vn):
    top = vn.classes["test-write-1"]
    assert [r.type for r in top.roles] == ["Agent", "Theme"]
    assert top.roles[0].restrictions[0].value == "+" and top.roles[0].restrictions[0].type == "animate"
    sub = vn.classes["test-write-1-1"]
    assert [r.type for r in sub.inherited_roles(vn)] == ["Recipient", "Agent", "Theme"]


def test_frames_syntax_semantics(vn):
    f0, f1 = vn.classes["test-write-1"].frames
    assert f0.primary == "NP V NP" and f0.secondary == "Basic Transitive" and f0.description_number == "0.1"
    assert f0.examples == ["كتب الولد الرسالة"]
    assert [s.tag for s in f0.syntax] == ["VERB", "NP", "NP"]
    assert f0.noun_phrase_roles == ["Agent", "Theme"]
    assert [(p.value, p.negated) for p in f0.semantics] == [("cause", False), ("exist", True)]
    assert f0.semantics[0].args[1].type == "Event"
    assert f1.prepositions == ["على"]


def test_indexes_and_stats(vn):
    assert vn.verbs_by_root()["ك ت ب"] == ["كتب"]
    assert [c.class_id for c in vn.classes_for_verb("ذهب")] == ["synthetic_class_b#2"]
    assert vn.stats() == {"files": 2, "classes": 3, "members": 4, "frames": 3, "roles": 3,
                          "members_with_root": 4}


def test_legacy_rows_shape(vn):
    rows = vn.legacy_rows()
    kataba = next(r for r in rows if r["Verb"] == "كتب")
    assert kataba == {"Verb": "كتب", "Root": "ك ت ب", "Deverbal_1": "كتابة", "Deverbal_2": "كتب",
                      "Deverbal_3": None, "Pariciple": "كاتب", "Catagory": "1"}
    assert next(r for r in rows if r["Verb"] == "ذهب")["Catagory"] == "2"


def test_deterministic_order(fixtures):
    p = sorted((fixtures / "verbnet").glob("*.xml"))
    assert list(ArabicVerbNet.from_paths(p).classes) == list(ArabicVerbNet.from_paths(p[::-1]).classes)


def test_rejects_entity_declarations(fixtures):
    with pytest.raises(UnsafeXMLError):
        ArabicVerbNet.from_paths([fixtures / "verbnet" / "unsafe_entity.xml.txt"])


def test_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        ArabicVerbNet.from_directory(tmp_path)
