import json

import pytest

from ansg.data.anhc import import_annotations, read_jsonl, write_jsonl
from ansg.data.records import (
    AnnotationSource, AnnotationStatus, CopyrightStatus, DialectAnnotation, LabelInventory,
    MotifAnnotation, NarrativeRecord, SourceProvenance, Span, validate_record,
)


def make(rid="r1", text="كان يا ما كان", redistributable=True):
    return NarrativeRecord(record_id=rid, text=text, source=SourceProvenance(
        source_id="s", title="t", copyright_status=CopyrightStatus.PUBLIC_DOMAIN,
        redistributable=redistributable))


def test_new_record_is_unannotated_and_valid():
    r = make()
    assert all(s == AnnotationStatus.UNANNOTATED for s in r.annotation_status.values())
    assert validate_record(r) == []


def test_annotations_with_unannotated_status_rejected():
    r = make()
    r.dia.append(DialectAnnotation(span=None, annotator_id="a", source=AnnotationSource.HUMAN_SINGLE, label="X"))
    assert any(i.severity == "error" for i in validate_record(r))


def test_gold_layer_rules():
    r = make()
    r.annotation_status["motif"] = AnnotationStatus.GOLD
    r.motif.append(MotifAnnotation(span=Span(0, 3), annotator_id=None,
                                   source=AnnotationSource.AUTOMATIC, motif_id="m"))
    msgs = [i.message for i in validate_record(r)]
    assert any("non-adjudicated" in m for m in msgs)
    assert any("automatic annotation" in m for m in msgs)


def test_span_and_label_inventory_checks():
    r = make()
    r.annotation_status["dia"] = AnnotationStatus.SILVER
    r.dia.append(DialectAnnotation(span=Span(0, 999), annotator_id="a",
                                   source=AnnotationSource.HUMAN_SINGLE, label="ZZ"))
    msgs = [i.message for i in validate_record(r, LabelInventory(dia=frozenset({"MSA"})))]
    assert any("exceeds text length" in m for m in msgs)
    assert any("not in configured inventory" in m for m in msgs)
    # without an inventory: warning only
    sev = {i.severity for i in validate_record(make()) }
    assert "error" not in sev


def test_jsonl_roundtrip_and_restricted_text_withheld(tmp_path):
    a, b = make("a"), make("b", redistributable=False)
    p = tmp_path / "x.jsonl"
    assert write_jsonl([a, b], p) == 2
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == a.text and rows[1]["text"] is None
    with pytest.raises(ValueError, match="no text"):
        list(read_jsonl(p))
    write_jsonl([a, b], p, allow_restricted_text=True)
    back = list(read_jsonl(p))
    assert [r.record_id for r in back] == ["a", "b"] and back[1].text == b.text


def test_import_annotations(tmp_path):
    recs = [make("a"), make("b")]
    p = tmp_path / "dia.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in [
        {"record_id": "a", "span": None, "annotator_id": "ann1", "source": "human_adjudicated", "label": "MSA"},
        {"record_id": "zzz", "span": None, "label": "MSA"},
    ]), encoding="utf-8")
    counts = import_annotations(recs, p, "dia", AnnotationStatus.GOLD)
    assert counts == {"imported": 1, "unknown_record": 1}
    assert recs[0].annotation_status["dia"] == AnnotationStatus.GOLD and recs[0].dia[0].label == "MSA"
    assert recs[1].annotation_status["dia"] == AnnotationStatus.UNANNOTATED
    assert validate_record(recs[0], LabelInventory(dia=frozenset({"MSA"}))) == []


def test_import_refuses_automatic_as_gold(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"record_id": "a", "source": "automatic", "motif_id": "m"}), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be gold"):
        import_annotations([make("a")], p, "motif", AnnotationStatus.GOLD)
