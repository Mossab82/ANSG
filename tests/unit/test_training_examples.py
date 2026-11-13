"""ANHC -> training-example adapter (Issue #11). Torch-free; synthetic fixtures only."""
import json

import pytest

from ansg.data.anhc import write_jsonl
from ansg.data.nights import ingest_file
from ansg.data.records import (
    AnnotationSource, AnnotationStatus, CopyrightStatus, DialectAnnotation, NarrativeRecord,
    SourceProvenance, Span,
)
from ansg.data.training_examples import (
    DIALECT_TABLE, SchemaVocabulary, build_example, build_examples_from_jsonl, dialect_code,
    document_dialect, main,
)
from ansg.knowledge.ckg import CulturalKnowledgeGraph

PROV = SourceProvenance(source_id="synth", title="synthetic fixture",
                        copyright_status=CopyrightStatus.PUBLIC_DOMAIN, redistributable=True)


@pytest.fixture
def graph(fixtures):
    return CulturalKnowledgeGraph.from_json(fixtures / "ckg" / "synthetic_ckg.json")


@pytest.fixture
def vocab(graph, fixtures):
    return SchemaVocabulary.from_graph(graph, source_path=fixtures / "ckg" / "synthetic_ckg.json")


@pytest.fixture
def tales(fixtures):
    recs, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    return [r for r in recs if r.narrative.title]


def _dia(rec, label, span=None):
    rec.dia.append(DialectAnnotation(span=span, annotator_id=None, source=AnnotationSource.AUTOMATIC,
                                     attributes={"synthetic": True}, label=label))
    rec.annotation_status["dia"] = AnnotationStatus.AUTOMATIC


def test_dialect_table_matches_thesis_labels_and_dal_order():
    assert [(l, c, i) for l, c, i in DIALECT_TABLE] == [
        ("MSA", "msa", 0), ("Egyptian", "egyptian", 1), ("Levantine", "levantine", 2),
        ("Gulf", "gulf", 3), ("Maghrebi", "maghrebi", 4)]
    assert dialect_code("Gulf") == dialect_code("gulf") == ("gulf", 3)
    for unknown in ("Iraqi", "Mixed", "msa ", ""):
        with pytest.raises(ValueError):
            dialect_code(unknown)


def test_document_dialect_never_invents(tales):
    rec = tales[0]
    assert document_dialect(rec) == (None, "dia_unannotated")
    _dia(rec, "Levantine", span=Span(0, 5))
    assert document_dialect(rec) == (None, "span_level_dia_only")
    _dia(rec, "MSA")
    assert document_dialect(rec) == ("msa", "document_level_dia")
    _dia(rec, "Egyptian")
    assert document_dialect(rec) == (None, "conflicting_document_level_dia")


def test_schema_vocabulary_reserves_pad(graph, vocab):
    assert vocab.schema_count == len(graph.nodes) + 1
    assert sorted(vocab.node_to_id.values()) == list(range(1, len(graph.nodes) + 1))
    assert 0 not in vocab.node_to_id.values()
    assert vocab.encode(["motif:test_magic_ring"]) == [vocab.node_to_id["motif:test_magic_ring"]]
    with pytest.raises(KeyError):
        vocab.encode(["motif:not_in_graph"])
    assert vocab.decode([0, vocab.node_to_id["schema:test_hospitality"], 0]) == ["schema:test_hospitality"]
    assert SchemaVocabulary.from_dict(vocab.to_dict()) == vocab
    assert len(vocab.source_sha256) == 64
    with pytest.raises(ValueError):
        SchemaVocabulary(node_to_id={"a": 0})


def test_build_example_keeps_unknowns_unknown(tales, graph, vocab):
    first, second = tales
    motif_before = list(second.motif)
    ex1 = build_example(first, graph, vocab)
    ex2 = build_example(second, graph, vocab)
    # First tale: no trigger -> no schema; no DIA -> unknown dialect (not MSA).
    assert ex1.schema_ids == () and ex1.dialect_id is None and ex1.dialect_reason == "dia_unannotated"
    # Second tale contains the synthetic trigger "خاتمًا".
    assert ex2.schema_node_ids == ("motif:test_magic_ring",)
    assert ex2.schema_ids == (vocab.node_to_id["motif:test_magic_ring"],)
    assert ex2.activation_method == "lexical_trigger_count"
    assert second.motif == motif_before  # activation is never written back as MOTIF annotation
    row = ex1.to_core_row()
    assert row["dialect_id"] is None and row["schema_ids"] == []


def test_dialect_from_document_level_dia(tales, graph, vocab):
    _dia(tales[1], "Gulf")
    ex = build_example(tales[1], graph, vocab)
    assert (ex.dialect, ex.dialect_id, ex.dialect_label_status) == ("gulf", 3, "automatic")


def test_withheld_text_is_skipped(tmp_path, tales, graph, vocab):
    restricted = NarrativeRecord(record_id="r-x", text="نص محمي", source=SourceProvenance("s", "t"))
    p = tmp_path / "train.jsonl"
    write_jsonl(tales + [restricted], p)  # restricted record is written with text = null
    examples, stats = build_examples_from_jsonl(p, graph, vocab)
    assert stats == {"records": 3, "skipped_withheld_text": 1, "with_dialect": 0, "with_schemas": 1}
    assert [e.record_id for e in examples] == [t.record_id for t in tales]


def test_cli_writes_examples_vocabulary_and_manifest(tmp_path, tales, fixtures):
    rec_dir = tmp_path / "records"
    write_jsonl(tales, rec_dir / "train.jsonl")
    out = tmp_path / "examples"
    assert main(["--records-dir", str(rec_dir), "--ckg", str(fixtures / "ckg" / "synthetic_ckg.json"),
                 "--output", str(out)]) == 0
    rows = [json.loads(l) for l in (out / "train.examples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {"text", "schema_ids", "dialect_id", "record_id"} <= set(rows[0])
    vocab = json.loads((out / "schema_vocabulary.json").read_text(encoding="utf-8"))
    assert vocab["pad_id"] == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert any("inspection baseline" in w for w in manifest["warnings"])
    assert main(["--records-dir", str(tmp_path / "empty"), "--ckg", str(fixtures / "ckg" / "synthetic_ckg.json"),
                 "--output", str(tmp_path / "o2")]) == 2
