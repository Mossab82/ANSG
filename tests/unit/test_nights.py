from ansg.data.nights import NightsConfig, find_night_markers, ingest_file, segment_stories
from ansg.data.records import CopyrightStatus, SourceProvenance

PROV = SourceProvenance(source_id="synth", title="synthetic fixture",
                        copyright_status=CopyrightStatus.PUBLIC_DOMAIN, redistributable=True)


def test_story_and_night_segmentation(fixtures):
    recs, rep = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    assert [r.narrative.title for r in recs] == [None, "حكاية الحطاب والطائر", "حكاية الصياد والخاتم"]
    assert [r.narrative.story_index for r in recs] == [0, 1, 2]
    assert recs[1].narrative.night_numbers == [1, 2]
    # second tale starts during night 2 and spans nights 11 and 121
    assert recs[2].narrative.night_numbers == [2, 11, 121]
    assert rep.n_night_headings == 4 and rep.n_unparsed_night_headings == 0
    assert rep.night_number_gaps == [(2, 11), (11, 121)]
    assert rep.night_number_regressions == []


def test_formulae_detected_not_stripped(fixtures):
    recs, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    tale = recs[1]
    types = [f["type"] for f in tale.narrative.formulae]
    assert types == ["night_opening", "night_closing", "night_opening"]
    for f in tale.narrative.formulae:
        assert tale.text[f["start"]:f["end"]] == f["text"]
    assert "بلغني أيها الملك السعيد" in tale.text


def test_normalization_applied_and_provenance_recorded(fixtures):
    recs, rep = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    assert "ـ" not in recs[1].text  # tatweel removed
    assert "لا يمل" in recs[2].text  # presentation-form ligature folded
    assert all(r.source.file_sha256 == rep.file_sha256 for r in recs)
    assert all(r.normalization_config_sha256 for r in recs)
    assert all(r.annotation_status["motif"].value == "unannotated" for r in recs)


def test_record_ids_deterministic(fixtures):
    a, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    b, _ = ingest_file(fixtures / "nights" / "synthetic_nights.txt", PROV)
    assert [r.record_id for r in a] == [r.record_id for r in b]


def test_unparsed_heading_reported_not_used():
    text = "الليلة الأولى\nنص\nليلة سعيدة جدا\nنص"
    markers = find_night_markers(text, NightsConfig())
    assert [m.number for m in markers] == [1, None]
    units, _ = segment_stories(text, NightsConfig())
    assert units[0].nights == [1]
    assert units[0].unparsed_night_headings == ["ليلة سعيدة جدا"]


def test_no_markers_warns(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("نص بلا عناوين.", encoding="utf-8")
    _, rep = ingest_file(p, PROV)
    assert any("no night headings" in w for w in rep.warnings)
    assert any("no tale titles" in w for w in rep.warnings)
