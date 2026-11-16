import json
import sys
import types

import numpy as np
import pytest

from ansg.dialects.qcri_classifier import QCRIDialectClassifier, SUPPORTED_LABELS
from scripts.audit_qcri_story_dialects import load_story_rows, summarize


class FakeVectorizer:
    def transform(self, texts):
        return texts


class FakeClassifier:
    classes_ = np.array(["Egyptian", "Gulf", "Levantine", "Maghrebi"])

    def predict(self, features):
        return ["Levantine" for _ in features]

    def decision_function(self, features):
        return np.array([[0.1, 0.2, 1.5, -0.4] for _ in features])


def test_classifier_returns_supported_label_and_raw_margins():
    classifier = QCRIDialectClassifier({
        "vectorizer": FakeVectorizer(),
        "classifier": FakeClassifier(),
        "labels": list(SUPPORTED_LABELS),
    })

    result = classifier.predict("كان يا ما كان")

    assert result.label == "Levantine"
    assert result.score_type == "raw_linear_svc_decision_margin"
    assert result.decision_scores["Levantine"] == 1.5
    assert "probability" not in result.score_type


def test_classifier_rejects_artifact_with_msa_instead_of_qcri_labels():
    with pytest.raises(ValueError, match="expected QCRI labels"):
        QCRIDialectClassifier({
            "vectorizer": FakeVectorizer(),
            "classifier": FakeClassifier(),
            "labels": ["MSA", *SUPPORTED_LABELS[:3]],
        })


def test_classifier_rejects_empty_text():
    classifier = QCRIDialectClassifier({
        "vectorizer": FakeVectorizer(),
        "classifier": FakeClassifier(),
        "labels": list(SUPPORTED_LABELS),
    })
    with pytest.raises(ValueError, match="non-empty"):
        classifier.predict("  ")


def test_story_loader_rejects_qcri_as_story_source(tmp_path):
    source = tmp_path / "stories.jsonl"
    source.write_text(json.dumps({
        "story_id": "1", "source_corpus": "QCRI/arabic_pos_dialect", "text": "نص"
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="distinct from QCRI"):
        load_story_rows(source)


def test_summary_excludes_msa_from_four_way_metrics():
    rows = [
        {"_target": "Egyptian"},
        {"_target": "MSA"},
        {"_target": None},
    ]
    predictions = [
        {"prediction": "Egyptian"},
        {"prediction": "Maghrebi"},
        {"prediction": "Gulf"},
    ]

    metrics = summarize(rows, predictions)

    assert metrics["scored_story_count"] == 1
    assert metrics["accuracy"] == 1.0
    assert metrics["unsupported_target_count"] == 1
    assert metrics["unlabeled_story_count"] == 1


def test_loader_rejects_duplicate_story_ids(tmp_path):
    row = {"story_id": "same", "source_corpus": "independent_stories", "text": "نص"}
    source = tmp_path / "stories.jsonl"
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate story_id"):
        load_story_rows(source)
