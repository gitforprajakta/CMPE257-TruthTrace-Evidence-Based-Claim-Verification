"""Preprocessing contract tests on synthetic AVeriTeC-like records."""
from __future__ import annotations

import pytest

from src import config as C
from src.preprocess import PROCESSED_COLUMNS, normalize_label, preprocess_records

MARKER = "JUSTIFICATION_MARKER_XYZ"


def _answer(text: str, answer_type: str = "Extractive", explanation: str = "") -> dict:
    return {
        "answer": text,
        "answer_type": answer_type,
        "boolean_explanation": explanation,
        "source_url": "https://example.test",
        "source_medium": "web",
    }


def _record(claim: str, label: str, questions=None, justification: str = "") -> dict:
    return {
        "claim": claim,
        "label": label,
        "justification": justification,
        "questions": questions if questions is not None else [],
        "speaker": "Ada",
        "claim_date": "2020-01-02",
    }


def test_label_normalization_variants():
    assert normalize_label("SUPPORTED") == "Supported"
    assert normalize_label("refuted") == "Refuted"
    assert normalize_label("Not Enough Information") == "Not Enough Evidence"
    assert normalize_label("Not Enough Evidence") == "Not Enough Evidence"
    assert normalize_label("Conflicting Evidence") == "Conflicting Evidence/Cherrypicking"
    assert normalize_label("Conflicting Evidence/Cherrypicking") == "Conflicting Evidence/Cherrypicking"
    with pytest.raises(ValueError, match="Unmappable label"):
        normalize_label("Maybe true")


def test_justification_marker_never_enters_evidence():
    record = _record(
        "The river flows north.",
        "Refuted",
        questions=[{
            "question": "Which way does the river flow?",
            "answers": [_answer("It flows south.")],
        }],
        justification=f"{MARKER} leaks the verdict if copied",
    )
    train, test, _report = preprocess_records([record], [], source="fixture")
    assert MARKER not in train.loc[0, "evidence_text"]
    assert MARKER not in " ".join(train.loc[0, "evidence_pairs"])
    assert test.empty


def test_no_train_test_overlap_and_dedup():
    train_records = [
        _record("Shared claim text", "Supported"),
        _record("Shared claim text", "Refuted"),
        _record("Train only claim", "Refuted"),
    ]
    test_records = [
        _record("shared claim text", "Supported"),
        _record("Test only claim", "Not Enough Evidence"),
        _record("Test only claim", "Conflicting Evidence"),
    ]
    train, test, report = preprocess_records(train_records, test_records, source="fixture")
    train_keys = set(train["claim"].str.casefold())
    test_keys = set(test["claim"].str.casefold())
    assert train_keys.isdisjoint(test_keys)
    assert "shared claim text" not in test_keys
    assert report["train"]["dropped_duplicates"] == 1
    assert report["test"]["dropped_duplicates"] == 1
    assert report["test"]["removed_train_overlap"] == 1
    assert list(train["claim_id"]) == list(range(len(train)))
    assert list(test["claim_id"]) == list(range(len(test)))


def test_missing_evidence_uses_placeholder():
    records = [_record("A claim without questions", "Not Enough Information", questions=[])]
    train, _test, _report = preprocess_records(records, [], source="fixture")
    row = train.iloc[0]
    assert row["evidence_text"] == C.NO_EVIDENCE_TOKEN
    assert list(row["evidence_pairs"]) == [C.NO_EVIDENCE_TOKEN]
    assert int(row["has_evidence"]) == 0
    assert int(row["n_qa"]) == 0
    assert int(row["evidence_len"]) == 0
    assert row["label"] == "Not Enough Evidence"


def test_output_columns_match_contract():
    questions = [{
        "question": "Is the flag red?",
        "answers": [
            _answer("Yes", answer_type="Boolean", explanation="The photo shows red."),
            _answer(""),
            _answer("A second answer"),
            _answer("A fourth answer that must be ignored"),
        ],
    }]
    records = [_record("  The   flag is red.  ", "supported", questions=questions)]
    train, _test, report = preprocess_records(records, [], source="fixture")
    assert list(train.columns) == PROCESSED_COLUMNS
    text = train.loc[0, "evidence_text"]
    assert "Explanation: The photo shows red." in text
    assert "fourth answer" not in text
    assert train.loc[0, "claim"] == "The flag is red."
    assert "justification" not in train.columns
    assert report["source"] == "fixture"
    assert set(report["train"]) == {
        "loaded", "dropped_invalid", "dropped_duplicates", "final",
        "missing_evidence_share", "class_counts",
    }
