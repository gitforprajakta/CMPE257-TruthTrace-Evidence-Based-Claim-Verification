"""Schema checks for the shared evaluation helpers."""
from __future__ import annotations

import numpy as np

from src import config as C
from src.evaluate import align_proba, bootstrap_ci, evaluate_predictions

REQUIRED_KEYS = {
    "model",
    "input_mode",
    "split",
    "seed",
    "n_train",
    "n_test",
    "labels",
    "best_params",
    "cv_macro_f1",
    "train_seconds",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "macro_f1_ci95",
    "per_class",
    "confusion_matrix",
    "confusion_matrix_normalized",
}


def test_metrics_schema_and_confusion_rows():
    y_true = ["Supported"] * 4 + ["Refuted"] * 2
    y_pred = ["Supported"] * 3 + ["Refuted", "Refuted", "Supported"]
    metrics = evaluate_predictions(
        y_true, y_pred, model="toy", input_mode="full", n_train=12,
        best_params={"C": 1.0}, cv_macro_f1=0.5, train_seconds=1.5,
        extra={"sbert_model": "toy"},
    )
    assert REQUIRED_KEYS <= set(metrics)
    assert metrics["extra"]["sbert_model"] == "toy"
    matrix = np.asarray(metrics["confusion_matrix"])
    normalized = np.asarray(metrics["confusion_matrix_normalized"])
    assert matrix.shape == (4, 4)
    assert normalized.shape == (4, 4)
    row_sums = normalized.sum(axis=1)
    support = matrix.sum(axis=1)
    for total, share in zip(support, row_sums):
        if total == 0:
            assert share == 0
        else:
            assert abs(share - 1) < 1e-6
    low, high = bootstrap_ci(y_true, y_pred, n=40)
    assert low <= high
    assert metrics["macro_f1_ci95"][0] <= metrics["macro_f1_ci95"][1]


def test_align_proba_reorders_to_label_order():
    classes = [C.LABELS[2], C.LABELS[0], C.LABELS[3], C.LABELS[1]]
    proba = np.array([[0.1, 0.2, 0.3, 0.4]])
    aligned = align_proba(proba, classes)
    assert aligned.shape == (1, 4)
    assert aligned[0, C.LABELS.index(classes[0])] == 0.1
    assert aligned[0, 0] == 0.2
    assert aligned[0, 1] == 0.4
    assert aligned[0, 2] == 0.1
    assert aligned[0, 3] == 0.3
