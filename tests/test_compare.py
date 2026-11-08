"""McNemar, paired bootstrap, and the verdict paragraph on hand-made tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chi2

from src import config as C
from src.compare import build_verdict, mcnemar_from_predictions, paired_bootstrap_diff


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_mcnemar_exact_matches_binomial_counts():
    rows_a = []
    rows_b = []
    # 2 both correct, 1 TF-IDF only, 5 SBERT only, 1 both wrong.
    pattern = (
        [(1, 1)] * 2
        + [(1, 0)] * 1
        + [(0, 1)] * 5
        + [(0, 0)] * 1
    )
    for claim_id, (correct_a, correct_b) in enumerate(pattern):
        label = C.LABELS[claim_id % len(C.LABELS)]
        rows_a.append({
            "claim_id": claim_id,
            "y_true": label,
            "y_pred": label if correct_a else C.LABELS[(C.LABELS.index(label) + 1) % 4],
            "correct": correct_a,
        })
        rows_b.append({
            "claim_id": claim_id,
            "y_true": label,
            "y_pred": label if correct_b else C.LABELS[(C.LABELS.index(label) + 1) % 4],
            "correct": correct_b,
        })
    result = mcnemar_from_predictions(_frame(rows_a), _frame(rows_b))
    assert result["table"] == [[2, 1], [5, 1]]
    assert result["n_discordant"] == 6
    assert result["test"] == "exact"
    expected = float(binomtest(min(1, 5), 6, 0.5).pvalue)
    assert result["p_value"] == pytest.approx(expected)


def test_mcnemar_zero_discordant_pairs():
    rows = []
    for claim_id, label in enumerate(C.LABELS):
        rows.append({"claim_id": claim_id, "y_true": label, "y_pred": label, "correct": 1})
    result = mcnemar_from_predictions(_frame(rows), _frame(rows))
    assert result["table"] == [[4, 0], [0, 0]]
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0
    assert result["test"] == "exact"


def test_mcnemar_uses_chi_square_when_many_pairs_disagree():
    rows_a = []
    rows_b = []
    pattern = [(1, 0)] * 20 + [(0, 1)] * 10 + [(1, 1)] * 5
    for claim_id, (correct_a, correct_b) in enumerate(pattern):
        rows_a.append({"claim_id": claim_id, "correct": correct_a, "y_true": "Supported", "y_pred": "Supported"})
        rows_b.append({"claim_id": claim_id, "correct": correct_b, "y_true": "Supported", "y_pred": "Refuted"})
    result = mcnemar_from_predictions(_frame(rows_a), _frame(rows_b))
    assert result["table"] == [[5, 20], [10, 0]]
    assert result["n_discordant"] == 30
    assert result["test"] == "chi2"
    # Chi-square with continuity correction: (|20-10|-1)^2 / 30.
    expected = float(chi2.sf((abs(20 - 10) - 1) ** 2 / 30, df=1))
    assert result["p_value"] == pytest.approx(expected)


def test_bootstrap_positive_when_sbert_is_always_right():
    rows_a = []
    rows_b = []
    for claim_id, label in enumerate(C.LABELS * 5):
        other = C.LABELS[(C.LABELS.index(label) + 1) % 4]
        rows_a.append({"claim_id": claim_id, "y_true": label, "y_pred": other, "correct": 0})
        rows_b.append({"claim_id": claim_id, "y_true": label, "y_pred": label, "correct": 1})
    result = paired_bootstrap_diff(_frame(rows_a), _frame(rows_b), n=40, seed=42)
    assert result["diff_mean"] > 0
    assert result["ci95"][0] > 0
    assert result["ci95"][0] <= result["ci95"][1]
    assert result["p_sbert_better"] == 1.0
    assert result["n_resamples"] == 40


def test_bootstrap_is_zero_when_predictions_match():
    rows = []
    for claim_id, label in enumerate(C.LABELS * 4):
        rows.append({"claim_id": claim_id, "y_true": label, "y_pred": label, "correct": 1})
    frame = _frame(rows)
    result = paired_bootstrap_diff(frame, frame, n=25, seed=42)
    assert result["diff_mean"] == pytest.approx(0.0)
    assert result["ci95"] == pytest.approx([0.0, 0.0])
    assert result["p_sbert_better"] == 0.0


def test_verdict_repeats_the_table_numbers():
    comparison = pd.DataFrame([
        {
            "model": "majority",
            "accuracy": 0.6263,
            "macro_precision": 0.15,
            "macro_recall": 0.25,
            "macro_f1": 0.1926,
            "macro_f1_ci_low": 0.17,
            "macro_f1_ci_high": 0.21,
            "weighted_f1": 0.48,
        },
        {
            "model": "tfidf_lr",
            "accuracy": 0.6468,
            "macro_precision": 0.57,
            "macro_recall": 0.44,
            "macro_f1": 0.4693,
            "macro_f1_ci_low": 0.40,
            "macro_f1_ci_high": 0.53,
            "weighted_f1": 0.62,
        },
        {
            "model": "sbert_lr",
            "accuracy": 0.70,
            "macro_precision": 0.60,
            "macro_recall": 0.55,
            "macro_f1": 0.5612,
            "macro_f1_ci_low": 0.50,
            "macro_f1_ci_high": 0.62,
            "weighted_f1": 0.68,
        },
    ])
    mcnemar_result = {"p_value": 0.0123456, "n_discordant": 40, "test": "chi2", "table": [[1, 2], [3, 4]]}
    bootstrap_result = {"diff_mean": 0.0919, "ci95": [0.0311, 0.1522], "p_sbert_better": 0.99, "n_resamples": 1000}
    text = build_verdict(comparison, mcnemar_result, bootstrap_result)
    assert "0.1926" in text
    assert "0.4693" in text
    assert "0.5612" in text
    assert "0.0919" in text
    assert "0.0311" in text
    assert "0.1522" in text
    assert "0.012346" in text
    assert "majority" in text.lower()
    assert "SBERT is better than TF-IDF at alpha = 0.05." in text
    tiny = build_verdict(
        comparison,
        {"p_value": 1.37e-7},
        {"diff_mean": -0.1095, "ci95": [-0.1796, -0.0392]},
    )
    assert "0.000000" not in tiny
    assert "1.37e-07" in tiny
    assert "SBERT is worse than TF-IDF at alpha = 0.05." in tiny
    worse = build_verdict(
        comparison,
        {"p_value": 0.001},
        {"diff_mean": -0.0400, "ci95": [-0.0800, -0.0100]},
    )
    assert "SBERT is worse than TF-IDF at alpha = 0.05." in worse
    assert "-0.0400" in worse
    tied = build_verdict(
        comparison,
        {"p_value": 0.40},
        {"diff_mean": 0.0100, "ci95": [-0.0200, 0.0300]},
    )
    assert "not significantly different at alpha = 0.05." in tied
    pending = build_verdict(comparison.iloc[:2], None, None)
    assert "sbert_lr" in pending
    assert "pending" in pending.lower()
    assert np.isfinite(bootstrap_result["diff_mean"])
