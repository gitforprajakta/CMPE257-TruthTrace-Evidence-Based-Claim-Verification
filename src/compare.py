"""Compare majority, TF-IDF, and SBERT on the saved test predictions.

Statistics are pure functions of the prediction tables so tests do not need
the results directory. Missing models are skipped with a warning.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

from . import config as C
from .evaluate import load_metrics, load_predictions
from .utils import get_logger, write_json

logger = get_logger(__name__)

FULL_MODELS = ("majority", "tfidf_lr", "sbert_lr")
COMPARISON_COLUMNS = [
    "model",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "weighted_f1",
]
ALPHA = 0.05


def _fmt(value: float) -> str:
    return f"{float(value):.4f}"


def _fmt_p(value: float) -> str:
    """Six decimals, or scientific notation when six decimals would print zero."""
    number = float(value)
    text = f"{number:.6f}"
    if text == "0.000000" and number != 0.0:
        return f"{number:.2e}"
    return text


def comparison_row(metrics: dict) -> dict:
    """One main-comparison row from a metrics JSON object."""
    low, high = metrics["macro_f1_ci95"]
    return {
        "model": metrics["model"],
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "macro_f1_ci_low": float(low),
        "macro_f1_ci_high": float(high),
        "weighted_f1": float(metrics["weighted_f1"]),
    }


def mcnemar_from_predictions(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """McNemar test on per-example correctness. `df_a` is model A, `df_b` is model B.

    The 2x2 table is [[both_correct, a_only], [b_only, both_wrong]]. Exact
    binomial is used when fewer than 25 pairs disagree; otherwise chi-square
    with continuity correction.
    """
    left = df_a[["claim_id", "correct"]].copy()
    right = df_b[["claim_id", "correct"]].copy()
    left["claim_id"] = left["claim_id"].astype(int)
    right["claim_id"] = right["claim_id"].astype(int)
    left = left.rename(columns={"correct": "correct_a"})
    right = right.rename(columns={"correct": "correct_b"})
    merged = left.merge(right, on="claim_id", how="inner")
    if merged.empty:
        raise ValueError("The two prediction tables have no claim_id values in common")
    correct_a = merged["correct_a"].astype(int).to_numpy()
    correct_b = merged["correct_b"].astype(int).to_numpy()
    both_correct = int(np.sum((correct_a == 1) & (correct_b == 1)))
    a_only = int(np.sum((correct_a == 1) & (correct_b == 0)))
    b_only = int(np.sum((correct_a == 0) & (correct_b == 1)))
    both_wrong = int(np.sum((correct_a == 0) & (correct_b == 0)))
    table = [[both_correct, a_only], [b_only, both_wrong]]
    n_discordant = a_only + b_only
    if n_discordant == 0:
        return {
            "model_a": "tfidf_lr",
            "model_b": "sbert_lr",
            "table": table,
            "n_discordant": 0,
            "p_value": 1.0,
            "test": "exact",
        }
    exact = n_discordant < 25
    result = mcnemar(np.asarray(table, dtype=int), exact=exact, correction=True)
    return {
        "model_a": "tfidf_lr",
        "model_b": "sbert_lr",
        "table": table,
        "n_discordant": int(n_discordant),
        "p_value": float(result.pvalue),
        "test": "exact" if exact else "chi2",
    }


def paired_bootstrap_diff(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    n: int = C.N_BOOTSTRAP,
    seed: int = C.SEED,
) -> dict:
    """Paired bootstrap of macro-F1(model B) minus macro-F1(model A).

    The same resampled indices are used for both models. `p_sbert_better` is
    the fraction of resamples whose difference is positive.
    """
    left = df_a[["claim_id", "y_true", "y_pred"]].copy()
    right = df_b[["claim_id", "y_pred"]].copy()
    left["claim_id"] = left["claim_id"].astype(int)
    right["claim_id"] = right["claim_id"].astype(int)
    right = right.rename(columns={"y_pred": "y_pred_b"})
    merged = left.merge(right, on="claim_id", how="inner")
    if merged.empty:
        raise ValueError("The two prediction tables have no claim_id values in common")
    y_true = merged["y_true"].to_numpy()
    pred_a = merged["y_pred"].to_numpy()
    pred_b = merged["y_pred_b"].to_numpy()
    rng = np.random.default_rng(seed)
    size = len(merged)
    diffs = np.empty(n, dtype=float)
    for i in range(n):
        index = rng.integers(0, size, size)
        score_a = f1_score(
            y_true[index], pred_a[index], labels=C.LABELS, average="macro", zero_division=0,
        )
        score_b = f1_score(
            y_true[index], pred_b[index], labels=C.LABELS, average="macro", zero_division=0,
        )
        diffs[i] = score_b - score_a
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "metric": "macro_f1",
        "diff_mean": float(diffs.mean()),
        "ci95": [float(low), float(high)],
        "p_sbert_better": float(np.mean(diffs > 0)),
        "n_resamples": int(n),
    }


def build_verdict(
    comparison: pd.DataFrame | None,
    mcnemar_result: dict | None,
    bootstrap_result: dict | None,
) -> str:
    """One paragraph. Every number is formatted from the objects passed in."""
    if comparison is None or comparison.empty:
        return (
            "Results pending: the main comparison table is not available yet, "
            "so this paragraph does not state a winner."
        )
    by_model = {str(row["model"]): row for _, row in comparison.iterrows()}
    missing = [name for name in FULL_MODELS if name not in by_model]
    sentences: list[str] = []
    if "majority" in by_model:
        sentences.append(f"The majority baseline reaches a macro-F1 of {_fmt(by_model['majority']['macro_f1'])}.")
    else:
        sentences.append("The majority baseline result is pending.")
    if "tfidf_lr" in by_model:
        sentences.append(
            f"TF-IDF + logistic regression reaches a macro-F1 of {_fmt(by_model['tfidf_lr']['macro_f1'])}."
        )
    else:
        sentences.append("The TF-IDF + logistic regression result is pending.")
    if "sbert_lr" in by_model:
        sentences.append(
            "Sentence-BERT + logistic regression reaches a macro-F1 of "
            f"{_fmt(by_model['sbert_lr']['macro_f1'])}."
        )
    else:
        sentences.append("The Sentence-BERT + logistic regression result is pending.")

    both_models = "tfidf_lr" in by_model and "sbert_lr" in by_model
    if both_models and mcnemar_result is not None and bootstrap_result is not None:
        diff = float(bootstrap_result["diff_mean"])
        low, high = bootstrap_result["ci95"]
        p_value = float(mcnemar_result["p_value"])
        sentences.append(
            "The macro-F1 difference (SBERT minus TF-IDF) is "
            f"{_fmt(diff)} with a 95% bootstrap interval of [{_fmt(low)}, {_fmt(high)}]."
        )
        sentences.append(f"McNemar's test on paired correctness gives p = {_fmt_p(p_value)}.")
        interval_above = float(low) > 0
        interval_below = float(high) < 0
        mcnemar_significant = p_value < ALPHA
        if interval_above and mcnemar_significant:
            sentences.append("SBERT is better than TF-IDF at alpha = 0.05.")
        elif interval_below and mcnemar_significant:
            sentences.append("SBERT is worse than TF-IDF at alpha = 0.05.")
        else:
            sentences.append("SBERT and TF-IDF are not significantly different at alpha = 0.05.")
    else:
        sentences.append(
            "A significance comparison is pending until both model predictions "
            "and the McNemar and bootstrap results are available."
        )
    if missing:
        sentences.append("Pending results: " + ", ".join(missing) + ".")
    return " ".join(sentences)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _per_class_rows(metrics: dict) -> list[dict]:
    rows = []
    for label in C.LABELS:
        stats = metrics["per_class"][label]
        rows.append({
            "model": metrics["model"],
            "label": label,
            "precision": float(stats["precision"]),
            "recall": float(stats["recall"]),
            "f1": float(stats["f1"]),
            "support": int(stats["support"]),
        })
    return rows


def run() -> None:
    """Read saved metrics and predictions and write the comparison tables."""
    C.ensure_dirs()
    comparison_rows = []
    per_class_rows = []
    ablation_rows = []
    for model in FULL_MODELS:
        for mode in C.INPUT_MODES:
            if model == "majority" and mode != "full":
                continue
            metrics = load_metrics(model, mode)
            if metrics is None:
                logger.warning("Skipping missing metrics for %s %s", model, mode)
                continue
            if mode == "full":
                comparison_rows.append(comparison_row(metrics))
                per_class_rows.extend(_per_class_rows(metrics))
            ablation_rows.append({
                "model": model,
                "input_mode": mode,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
            })

    comparison = pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS)
    comparison_path = C.TABLES_DIR / "main_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    (C.TABLES_DIR / "main_comparison.md").write_text(_markdown_table(comparison), encoding="utf-8")
    logger.info("Wrote %s", comparison_path)

    per_class = pd.DataFrame(
        per_class_rows,
        columns=["model", "label", "precision", "recall", "f1", "support"],
    )
    per_class.to_csv(C.TABLES_DIR / "per_class_f1.csv", index=False)
    ablation = pd.DataFrame(ablation_rows, columns=["model", "input_mode", "accuracy", "macro_f1"])
    ablation.to_csv(C.TABLES_DIR / "ablation.csv", index=False)

    tfidf_pred = load_predictions("tfidf_lr", "full")
    sbert_pred = load_predictions("sbert_lr", "full")
    mcnemar_result = None
    bootstrap_result = None
    if tfidf_pred is None or sbert_pred is None:
        missing = []
        if tfidf_pred is None:
            missing.append("tfidf_lr")
        if sbert_pred is None:
            missing.append("sbert_lr")
        logger.warning("Skipping McNemar and bootstrap because predictions are missing: %s", ", ".join(missing))
    else:
        mcnemar_result = mcnemar_from_predictions(tfidf_pred, sbert_pred)
        bootstrap_result = paired_bootstrap_diff(tfidf_pred, sbert_pred)
        write_json(mcnemar_result, C.TABLES_DIR / "mcnemar.json")
        write_json(bootstrap_result, C.TABLES_DIR / "bootstrap_diff.json")
        logger.info(
            "McNemar p=%.3e (%s), bootstrap diff=%.4f",
            mcnemar_result["p_value"], mcnemar_result["test"], bootstrap_result["diff_mean"],
        )

    verdict = build_verdict(comparison, mcnemar_result, bootstrap_result)
    verdict_path = C.TABLES_DIR / "verdict.md"
    verdict_path.write_text(verdict + "\n", encoding="utf-8")
    logger.info("Wrote %s", verdict_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison tables and significance tests.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
