"""Comparison figures from saved metrics. A missing input is skipped with a warning."""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from .evaluate import load_metrics
from .utils import get_logger

logger = get_logger(__name__)

DISPLAY = {
    "majority": "Majority",
    "tfidf_lr": "TF-IDF",
    "sbert_lr": "SBERT",
}
MODE_ORDER = ["claim_only", "evidence_only", "full"]
OVERALL_METRICS = [
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro-F1"),
    ("weighted_f1", "Weighted-F1"),
]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")


def _save(fig: plt.Figure, name: str) -> None:
    path = C.FIGURES_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def _short_labels() -> list[str]:
    return [C.SHORT_LABELS[label] for label in C.LABELS]


def plot_confusion(metrics: dict, normalized: bool, name: str) -> None:
    """Heatmap of the saved confusion matrix. Counts are annotated."""
    key = "confusion_matrix_normalized" if normalized else "confusion_matrix"
    matrix = np.asarray(metrics[key], dtype=float)
    _style()
    fig, axis = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalized else ".0f",
        cmap="Blues",
        xticklabels=_short_labels(),
        yticklabels=_short_labels(),
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    title = "Row-normalized confusion matrix" if normalized else "Confusion matrix"
    axis.set_title(f"{title}: {DISPLAY.get(metrics['model'], metrics['model'])}")
    _save(fig, name)


def plot_per_class_prf(metrics: dict, name: str) -> None:
    """Grouped bars of precision, recall, and F1 for one model."""
    rows = []
    for label in C.LABELS:
        stats = metrics["per_class"][label]
        short = C.SHORT_LABELS[label]
        for metric in ("precision", "recall", "f1"):
            rows.append({"label": short, "metric": metric, "score": float(stats[metric])})
    frame = pd.DataFrame(rows)
    _style()
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=frame,
        x="label",
        y="score",
        hue="metric",
        order=_short_labels(),
        hue_order=["precision", "recall", "f1"],
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Class")
    axis.set_ylabel("Score")
    axis.set_title(f"Per-class precision, recall, and F1: {DISPLAY.get(metrics['model'], metrics['model'])}")
    _save(fig, name)


def plot_comparison_overall(metrics_by_model: dict[str, dict]) -> None:
    """Accuracy, macro-F1, and weighted-F1. Macro-F1 bars carry the bootstrap interval."""
    models = [model for model in ("majority", "tfidf_lr", "sbert_lr") if model in metrics_by_model]
    if not models:
        logger.warning("Skipping comparison_overall.png because no full-mode metrics were found")
        return
    _style()
    fig, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(OVERALL_METRICS))
    width = 0.8 / max(len(models), 1)
    for index, model in enumerate(models):
        metrics = metrics_by_model[model]
        values = [float(metrics[key]) for key, _ in OVERALL_METRICS]
        offset = (index - (len(models) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=DISPLAY[model])
        low, high = metrics["macro_f1_ci95"]
        macro = float(metrics["macro_f1"])
        axis.errorbar(
            x[1] + offset,
            macro,
            yerr=[[max(0.0, macro - float(low))], [max(0.0, float(high) - macro)]],
            fmt="none",
            ecolor="black",
            capsize=3,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in OVERALL_METRICS])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Overall test scores (interval on macro-F1)")
    axis.legend()
    _save(fig, "comparison_overall.png")


def plot_comparison_per_class(tfidf: dict | None, sbert: dict | None) -> None:
    """Per-class F1 for TF-IDF and SBERT."""
    rows = []
    for model, metrics in (("tfidf_lr", tfidf), ("sbert_lr", sbert)):
        if metrics is None:
            continue
        for label in C.LABELS:
            rows.append({
                "label": C.SHORT_LABELS[label],
                "model": DISPLAY[model],
                "f1": float(metrics["per_class"][label]["f1"]),
            })
    if not rows:
        logger.warning("Skipping comparison_per_class_f1.png because both model metrics are missing")
        return
    _style()
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=pd.DataFrame(rows),
        x="label",
        y="f1",
        hue="model",
        order=_short_labels(),
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Class")
    axis.set_ylabel("F1")
    axis.set_title("Per-class F1")
    _save(fig, "comparison_per_class_f1.png")


def plot_ablation(rows: list[dict]) -> None:
    """Macro-F1 by input mode for each model that has results."""
    if not rows:
        logger.warning("Skipping ablation_input_modes.png because no ablation rows were found")
        return
    frame = pd.DataFrame(rows)
    frame["model_name"] = frame["model"].map(lambda name: DISPLAY.get(name, name))
    _style()
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=frame,
        x="input_mode",
        y="macro_f1",
        hue="model_name",
        order=MODE_ORDER,
        ax=axis,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Input mode")
    axis.set_ylabel("Macro-F1")
    axis.set_title("Input-mode ablation")
    _save(fig, "ablation_input_modes.png")


def run() -> None:
    """Write every comparison figure whose inputs exist."""
    C.ensure_dirs()
    tfidf = load_metrics("tfidf_lr", "full")
    sbert = load_metrics("sbert_lr", "full")
    confusion_specs = (
        (tfidf, False, "cm_tfidf_lr_full.png"),
        (sbert, False, "cm_sbert_lr_full.png"),
        (tfidf, True, "cm_normalized_tfidf_lr_full.png"),
        (sbert, True, "cm_normalized_sbert_lr_full.png"),
    )
    for metrics, normalized, name in confusion_specs:
        if metrics is None:
            logger.warning("Skipping %s because metrics are missing", name)
            continue
        plot_confusion(metrics, normalized, name)
    if tfidf is None:
        logger.warning("Skipping per_class_prf_tfidf_lr.png because metrics are missing")
    else:
        plot_per_class_prf(tfidf, "per_class_prf_tfidf_lr.png")
    if sbert is None:
        logger.warning("Skipping per_class_prf_sbert_lr.png because metrics are missing")
    else:
        plot_per_class_prf(sbert, "per_class_prf_sbert_lr.png")

    overall = {}
    for model in ("majority", "tfidf_lr", "sbert_lr"):
        metrics = load_metrics(model, "full")
        if metrics is None:
            logger.warning("No full-mode metrics for %s; leaving it off comparison_overall.png", model)
            continue
        overall[model] = metrics
    plot_comparison_overall(overall)
    plot_comparison_per_class(tfidf, sbert)

    ablation_rows = []
    for model in ("majority", "tfidf_lr", "sbert_lr"):
        for mode in MODE_ORDER:
            if model == "majority" and mode != "full":
                continue
            metrics = load_metrics(model, mode)
            if metrics is None:
                logger.warning("No metrics for %s %s; leaving it off the ablation figure", model, mode)
                continue
            ablation_rows.append({
                "model": model,
                "input_mode": mode,
                "macro_f1": float(metrics["macro_f1"]),
            })
    plot_ablation(ablation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw comparison figures from saved metrics.")
    return parser.parse_args()


def main() -> None:
    parse_args()
    run()


if __name__ == "__main__":
    main()
