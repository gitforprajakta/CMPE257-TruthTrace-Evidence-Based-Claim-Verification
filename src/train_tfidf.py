"""Train TF-IDF + logistic regression for one input mode.

Outer loop: ngram_range in {(1, 1), (1, 2)}. Inner loop: GridSearchCV over C.
The best pipeline is the GridSearchCV refit on all training rows. Test is
predicted once.
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as C
from .evaluate import evaluate_predictions, metrics_path, predictions_path, save_metrics, save_predictions
from .features_tfidf import build_pipeline, input_frame, top_positive_features
from .utils import get_logger, set_seed, write_json

logger = get_logger(__name__)

MODEL = "tfidf_lr"
NGRAMS = ((1, 1), (1, 2))


def model_path(mode: str):
    return C.MODELS_DIR / f"{MODEL}_{mode}.joblib"


def outputs_ready(mode: str) -> bool:
    ready = (
        metrics_path(MODEL, mode).exists()
        and predictions_path(MODEL, mode).exists()
        and model_path(mode).exists()
    )
    if mode == "full":
        ready = (
            ready
            and (C.TABLES_DIR / "tfidf_top_words_full.json").exists()
            and (C.FIGURES_DIR / "tfidf_top_words.png").exists()
        )
    return ready


def _search(frame: pd.DataFrame, labels: pd.Series, mode: str):
    """Return the best fitted pipeline, its params, and its CV macro-F1."""
    features = input_frame(frame)
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)
    best_pipeline = None
    best_score = -1.0
    best_params: dict = {}
    for ngram in NGRAMS:
        pipeline = build_pipeline(mode, ngram)
        search = GridSearchCV(
            pipeline,
            param_grid={"clf__C": C.C_GRID},
            cv=cv,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True,
        )
        logger.info("Grid search mode=%s ngram_range=%s", mode, ngram)
        search.fit(features, labels)
        logger.info("CV macro-F1=%.4f best C=%s", search.best_score_, search.best_params_["clf__C"])
        if search.best_score_ > best_score:
            best_score = float(search.best_score_)
            best_pipeline = search.best_estimator_
            best_params = {
                "ngram_range": [int(ngram[0]), int(ngram[1])],
                "C": float(search.best_params_["clf__C"]),
            }
    if best_pipeline is None:
        raise RuntimeError(f"Grid search produced no pipeline for mode={mode}")
    return best_pipeline, best_params, best_score


def plot_top_words(ranked: dict) -> None:
    """2x2 bar chart of the top 10 positive features per class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, label in zip(axes.ravel(), C.LABELS):
        items = ranked.get(label, [])[:10]
        if not items:
            axis.set_title(C.SHORT_LABELS[label])
            axis.text(0.5, 0.5, "No positive features", ha="center", va="center")
            axis.set_axis_off()
            continue
        shown = list(reversed(items))
        colors = ["#0f6e6b" if item["source"] == "claim" else "#c47b2b" for item in shown]
        axis.barh([item["feature"] for item in shown], [item["weight"] for item in shown], color=colors)
        axis.set_title(C.SHORT_LABELS[label])
        axis.set_xlabel("Coefficient")
    fig.suptitle("Top TF-IDF features (teal = claim, orange = evidence)")
    fig.tight_layout()
    path = C.FIGURES_DIR / "tfidf_top_words.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def run(mode: str, force: bool = False) -> dict:
    if mode not in C.INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode!r}. Expected one of {C.INPUT_MODES}.")
    set_seed()
    C.ensure_dirs()
    if outputs_ready(mode) and not force:
        logger.info("TF-IDF outputs for %s already present. Use --force to recompute.", mode)
        from .evaluate import load_metrics
        return load_metrics(MODEL, mode)

    train = pd.read_parquet(C.TRAIN_PARQUET)
    test = pd.read_parquet(C.TEST_PARQUET)
    started = time.perf_counter()
    pipeline, best_params, cv_macro_f1 = _search(train, train["label"], mode)
    train_seconds = time.perf_counter() - started

    test_x = input_frame(test)
    y_pred = pipeline.predict(test_x)
    proba = pipeline.predict_proba(test_x)
    metrics = evaluate_predictions(
        test["label"].to_numpy(),
        y_pred,
        model=MODEL,
        input_mode=mode,
        n_train=len(train),
        best_params=best_params,
        cv_macro_f1=cv_macro_f1,
        train_seconds=train_seconds,
    )
    save_metrics(metrics)
    save_predictions(
        test["claim_id"].tolist(),
        test["label"].tolist(),
        list(y_pred),
        MODEL,
        mode,
        proba=proba,
        classes=list(pipeline.named_steps["clf"].classes_),
    )
    joblib.dump(pipeline, model_path(mode))
    logger.info(
        "Saved %s mode=%s macro_f1=%.4f cv_macro_f1=%.4f params=%s",
        MODEL, mode, metrics["macro_f1"], cv_macro_f1, best_params,
    )
    if mode == "full":
        ranked = top_positive_features(pipeline, k=20)
        write_json(ranked, C.TABLES_DIR / "tfidf_top_words_full.json")
        plot_top_words(ranked)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF + logistic regression.")
    parser.add_argument("--mode", required=True, choices=C.INPUT_MODES, help="Input mode to train.")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(mode=args.mode, force=args.force)


if __name__ == "__main__":
    main()
