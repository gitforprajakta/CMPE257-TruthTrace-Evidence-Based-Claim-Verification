"""Majority-class baseline on the full input mode.

Predictions always go through src.evaluate so the metrics JSON and predictions
CSV match the shared schema. The classifier ignores the text and predicts the
most frequent training label.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from . import config as C
from .evaluate import evaluate_predictions, metrics_path, predictions_path, save_metrics, save_predictions
from .utils import get_logger, set_seed

logger = get_logger(__name__)

MODEL = "majority"
MODE = "full"


def outputs_ready() -> bool:
    return metrics_path(MODEL, MODE).exists() and predictions_path(MODEL, MODE).exists()


def run(force: bool = False) -> dict:
    set_seed()
    C.ensure_dirs()
    if outputs_ready() and not force:
        logger.info("Majority baseline already present. Use --force to recompute.")
        from .evaluate import load_metrics
        return load_metrics(MODEL, MODE)

    train = pd.read_parquet(C.TRAIN_PARQUET)
    test = pd.read_parquet(C.TEST_PARQUET)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    x_train = np.zeros((len(train), 1))
    x_test = np.zeros((len(test), 1))

    clf = DummyClassifier(strategy="most_frequent", random_state=C.SEED)
    started = time.perf_counter()
    clf.fit(x_train, y_train)
    train_seconds = time.perf_counter() - started
    y_pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)

    metrics = evaluate_predictions(
        y_test, y_pred, model=MODEL, input_mode=MODE, n_train=len(train),
        best_params={"strategy": "most_frequent"},
        cv_macro_f1=None,
        train_seconds=train_seconds,
    )
    save_metrics(metrics)
    save_predictions(
        test["claim_id"].tolist(), y_test, y_pred, MODEL, MODE,
        proba=proba, classes=list(clf.classes_),
    )
    logger.info(
        "Majority label=%s accuracy=%.4f macro_f1=%.4f",
        y_pred[0], metrics["accuracy"], metrics["macro_f1"],
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the majority-class baseline.")
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
