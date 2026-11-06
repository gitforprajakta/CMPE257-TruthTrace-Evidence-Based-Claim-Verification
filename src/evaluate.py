"""Shared evaluation. Created in Stage 1; do not change afterwards.

Every model (majority, tfidf_lr, sbert_lr) MUST report through these functions so that
metrics JSON and predictions CSV files have exactly the same schema.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_fscore_support, accuracy_score)

from . import config as C
from .utils import read_json, write_json


def metrics_path(model: str, mode: str) -> Path:
    return C.METRICS_DIR / f"{model}_{mode}.json"


def predictions_path(model: str, mode: str) -> Path:
    return C.PREDICTIONS_DIR / f"{model}_{mode}.csv"


def align_proba(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Reorder predict_proba columns (clf.classes_ order) into C.LABELS order."""
    proba = np.asarray(proba)
    out = np.zeros((proba.shape[0], len(C.LABELS)), dtype=float)
    for j, c in enumerate(classes):
        out[:, C.LABELS.index(c)] = proba[:, j]
    return out


def bootstrap_ci(y_true, y_pred, n: int = C.N_BOOTSTRAP, seed: int = C.SEED,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for macro-F1."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        scores.append(f1_score(y_true[idx], y_pred[idx], labels=C.LABELS,
                               average="macro", zero_division=0))
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate_predictions(y_true, y_pred, model: str, input_mode: str, n_train: int,
                         best_params: Optional[dict] = None,
                         cv_macro_f1: Optional[float] = None,
                         train_seconds: Optional[float] = None,
                         extra: Optional[dict] = None) -> dict:
    """Return the metrics dict (schema in Section 5). Does not write anything."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=C.LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=C.LABELS, average="macro", zero_division=0)
    wf = f1_score(y_true, y_pred, labels=C.LABELS, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=C.LABELS)
    row = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros(cm.shape, dtype=float), where=row > 0)
    lo, hi = bootstrap_ci(y_true, y_pred)
    d = {
        "model": model,
        "input_mode": input_mode,
        "split": "test",
        "seed": C.SEED,
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
        "labels": list(C.LABELS),
        "best_params": best_params or {},
        "cv_macro_f1": None if cv_macro_f1 is None else float(cv_macro_f1),
        "train_seconds": None if train_seconds is None else float(train_seconds),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(wf),
        "macro_f1_ci95": [lo, hi],
        "per_class": {
            lab: {"precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(C.LABELS)
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cmn.tolist(),
    }
    if extra:
        d["extra"] = extra
    return d


def save_metrics(d: dict) -> Path:
    path = metrics_path(d["model"], d["input_mode"])
    write_json(d, path)
    return path


def save_predictions(claim_ids, y_true, y_pred, model: str, mode: str,
                     proba: Optional[np.ndarray] = None,
                     classes: Optional[Sequence[str]] = None) -> Path:
    """Write predictions CSV. `proba` columns follow `classes` (e.g. clf.classes_)."""
    df = pd.DataFrame({"claim_id": list(claim_ids),
                       "y_true": list(y_true), "y_pred": list(y_pred)})
    df["correct"] = (df["y_true"] == df["y_pred"]).astype(int)
    if proba is not None:
        aligned = align_proba(proba, classes if classes is not None else C.LABELS)
        for i, lab in enumerate(C.LABELS):
            df[f"proba_{lab}"] = aligned[:, i]
    path = predictions_path(model, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_metrics(model: str, mode: str) -> Optional[dict]:
    p = metrics_path(model, mode)
    return read_json(p) if p.exists() else None


def load_predictions(model: str, mode: str) -> Optional[pd.DataFrame]:
    p = predictions_path(model, mode)
    return pd.read_csv(p) if p.exists() else None
