"""Sklearn-only TF-IDF pipelines for the three input modes.

The saved model is a plain scikit-learn Pipeline. `predict` and `predict_proba`
accept a DataFrame with columns `claim`, `evidence_text`, and `has_evidence`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config as C


def make_tfidf(ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """Word TF-IDF vectorizer shared by the claim and evidence columns."""
    return TfidfVectorizer(
        min_df=2,
        max_features=50000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
        ngram_range=ngram_range,
    )


def build_pipeline(mode: str, ngram_range: tuple[int, int] = (1, 1)) -> Pipeline:
    """Build the ColumnTransformer + logistic regression pipeline for one mode."""
    if mode not in C.INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode!r}. Expected one of {C.INPUT_MODES}.")
    transformers = []
    if mode in ("full", "claim_only"):
        transformers.append(("claim", make_tfidf(ngram_range), "claim"))
    if mode in ("full", "evidence_only"):
        transformers.append(("evidence", make_tfidf(ngram_range), "evidence_text"))
        transformers.append(("has_evidence", "passthrough", ["has_evidence"]))
    features = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3)
    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=C.SEED,
    )
    return Pipeline([("features", features), ("clf", classifier)])


def _feature_source(name: str) -> tuple[str, str]:
    if name.startswith("claim__"):
        return "claim", name[len("claim__"):]
    if name.startswith("evidence__"):
        return "evidence", name[len("evidence__"):]
    if "has_evidence" in name:
        return "has_evidence", "has_evidence"
    if "__" in name:
        source, feature = name.split("__", 1)
        return source, feature
    return "other", name


def top_positive_features(pipeline: Pipeline, k: int = 20) -> dict[str, list[dict]]:
    """Top positive-coefficient features per class, with claim/evidence source."""
    names = [str(name) for name in pipeline.named_steps["features"].get_feature_names_out()]
    classifier = pipeline.named_steps["clf"]
    coef = np.asarray(classifier.coef_)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    class_list = [str(label) for label in classifier.classes_]
    ranked: dict[str, list[dict]] = {}
    for label in C.LABELS:
        if label not in class_list:
            ranked[label] = []
            continue
        weights = coef[class_list.index(label)]
        positive = np.flatnonzero(weights > 0)
        if len(positive) == 0:
            ranked[label] = []
            continue
        order = positive[np.argsort(weights[positive])[::-1][:k]]
        ranked[label] = [
            {
                "feature": _feature_source(names[index])[1],
                "source": _feature_source(names[index])[0],
                "weight": float(weights[index]),
            }
            for index in order
        ]
    return ranked


def input_frame(frame, columns: Sequence[str] = ("claim", "evidence_text", "has_evidence")):
    """Return the columns the saved pipeline reads, in a stable order."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Input frame is missing columns: {missing}")
    return frame.loc[:, list(columns)]
