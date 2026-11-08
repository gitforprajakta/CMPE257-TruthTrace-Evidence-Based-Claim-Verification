"""Train Sentence-BERT embeddings + logistic regression for one input mode.

The tuning protocol matches TF-IDF: the same C grid, the same stratified
5-fold split, and macro-F1. Test rows are predicted once.
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .evaluate import (
    evaluate_predictions,
    load_metrics,
    metrics_path,
    predictions_path,
    save_metrics,
    save_predictions,
)
from .features_sbert import (
    build_features,
    cache_is_ready,
    embed_claims,
    embed_evidence,
    resolve_device,
)
from .utils import get_logger, set_seed, write_json

logger = get_logger(__name__)


def result_model_name(sbert_model: str) -> str:
    """Default MiniLM writes `sbert_lr`. Any other encoder writes `sbert_mpnet_lr`."""
    if sbert_model == C.SBERT_MODEL:
        return "sbert_lr"
    return "sbert_mpnet_lr"


def model_path(model: str, mode: str):
    return C.MODELS_DIR / f"{model}_{mode}.joblib"


def sidecar_path(model: str, mode: str):
    return C.MODELS_DIR / f"{model}_{mode}.json"


def build_classifier() -> Pipeline:
    """Scaler plus class-balanced logistic regression. Sklearn classes only."""
    classifier = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=C.SEED,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def outputs_ready(model: str, mode: str) -> bool:
    ready = (
        metrics_path(model, mode).exists()
        and predictions_path(model, mode).exists()
        and model_path(model, mode).exists()
        and sidecar_path(model, mode).exists()
    )
    if mode == "full" and model == "sbert_lr":
        ready = ready and (C.FIGURES_DIR / "sbert_umap_claims.png").exists()
    return ready


def plot_claim_projection(embeddings: np.ndarray, labels: pd.Series) -> None:
    """2-D view of test claim embeddings, colored by the true label."""
    try:
        import umap

        n_neighbors = min(15, max(2, len(embeddings) - 1))
        projected = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
            random_state=C.SEED,
        ).fit_transform(embeddings)
        title = "SBERT test claim embeddings (UMAP)"
    except Exception as exc:
        logger.warning("UMAP failed (%s). Falling back to t-SNE.", exc)
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, len(embeddings) - 1))
        projected = TSNE(
            n_components=2,
            random_state=C.SEED,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(embeddings)
        title = "SBERT test claim embeddings (t-SNE)"

    frame = pd.DataFrame({
        "x": projected[:, 0],
        "y": projected[:, 1],
        "label": [C.SHORT_LABELS.get(str(label), str(label)) for label in labels.tolist()],
    })
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=frame,
        x="x",
        y="y",
        hue="label",
        hue_order=[C.SHORT_LABELS[label] for label in C.LABELS],
        ax=axis,
        s=28,
    )
    axis.set_title(title)
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    fig.tight_layout()
    path = C.FIGURES_DIR / "sbert_umap_claims.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)


def run(
    mode: str,
    sbert_model: str = C.SBERT_MODEL,
    force: bool = False,
    fake: bool = False,
) -> dict:
    """Train one input mode and write metrics, predictions, and the sklearn pipeline."""
    if mode not in C.INPUT_MODES:
        raise ValueError(f"Unknown input mode {mode!r}. Expected one of {C.INPUT_MODES}.")
    set_seed()
    C.ensure_dirs()
    model = result_model_name(sbert_model)
    if fake:
        logger.warning(
            "Using the fake encoder. These outputs are for smoke tests only and are not real results."
        )
    if outputs_ready(model, mode) and not force:
        logger.info("%s outputs for %s already present. Use --force to recompute.", model, mode)
        return load_metrics(model, mode)

    if not C.TRAIN_PARQUET.exists() or not C.TEST_PARQUET.exists():
        raise FileNotFoundError("Processed parquet files are missing. Run python -m src.preprocess first.")

    train = pd.read_parquet(C.TRAIN_PARQUET)
    test = pd.read_parquet(C.TEST_PARQUET)
    device = None if fake else resolve_device(None)
    encoder = None
    caches_ready = all(
        cache_is_ready(sbert_model, split, kind, n_rows, fake=fake, force=force)
        for split, n_rows in (("train", len(train)), ("test", len(test)))
        for kind in ("claim", "evidence")
    )
    if not fake and not caches_ready:
        from sentence_transformers import SentenceTransformer

        logger.info("Encoding with %s on %s", sbert_model, device)
        encoder = SentenceTransformer(sbert_model, device=device)
    elif caches_ready:
        logger.info("Embedding caches already match the processed splits. Reusing them.")

    train_claims = embed_claims(
        train, sbert_model, "train", force=force, fake=fake, device=device, encoder=encoder,
    )
    test_claims = embed_claims(
        test, sbert_model, "test", force=force, fake=fake, device=device, encoder=encoder,
    )
    train_evidence = embed_evidence(
        train, sbert_model, "train", force=force, fake=fake, device=device, encoder=encoder,
    )
    test_evidence = embed_evidence(
        test, sbert_model, "test", force=force, fake=fake, device=device, encoder=encoder,
    )
    train_x = build_features(train_claims, train_evidence, train["has_evidence"], mode)
    test_x = build_features(test_claims, test_evidence, test["has_evidence"], mode)
    # Drop the encoder before the parallel grid search so worker processes
    # do not inherit a loaded PyTorch model.
    del encoder
    import gc
    gc.collect()

    pipeline = build_classifier()
    search = GridSearchCV(
        pipeline,
        param_grid={"clf__C": C.C_GRID},
        cv=StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED),
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )
    logger.info("Grid search model=%s mode=%s C=%s", model, mode, C.C_GRID)
    started = time.perf_counter()
    search.fit(train_x, train["label"])
    train_seconds = time.perf_counter() - started
    best = search.best_estimator_
    best_params = {"C": float(search.best_params_["clf__C"])}
    cv_macro_f1 = float(search.best_score_)
    logger.info("CV macro-F1=%.4f best C=%s", cv_macro_f1, best_params["C"])

    y_pred = best.predict(test_x)
    proba = best.predict_proba(test_x)
    extra = {"sbert_model": sbert_model}
    if fake:
        extra["fake_encoder"] = True
    metrics = evaluate_predictions(
        test["label"].to_numpy(),
        y_pred,
        model=model,
        input_mode=mode,
        n_train=len(train),
        best_params=best_params,
        cv_macro_f1=cv_macro_f1,
        train_seconds=train_seconds,
        extra=extra,
    )
    save_metrics(metrics)
    save_predictions(
        test["claim_id"].tolist(),
        test["label"].tolist(),
        list(y_pred),
        model,
        mode,
        proba=proba,
        classes=list(best.classes_),
    )
    model_path(model, mode).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, model_path(model, mode))
    write_json(
        {"sbert_model": sbert_model, "feature_mode": mode},
        sidecar_path(model, mode),
    )
    logger.info(
        "Saved %s mode=%s macro_f1=%.4f cv_macro_f1=%.4f C=%s",
        model, mode, metrics["macro_f1"], cv_macro_f1, best_params["C"],
    )
    if mode == "full" and model == "sbert_lr":
        plot_claim_projection(test_claims, test["label"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Sentence-BERT + logistic regression.")
    parser.add_argument("--mode", required=True, choices=C.INPUT_MODES, help="Input mode to train.")
    parser.add_argument(
        "--sbert-model",
        default=C.SBERT_MODEL,
        help="SentenceTransformer model name. The default writes sbert_lr files.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute even if outputs exist.")
    parser.add_argument(
        "--fake-encoder",
        action="store_true",
        help="Use deterministic hash vectors instead of downloading a model. For tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(mode=args.mode, sbert_model=args.sbert_model, force=args.force, fake=args.fake_encoder)


if __name__ == "__main__":
    main()
