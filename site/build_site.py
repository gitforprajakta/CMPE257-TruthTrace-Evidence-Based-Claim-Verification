"""Build the static TruthTrace site into docs/.

Stage 2 artifacts are optional. A missing file becomes a pending box instead of
a failed build. No result number is written into a template; values are passed
in from results files at build time.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import config as C
from src.evaluate import load_metrics
from src.utils import get_logger, read_json

logger = get_logger("site.build_site")

SITE_DIR = ROOT / "site"
TEMPLATE_DIR = SITE_DIR / "templates"
STATIC_DIR = SITE_DIR / "static"
DOCS_DIR = ROOT / "docs"

NAV = [
    ("index.html", "Home"),
    ("data.html", "Data"),
    ("methods.html", "Methods"),
    ("results.html", "Results"),
    ("errors.html", "Errors"),
    ("demo.html", "Demo"),
    ("team.html", "Team"),
]

MODEL_NAMES = {
    "majority": "Majority",
    "tfidf_lr": "TF-IDF + LR",
    "sbert_lr": "SBERT + LR",
}

METRIC_SLOTS = [
    ("majority", "full"),
    ("tfidf_lr", "full"),
    ("tfidf_lr", "claim_only"),
    ("tfidf_lr", "evidence_only"),
    ("sbert_lr", "full"),
    ("sbert_lr", "claim_only"),
    ("sbert_lr", "evidence_only"),
]

HEADLINE_MODELS = ("majority", "tfidf_lr", "sbert_lr")
MODEL_SLUGS = {"majority": "majority", "tfidf_lr": "tfidf", "sbert_lr": "sbert"}
LABEL_SLUGS = {
    "Supported": "supported",
    "Refuted": "refuted",
    "Not Enough Evidence": "nee",
    "Conflicting Evidence/Cherrypicking": "conflicting",
    "NEE": "nee",
    "Conflicting": "conflicting",
}

EDA_FIGURES = [
    ("eda_class_distribution.png", "Class counts"),
    ("eda_claim_length.png", "Claim length"),
    ("eda_evidence_length.png", "Evidence length"),
    ("eda_missing_evidence_by_class.png", "Missing evidence"),
    ("eda_n_qa_by_class.png", "Question-answer pairs"),
    ("eda_nee_phrase_check.png", "Stock 'no answer' phrase"),
]

RESEARCH_QUESTION = (
    "Can a semantic model (Sentence-BERT + logistic regression) beat a traditional "
    "model (TF-IDF + logistic regression) for claim verification, given identical "
    "data, splits, tuning budget, and metrics?"
)


def label_slug(label: str) -> str:
    return LABEL_SLUGS.get(label, "other")


def short_label(label: str) -> str:
    return C.SHORT_LABELS.get(label, label)


def pending(relpath: str) -> dict:
    return {"pending": True, "message": f"Pending: waiting for {relpath}"}


def ready(**kwargs) -> dict:
    data = {"pending": False}
    data.update(kwargs)
    return data


def fmt_float(value, digits: int = 4) -> str:
    """Fixed decimals. A non-zero value that would print as zero uses scientific notation."""
    number = float(value)
    text = f"{number:.{digits}f}"
    if text == f"{0:.{digits}f}" and number != 0.0:
        return f"{number:.2e}"
    return text


def figure(name: str, caption: str) -> dict:
    path = C.FIGURES_DIR / name
    if not path.exists():
        return pending(f"results/figures/{name}")
    return ready(src=f"assets/figures/{name}", alt=caption, caption=caption)


def table_from_frame(frame: pd.DataFrame) -> dict:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(fmt_float(value))
            else:
                cells.append(str(value))
        rows.append(cells)
    return ready(columns=columns, rows=rows)


def optional_table(path: Path) -> dict:
    if not path.exists():
        rel = path.relative_to(C.ROOT).as_posix()
        return pending(rel)
    return table_from_frame(pd.read_csv(path))


def confusion_from_metrics(metrics: dict | None, normalized: bool, relpath: str) -> dict:
    if metrics is None:
        return pending(relpath)
    key = "confusion_matrix_normalized" if normalized else "confusion_matrix"
    matrix = metrics[key]
    labels = [C.SHORT_LABELS.get(label, label) for label in metrics["labels"]]
    rows = []
    for index, label in enumerate(labels):
        if normalized:
            cells = [fmt_float(value, 3) for value in matrix[index]]
        else:
            cells = [str(int(value)) for value in matrix[index]]
        rows.append({"label": label, "cells": cells})
    return ready(labels=labels, rows=rows)


def metric_row(model: str, mode: str) -> dict:
    relpath = f"results/metrics/{model}_{mode}.json"
    metrics = load_metrics(model, mode)
    if metrics is None:
        return {
            "model": model,
            "input_mode": mode,
            "pending": True,
            "message": f"Pending: waiting for {relpath}",
        }
    lo, hi = metrics["macro_f1_ci95"]
    return {
        "model": model,
        "input_mode": mode,
        "pending": False,
        "accuracy": fmt_float(metrics["accuracy"]),
        "macro_f1": fmt_float(metrics["macro_f1"]),
        "weighted_f1": fmt_float(metrics["weighted_f1"]),
        "ci": f"[{fmt_float(lo)}, {fmt_float(hi)}]",
    }


def headline() -> dict:
    comparison_path = C.TABLES_DIR / "main_comparison.csv"
    by_model = {}
    source = "results/metrics"
    if comparison_path.exists():
        frame = pd.read_csv(comparison_path)
        for _, row in frame.iterrows():
            by_model[str(row["model"])] = row
        source = "results/tables/main_comparison.csv"
    rows = []
    for model in HEADLINE_MODELS:
        name = MODEL_NAMES[model]
        slug = MODEL_SLUGS[model]
        if model in by_model:
            row = by_model[model]
            rows.append(ready(
                name=name,
                slug=slug,
                accuracy=fmt_float(row["accuracy"], 3),
                macro_f1=fmt_float(row["macro_f1"], 3),
            ))
            continue
        metrics = load_metrics(model, "full")
        if metrics is None:
            item = pending(f"results/metrics/{model}_full.json")
            item["name"] = name
            item["slug"] = slug
            rows.append(item)
            continue
        rows.append(ready(
            name=name,
            slug=slug,
            accuracy=fmt_float(metrics["accuracy"], 3),
            macro_f1=fmt_float(metrics["macro_f1"], 3),
        ))
    return {"note": f"Source: {source}.", "rows": rows}


def verdict() -> dict:
    path = C.TABLES_DIR / "verdict.md"
    if not path.exists():
        return pending("results/tables/verdict.md")
    text = path.read_text(encoding="utf-8").strip()
    parts = re.split(r"(?<=\.)\s+(?=[A-Z])", text)
    return ready(text=text, line=parts[-1] if parts else text)


def preprocess_view() -> dict:
    if not C.PREPROCESS_REPORT.exists():
        return pending("data/processed/preprocess_report.json")
    report = read_json(C.PREPROCESS_REPORT)
    splits = []
    class_rows = []
    for split in ("train", "test"):
        block = report[split]
        overlap = "—" if split == "train" else str(block.get("removed_train_overlap", 0))
        splits.append({
            "split": split,
            "loaded": str(block["loaded"]),
            "dropped_invalid": str(block["dropped_invalid"]),
            "dropped_duplicates": str(block["dropped_duplicates"]),
            "removed_train_overlap": overlap,
            "final": str(block["final"]),
            "missing_evidence_share": f"{100 * float(block['missing_evidence_share']):.1f}%",
        })
        counts = block.get("class_counts", {})
        for label in C.LABELS:
            class_rows.append({"split": split, "label": label, "count": str(counts.get(label, 0))})
    return ready(source=str(report.get("source", "unknown")), splits=splits, class_rows=class_rows)


def per_class_view() -> dict:
    path = C.TABLES_DIR / "per_class_f1.csv"
    if path.exists():
        return table_from_frame(pd.read_csv(path))
    rows = []
    found = False
    for model, mode in METRIC_SLOTS:
        metrics = load_metrics(model, mode)
        if metrics is None or mode != "full":
            continue
        found = True
        for label in C.LABELS:
            stats = metrics["per_class"][label]
            rows.append([
                model, label,
                fmt_float(stats["precision"]),
                fmt_float(stats["recall"]),
                fmt_float(stats["f1"]),
                str(stats["support"]),
            ])
    if not found:
        return pending("results/tables/per_class_f1.csv")
    return ready(
        columns=["model", "label", "precision", "recall", "f1", "support"],
        rows=rows,
    )


def ablation_view() -> dict:
    path = C.TABLES_DIR / "ablation.csv"
    if path.exists():
        return table_from_frame(pd.read_csv(path))
    rows = []
    for model, mode in METRIC_SLOTS:
        metrics = load_metrics(model, mode)
        if metrics is None:
            continue
        rows.append([model, mode, fmt_float(metrics["accuracy"]), fmt_float(metrics["macro_f1"])])
    if not rows:
        return pending("results/tables/ablation.csv")
    return ready(columns=["model", "input_mode", "accuracy", "macro_f1"], rows=rows)


def stats_view(name: str) -> dict:
    path = C.TABLES_DIR / name
    if not path.exists():
        return pending(f"results/tables/{name}")
    payload = read_json(path)
    if name == "mcnemar.json":
        return ready(
            test=str(payload.get("test", "")),
            n_discordant=str(payload.get("n_discordant", "")),
            p_value=fmt_float(payload.get("p_value", 0.0), 6),
            table=str(payload.get("table", "")),
        )
    ci = payload.get("ci95", [None, None])
    return ready(
        diff_mean=fmt_float(payload.get("diff_mean", 0.0)),
        ci=f"[{fmt_float(ci[0])}, {fmt_float(ci[1])}]",
        p_sbert_better=fmt_float(payload.get("p_sbert_better", 0.0)),
        n_resamples=str(payload.get("n_resamples", "")),
    )


def error_block(title: str, filename: str) -> dict:
    path = C.ERRORS_DIR / filename
    if not path.exists():
        return {"title": title, **pending(f"results/error_analysis/{filename}")}
    frame = pd.read_csv(path).head(10)
    rows = []
    for _, row in frame.iterrows():
        evidence = str(row.get("evidence_text", ""))
        if len(evidence) > 600:
            evidence = evidence[:600]
        reason = str(row.get("auto_reason", ""))
        if "who_wins" in frame.columns and pd.notna(row.get("who_wins")):
            reason = f"{reason} ({row['who_wins']})"
        true_label = str(row.get("true", ""))
        pred_tfidf = str(row.get("pred_tfidf", ""))
        pred_sbert = str(row.get("pred_sbert", ""))
        rows.append({
            "claim_id": str(row.get("claim_id", "")),
            "claim": str(row.get("claim", "")),
            "true_label": short_label(true_label),
            "true_slug": label_slug(true_label),
            "pred_tfidf": short_label(pred_tfidf),
            "tfidf_slug": label_slug(pred_tfidf),
            "pred_sbert": short_label(pred_sbert),
            "sbert_slug": label_slug(pred_sbert),
            "auto_reason": reason.replace("_", " "),
            "evidence": evidence,
        })
    return ready(title=title, rows=rows)


def _proba_items(row: pd.Series, prefix: str) -> list[dict]:
    items = []
    for label in C.LABELS:
        column = f"{prefix}proba_{label}"
        if column not in row.index or pd.isna(row[column]):
            continue
        items.append({
            "label": C.SHORT_LABELS[label],
            "slug": label_slug(label),
            "value": fmt_float(row[column], 3),
        })
    return items


def _model_card(row: pd.Series, name: str, pred_col: str, correct_col: str, prefix: str, relpath: str) -> dict:
    if pred_col not in row.index or pd.isna(row[pred_col]):
        card = pending(relpath)
        card["name"] = name
        card["slug"] = "tfidf" if "TF-IDF" in name else "sbert"
        return card
    correct = int(row[correct_col]) == 1
    prediction = str(row[pred_col])
    return ready(
        name=name,
        slug="tfidf" if "TF-IDF" in name else "sbert",
        pred=short_label(prediction),
        pred_slug=label_slug(prediction),
        outcome="correct" if correct else "incorrect",
        probabilities=_proba_items(row, prefix),
    )


def demo_examples() -> dict:
    if not C.TEST_PARQUET.exists():
        return pending("data/processed/test.parquet")
    tfidf_path = C.PREDICTIONS_DIR / "tfidf_lr_full.csv"
    if not tfidf_path.exists():
        return pending("results/predictions/tfidf_lr_full.csv")
    test = pd.read_parquet(C.TEST_PARQUET)
    tfidf = pd.read_csv(tfidf_path)
    merged = test.merge(tfidf, on="claim_id", how="inner", suffixes=("", "_pred"))
    sbert_path = C.PREDICTIONS_DIR / "sbert_lr_full.csv"
    if sbert_path.exists():
        sbert = pd.read_csv(sbert_path)
        rename = {column: f"sbert_{column}" for column in sbert.columns if column != "claim_id"}
        merged = merged.merge(sbert.rename(columns=rename), on="claim_id", how="left")
    if merged.empty:
        return pending("results/predictions/tfidf_lr_full.csv")

    wrong = merged[merged["correct"] == 0]
    right = merged[merged["correct"] == 1]
    target = 10
    n_wrong = min(len(wrong), target // 2)
    n_right = min(len(right), target - n_wrong)
    parts = []
    if n_wrong:
        parts.append(wrong.sample(n=n_wrong, random_state=C.SEED))
    if n_right:
        parts.append(right.sample(n=n_right, random_state=C.SEED))
    picked = pd.concat(parts) if parts else merged.iloc[0:0]
    shortfall = target - len(picked)
    if shortfall > 0:
        rest = merged.drop(index=picked.index, errors="ignore")
        extra = min(shortfall, len(rest))
        if extra:
            picked = pd.concat([picked, rest.sample(n=extra, random_state=C.SEED)])
    picked = picked.sort_values("claim_id")

    sbert_rel = "results/predictions/sbert_lr_full.csv"
    rows = []
    for _, row in picked.iterrows():
        evidence = str(row["evidence_text"])
        if len(evidence) > 600:
            evidence = evidence[:600]
        models = [
            _model_card(row, "TF-IDF + LR", "y_pred", "correct", "", "results/predictions/tfidf_lr_full.csv"),
            _model_card(row, "SBERT + LR", "sbert_y_pred", "sbert_correct", "sbert_", sbert_rel),
        ]
        true_label = str(row["label"])
        predictions = [model["pred"] for model in models if not model.get("pending") and model.get("pred")]
        rows.append({
            "claim_id": str(int(row["claim_id"])),
            "claim": str(row["claim"]),
            "true_label": true_label,
            "true_short": short_label(true_label),
            "true_slug": label_slug(true_label),
            "evidence": evidence,
            "models": models,
            "disagree": "yes" if len(predictions) >= 2 and len(set(predictions)) > 1 else "no",
        })
    return ready(rows=rows)


def protocol() -> dict:
    return {
        "cv_folds": str(C.CV_FOLDS),
        "seed": str(C.SEED),
        "c_grid": ", ".join(str(value) for value in C.C_GRID),
        "n_bootstrap": str(C.N_BOOTSTRAP),
        "sbert_model": C.SBERT_MODEL,
    }


def copy_assets() -> list[dict]:
    figures_out = DOCS_DIR / "assets" / "figures"
    metrics_out = DOCS_DIR / "assets" / "metrics"
    for folder in (figures_out, metrics_out):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    static_out = DOCS_DIR / "assets"
    static_out.mkdir(parents=True, exist_ok=True)
    for name in ("style.css", "tabs.js", "interact.js"):
        shutil.copyfile(STATIC_DIR / name, static_out / name)
    if C.FIGURES_DIR.exists():
        for path in sorted(C.FIGURES_DIR.glob("*.png")):
            shutil.copyfile(path, figures_out / path.name)
    downloads = []
    if C.METRICS_DIR.exists():
        for path in sorted(C.METRICS_DIR.glob("*.json")):
            shutil.copyfile(path, metrics_out / path.name)
            downloads.append({"href": f"assets/metrics/{path.name}", "label": path.name})
    return downloads


def confusion_blocks() -> list[dict]:
    specs = [
        ("TF-IDF + LR counts", "tfidf_lr", False, "cm_tfidf_lr_full.png", "TF-IDF confusion matrix."),
        ("TF-IDF + LR row-normalized", "tfidf_lr", True, "cm_normalized_tfidf_lr_full.png", "Row-normalized TF-IDF confusion matrix."),
        ("SBERT + LR counts", "sbert_lr", False, "cm_sbert_lr_full.png", "SBERT confusion matrix."),
        ("SBERT + LR row-normalized", "sbert_lr", True, "cm_normalized_sbert_lr_full.png", "Row-normalized SBERT confusion matrix."),
    ]
    blocks = []
    for title, model, normalized, figure_name, caption in specs:
        rel = f"results/metrics/{model}_full.json"
        blocks.append({
            "title": title,
            "table": confusion_from_metrics(load_metrics(model, "full"), normalized, rel),
            "figure": figure(figure_name, caption),
        })
    return blocks


def build() -> None:
    C.ensure_dirs()
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    downloads = copy_assets()
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    shared = {
        "nav": [{"href": href, "label": label} for href, label in NAV],
        "research_question": RESEARCH_QUESTION,
        "headline": headline(),
        "verdict": verdict(),
        "protocol": protocol(),
    }
    pages = {
        "index.html": ("Home", {}),
        "data.html": ("Data", {
            "report": preprocess_view(),
            "summary": optional_table(C.TABLES_DIR / "eda_summary.csv"),
            "figures": [figure(name, caption) for name, caption in EDA_FIGURES],
        }),
        "methods.html": ("Methods", {}),
        "results.html": ("Results", {
            "metric_rows": [metric_row(model, mode) for model, mode in METRIC_SLOTS],
            "comparison": optional_table(C.TABLES_DIR / "main_comparison.csv"),
            "comparison_fig": figure("comparison_overall.png", "Accuracy, macro-F1, and weighted-F1."),
            "per_class_fig": figure("comparison_per_class_f1.png", "Per-class F1 for TF-IDF and SBERT."),
            "per_class": per_class_view(),
            "ablation": ablation_view(),
            "ablation_fig": figure("ablation_input_modes.png", "Macro-F1 by input mode."),
            "confusion_blocks": confusion_blocks(),
            "top_words": figure("tfidf_top_words.png", "Highest positive TF-IDF coefficients per class."),
            "umap": figure("sbert_umap_claims.png", "Test claims embedded and projected."),
            "mcnemar": stats_view("mcnemar.json"),
            "bootstrap": stats_view("bootstrap_diff.json"),
            "downloads": downloads,
        }),
        "errors.html": ("Error analysis", {
            "reason_fig": figure("error_reason_counts.png", "Automatic error reasons per model."),
            "error_tables": [
                error_block("TF-IDF mistakes", "tfidf_lr_errors.csv"),
                error_block("SBERT mistakes", "sbert_lr_errors.csv"),
                error_block("Disagreements", "disagreements.csv"),
            ],
            "summary": (
                ready(text=(C.ERRORS_DIR / "summary.md").read_text(encoding="utf-8").strip())
                if (C.ERRORS_DIR / "summary.md").exists()
                else pending("results/error_analysis/summary.md")
            ),
        }),
        "demo.html": ("Demo", {
            "examples": demo_examples(),
            "verdict_choices": [
                {"label": label, "short": C.SHORT_LABELS[label], "slug": label_slug(label)}
                for label in C.LABELS
            ],
        }),
        "team.html": ("Team", {}),
    }
    for filename, (title, extra) in pages.items():
        html = env.get_template(filename).render(title=title, active=filename, **shared, **extra)
        (DOCS_DIR / filename).write_text(html, encoding="utf-8")
        logger.info("Wrote %s", DOCS_DIR / filename)
    for path in sorted(DOCS_DIR.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        if "{{" in text or "{%" in text:
            raise RuntimeError(f"Unrendered template syntax remains in {path.name}")
    logger.info("Site build finished: %d pages", len(pages))


if __name__ == "__main__":
    build()
