"""Build the TruthTrace single-page site into docs/.

Every result number is collected here and injected as window.TT_DATA.
Templates contain structure only. A missing Stage 2 file becomes a pending
flag in that blob; the page renders a Pending card and the build still exits 0.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from src import config as C
from src.evaluate import load_metrics
from src.utils import get_logger, read_json

logger = get_logger("site.build_site")

SITE_DIR = ROOT / "site"
TEMPLATE_DIR = SITE_DIR / "templates"
DOCS_DIR = ROOT / "docs"
VENDOR_DIR = SITE_DIR / "vendor"

EVIDENCE_LIMIT = 600
MODEL_ORDER = ["majority", "tfidf_lr", "sbert_lr"]
PREDICTION_MODELS = ["tfidf_lr", "sbert_lr"]
MODEL_NAMES = {
    "majority": "Majority",
    "tfidf_lr": "TF-IDF + LR",
    "sbert_lr": "SBERT + LR",
}
LABEL_SLUGS = {
    "Supported": "supported",
    "Refuted": "refuted",
    "Not Enough Evidence": "nee",
    "Conflicting Evidence/Cherrypicking": "conflicting",
}
REDIRECTS = {
    "data.html": "index.html#data",
    "methods.html": "index.html#methods",
    "results.html": "index.html#results",
    "errors.html": "index.html#errors",
    "demo.html": "index.html#claims",
    "team.html": "index.html#team",
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NA:
        return None
    return value


def _read_json(path: Path):
    if not path.exists():
        logger.info("Missing %s", path)
        return None
    return read_json(path)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.info("Missing %s", path)
        return None
    return pd.read_csv(path)


def _truncate(text, limit: int = EVIDENCE_LIMIT) -> str:
    cleaned = "" if text is None else str(text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _metrics_tree() -> dict:
    tree = {}
    for model in MODEL_ORDER:
        tree[model] = {}
        for mode in C.INPUT_MODES:
            tree[model][mode] = load_metrics(model, mode)
    return tree


def _frame_records(frame: pd.DataFrame | None):
    if frame is None:
        return None
    return _json_ready(frame.to_dict(orient="records"))


def _histogram(frames: list[pd.DataFrame], column: str, width: int) -> dict | None:
    series = []
    for frame in frames:
        if frame is None or column not in frame.columns:
            series.append(pd.Series(dtype=float))
            continue
        series.append(pd.to_numeric(frame[column], errors="coerce").dropna())
    if all(values.empty for values in series):
        return None
    high = max(float(values.max()) if not values.empty else 0.0 for values in series)
    top = max(int(np.ceil((high + 1) / width) * width), width)
    edges = np.arange(0, top + width, width)
    labels = [f"{int(edges[index])}\u2013{int(edges[index + 1] - 1)}" for index in range(len(edges) - 1)]
    counts = []
    for values in series:
        hist, _ = np.histogram(values.to_numpy(dtype=float), bins=edges)
        counts.append([int(bin_count) for bin_count in hist])
    return {"labels": labels, "train": counts[0], "test": counts[1]}


def _load_splits():
    train = pd.read_parquet(C.TRAIN_PARQUET) if C.TRAIN_PARQUET.exists() else None
    test = pd.read_parquet(C.TEST_PARQUET) if C.TEST_PARQUET.exists() else None
    if train is None and test is None:
        logger.info("Missing processed parquet; length histograms will be pending")
    return train, test


def _prediction_frame(model: str) -> pd.DataFrame | None:
    return _read_csv(C.PREDICTIONS_DIR / f"{model}_full.csv")


def _proba_list(row: pd.Series) -> list[float] | None:
    values = []
    for label in C.LABELS:
        column = f"proba_{label}"
        if column not in row.index or pd.isna(row[column]):
            return None
        values.append(float(row[column]))
    return values


def _claims(test: pd.DataFrame | None) -> list | None:
    if test is None or "claim" not in test.columns:
        return None
    predictions = {model: _prediction_frame(model) for model in PREDICTION_MODELS}
    if all(frame is None for frame in predictions.values()):
        return None
    indexed = {}
    for model, frame in predictions.items():
        if frame is None:
            continue
        frame = frame.copy()
        frame["claim_id"] = frame["claim_id"].astype(int)
        indexed[model] = frame.set_index("claim_id")
    claims = []
    for _, row in test.iterrows():
        claim_id = int(row["claim_id"])
        models = {}
        for model in PREDICTION_MODELS:
            table = indexed.get(model)
            if table is None or claim_id not in table.index:
                models[model] = None
                continue
            pred = table.loc[claim_id]
            if isinstance(pred, pd.DataFrame):
                pred = pred.iloc[0]
            models[model] = {
                "pred": None if pd.isna(pred["y_pred"]) else str(pred["y_pred"]),
                "correct": bool(int(pred["correct"])) if "correct" in pred.index and not pd.isna(pred["correct"]) else None,
                "proba": _proba_list(pred),
            }
        evidence = "" if pd.isna(row.get("evidence_text", "")) else str(row.get("evidence_text", ""))
        claims.append({
            "id": claim_id,
            "claim": "" if pd.isna(row["claim"]) else str(row["claim"]),
            "evidence": _truncate(evidence),
            "evidence_shortened": len(evidence.strip()) > EVIDENCE_LIMIT,
            "true": None if pd.isna(row["label"]) else str(row["label"]),
            "models": models,
        })
    claims.sort(key=lambda item: item["id"])
    return claims


def _error_rows(model: str) -> list | None:
    frame = _read_csv(C.ERRORS_DIR / f"{model}_errors.csv")
    if frame is None:
        return None
    rows = []
    for _, row in frame.iterrows():
        rows.append({
            "id": None if pd.isna(row.get("claim_id")) else int(row["claim_id"]),
            "claim": "" if pd.isna(row.get("claim")) else str(row["claim"]),
            "true": None if pd.isna(row.get("true")) else str(row["true"]),
            "pred_tfidf": None if pd.isna(row.get("pred_tfidf")) else str(row["pred_tfidf"]),
            "pred_sbert": None if pd.isna(row.get("pred_sbert")) else str(row["pred_sbert"]),
            "evidence": _truncate(row.get("evidence_text", "")),
            "reason": "" if pd.isna(row.get("auto_reason")) else str(row["auto_reason"]),
            "manual_reason": "" if pd.isna(row.get("manual_reason")) else str(row["manual_reason"]),
            "model": model,
        })
    return rows


def build_payload() -> dict:
    train, test = _load_splits()
    lengths = None
    if train is not None or test is not None:
        lengths = {
            "claim_len": _histogram([train, test], "claim_len", 5),
            "evidence_len": _histogram([train, test], "evidence_len", 40),
        }
    return {
        "labels": list(C.LABELS),
        "short_labels": dict(C.SHORT_LABELS),
        "label_slugs": LABEL_SLUGS,
        "modes": list(C.INPUT_MODES),
        "model_order": MODEL_ORDER,
        "prediction_models": PREDICTION_MODELS,
        "model_names": MODEL_NAMES,
        "evidence_limit": EVIDENCE_LIMIT,
        "copy": {
            "kicker": "CMPE 257",
            "title": "A verdict is only as useful as the evidence beside it.",
            "question": (
                "Can a semantic model (Sentence-BERT + logistic regression) beat a "
                "traditional model (TF-IDF + logistic regression) for claim verification, "
                "given identical data, splits, tuning budget, and metrics?"
            ),
            "license": "AVeriTeC is used under CC BY-NC 4.0 for non-commercial research and teaching.",
        },
        "protocol": {
            "seed": C.SEED,
            "cv_folds": C.CV_FOLDS,
            "c_grid": list(C.C_GRID),
            "n_bootstrap": C.N_BOOTSTRAP,
            "sbert_model": C.SBERT_MODEL,
        },
        "references": [
            {
                "label": "Schlichtkrull, Guo, Vlachos. AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web. NeurIPS 2023.",
                "url": "https://arxiv.org/abs/2305.13117",
            },
            {
                "label": "Reimers and Gurevych. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.",
                "url": "https://arxiv.org/abs/1908.10084",
            },
            {
                "label": "Thorne, Vlachos, Christodoulopoulos, Mittal. FEVER: a large-scale dataset for fact extraction and verification. NAACL 2018.",
                "url": "https://aclanthology.org/N18-1074/",
            },
            {
                "label": "Official AVeriTeC repository",
                "url": "https://github.com/MichSchli/AVeriTeC",
            },
        ],
        "preprocess": _read_json(C.PREPROCESS_REPORT),
        "eda": _frame_records(_read_csv(C.TABLES_DIR / "eda_summary.csv")),
        "metrics": _json_ready(_metrics_tree()),
        "mcnemar": _read_json(C.TABLES_DIR / "mcnemar.json"),
        "bootstrap": _read_json(C.TABLES_DIR / "bootstrap_diff.json"),
        "verdict": None if not (C.TABLES_DIR / "verdict.md").exists() else (C.TABLES_DIR / "verdict.md").read_text(encoding="utf-8").strip(),
        "claims": _claims(test),
        "errors": {model: _error_rows(model) for model in PREDICTION_MODELS},
        "lengths": lengths,
        "files": {
            "preprocess": "data/processed/preprocess_report.json",
            "lengths": "data/processed/train.parquet",
            "claims": "data/processed/test.parquet",
            "mcnemar": "results/tables/mcnemar.json",
            "bootstrap": "results/tables/bootstrap_diff.json",
            "verdict": "results/tables/verdict.md",
            "errors": {
                "tfidf_lr": "results/error_analysis/tfidf_lr_errors.csv",
                "sbert_lr": "results/error_analysis/sbert_lr_errors.csv",
            },
        },
    }


def _script_payload(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("<", "\\u003c")
        .replace("{{", "{\\u007b")
        .replace("{%", "{\\u0025")
    )


def _reset_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    assets = DOCS_DIR / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    for name in ["index.html", *REDIRECTS]:
        path = DOCS_DIR / name
        if path.exists():
            path.unlink()


def _copy_assets() -> None:
    assets = DOCS_DIR / "assets"
    vendor = assets / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SITE_DIR / "static" / "style.css", assets / "style.css")
    shutil.copyfile(SITE_DIR / "static" / "app.js", assets / "app.js")
    chart = VENDOR_DIR / "chart.umd.js"
    if not chart.exists():
        raise FileNotFoundError(f"Vendored Chart.js is missing: {chart}")
    shutil.copyfile(chart, vendor / "chart.umd.js")
    license_path = VENDOR_DIR / "LICENSE.md"
    if license_path.exists():
        shutil.copyfile(license_path, vendor / "LICENSE.md")


def _write_redirects() -> None:
    for name, target in REDIRECTS.items():
        html = (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>TruthTrace</title>\n<script>location.replace({json.dumps(target)});</script>\n"
            "</head>\n<body>\n"
            f'<p><a href="{target}">Continue to TruthTrace</a></p>\n'
            "</body>\n</html>\n"
        )
        (DOCS_DIR / name).write_text(html, encoding="utf-8")


def _assert_clean(html: str) -> None:
    if "{{" in html or "{%" in html:
        raise SystemExit("Generated HTML still contains Jinja delimiters.")


def build() -> None:
    C.ensure_dirs()
    payload = build_payload()
    _reset_docs()
    _copy_assets()
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
    )
    html = env.get_template("index.html").render(data_json=_script_payload(payload))
    _assert_clean(html)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    _write_redirects()
    for name in REDIRECTS:
        _assert_clean((DOCS_DIR / name).read_text(encoding="utf-8"))
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
    present = []
    missing = []
    for model, modes in payload["metrics"].items():
        for mode, metrics in modes.items():
            slot = f"{model}/{mode}"
            (present if metrics else missing).append(slot)
    logger.info("Site written to %s", DOCS_DIR)
    logger.info("Metrics present: %s", ", ".join(present) or "none")
    logger.info("Metrics missing: %s", ", ".join(missing) or "none")


def main() -> None:
    build()


if __name__ == "__main__":
    main()
