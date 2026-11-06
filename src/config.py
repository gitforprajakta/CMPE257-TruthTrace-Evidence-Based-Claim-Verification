"""Shared constants and paths. Created in Stage 1; do not change afterwards."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
SHORT_LABELS = {
    "Supported": "Supported",
    "Refuted": "Refuted",
    "Not Enough Evidence": "NEE",
    "Conflicting Evidence/Cherrypicking": "Conflicting",
}

INPUT_MODES = ["full", "claim_only", "evidence_only"]
NO_EVIDENCE_TOKEN = "[NO_EVIDENCE]"

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
C_GRID = [0.01, 0.1, 1, 10, 100]  # SAME grid for TF-IDF and SBERT
CV_FOLDS = 5
N_BOOTSTRAP = 1000

ROOT = Path(__file__).resolve().parents[1]
# Optional env vars to relocate data/results (default: ./data and ./results).
DATA_DIR = Path(os.environ.get("TRUTHTRACE_DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("TRUTHTRACE_RESULTS_DIR", ROOT / "results"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR = DATA_DIR / "embeddings"

METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ERRORS_DIR = RESULTS_DIR / "error_analysis"

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
PREPROCESS_REPORT = PROCESSED_DIR / "preprocess_report.json"


def ensure_dirs() -> None:
    """Create every output directory (safe to call repeatedly)."""
    for d in (RAW_DIR, PROCESSED_DIR, EMBED_DIR, METRICS_DIR, PREDICTIONS_DIR,
              MODELS_DIR, FIGURES_DIR, TABLES_DIR, ERRORS_DIR):
        d.mkdir(parents=True, exist_ok=True)
