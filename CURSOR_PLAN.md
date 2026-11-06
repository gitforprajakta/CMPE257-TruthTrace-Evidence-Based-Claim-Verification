# TruthTrace: Single End-to-End Implementation Plan (Stage 1: Krati, then Stage 2: Prajakta)

> **How to use this file:** put it in the root of the project repo and give it to Cursor.
>
> - **Krati** says: *"Read CURSOR_PLAN.md fully. Implement ONLY Stage 1. Do not start Stage 2. Run each acceptance check and fix failures before moving on. Do not ask questions."*
> - **Prajakta** (after Krati has committed and pushed Stage 1; pull it first) says: *"Read CURSOR_PLAN.md fully. Implement ONLY Stage 2. First do Milestone P0 to verify Stage 1 is done. Run each acceptance check and fix failures before moving on. Do not ask questions."*
>
> The work is **sequential on purpose**: Stage 2 starts only after Stage 1 is finished and committed, so the two people never work on the same files at the same time and never wait on stubs or half-finished code.

---

# 1. Project summary

**TruthTrace** (CMPE 257, Krati Jain and Prajakta Jivanrao Ketkar): evidence-based claim verification on the **AVeriTeC** dataset. Predict one of four verdicts: `Supported`, `Refuted`, `Not Enough Evidence`, `Conflicting Evidence/Cherrypicking`.

Research question: *Can a semantic model (Sentence-BERT + Logistic Regression) beat a traditional model (TF-IDF + Logistic Regression) for claim verification, given identical data, splits, tuning budget, and metrics?*

This is a **decision-support** system, not a truth oracle. The website and demo must say so.

# 2. Who does what

| Stage | Person | Delivers |
|---|---|---|
| **Stage 1** | **Krati** | Repo scaffold and foundation code, data download, preprocessing, EDA and figures, majority baseline, **TF-IDF + LR (3 input modes)**, tests, notebooks 01-02, Makefile, README, **website generator** (tolerates missing Stage 2 results) |
| **Stage 2** | **Prajakta** | **SBERT + LR (3 input modes)**, comparison tables and statistics (McNemar, bootstrap), all comparison figures, error analysis, Streamlit demo, tests, notebooks 03-04, final full run, website and README refresh |

Stage 2 reads everything Stage 1 produced and writes only new files, so nothing from Stage 1 is overwritten.

# 3. Rules for the Cursor agent (both stages)

1. Implement **only your stage**. Stage 1 must not build Stage 2 modules; Stage 2 must not rewrite Stage 1 modules (bug fixes and the website refresh in P7 excepted, and noted in your notes file).
2. **Do not ask questions.** Make reasonable choices and log them in your notes file (`notes/KRATI_NOTES.md` in Stage 1, `notes/PRAJAKTA_NOTES.md` in Stage 2).
3. **Never hardcode or invent results.** Every number on the site, README and figures comes from files in `results/`.
4. Seed `42` everywhere. Python 3.10+. Type hints, docstrings, `argparse` CLIs, `logging` (no bare prints). Every script runs as `python -m <module>` from the repo root and is idempotent (`--force` to recompute).
5. Do not commit big files (`data/`, `results/models/`, `*.npy`, `*.joblib`). Do commit `results/metrics`, `results/predictions`, `results/tables`, `results/figures`, `results/error_analysis`, and `docs/`.
6. Run each milestone's acceptance check before moving on and fix failures.

# Repository layout
```
truthtrace/
  CURSOR_PLAN.md, README.md, Makefile, requirements.txt, .gitignore
  data/{raw,processed,embeddings}/           (gitignored)
  src/   config.py utils.py evaluate.py                     [Stage 1]
         data_loading.py preprocess.py eda.py               [Stage 1]
         train_majority.py features_tfidf.py train_tfidf.py [Stage 1]
         readme_results.py                                  [Stage 1]
         features_sbert.py train_sbert.py                   [Stage 2]
         compare.py figures.py error_analysis.py            [Stage 2]
  app/streamlit_app.py                                      [Stage 2]
  site/build_site.py, site/templates/                       [Stage 1]
  docs/                                                     [generated site]
  notebooks/ 01_eda 02_tfidf [Stage 1]   03_sbert 04_comparison [Stage 2]
  tests/ test_preprocess test_tfidf [Stage 1]
         test_evaluate test_sbert_features test_compare [Stage 2]
  results/{metrics,predictions,models,figures,tables,error_analysis}/
  notes/
```

## 4. Foundation code (Stage 1 creates these files EXACTLY as shown)

### `src/__init__.py`
Empty file.

### `src/config.py`
```python
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
```

### `src/utils.py`
```python
"""Small helpers. Created in Stage 1; do not change afterwards."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C


def set_seed(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

### `src/evaluate.py`
```python
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
```


## 5. Data and file contracts (schemas every stage must follow)

### Processed data (`data/processed/train.parquet`, `test.parquet`) written in Stage 1, read in Stage 2
`test` = the official AVeriTeC **dev** split (test labels are not public). Columns (exact names):

| column | type | meaning |
|---|---|---|
| `claim_id` | int | unique within each split, running index starting at 0 |
| `claim` | str | cleaned claim text |
| `label` | str | one of `config.LABELS` |
| `label_id` | int | `LABEL2ID[label]` |
| `evidence_text` | str | `"Q: .. A: .."` pairs joined by `" [SEP] "`; or `NO_EVIDENCE_TOKEN` |
| `evidence_pairs` | list[str] | the individual pair strings; `[NO_EVIDENCE_TOKEN]` if none |
| `has_evidence` | int (0/1) | 0 when there are no usable QA pairs |
| `n_qa` | int | number of usable QA pairs (0 if none) |
| `claim_len` | int | words in claim |
| `evidence_len` | int | words in evidence_text (0 if none) |
| `speaker` | str | may be empty |
| `claim_date` | str | may be empty |
| `combined_text` | str | `claim + " [SEP] " + evidence_text` |

The `justification` field is **never** part of any column (it leaks the verdict).

### `data/processed/preprocess_report.json` (written in Stage 1)
```json
{"source": "chenxwh_hf | pminervini_hf | github",
 "train": {"loaded": 0, "dropped_invalid": 0, "dropped_duplicates": 0, "final": 0,
           "missing_evidence_share": 0.0, "class_counts": {"<label>": 0}},
 "test":  {"loaded": 0, "dropped_invalid": 0, "dropped_duplicates": 0,
           "removed_train_overlap": 0, "final": 0,
           "missing_evidence_share": 0.0, "class_counts": {"<label>": 0}}}
```

### Metrics JSON (`results/metrics/{model}_{mode}.json`)
Produced only via `evaluate.evaluate_predictions` + `evaluate.save_metrics`. Keys: `model, input_mode, split, seed, n_train, n_test, labels, best_params, cv_macro_f1, train_seconds, accuracy, macro_precision, macro_recall, macro_f1, weighted_f1, macro_f1_ci95, per_class{label:{precision,recall,f1,support}}, confusion_matrix, confusion_matrix_normalized` (+ optional `extra`).
`model` is one of `majority`, `tfidf_lr`, `sbert_lr` (an optional second encoder may use `sbert_mpnet_lr`). `input_mode` is one of `full`, `claim_only`, `evidence_only` (majority only has `full`).

### Predictions CSV (`results/predictions/{model}_{mode}.csv`)
Produced only via `evaluate.save_predictions`. Columns: `claim_id, y_true, y_pred, correct, proba_<label>` for each label (if probabilities available). Rows are in the same order as `test.parquet`.

### Model files
- `results/models/tfidf_lr_{mode}.joblib`: a plain **scikit-learn `Pipeline`** using only sklearn classes (no custom classes, so it unpickles anywhere). Its `predict` / `predict_proba` accept a **pandas DataFrame with columns `claim`, `evidence_text`, `has_evidence`**.
- `results/models/sbert_lr_{mode}.joblib` + sidecar `results/models/sbert_lr_{mode}.json` (`{"sbert_model": "...", "feature_mode": "full|claim_only|evidence_only"}`): a sklearn `Pipeline` that takes the feature matrix produced by `src.features_sbert.build_features`.

### Tables written in Stage 2 (the website, built in Stage 1, must tolerate their absence)
- `results/tables/main_comparison.csv`: columns `model, accuracy, macro_precision, macro_recall, macro_f1, macro_f1_ci_low, macro_f1_ci_high, weighted_f1`; rows for `majority`, `tfidf_lr`, `sbert_lr` (full mode). A `.md` copy too.
- `results/tables/per_class_f1.csv`: `model, label, precision, recall, f1, support`.
- `results/tables/ablation.csv`: `model, input_mode, accuracy, macro_f1`.
- `results/tables/mcnemar.json`: `{"model_a":"tfidf_lr","model_b":"sbert_lr","table":[[both_correct,a_only],[b_only,both_wrong]],"n_discordant":0,"p_value":0.0,"test":"exact|chi2"}`.
- `results/tables/bootstrap_diff.json`: `{"metric":"macro_f1","diff_mean":0.0,"ci95":[0.0,0.0],"p_sbert_better":0.0,"n_resamples":1000}` (diff = sbert minus tfidf).
- `results/tables/verdict.md`: one plain-English paragraph generated from the real numbers.

### Error analysis files written in Stage 2
- `results/error_analysis/tfidf_lr_errors.csv`, `sbert_lr_errors.csv`, `disagreements.csv`: columns `claim_id, claim, true, pred_tfidf, pred_sbert, evidence_text, has_evidence, n_qa, claim_len, auto_reason, manual_reason` (`evidence_text` truncated to 600 chars; `manual_reason` left empty).
- `results/error_analysis/summary.md`.

### Figure file names (PNG, dpi 150)
- Stage 1: `eda_class_distribution.png`, `eda_claim_length.png`, `eda_evidence_length.png`, `eda_missing_evidence_by_class.png`, `eda_n_qa_by_class.png`, `eda_nee_phrase_check.png`, `tfidf_top_words.png`.
- Stage 2: `cm_tfidf_lr_full.png`, `cm_sbert_lr_full.png`, `cm_normalized_tfidf_lr_full.png`, `cm_normalized_sbert_lr_full.png`, `per_class_prf_tfidf_lr.png`, `per_class_prf_sbert_lr.png`, `comparison_overall.png`, `comparison_per_class_f1.png`, `ablation_input_modes.png`, `sbert_umap_claims.png`, `error_reason_counts.png`.

### CLI contract (so the Makefile and notebooks can call each stage's code)
| Module | CLI |
|---|---|
| `src.data_loading` | `python -m src.data_loading [--force]` |
| `src.preprocess` | `python -m src.preprocess [--force]` |
| `src.eda` | `python -m src.eda` |
| `src.train_majority` | `python -m src.train_majority` |
| `src.train_tfidf` | `python -m src.train_tfidf --mode {full,claim_only,evidence_only} [--force]` |
| `src.train_sbert` | `python -m src.train_sbert --mode {full,claim_only,evidence_only} [--sbert-model NAME] [--force]` |
| `src.compare` | `python -m src.compare` |
| `src.figures` | `python -m src.figures` |
| `src.error_analysis` | `python -m src.error_analysis` |
| `site/build_site.py` | `python site/build_site.py` |

### Shared `requirements.txt` content
```
numpy
pandas
pyarrow
scikit-learn
scipy
matplotlib
seaborn
sentence-transformers
torch
huggingface_hub
datasets
statsmodels
umap-learn
jinja2
joblib
tqdm
pytest
streamlit
jupyter
nbformat
nbconvert
```

---


# STAGE 1: KRATI

**Stage 1 delivers:** foundation, data, preprocessing, EDA, majority baseline, **Model 1 (TF-IDF + Logistic Regression)**, tests, notebooks 01-02, and the **Makefile, README and website generator**. Stage 1 must be fully finished and committed before Stage 2 starts.

Do NOT implement anything from Stage 2 (SBERT, comparison, figures other than EDA/TF-IDF ones, error analysis, demo). Your code must simply not depend on those files existing.

Log decisions and anything surprising in `notes/KRATI_NOTES.md`.

---

## Milestone K1: Foundation, data, preprocessing

### K1.1 Scaffold
Create the repo skeleton and these files:
- `src/__init__.py`, `src/config.py`, `src/utils.py`, `src/evaluate.py`: copy **exactly** from Section 4.
- `requirements.txt`: exactly the list in Section 5.
- `.gitignore`: `data/`, `results/models/`, `__pycache__/`, `.venv/`, `.ipynb_checkpoints/`, `*.joblib`, `*.npy`. Keep `.gitkeep` files in the empty tracked directories (`results/metrics`, `results/predictions`, `results/figures`, `results/tables`, `results/error_analysis`, `notes`, `tests`, `notebooks`).
- `Makefile` with these targets (use tabs):
  ```
  setup      pip install -r requirements.txt
  data       python -m src.data_loading && python -m src.preprocess
  eda        python -m src.eda
  majority   python -m src.train_majority
  tfidf      for each mode in full claim_only evidence_only: python -m src.train_tfidf --mode $$m
  sbert      for each mode: python -m src.train_sbert --mode $$m        # Stage 2 module
  compare    python -m src.compare                                       # Stage 2 module
  figures    python -m src.figures                                       # Stage 2 module
  errors     python -m src.error_analysis                                # Stage 2 module
  readme     python -m src.readme_results
  site       python site/build_site.py
  demo       streamlit run app/streamlit_app.py                          # Stage 2 file
  test       pytest -q
  all        setup data eda majority tfidf sbert compare figures errors readme site test
  ```
  The Makefile only calls Stage 2 modules by the CLI names in Section 5; it must not import them. Those targets will fail until Stage 2 is done, which is expected.

**Acceptance:** `python -c "import src.config, src.evaluate, src.utils"` works. Files match Section 4 exactly (a diff-based check is fine).

### K1.2 Data loading (`src/data_loading.py`)
Download the official AVeriTeC **train** and **dev** JSON into `config.RAW_DIR` using this fallback chain (log which source worked, record it for `preprocess_report.json`):
1. `huggingface_hub.hf_hub_download(repo_id="chenxwh/AVeriTeC", filename="data/train.json")` and `"data/dev.json"` (this is a *model-type* repo, so use the default `repo_type`; if that fails retry with `repo_type="dataset"`). Source name: `chenxwh_hf`.
2. `datasets.load_dataset("pminervini/averitec")`, exported to JSON records. Source name: `pminervini_hf`.
3. Raw files from the GitHub repo `MichSchli/AVeriTeC` (`data/train.json`, `data/dev.json`) via `urllib`. Source name: `github`.

Save as `train.json` and `dev.json` in `RAW_DIR` plus `source.txt` with the source name. Skip if files exist (unless `--force`). Expected record fields (inspect and access defensively with `.get`):
```
claim, label, justification (NEVER USE), questions: [ {question, answers: [ {answer, answer_type,
boolean_explanation, source_url, source_medium} ]} ], speaker, claim_date, reporting_source, location_ISO_code
```
Test labels are not public, so the official **dev** split is our **test** set.

### K1.3 Preprocessing (`src/preprocess.py`)
Produce `config.TRAIN_PARQUET`, `config.TEST_PARQUET`, and `config.PREPROCESS_REPORT`, matching the schemas in Section 5 **exactly**.

Steps (deterministic):
1. Normalize unicode (`NFKC`), collapse whitespace, strip.
2. Normalize labels to the four in `config.LABELS`. Handle case and variants (for example `Conflicting Evidence`, `Conflicting Evidence/Cherrypicking`, `Not Enough Information`, `Not Enough Evidence`). Raise a clear error listing any unmappable label.
3. Drop rows with an empty claim or an unmappable label (count them as `dropped_invalid`).
4. Deduplicate by normalized (lowercased) claim text within each split, keeping the first (`dropped_duplicates`). Then **remove from test any claim that also appears in train** (`removed_train_overlap`).
5. Build `evidence_pairs`: for each question, for the **first 3 answers**, create `"Q: {question} A: {answer}"`; if `answer_type == "Boolean"` and `boolean_explanation` is non-empty, append `" Explanation: {boolean_explanation}"`. Skip empty answers. `evidence_text = " [SEP] ".join(pairs)`. **Never** use `justification`.
6. No usable pairs: `evidence_pairs = [NO_EVIDENCE_TOKEN]`, `evidence_text = NO_EVIDENCE_TOKEN`, `has_evidence = 0`, `n_qa = 0`, `evidence_len = 0`.
7. Add all other columns from Section 5 (`claim_id` reset to `0..n-1` per split, `label_id`, `claim_len`, `evidence_len`, `speaker`, `claim_date`, `combined_text`).
8. Save parquet (pyarrow) and the JSON report.

**Acceptance:**
- Both parquet files and the report exist, with exactly the Section 5 columns; all four labels appear in train.
- No claim text overlaps between train and test.
- No `justification` text appears in `evidence_text` (unit test with a marker string).


---

## Milestone K2: EDA and baseline

### K2.1 EDA (`src/eda.py`)
Write PNGs (dpi 150, consistent seaborn style, `SHORT_LABELS` on axes) with exactly the Stage 1 figure names in Section 5:
- `eda_class_distribution.png`: train vs test counts per class with percentage labels.
- `eda_claim_length.png`, `eda_evidence_length.png`: histograms per split.
- `eda_missing_evidence_by_class.png`: share with `has_evidence == 0` per label.
- `eda_n_qa_by_class.png`: boxplot of `n_qa` per label.
- `eda_nee_phrase_check.png`: share of evidence containing phrases such as "no answer could be found" / "no answer" per label. This documents the annotation shortcut risk.

Also write `results/tables/eda_summary.csv` (per split and label: count, share, mean claim_len, mean evidence_len, missing-evidence share).

### K2.2 Majority baseline (`src/train_majority.py`)
`DummyClassifier(strategy="most_frequent")` on the `full` input mode. Report only through `evaluate.evaluate_predictions`, `save_metrics`, and `save_predictions`, with `model="majority"`.

**Acceptance:** all EDA PNGs and `eda_summary.csv` exist; `majority_full.json` and `.csv` exist and pass the Section 5 schema.

---

## Milestone K3: Model 1, TF-IDF + Logistic Regression

Files: `src/features_tfidf.py`, `src/train_tfidf.py`. CLI per Section 5.

**Pipeline** (use **only sklearn classes** so the saved model has no custom code):
- `ColumnTransformer` over a DataFrame with columns `claim`, `evidence_text`, `has_evidence`:
  - `claim` -> `TfidfVectorizer(min_df=2, max_features=50000, sublinear_tf=True, strip_accents="unicode", lowercase=True)`
  - `evidence_text` -> same settings
  - `has_evidence` -> passthrough numeric
- Input modes: `claim_only` uses only the claim vectorizer; `evidence_only` uses the evidence vectorizer + `has_evidence`; `full` uses both + `has_evidence`.
- Classifier: `LogisticRegression(max_iter=3000, class_weight="balanced", random_state=C.SEED)`.
- **Tuning:** outer loop over `ngram_range` in `[(1,1), (1,2)]` (applied to every vectorizer in the pipeline), inner `GridSearchCV` over `C` in `config.C_GRID`, `StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)`, scoring `f1_macro`, `n_jobs=-1`. Choose the combination with the best CV macro-F1. Record both in `best_params`.
- Fit the best pipeline on all of train; predict test **once**.
- Save to `results/models/tfidf_lr_{mode}.joblib`.
- For `full` mode also write `results/tables/tfidf_top_words_full.json`: top 20 positive-coefficient features per class, each with its source (`claim` or `evidence`) and weight.
- Write the metrics/predictions with `model="tfidf_lr"`, including `best_params`, `cv_macro_f1`, `train_seconds`, and probabilities (`save_predictions` with `proba=` and `classes=clf.classes_`).
- Also produce `results/figures/tfidf_top_words.png` (2x2 grid, top 10 words per class).

**Acceptance:** for all three modes there is a metrics JSON, a predictions CSV and a joblib file; the top-words JSON and PNG exist; loading the `full` joblib and calling `predict_proba` on a one-row DataFrame with the three columns works (test it).

---

## Milestone K4: Tests, notebooks

**Tests (yours only):**
- `tests/test_preprocess.py`: label normalization variants; justification marker never appears in `evidence_text`; no train/test overlap on a synthetic fixture; missing evidence gives the placeholder and `has_evidence == 0`; output columns equal the Section 5 list.
- `tests/test_tfidf.py`: the pipeline fits on a tiny toy DataFrame for all three modes; the pickled pipeline works on a 1-row DataFrame.

**Notebooks:** `notebooks/01_eda.ipynb` and `notebooks/02_tfidf.ipynb`, built with `nbformat`. They only import from `src` and display saved outputs (tables, figures, a few example predictions) with short markdown explanations. They must execute top to bottom after K1-K3.

**Acceptance:** `pytest -q tests/test_preprocess.py tests/test_tfidf.py` passes; both notebooks run with `jupyter nbconvert --execute`.

---

## Milestone K5: README, results injector, website generator

All of this reads Stage 2 output files via the Section 5 contract. **Every Stage 2 input will be missing during Stage 1, so each is optional:** if a file is missing, render a clear "Pending: waiting for <file>" placeholder box instead of failing.

### K5.1 `src/readme_results.py`
Reads `results/tables/main_comparison.csv` and rewrites the block between `<!-- RESULTS_START -->` and `<!-- RESULTS_END -->` in `README.md` with a Markdown table and the contents of `verdict.md`. If the files are absent, write "Results pending".

### K5.2 `README.md`
Project summary; research question; dataset and license note (AVeriTeC is CC BY-NC 4.0, non-commercial use); setup and `make all`; repo layout; **who does what** (Stage 1 by Krati, Stage 2 by Prajakta); how to reproduce each result; the results markers block above; limitations (annotation shortcut risk, small minority classes, evidence is human-annotated QA rather than retrieved, no claim of real-world truth detection); ethics note (decision-support only); team (Krati Jain, Prajakta Jivanrao Ketkar); references; related repos:
- https://github.com/MichSchli/AVeriTeC
- https://huggingface.co/chenxwh/AVeriTeC
- https://huggingface.co/datasets/pminervini/averitec
- https://github.com/UKPLab/sentence-transformers
- https://fever.ai/

Include GitHub Pages instructions (Settings -> Pages -> branch `main`, folder `/docs`) 

### K5.3 Website (`site/build_site.py` + `site/templates/*.html` -> `docs/`)
Static, responsive, self-contained site built with Jinja2. One `docs/assets/style.css` (clean modern look, light and dark via `prefers-color-scheme`), minimal vanilla JS only for tabs. Write `docs/.nojekyll`. Copy every PNG from `results/figures/` to `docs/assets/figures/` and use **relative paths** so it works from `file://` and from GitHub Pages.

**No number may be typed into a template.** Read from: `data/processed/preprocess_report.json`, `results/metrics/*.json`, `results/tables/*`, `results/error_analysis/*.csv`, `results/predictions/*.csv`. Add a post-build check that no `{{` or `{%` remains in any generated HTML.

Pages (shared nav on all):
1. **Home (`index.html`):** one-sentence problem, the research question, a headline-results card (accuracy and macro-F1 for majority / TF-IDF / SBERT from `main_comparison.csv`, with the `verdict.md` text), and the decision-support disclaimer.
2. **Data (`data.html`):** AVeriTeC description (4,568 claims, 50 fact-checking organizations; link the paper arXiv:2305.13117 and the official repo), preprocessing steps, counts from `preprocess_report.json`, the EDA figures, and a callout on the annotation shortcut and the leakage precautions (justification excluded, cross-split dedup).
3. **Methods (`methods.html`):** two pipeline diagrams as inline SVG or CSS boxes (Claim + Evidence -> TF-IDF -> LR; Claim + Evidence -> SBERT embeddings -> `[u, v, |u-v|, u*v]` -> LR), plus the evaluation protocol: official train/dev split, 5-fold stratified CV, identical `C` grid, class weighting, macro-F1, bootstrap CI, McNemar test.
4. **Results (`results.html`):** main comparison table, `comparison_overall.png`, `comparison_per_class_f1.png`, `ablation_input_modes.png`, the confusion matrices side by side, `tfidf_top_words.png`, `sbert_umap_claims.png`, McNemar and bootstrap results (from the JSONs), the verdict, and download links to each metrics JSON (copy them to `docs/assets/metrics/`).
5. **Error analysis (`errors.html`):** `error_reason_counts.png`, a table of about 10 misclassified claims per model (evidence in a collapsible `<details>`), and disagreement examples.
6. **Demo (`demo.html`):** 8-10 precomputed test claims (mix of correct and incorrect) with the true label and both models' predictions and probabilities, read from the predictions CSVs joined to `test.parquet` on `claim_id`. Show how to launch the live demo (`make demo`) and repeat the disclaimer.
7. **Team and references (`team.html`):** Krati Jain (preprocessing, EDA, TF-IDF + LR, website), Prajakta Jivanrao Ketkar (SBERT + LR, comparison, error analysis, demo); CMPE 257; references (AVeriTeC arXiv:2305.13117, Reimers & Gurevych 2019 Sentence-BERT, Thorne et al. 2018 FEVER).

**Acceptance:** `make site` generates all 7 pages with the Stage 1 data (EDA, TF-IDF, majority) filled in and "Pending" boxes where Stage 2 results will go; navigation and images work from `file://`.

---

## Stage 1 exit checklist (all must be true before Stage 2)
1. `pytest -q tests/test_preprocess.py tests/test_tfidf.py` passes.
2. Processed parquet files and `preprocess_report.json` exist and match Section 5.
3. Metrics, predictions and joblib files exist for `majority_full` and `tfidf_lr_{full,claim_only,evidence_only}`.
4. EDA figures, `tfidf_top_words.png`, `eda_summary.csv`, `tfidf_top_words_full.json` exist.
5. `make site` and `make readme` run and produce a site with "Pending" boxes for Stage 2 content.
6. Write `notes/STAGE1_DONE.md` listing the checklist above with the actual metrics-file names and the preprocess counts, then commit everything.

---


# STAGE 2: PRAJAKTA

**Stage 2 delivers:** **Model 2 (Sentence-BERT + Logistic Regression)**, the **comparison and statistics**, all **comparison figures**, **error analysis**, the **live demo**, notebooks 03-04, and the final full run plus website refresh.


Log decisions and anything surprising in `notes/PRAJAKTA_NOTES.md`.

---

## Milestone P0: Verify Stage 1 is done

Stage 1 is already merged into your working copy. Before writing anything:
1. Confirm `notes/STAGE1_DONE.md` exists. If not, stop and report that Stage 1 is incomplete.
2. Run `pytest -q tests/test_preprocess.py tests/test_tfidf.py` and confirm the files in the Stage 1 exit checklist exist.
3. If the data or `tfidf_lr_*` results are missing (for example `data/` is gitignored and not on your machine), run `make data eda majority tfidf` to regenerate them before continuing.
4. Do not modify Stage 1 files, except for a genuine bug fix: describe any such fix in `notes/PRAJAKTA_NOTES.md`.

---

## Milestone P1: Model 2, SBERT + Logistic Regression

Files: `src/features_sbert.py`, `src/train_sbert.py`. CLI per Section 5. Input: `config.TRAIN_PARQUET`, `config.TEST_PARQUET` (Section 5 schema), produced by Stage 1.

### P1.1 `src/features_sbert.py`
- `encode_texts(texts, model_name, batch_size=64, device=None) -> np.ndarray`: `SentenceTransformer(model_name)`, `normalize_embeddings=True`, progress bar; device `cuda` > `mps` > `cpu`. Support a `fake=True` option (hash-based deterministic random vectors, 384 dims) for offline tests and smoke runs; expose it via a `--fake-encoder` flag in the CLI.
- `embed_claims(df, model_name, split, force=False) -> np.ndarray`: encode `df["claim"]`.
- `embed_evidence(df, model_name, split, force=False) -> np.ndarray`: **encode each string in `evidence_pairs` separately, then mean-pool per claim** (avoids truncation of long evidence). Flatten all pairs, encode once, then group back with an index map or `np.add.reduceat`. For rows without evidence the single pair is `NO_EVIDENCE_TOKEN`.
- **Caching:** save `.npy` files in `config.EMBED_DIR` named `{model_slug}_{split}_{claim|evidence}.npy` (`model_slug` = model name with `/` replaced by `_`). Skip when the cache exists and its row count matches (unless `force`).
- `build_features(claim_emb, evid_emb, has_evidence, mode) -> np.ndarray`:
  - `full`: `[u, v, |u-v|, u*v, has_evidence]`
  - `claim_only`: `[u]`
  - `evidence_only`: `[v, has_evidence]`
  (`u` = claim embedding, `v` = evidence embedding.) This function is the single source of truth, also used by the demo.

### P1.2 `src/train_sbert.py`
- Load train and test parquet. Build features for the chosen `--mode`.
- Classifier: `Pipeline([StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=C.SEED)])`.
- **Tuning (must be identical in protocol to Model 1):** `GridSearchCV` over `C` in `config.C_GRID`, `StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)`, scoring `f1_macro`, `n_jobs=-1`. Fit on all train, predict test **once**.
- Report only through `evaluate.evaluate_predictions`, `save_metrics`, `save_predictions` (with `proba=` and `classes=pipeline.classes_`), `model="sbert_lr"` (or `sbert_mpnet_lr` if `--sbert-model` is not the default), and put the SBERT model name in `extra={"sbert_model": ...}`.
- Save `results/models/sbert_lr_{mode}.joblib` (sklearn pipeline only) and the sidecar `results/models/sbert_lr_{mode}.json` (`{"sbert_model": ..., "feature_mode": ...}`).
- In `full` mode also save `results/figures/sbert_umap_claims.png`: UMAP of the **test claim embeddings** colored by true label (fall back to t-SNE if UMAP fails; `SHORT_LABELS` in the legend).
- Optional model: `--sbert-model sentence-transformers/all-mpnet-base-v2` writes `sbert_mpnet_lr_*` files. It is not required for the default run.

**Acceptance:** for all three modes there are a metrics JSON, a predictions CSV, a joblib, and a sidecar JSON; embedding caches exist; a second run reuses the cache and is much faster. `--fake-encoder` is for quick smoke runs and tests only; real results must use the real encoder.

---

## Milestone P2: Comparison and statistics (`src/compare.py`)

Reads `results/metrics/*.json` and `results/predictions/*.csv` via `evaluate.load_metrics/load_predictions`. **Missing models are skipped with a warning** (so it runs even if only SBERT exists). Write the Section 5 table files exactly:
- `main_comparison.csv` and `.md`: rows `majority`, `tfidf_lr`, `sbert_lr` (full mode) with columns `model, accuracy, macro_precision, macro_recall, macro_f1, macro_f1_ci_low, macro_f1_ci_high, weighted_f1`.
- `per_class_f1.csv`, `ablation.csv` (model x input_mode).
- `mcnemar.json`: McNemar test on per-example correctness between `tfidf_lr` and `sbert_lr` (full mode), joined on `claim_id`. Use `statsmodels.stats.contingency_tables.mcnemar` (exact binomial when discordant pairs < 25, else chi-square with continuity correction). Record the 2x2 table `[[both_correct, a_only],[b_only, both_wrong]]`, `n_discordant`, `p_value`, `test`.
- `bootstrap_diff.json`: paired bootstrap (same resampled indices for both models, 1000 resamples, seed 42) of macro-F1 difference (sbert minus tfidf): `diff_mean`, `ci95`, `p_sbert_better` (fraction of resamples with a positive difference).
- `verdict.md`: **one plain-English paragraph generated by conditional logic on the real numbers**. It states the macro-F1 for each model, the difference and its CI, the McNemar p-value, and whether SBERT is better, worse, or not significantly different at alpha = 0.05. It must never claim something the numbers do not support, and it must mention the majority baseline. If a model is missing, say which results are pending.
- Make the statistics **pure functions** taking DataFrames/arrays (for example `mcnemar_from_predictions(df_a, df_b)`, `paired_bootstrap_diff(df_a, df_b)`, `build_verdict(...)`), so tests do not depend on global paths.

**Acceptance:** writes every file above from the real Stage 1 + Stage 2 results; `verdict.md` numbers equal the table numbers (test it).

---

## Milestone P3: Figures (`src/figures.py`)

All PNG, dpi 150, tight layout, consistent seaborn style, `SHORT_LABELS` on axes. Skip a figure with a warning if its inputs are missing. Exact output names (Section 5):
- `cm_tfidf_lr_full.png`, `cm_sbert_lr_full.png`: confusion matrices with raw counts annotated.
- `cm_normalized_tfidf_lr_full.png`, `cm_normalized_sbert_lr_full.png`: row-normalized.
- `per_class_prf_tfidf_lr.png`, `per_class_prf_sbert_lr.png`: grouped bars of precision / recall / F1 per class.
- `comparison_overall.png`: grouped bars of accuracy, macro-F1, weighted-F1 for majority / TF-IDF / SBERT, with bootstrap CI error bars on macro-F1.
- `comparison_per_class_f1.png`: per-class F1, TF-IDF vs SBERT.
- `ablation_input_modes.png`: macro-F1 by input mode (`claim_only`, `evidence_only`, `full`) for each model.

(`sbert_umap_claims.png` comes from P1; `error_reason_counts.png` from P4.)

**Acceptance:** all names above exist after a full run.

---

## Milestone P4: Error analysis (`src/error_analysis.py`)

Inputs: `test.parquet` plus predictions of `tfidf_lr` and `sbert_lr` (full mode), joined on `claim_id`. If one model is missing, analyze the other and leave its columns empty.
- Sample **15 misclassified claims per model**, stratified over true classes, seed 42.
- Sample up to **10** claims where SBERT is right and TF-IDF is wrong, and up to **10** the reverse (`disagreements.csv`, with a `who_wins` column).
- Columns exactly as in Section 5: `claim_id, claim, true, pred_tfidf, pred_sbert, evidence_text (truncated to 600 chars), has_evidence, n_qa, claim_len, auto_reason, manual_reason` (leave `manual_reason` empty).
- `auto_reason`, first match wins:
  1. `no_evidence` if `has_evidence == 0`
  2. `shortcut_phrase` if the evidence contains "no answer could be found" (or similar) and the true label is Not Enough Evidence
  3. `conflicting_confusion` if the true or predicted label is Conflicting
  4. `thin_evidence` if `n_qa <= 1`
  5. `short_claim` if `claim_len < 8`
  6. `other`
- Write `results/error_analysis/{tfidf_lr,sbert_lr}_errors.csv`, `disagreements.csv`, `summary.md` (counts per reason per model, plus a short readable list of examples), and `results/figures/error_reason_counts.png`.

**Acceptance:** CSVs have the required columns; the PNG exists.

---

## Milestone P5: Live demo (`app/streamlit_app.py`)

- Two text areas: claim and evidence (QA pairs or free text); a dropdown to load an example from `test.parquet`.
- Loads `results/models/tfidf_lr_full.joblib` (optional; if missing show "TF-IDF model not available") and `sbert_lr_full.joblib` plus its sidecar JSON. Loads the SBERT encoder lazily with `st.cache_resource`.
- For TF-IDF: build a one-row DataFrame with `claim`, `evidence_text`, `has_evidence`. For SBERT: embed the claim; split evidence on `" [SEP] "` into pairs, embed and mean-pool (reuse the helpers in `src/features_sbert.py`; **do not duplicate logic**), then call `build_features`.
- Empty evidence uses the `NO_EVIDENCE_TOKEN` path and `has_evidence = 0`.
- Show each model's predicted verdict and class-probability bars, plus a prominent banner: *"This is a decision-support prototype trained on AVeriTeC. It is not a source of truth."*

**Acceptance:** starts without error and returns predictions for both models on an example.

---

## Milestone P6: Tests and notebooks

**Tests (yours only):**
- `tests/test_evaluate.py`: schema keys, 4x4 confusion matrix, normalized rows sum to 1 (or 0 for empty classes), `bootstrap_ci` returns lo <= hi, `align_proba` reorders correctly.
- `tests/test_sbert_features.py`: `build_features` output shapes for each mode on random arrays; mean-pooling of evidence pairs groups rows correctly (use the fake encoder, no download).
- `tests/test_compare.py`: McNemar and bootstrap functions on small hand-made DataFrames with known answers; `build_verdict` text agrees with the numbers it is given.

**Notebooks:** `notebooks/03_sbert.ipynb` and `notebooks/04_comparison.ipynb`, built with `nbformat`. They import from `src` and display saved outputs (tables, figures, examples) with short markdown explanations. 04 shows the main comparison, the stats, the figures, and the error-analysis summary. They must execute top to bottom.

**Acceptance:** `pytest -q tests/test_evaluate.py tests/test_sbert_features.py tests/test_compare.py` passes; notebooks execute top to bottom.

---

## Milestone P7: Full run, website and README refresh

1. Run in order: `make sbert compare figures errors readme site test` (or `make all` if the environment is clean).
2. Check that `results/tables/main_comparison.csv` contains `majority`, `tfidf_lr` and `sbert_lr`, and that `verdict.md` agrees with it.
3. Open `docs/index.html`: the Stage 1 "Pending" boxes must now show real tables, figures, McNemar and bootstrap results, error examples and the demo examples. If any page renders wrongly or the site build fails with the full results present, **fix `site/build_site.py` or its templates** (this is the one allowed edit to Stage 1 files besides bug fixes) and note it in `notes/PRAJAKTA_NOTES.md`.
4. Check that the README results block was filled in and that no number on the site or README is typed by hand.
5. Commit everything, including `docs/`.

---

## Stage 2 exit checklist (project complete)
1. `sbert_lr_{full,claim_only,evidence_only}` metrics, predictions and joblibs exist.
2. `main_comparison`, `per_class_f1`, `ablation`, `mcnemar`, `bootstrap_diff`, `verdict` exist and agree with each other.
3. Every Stage 2 figure exists (Section 5 list), including `sbert_umap_claims.png` and `error_reason_counts.png`.
4. Error analysis CSVs exist, ready for the team to fill `manual_reason` by hand.
5. `pytest -q` passes for the whole repo; notebooks 01-04 execute; the demo runs.
6. `docs/` is fully populated and ready for GitHub Pages.
