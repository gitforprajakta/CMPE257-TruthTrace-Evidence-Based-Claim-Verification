# Stage 1 complete

Krati Jain. Stage 2 has not been started.

## Exit checklist

1. `pytest -q tests/test_preprocess.py tests/test_tfidf.py` — 6 passed.
2. Processed files match the Section 5 contract (local `data/processed/`, gitignored):
   - source `chenxwh_hf`
   - train: loaded 3068, dropped_invalid 1, dropped_duplicates 71, final 2996, missing_evidence_share 0
   - train class counts: Supported 809, Refuted 1727, Not Enough Evidence 268, Conflicting Evidence/Cherrypicking 192
   - test (official dev): loaded 500, dropped_invalid 0, dropped_duplicates 9, removed_train_overlap 4, final 487, missing_evidence_share 0
   - test class counts: Supported 111, Refuted 305, Not Enough Evidence 33, Conflicting Evidence/Cherrypicking 38
   - no shared claim text between train and test; columns match the contract
3. Metrics, predictions, and local joblib files:
   - `results/metrics/majority_full.json`, `results/predictions/majority_full.csv`
   - `results/metrics/tfidf_lr_full.json`, `results/predictions/tfidf_lr_full.csv`
   - `results/metrics/tfidf_lr_claim_only.json`, `results/predictions/tfidf_lr_claim_only.csv`
   - `results/metrics/tfidf_lr_evidence_only.json`, `results/predictions/tfidf_lr_evidence_only.csv`
   - `results/models/tfidf_lr_{full,claim_only,evidence_only}.joblib` (gitignored). Loading `tfidf_lr_full.joblib` and calling `predict_proba` on a one-row frame returns shape (1, 4).
4. Figures and tables:
   - `results/figures/eda_class_distribution.png`
   - `results/figures/eda_claim_length.png`
   - `results/figures/eda_evidence_length.png`
   - `results/figures/eda_missing_evidence_by_class.png`
   - `results/figures/eda_n_qa_by_class.png`
   - `results/figures/eda_nee_phrase_check.png`
   - `results/figures/tfidf_top_words.png`
   - `results/tables/eda_summary.csv`
   - `results/tables/tfidf_top_words_full.json`
5. `python -m src.readme_results` and `python site/build_site.py` succeed. The site shows Stage 1 metrics, EDA, and TF-IDF, and pending boxes for Stage 2 (SBERT metrics, comparison table, McNemar, bootstrap, verdict, error analysis, UMAP).
6. This file is that checklist.

## Test-set scores (from the metrics JSON files)

| File | Accuracy | Macro-F1 | CV macro-F1 | Chosen params |
|---|---:|---:|---:|---|
| `majority_full.json` | 0.6263 | 0.1926 | — | most_frequent |
| `tfidf_lr_full.json` | 0.6468 | 0.4693 | 0.5404 | ngram (1, 2), C = 1 |
| `tfidf_lr_claim_only.json` | 0.5380 | 0.3317 | 0.3993 | ngram (1, 2), C = 1 |
| `tfidf_lr_evidence_only.json` | 0.6242 | 0.4462 | 0.5594 | ngram (1, 2), C = 1 |

Macro-F1 intervals are stored on each metrics file as `macro_f1_ci95`. n_train = 2996, n_test = 487, seed = 42.
