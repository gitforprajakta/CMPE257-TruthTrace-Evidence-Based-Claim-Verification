# Stage 2 notes

Prajakta Jivanrao Ketkar. Stage 1 modules were left as Krati wrote them, except `site/build_site.py` (see below).

## Decisions

- Sentence-BERT model is `sentence-transformers/all-MiniLM-L6-v2`. This machine used the Apple MPS device. The fake encoder is hash-based, 384 dimensions, and its cache files use a `_fake` suffix so a smoke run cannot be reused as a real MiniLM cache.
- Evidence is encoded pair by pair and mean-pooled with `np.add.reduceat`. `build_features` is the only definition of the three input modes, and the Streamlit demo calls it.
- Tuning matches Model 1: `C` in `{0.01, 0.1, 1, 10, 100}`, stratified 5-fold CV, seed 42, macro-F1, `class_weight="balanced"`. Every mode selected `C = 0.01`.
- `train_seconds` is the grid-search fit only. Embedding time is separate and is skipped when the `.npy` cache row count matches.
- The UMAP figure uses cosine distance and seed 42. The code falls back to t-SNE if UMAP fails. This run used UMAP.
- A result is called significant only when McNemar's p-value is below 0.05 and the bootstrap interval for SBERT minus TF-IDF lies entirely above 0 or entirely below 0. Otherwise the verdict says the models are not significantly different.
- Very small p-values are printed in scientific notation. Six fixed decimals would have shown McNemar's p-value as 0.000000.
- Disagreement rows use `who_wins` values `sbert` and `tfidf`. `manual_reason` is blank.
- `data/` and `results/models/` are gitignored. The processed counts match `notes/STAGE1_DONE.md` (train 2996, test 487, source `chenxwh_hf`). The TF-IDF joblib files were refit from the saved `best_params` so Krati's metrics files were not rewritten. Agreement with her saved test predictions is 1.0000 on all three modes.
- Pytest 9 does not put the repository root on `sys.path`, so `pytest -q` could not import `src`. `pytest.ini` sets `pythonpath = .`. No Stage 1 test was edited.

## Site change

`site/build_site.py` `fmt_float` now prints a non-zero number in scientific notation when the requested fixed decimals would display zero. Without that, the results page showed McNemar's p-value as 0.000000. The stored JSON is unchanged (`1.3767704755836865e-07`).

## Test-set scores

| mode | accuracy | macro-F1 | CV macro-F1 | C |
|---|---:|---:|---:|---:|
| full | 0.4949 | 0.3596 | 0.4072 | 0.01 |
| claim_only | 0.4353 | 0.3165 | 0.3337 | 0.01 |
| evidence_only | 0.4517 | 0.3520 | 0.4081 | 0.01 |

Full-mode comparison, from `results/tables/main_comparison.csv`:

- majority macro-F1 0.1926
- TF-IDF + LR macro-F1 0.4693
- SBERT + LR macro-F1 0.3596
- difference (SBERT minus TF-IDF) -0.1095, 95% interval [-0.1796, -0.0392]
- McNemar p = 1.38e-07, 192 discordant pairs, chi-square with continuity correction

SBERT beats the majority baseline and is worse than TF-IDF at alpha = 0.05. That is what the numbers support.

## Exit checklist

1. `sbert_lr_{full,claim_only,evidence_only}` metrics, predictions, joblibs, and sidecar JSON files exist. Embedding caches exist and the second and third modes reused them.
2. `main_comparison`, `per_class_f1`, `ablation`, `mcnemar`, `bootstrap_diff`, and `verdict` exist. The verdict numbers are the table numbers rounded to 4 decimals, and the p-value is the McNemar JSON value.
3. Stage 2 figures exist, including `sbert_umap_claims.png` and `error_reason_counts.png`.
4. Error CSVs have 15 mistakes per model and 10 disagreements in each direction. `manual_reason` is empty.
5. `pytest -q` passes (18 tests). Notebooks 01 and 02 execute without modifying those files. Notebooks 03 and 04 execute and keep their outputs. The Streamlit demo returns a TF-IDF verdict and an SBERT verdict for a loaded example, with the decision-support banner.
6. `docs/` is rebuilt from the results files. No page still says Pending.
