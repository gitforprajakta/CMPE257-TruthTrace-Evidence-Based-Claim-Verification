# Stage 1 notes (Krati)

Decisions and surprises while building Stage 1. Stage 2 files were not created.

## Data

- The first source in the fallback chain worked: `chenxwh/AVeriTeC` via `hf_hub_download` with the default repo type. `source.txt` is `chenxwh_hf`. The pminervini and GitHub fallbacks are implemented and were not needed.
- Official dev is the evaluation split (`test.parquet`) because test labels are not public.
- Train record 1948 has an empty claim and label Supported. It is the one `dropped_invalid` row. Blank labels would be dropped the same way. A non-empty label that is not one of the documented variants raises `ValueError` and lists the strings, so a new label name cannot disappear into the invalid count.
- `answer_type == "Boolean"` is matched case-insensitively. The released file uses `Boolean`, and the boolean explanation is appended as specified.
- After cleaning, every kept train and dev claim has at least one usable QA pair, so `missing_evidence_share` is 0. The no-evidence placeholder is still implemented and covered by `tests/test_preprocess.py`.
- Counts are in `data/processed/preprocess_report.json` (not committed; `data/` is gitignored): train 3068 loaded, 1 invalid, 71 duplicates, 2996 final; dev 500 loaded, 0 invalid, 9 duplicates, 4 overlapping train claims removed, 487 final.

## Modeling

- `GridSearchCV(..., refit=True)` is the fit on all training rows. Test is predicted once. If two n-gram ranges tie on CV macro-F1, the earlier one is kept (unigrams, then unigrams+bigrams).
- All three TF-IDF modes selected `ngram_range` (1, 2) and `C` 1.
- The string ` [SEP] ` is part of `evidence_text`, so TF-IDF can learn the token `sep`. The Not Enough Evidence coefficients are dominated by "no answer could be found", which is the annotation shortcut the EDA phrase figure is there to show.
- Saved TF-IDF pipelines are sklearn objects only. `results/models/*.joblib` is gitignored.

## Site and repo

- `main_comparison.csv` is a Stage 2 file. Until it exists, the homepage cards read `results/metrics/*_full.json`. The sentence "4,568 claims" and "50 fact-checking organizations" on the data page is the paper's description of AVeriTeC, not a number computed by this repo. Computed split counts come from `preprocess_report.json`.
- `.gitignore` also ignores `.pytest_cache/` and `*.pyc`. `.gitattributes` forces LF for the Makefile, Python, and Markdown so `make` does not see CRLF.
- Makefile recipes call one mode per line instead of a POSIX `for` loop, so the file runs under both `cmd.exe` and `sh`.
- GNU make was not installed on the machine used for this stage. The same commands the Makefile runs were executed directly (`python -m ...`, `python site/build_site.py`, `pytest -q`).
