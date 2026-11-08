# TruthTrace

Evidence-based claim verification on the [AVeriTeC](https://arxiv.org/abs/2305.13117) dataset (CMPE 257). Given a claim and human-written question-answer evidence, the system predicts one of four verdicts: Supported, Refuted, Not Enough Evidence, or Conflicting Evidence/Cherrypicking.

**Research question.** Can a semantic model (Sentence-BERT + logistic regression) beat a traditional model (TF-IDF + logistic regression) for claim verification, given identical data, splits, tuning budget, and metrics?

This is a **decision-support** prototype. It is not a source of truth and it does not detect real-world truth on its own.

## Dataset and license

AVeriTeC contains claims from fact-checking organizations, with questions and answers written by annotators. The official test labels are not public, so this project trains on the official train split and evaluates on the official dev split.

AVeriTeC is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Use it for non-commercial research and teaching only.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`make setup` runs the same install.

## Reproduce

From the repository root:

```bash
make data       # download AVeriTeC and preprocess it
make eda        # figures and results/tables/eda_summary.csv
make majority   # majority-class baseline
make tfidf      # TF-IDF + logistic regression, three input modes
make readme     # refresh the results block below
make site       # static site in docs/
make test       # unit tests for the finished stage
```

`make all` also calls the Stage 2 targets (`sbert`, `compare`, `figures`, `errors`). Those commands fail until Stage 2 adds the modules. That is expected during Stage 1.

Every training script is idempotent. Pass `--force` to recompute an existing output. The random seed is 42.

| Result | Command |
|---|---|
| Processed train/test and preprocess report | `python -m src.data_loading` then `python -m src.preprocess` |
| EDA figures and `eda_summary.csv` | `python -m src.eda` |
| Majority baseline metrics and predictions | `python -m src.train_majority` |
| TF-IDF + LR for `full`, `claim_only`, `evidence_only` | `python -m src.train_tfidf --mode full` (and the other modes) |
| Sentence-BERT + LR, comparison, error analysis, demo | Stage 2 (`make sbert compare figures errors demo`) |

Raw JSON and parquet files stay in `data/` and are not committed. Fitted models (`*.joblib`) stay in `results/models/` and are not committed. Metrics, predictions, tables, figures, and the generated site are committed.

## Repository layout

```
src/            shared config, data, EDA, TF-IDF (Stage 1) and SBERT/comparison (Stage 2)
site/           Jinja templates and the static-site generator
docs/           generated site (GitHub Pages)
notebooks/      01-02 Stage 1, 03-04 Stage 2
tests/          stage-specific unit tests
results/        metrics, predictions, figures, tables, error analysis
notes/          stage notes
app/            Streamlit demo (Stage 2)
```

## Who does what

| Stage | Author | Work |
|---|---|---|
| Stage 1 | Krati Jain | Repository foundation, data download, preprocessing, EDA, majority baseline, TF-IDF + logistic regression, tests, notebooks 01-02, Makefile, README, website generator |
| Stage 2 | Prajakta Jivanrao Ketkar | Sentence-BERT + logistic regression, comparison and statistics, comparison figures, error analysis, Streamlit demo, tests, notebooks 03-04, final site refresh |

Stage 2 reads Stage 1 outputs and adds new files. The site generator already leaves a visible pending box wherever a Stage 2 file is not on disk yet.

## Results

<!-- RESULTS_START -->
Numbers below are read from `results/tables/main_comparison.csv`.

| model | accuracy | macro_precision | macro_recall | macro_f1 | macro_f1_ci_low | macro_f1_ci_high | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority | 0.6263 | 0.1566 | 0.2500 | 0.1926 | 0.1838 | 0.2001 | 0.4824 |
| tfidf_lr | 0.6468 | 0.5721 | 0.4365 | 0.4693 | 0.4041 | 0.5333 | 0.6197 |
| sbert_lr | 0.4949 | 0.3530 | 0.3862 | 0.3596 | 0.3115 | 0.4098 | 0.5157 |

The majority baseline reaches a macro-F1 of 0.1926. TF-IDF + logistic regression reaches a macro-F1 of 0.4693. Sentence-BERT + logistic regression reaches a macro-F1 of 0.3596. The macro-F1 difference (SBERT minus TF-IDF) is -0.1095 with a 95% bootstrap interval of [-0.1796, -0.0392]. McNemar's test on paired correctness gives p = 1.38e-07. SBERT is worse than TF-IDF at alpha = 0.05.
<!-- RESULTS_END -->

## Limitations

- Some Not Enough Evidence annotations use stock phrases such as "no answer could be found". A model can latch onto that shortcut instead of weighing the claim.
- The minority verdicts, especially Conflicting Evidence/Cherrypicking, have less support than Supported and Refuted.
- Evidence is human-annotated question-answer text, not documents retrieved by the system. Results do not measure open-web retrieval.
- Nothing here is a claim of real-world truth detection. The dev split is a research benchmark.

## Ethics

Use the outputs as decision support: a second view over a claim and its evidence, with the benchmark label set and the model's confidence shown beside it. Do not treat a verdict as a fact, and do not use the dataset or models for commercial fact-checking. The non-commercial AVeriTeC license applies.

## Team

- Krati Jain
- Prajakta Jivanrao Ketkar

Course: CMPE 257.

## GitHub Pages

The static site is written to `docs/`.

1. Push the `main` branch.
2. On GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select branch `main` and folder `/docs`.
5. Save. The site is served from that folder. `docs/.nojekyll` is included so Pages does not rewrite the files.

Opening `docs/index.html` locally also works. Asset links are relative.

## References

- Schlichtkrull, Guo, Vlachos. *AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web.* arXiv:2305.13117.
- Reimers and Gurevych. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* 2019.
- Thorne, Vlachos, Christodoulopoulos, Mittal. *FEVER: a Large-scale Dataset for Fact Extraction and VERification.* 2018.

## Related repositories

- https://github.com/MichSchli/AVeriTeC
- https://huggingface.co/chenxwh/AVeriTeC
- https://huggingface.co/datasets/pminervini/averitec
- https://github.com/UKPLab/sentence-transformers
- https://fever.ai/
