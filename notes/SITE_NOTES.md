# Site redesign notes

The public site is one page, `docs/index.html`. `site/build_site.py` reads the result files and injects them as `window.TT_DATA`. The template has the layout and the prose. JavaScript draws every count, score, chart, and table from that blob.

## Choices

- One page instead of seven. The old paths (`results.html`, `data.html`, `methods.html`, `errors.html`, `demo.html`, `team.html`) redirect to the matching section so existing links still land.
- Chart.js 4.4.6 is vendored at `site/vendor/chart.umd.js` and copied to `docs/assets/vendor/`. The MIT notice is copied with it. No CDN. Bootstrap error bars are a small plugin in `app.js`, because the only saved interval is the macro-F1 interval. Accuracy and weighted-F1 have no interval in the metrics files, so they are drawn without whiskers.
- The majority baseline is full-mode only. Claim-only and evidence-only do not show a Pending card for it. A missing TF-IDF or Sentence-BERT file does.
- Histograms are counted at build time from `train.parquet` and `test.parquet` (word-length columns already on the rows) and stored in `TT_DATA`. The page does not use the EDA PNGs.
- Evidence in the claim explorer and the error sample is cut at 600 characters. A shortened claim is flagged in the data, and the drawer says so.
- The claim table pages through the full dev set in the browser. Filters apply to every row, not just the visible page.
- `combined_lr` is not on the site. It is an extra experiment and is not part of the locked comparison.
- The accent is teal. Majority, TF-IDF, and Sentence-BERT use ink, teal, and stone so the series stay distinct. Verdict chips use four separate colors so Supported and Refuted cannot be confused.
- The hero counts macro-F1 to three decimals. Tables use four. Both come from the same JSON.
- Theme follows `prefers-color-scheme` until the visitor picks Light or Dark. The choice is stored as `truthtrace-theme`.
- The page does not pretend to classify a new claim. Live inference stays `make demo`.

## Checks

- `python site/build_site.py` succeeds with the full results and with the Stage 2 metrics, predictions, comparison tables, and error files removed. In that second case the Sentence-BERT, McNemar, bootstrap, verdict, and error sections are pending, and the build still exits 0.
- Generated HTML contains no `{{` or `{%`.
- The loaded page requests only `assets/style.css`, `assets/vendor/chart.umd.js`, and `assets/app.js` from the same host.
- Per-class, claim-only, a confusion cell, a claim drawer, and the disagreement filter all update without a reload. At 375px the page does not scroll sideways.
- Lighthouse accessibility on desktop is 100. Hero model names are paragraphs, so the heading order stays h1 then h2.
