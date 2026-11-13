(function () {
  "use strict";

  const data = window.TT_DATA || {};
  const labels = data.labels || [];
  const shortLabels = data.short_labels || {};
  const slugs = data.label_slugs || {};
  const names = data.model_names || {};
  const modelOrder = data.model_order || [];
  const predictionModels = data.prediction_models || [];
  const files = data.files || {};
  const charts = {};
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const PAGE = 20;
  const MODE_ORDER = ["claim_only", "evidence_only", "full"];
  const MODE_LABEL = {
    claim_only: "Claim only",
    evidence_only: "Evidence only",
    full: "Full",
  };

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function formatNumber(value, digits) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    const text = number.toFixed(digits);
    if (Number(text) === 0 && number !== 0) return number.toExponential(2);
    return text;
  }

  function formatP(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number !== 0 && Math.abs(number) < 0.0001) return number.toExponential(2);
    return number.toFixed(4);
  }

  function short(label) {
    return shortLabels[label] || label || "—";
  }

  function chip(label) {
    const span = document.createElement("span");
    span.className = "chip " + (slugs[label] || "other");
    span.textContent = short(label);
    return span;
  }

  function pendingCard(file) {
    const card = document.createElement("article");
    card.className = "pending";
    card.setAttribute("role", "status");
    const title = document.createElement("h3");
    title.textContent = "Pending";
    const copy = document.createElement("p");
    copy.textContent = "Pending: waiting for " + file;
    card.append(title, copy);
    return card;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function metricsFile(model, mode) {
    return "results/metrics/" + model + "_" + mode + ".json";
  }

  function fullMetrics(model) {
    const bucket = data.metrics && data.metrics[model];
    return bucket ? bucket.full : null;
  }

  function seriesColor(model) {
    if (model === "tfidf_lr") return cssVar("--series-tfidf");
    if (model === "sbert_lr") return cssVar("--series-sbert");
    return cssVar("--series-majority");
  }

  function chartDefaults() {
    if (!window.Chart) return;
    const ink = cssVar("--ink");
    const muted = cssVar("--muted");
    const line = cssVar("--line");
    Chart.defaults.font.family = cssVar("--sans");
    Chart.defaults.color = muted;
    Chart.defaults.borderColor = line;
    Chart.defaults.plugins.legend.labels.color = ink;
    Chart.defaults.plugins.tooltip.backgroundColor = cssVar("--card");
    Chart.defaults.plugins.tooltip.titleColor = ink;
    Chart.defaults.plugins.tooltip.bodyColor = ink;
    Chart.defaults.plugins.tooltip.borderColor = line;
    Chart.defaults.plugins.tooltip.borderWidth = 1;
  }

  const ciPlugin = {
    id: "macroCi",
    afterDatasetsDraw(chart) {
      const spec = chart.$macroCi;
      if (!spec) return;
      const meta = chart.getDatasetMeta(spec.datasetIndex);
      if (!meta || meta.hidden) return;
      const scale = chart.scales.y;
      const context = chart.ctx;
      context.save();
      context.strokeStyle = cssVar("--ink");
      context.lineWidth = 1.5;
      spec.intervals.forEach(function (interval, index) {
        const bar = meta.data[index];
        if (!interval || !bar) return;
        const x = bar.x;
        const top = scale.getPixelForValue(interval[1]);
        const bottom = scale.getPixelForValue(interval[0]);
        context.beginPath();
        context.moveTo(x, top);
        context.lineTo(x, bottom);
        context.moveTo(x - 5, top);
        context.lineTo(x + 5, top);
        context.moveTo(x - 5, bottom);
        context.lineTo(x + 5, bottom);
        context.stroke();
      });
      context.restore();
    },
  };

  if (window.Chart) Chart.register(ciPlugin);

  function mountChart(key, host, config, label) {
    clear(host);
    const box = document.createElement("div");
    box.className = "chart-box";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", label);
    box.appendChild(canvas);
    host.appendChild(box);
    if (!window.Chart) {
      host.appendChild(pendingCard("docs/assets/vendor/chart.umd.js"));
      return null;
    }
    chartDefaults();
    if (charts[key]) charts[key].destroy();
    charts[key] = new Chart(canvas, config);
    return charts[key];
  }

  function dataTable(caption, columns, rows) {
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    const cap = document.createElement("caption");
    cap.textContent = caption;
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach(function (column) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = column;
      headRow.appendChild(cell);
    });
    head.appendChild(headRow);
    const body = document.createElement("tbody");
    rows.forEach(function (row) {
      const tr = document.createElement("tr");
      row.forEach(function (value, index) {
        const cell = document.createElement(index === 0 ? "th" : "td");
        if (index === 0) cell.scope = "row";
        if (value instanceof Node) cell.appendChild(value);
        else cell.textContent = value;
        tr.appendChild(cell);
      });
      body.appendChild(tr);
    });
    table.append(cap, head, body);
    wrap.appendChild(table);
    return wrap;
  }

  function baseScales() {
    return {
      y: {
        beginAtZero: true,
        suggestedMax: 1,
        ticks: { color: cssVar("--muted") },
        grid: { color: cssVar("--line") },
      },
      x: {
        ticks: { color: cssVar("--muted") },
        grid: { display: false },
      },
    };
  }

  function fillCopy() {
    const copy = data.copy || {};
    const kicker = document.getElementById("hero-kicker");
    const title = document.getElementById("hero-title");
    const question = document.getElementById("hero-question");
    const license = document.getElementById("footer-license");
    const course = document.getElementById("course-line");
    if (kicker) kicker.textContent = copy.kicker || "";
    if (title) title.textContent = copy.title || "";
    if (question) question.textContent = copy.question || "";
    if (license) license.textContent = copy.license || "";
    if (course) course.textContent = copy.kicker || "";
    const list = document.getElementById("references");
    (data.references || []).forEach(function (item) {
      const li = document.createElement("li");
      const link = document.createElement("a");
      link.href = item.url;
      link.rel = "noopener";
      link.textContent = item.label;
      li.appendChild(link);
      list.appendChild(li);
    });
  }

  function countUp(shown, value, digits) {
    const finalText = formatNumber(value, digits);
    if (reduceMotion) {
      shown.textContent = finalText;
      return;
    }
    const start = performance.now();
    function frame(now) {
      const t = Math.min(1, (now - start) / 900);
      const eased = 1 - Math.pow(1 - t, 3);
      shown.textContent = formatNumber(value * eased, digits);
      if (t < 1) requestAnimationFrame(frame);
      else shown.textContent = finalText;
    }
    requestAnimationFrame(frame);
  }

  function renderHero() {
    const host = document.getElementById("hero-metrics");
    clear(host);
    modelOrder.forEach(function (model) {
      const metrics = fullMetrics(model);
      if (!metrics) {
        host.appendChild(pendingCard(metricsFile(model, "full")));
        return;
      }
      const card = document.createElement("article");
      card.className = "card";
      const name = document.createElement("p");
      name.className = "metric-name";
      name.textContent = names[model] || model;
      const label = document.createElement("p");
      label.className = "metric-label";
      label.textContent = "Macro-F1";
      const value = document.createElement("p");
      value.className = "metric-value";
      const shown = document.createElement("span");
      shown.setAttribute("aria-hidden", "true");
      shown.textContent = formatNumber(0, 3);
      const sr = document.createElement("span");
      sr.className = "sr-only";
      sr.textContent = formatNumber(metrics.macro_f1, 3);
      value.append(shown, sr);
      const accuracy = document.createElement("p");
      accuracy.className = "metric-label";
      accuracy.textContent = "Accuracy " + formatNumber(metrics.accuracy, 3);
      card.append(name, label, value, accuracy);
      host.appendChild(card);
      countUp(shown, metrics.macro_f1, 3);
    });
  }

  let comparisonView = "overall";

  function renderComparison() {
    const host = document.getElementById("comparison-mount");
    const summary = document.getElementById("comparison-summary");
    clear(host);
    const available = modelOrder.filter(function (model) { return fullMetrics(model); });
    modelOrder.forEach(function (model) {
      if (!fullMetrics(model)) host.appendChild(pendingCard(metricsFile(model, "full")));
    });
    if (!available.length) {
      summary.textContent = "Comparison scores are not available yet.";
      return;
    }
    if (comparisonView === "overall") renderOverall(host, summary, available);
    else renderPerClass(host, summary, available);
  }

  function renderOverall(host, summary, available) {
    const metricNames = ["Accuracy", "Macro-F1", "Weighted-F1"];
    const keys = ["accuracy", "macro_f1", "weighted_f1"];
    const datasets = keys.map(function (key, index) {
      return {
        label: metricNames[index],
        data: available.map(function (model) { return fullMetrics(model)[key]; }),
        backgroundColor: index === 1 ? cssVar("--series-tfidf") : index === 0 ? cssVar("--series-majority") : cssVar("--series-sbert"),
        borderWidth: 0,
      };
    });
    const intervals = available.map(function (model) {
      const ci = fullMetrics(model).macro_f1_ci95;
      return ci && ci.length === 2 ? ci : null;
    });
    const config = {
      type: "bar",
      data: { labels: available.map(function (model) { return names[model] || model; }), datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: baseScales(),
        plugins: { legend: { position: "bottom" } },
      },
    };
    const chart = mountChart("comparison", host, config, "Grouped bar chart of accuracy, macro-F1, and weighted-F1");
    if (chart) chart.$macroCi = { datasetIndex: 1, intervals: intervals };
    const rows = available.map(function (model) {
      const metrics = fullMetrics(model);
      const ci = metrics.macro_f1_ci95 || [];
      return [
        names[model] || model,
        formatNumber(metrics.accuracy, 4),
        formatNumber(metrics.macro_f1, 4),
        "[" + formatNumber(ci[0], 4) + ", " + formatNumber(ci[1], 4) + "]",
        formatNumber(metrics.weighted_f1, 4),
      ];
    });
    host.appendChild(dataTable(
      "Overall scores. The interval is the bootstrap range for macro-F1.",
      ["Model", "Accuracy", "Macro-F1", "Macro-F1 interval", "Weighted-F1"],
      rows
    ));
    summary.textContent = available.map(function (model) {
      return (names[model] || model) + " macro-F1 " + formatNumber(fullMetrics(model).macro_f1, 4);
    }).join(". ");
  }

  function renderPerClass(host, summary, available) {
    const datasets = available.map(function (model) {
      const perClass = fullMetrics(model).per_class || {};
      return {
        label: names[model] || model,
        data: labels.map(function (label) { return perClass[label] ? perClass[label].f1 : null; }),
        backgroundColor: seriesColor(model),
        borderWidth: 0,
      };
    });
    mountChart("comparison", host, {
      type: "bar",
      data: { labels: labels.map(short), datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: baseScales(),
        plugins: { legend: { position: "bottom" } },
      },
    }, "Grouped bar chart of per-class F1");
    const rows = available.map(function (model) {
      const perClass = fullMetrics(model).per_class || {};
      return [names[model] || model].concat(labels.map(function (label) {
        return perClass[label] ? formatNumber(perClass[label].f1, 4) : "—";
      }));
    });
    host.appendChild(dataTable(
      "Per-class F1.",
      ["Model"].concat(labels.map(short)),
      rows
    ));
    summary.textContent = "Per-class F1 is showing for " + available.map(function (model) { return names[model] || model; }).join(", ");
  }

  let ablationMode = "full";

  function renderAblation() {
    const tabs = document.getElementById("mode-tabs");
    const host = document.getElementById("ablation-mount");
    if (!tabs.childElementCount) {
      MODE_ORDER.forEach(function (mode) {
        const button = document.createElement("button");
        button.type = "button";
        button.role = "tab";
        button.textContent = MODE_LABEL[mode] || mode;
        button.setAttribute("aria-selected", mode === ablationMode ? "true" : "false");
        button.addEventListener("click", function () {
          ablationMode = mode;
          renderAblation();
        });
        tabs.appendChild(button);
      });
    }
    Array.from(tabs.querySelectorAll("button")).forEach(function (button) {
      button.setAttribute("aria-selected", button.textContent === MODE_LABEL[ablationMode] ? "true" : "false");
    });
    clear(host);
    const models = ablationMode === "full" ? modelOrder : predictionModels;
    const available = [];
    models.forEach(function (model) {
      const bucket = data.metrics && data.metrics[model];
      const metrics = bucket ? bucket[ablationMode] : null;
      if (!metrics) host.appendChild(pendingCard(metricsFile(model, ablationMode)));
      else available.push(model);
    });
    if (!available.length) return;
    mountChart("ablation", host, {
      type: "bar",
      data: {
        labels: available.map(function (model) { return names[model] || model; }),
        datasets: [
          {
            label: "Accuracy",
            data: available.map(function (model) { return data.metrics[model][ablationMode].accuracy; }),
            backgroundColor: cssVar("--series-majority"),
            borderWidth: 0,
          },
          {
            label: "Macro-F1",
            data: available.map(function (model) { return data.metrics[model][ablationMode].macro_f1; }),
            backgroundColor: cssVar("--series-tfidf"),
            borderWidth: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: baseScales(),
        plugins: { legend: { position: "bottom" } },
      },
    }, "Accuracy and macro-F1 for the selected input mode");
    const rows = available.map(function (model) {
      const metrics = data.metrics[model][ablationMode];
      const params = metrics.best_params || {};
      const ngram = params.ngram_range ? params.ngram_range.join("–") : "—";
      const chosenC = params.C === undefined || params.C === null ? "—" : formatNumber(params.C, 2);
      return [
        names[model] || model,
        formatNumber(metrics.accuracy, 4),
        formatNumber(metrics.macro_f1, 4),
        formatNumber(metrics.weighted_f1, 4),
        chosenC,
        ngram,
      ];
    });
    host.appendChild(dataTable(
      MODE_LABEL[ablationMode] + " input.",
      ["Model", "Accuracy", "Macro-F1", "Weighted-F1", "Chosen C", "N-grams"],
      rows
    ));
  }

  let confusionModel = "tfidf_lr";
  let confusionNormalized = false;

  function renderConfusion() {
    const select = document.getElementById("cm-model");
    const host = document.getElementById("confusion-mount");
    const detail = document.getElementById("cm-detail");
    if (!select.childElementCount) {
      predictionModels.forEach(function (model) {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = names[model] || model;
        select.appendChild(option);
      });
      select.addEventListener("change", function () {
        confusionModel = select.value;
        renderConfusion();
      });
    }
    select.value = confusionModel;
    clear(host);
    detail.textContent = "Choose a cell to read its count and its share of that true label.";
    const metrics = fullMetrics(confusionModel);
    if (!metrics || !metrics.confusion_matrix) {
      host.appendChild(pendingCard(metricsFile(confusionModel, "full")));
      return;
    }
    const raw = metrics.confusion_matrix;
    const share = metrics.confusion_matrix_normalized;
    const grid = document.createElement("div");
    grid.className = "matrix";
    grid.style.gridTemplateColumns = "7.5rem repeat(" + labels.length + ", minmax(0, 1fr))";
    grid.appendChild(document.createElement("span"));
    labels.forEach(function (label) {
      const head = document.createElement("div");
      head.className = "axis";
      head.textContent = short(label);
      grid.appendChild(head);
    });
    const displayed = confusionNormalized ? share : raw;
    let maxValue = 0;
    displayed.forEach(function (row) {
      row.forEach(function (value) { maxValue = Math.max(maxValue, Number(value) || 0); });
    });
    raw.forEach(function (row, rowIndex) {
      const rowLabel = document.createElement("div");
      rowLabel.className = "axis";
      rowLabel.textContent = short(labels[rowIndex]);
      grid.appendChild(rowLabel);
      row.forEach(function (_, colIndex) {
        const count = Number(raw[rowIndex][colIndex]);
        const portion = Number(share[rowIndex][colIndex]);
        const shown = confusionNormalized ? portion : count;
        const intensity = maxValue > 0 ? (confusionNormalized ? portion : count / maxValue) : 0;
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.row = String(rowIndex);
        button.dataset.col = String(colIndex);
        button.textContent = confusionNormalized ? formatNumber(shown, 3) : String(count);
        button.style.background = "color-mix(in srgb, " + cssVar("--accent") + " " + Math.round(intensity * 100) + "%, " + cssVar("--card") + ")";
        button.style.color = intensity > 0.62 ? cssVar("--on-accent") : cssVar("--ink");
        const message = count + " predictions. " + formatNumber(portion * 100, 1) + "% of true " + short(labels[rowIndex]) + ".";
        button.setAttribute("aria-label", "True " + short(labels[rowIndex]) + ", predicted " + short(labels[colIndex]) + ". " + message);
        function describe() { detail.textContent = "True " + short(labels[rowIndex]) + ", predicted " + short(labels[colIndex]) + ". " + message; }
        button.addEventListener("mouseenter", describe);
        button.addEventListener("focus", describe);
        button.addEventListener("keydown", function (event) {
          const step = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[event.key];
          if (!step) return;
          const next = grid.querySelector('[data-row="' + (rowIndex + step[0]) + '"][data-col="' + (colIndex + step[1]) + '"]');
          if (next) {
            event.preventDefault();
            next.focus();
          }
        });
        grid.appendChild(button);
      });
    });
    host.appendChild(grid);
  }

  function renderStatistics() {
    const host = document.getElementById("statistics-mount");
    clear(host);
    host.appendChild(renderMcnemar());
    host.appendChild(renderBootstrap());
    const verdict = document.createElement("article");
    verdict.className = "stat-card";
    const heading = document.createElement("h3");
    heading.textContent = "Verdict";
    verdict.appendChild(heading);
    if (!data.verdict) verdict.appendChild(pendingCard(files.verdict || "results/tables/verdict.md"));
    else {
      const quote = document.createElement("p");
      quote.textContent = data.verdict;
      verdict.appendChild(quote);
    }
    verdict.style.gridColumn = "1 / -1";
    host.appendChild(verdict);
  }

  function renderMcnemar() {
    const card = document.createElement("article");
    card.className = "stat-card";
    const heading = document.createElement("h3");
    heading.textContent = "McNemar";
    card.appendChild(heading);
    const table = data.mcnemar;
    if (!table || !table.table) {
      card.appendChild(pendingCard(files.mcnemar || "results/tables/mcnemar.json"));
      return card;
    }
    const nameA = names[table.model_a] || table.model_a;
    const nameB = names[table.model_b] || table.model_b;
    const captions = [
      ["Both correct", "Only " + nameA],
      ["Only " + nameB, "Both wrong"],
    ];
    const grid = document.createElement("div");
    grid.className = "stat-grid";
    table.table.forEach(function (row, rowIndex) {
      row.forEach(function (value, colIndex) {
        const cell = document.createElement("div");
        cell.className = "stat-cell";
        const strong = document.createElement("strong");
        strong.textContent = String(value);
        const note = document.createElement("span");
        note.textContent = captions[rowIndex][colIndex];
        cell.append(strong, note);
        grid.appendChild(cell);
      });
    });
    const meta = document.createElement("p");
    meta.textContent = "Discordant pairs " + table.n_discordant + ". " + table.test + " test, p = " + formatP(table.p_value) + ".";
    card.append(grid, meta);
    return card;
  }

  function renderBootstrap() {
    const card = document.createElement("article");
    card.className = "stat-card";
    const heading = document.createElement("h3");
    heading.textContent = "Bootstrap difference";
    card.appendChild(heading);
    const boot = data.bootstrap;
    if (!boot || !boot.ci95) {
      card.appendChild(pendingCard(files.bootstrap || "results/tables/bootstrap_diff.json"));
      return card;
    }
    const mean = Number(boot.diff_mean);
    const lo = Number(boot.ci95[0]);
    const hi = Number(boot.ci95[1]);
    const span = Math.max(Math.abs(lo), Math.abs(hi), Math.abs(mean), 0.05);
    const min = -span * 1.2;
    const max = span * 1.2;
    function place(value) {
      return ((value - min) / (max - min)) * 100;
    }
    const range = document.createElement("div");
    range.className = "range";
    range.setAttribute("role", "img");
    range.setAttribute("aria-label", "Bootstrap interval for the macro-F1 difference, from " + formatNumber(lo, 4) + " to " + formatNumber(hi, 4));
    const track = document.createElement("div");
    track.className = "range-track";
    const interval = document.createElement("div");
    interval.className = "range-interval";
    interval.style.left = place(lo) + "%";
    interval.style.width = Math.max(place(hi) - place(lo), 0) + "%";
    const zero = document.createElement("div");
    zero.className = "range-zero";
    zero.style.left = place(0) + "%";
    const marker = document.createElement("div");
    marker.className = "range-mean";
    marker.style.left = place(mean) + "%";
    range.append(track, interval, zero, marker);
    const ticks = document.createElement("div");
    ticks.className = "range-labels";
    [min, 0, max].forEach(function (tick) {
      const spanTick = document.createElement("span");
      spanTick.textContent = formatNumber(tick, 3);
      ticks.appendChild(spanTick);
    });
    const meta = document.createElement("p");
    meta.textContent = "Mean difference " + formatNumber(mean, 4) + " on " + (boot.metric || "macro-F1") + ". Interval [" + formatNumber(lo, 4) + ", " + formatNumber(hi, 4) + "]. Share of resamples with SBERT ahead: " + formatNumber(boot.p_sbert_better, 3) + " of " + boot.n_resamples + ".";
    card.append(range, ticks, meta);
    return card;
  }

  const claimState = { page: 0, query: "", label: "", outcome: "", disagree: false, lastFocus: null };

  function claimModels(claim) {
    return claim.models || {};
  }

  function filteredClaims() {
    return (data.claims || []).filter(function (claim) {
      const query = claimState.query.trim().toLowerCase();
      if (query) {
        const haystack = (claim.claim + " " + claim.id).toLowerCase();
        if (!haystack.includes(query)) return false;
      }
      if (claimState.label && claim.true !== claimState.label) return false;
      const tfidf = claimModels(claim).tfidf_lr;
      const sbert = claimModels(claim).sbert_lr;
      if (claimState.outcome === "tfidf-right" && !(tfidf && tfidf.correct)) return false;
      if (claimState.outcome === "tfidf-wrong" && !(tfidf && tfidf.correct === false)) return false;
      if (claimState.outcome === "sbert-right" && !(sbert && sbert.correct)) return false;
      if (claimState.outcome === "sbert-wrong" && !(sbert && sbert.correct === false)) return false;
      if (claimState.disagree) {
        if (!tfidf || !sbert || tfidf.pred === sbert.pred) return false;
      }
      return true;
    });
  }

  function renderClaims() {
    const host = document.getElementById("claims-mount");
    clear(host);
    if (!data.claims) {
      host.appendChild(pendingCard(files.claims || "data/processed/test.parquet"));
      return;
    }
    const sbertReady = data.claims.some(function (claim) { return claim.models && claim.models.sbert_lr; });
    const tfidfReady = data.claims.some(function (claim) { return claim.models && claim.models.tfidf_lr; });
    if (!sbertReady) host.appendChild(pendingCard("results/predictions/sbert_lr_full.csv"));
    if (!tfidfReady) host.appendChild(pendingCard("results/predictions/tfidf_lr_full.csv"));

    const filters = document.createElement("div");
    filters.className = "filters";
    const searchLabel = document.createElement("label");
    searchLabel.textContent = "Search";
    const search = document.createElement("input");
    search.type = "search";
    search.value = claimState.query;
    search.setAttribute("aria-label", "Search claims");
    search.addEventListener("input", function () {
      claimState.query = search.value;
      claimState.page = 0;
      renderClaimTable();
    });
    searchLabel.appendChild(search);

    const labelLabel = document.createElement("label");
    labelLabel.textContent = "True label";
    const labelSelect = document.createElement("select");
    labelSelect.setAttribute("aria-label", "Filter by true label");
    const anyLabel = document.createElement("option");
    anyLabel.value = "";
    anyLabel.textContent = "Any";
    labelSelect.appendChild(anyLabel);
    labels.forEach(function (label) {
      const option = document.createElement("option");
      option.value = label;
      option.textContent = short(label);
      labelSelect.appendChild(option);
    });
    labelSelect.value = claimState.label;
    labelSelect.addEventListener("change", function () {
      claimState.label = labelSelect.value;
      claimState.page = 0;
      renderClaimTable();
    });
    labelLabel.appendChild(labelSelect);

    const outcomeLabel = document.createElement("label");
    outcomeLabel.textContent = "Correctness";
    const outcome = document.createElement("select");
    outcome.setAttribute("aria-label", "Filter by correctness");
    [["", "Any"], ["tfidf-right", "TF-IDF correct"], ["tfidf-wrong", "TF-IDF incorrect"]].concat(
      sbertReady ? [["sbert-right", "SBERT correct"], ["sbert-wrong", "SBERT incorrect"]] : []
    ).forEach(function (pair) {
      const option = document.createElement("option");
      option.value = pair[0];
      option.textContent = pair[1];
      outcome.appendChild(option);
    });
    outcome.value = claimState.outcome;
    outcome.addEventListener("change", function () {
      claimState.outcome = outcome.value;
      claimState.page = 0;
      renderClaimTable();
    });
    outcomeLabel.appendChild(outcome);

    const disagreeLabel = document.createElement("label");
    const disagree = document.createElement("input");
    disagree.type = "checkbox";
    disagree.checked = claimState.disagree;
    disagree.disabled = !sbertReady || !tfidfReady;
    disagree.addEventListener("change", function () {
      claimState.disagree = disagree.checked;
      claimState.page = 0;
      renderClaimTable();
    });
    disagreeLabel.append(disagree, document.createTextNode(" Models disagree"));

    filters.append(searchLabel, labelLabel, outcomeLabel, disagreeLabel);
    const status = document.createElement("p");
    status.id = "claim-status";
    status.setAttribute("aria-live", "polite");
    const tableHost = document.createElement("div");
    tableHost.id = "claim-table-host";
    const pager = document.createElement("div");
    pager.className = "pager";
    pager.id = "claim-pager";
    host.append(filters, status, tableHost, pager);
    renderClaimTable();
  }

  function renderClaimTable() {
    const tableHost = document.getElementById("claim-table-host");
    const status = document.getElementById("claim-status");
    const pager = document.getElementById("claim-pager");
    if (!tableHost) return;
    const rows = filteredClaims();
    const pages = Math.max(1, Math.ceil(rows.length / PAGE));
    if (claimState.page >= pages) claimState.page = pages - 1;
    const start = claimState.page * PAGE;
    const slice = rows.slice(start, start + PAGE);
    clear(tableHost);
    clear(pager);
    if (!rows.length) {
      status.textContent = "No claims match these filters.";
      return;
    }
    const end = start + slice.length;
    status.textContent = "Showing " + (start + 1) + "–" + end + " of " + rows.length + ".";
    const bodyRows = slice.map(function (claim) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "claim-button";
      button.textContent = claim.claim;
      button.addEventListener("click", function () { openDrawer(claim, button); });
      const tfidf = claimModels(claim).tfidf_lr;
      const sbert = claimModels(claim).sbert_lr;
      return [
        button,
        chip(claim.true),
        tfidf ? chip(tfidf.pred) : document.createTextNode("Pending"),
        sbert ? chip(sbert.pred) : document.createTextNode("Pending"),
      ];
    });
    tableHost.appendChild(dataTable("Dev claims and saved predictions.", ["Claim", "True", "TF-IDF", "SBERT"], bodyRows));
    const prev = document.createElement("button");
    prev.type = "button";
    prev.textContent = "Previous";
    prev.disabled = claimState.page === 0;
    prev.addEventListener("click", function () {
      claimState.page -= 1;
      renderClaimTable();
    });
    const next = document.createElement("button");
    next.type = "button";
    next.textContent = "Next";
    next.disabled = end >= rows.length;
    next.addEventListener("click", function () {
      claimState.page += 1;
      renderClaimTable();
    });
    pager.append(prev, next);
  }

  function probabilityBars(modelResult) {
    const wrap = document.createElement("div");
    if (!modelResult || !modelResult.proba) {
      const note = document.createElement("p");
      note.textContent = "Probabilities were not saved for this model.";
      wrap.appendChild(note);
      return wrap;
    }
    modelResult.proba.forEach(function (probability, index) {
      const row = document.createElement("div");
      const predicted = labels[index] === modelResult.pred;
      row.className = predicted ? "prob is-pred" : "prob";
      const name = document.createElement("span");
      name.textContent = short(labels[index]);
      const track = document.createElement("span");
      track.className = "prob-track";
      const fill = document.createElement("span");
      fill.className = "prob-fill";
      const width = Math.max(0, Math.min(100, probability * 100));
      fill.style.width = width + "%";
      track.appendChild(fill);
      track.setAttribute("role", "meter");
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "1");
      track.setAttribute("aria-valuenow", String(probability));
      track.setAttribute("aria-label", short(labels[index]));
      const value = document.createElement("span");
      value.textContent = formatNumber(probability, 3);
      row.append(name, track, value);
      wrap.appendChild(row);
    });
    return wrap;
  }

  function openDrawer(claim, opener) {
    const drawer = document.getElementById("drawer");
    const title = document.getElementById("drawer-title");
    const body = document.getElementById("drawer-body");
    claimState.lastFocus = opener;
    title.textContent = claim.claim;
    clear(body);
    const truth = document.createElement("p");
    truth.append(document.createTextNode("True label "), chip(claim.true));
    body.appendChild(truth);
    predictionModels.forEach(function (model) {
      const block = document.createElement("section");
      const heading = document.createElement("h4");
      heading.textContent = names[model] || model;
      const result = claimModels(claim)[model];
      block.appendChild(heading);
      if (!result) {
        block.appendChild(pendingCard("results/predictions/" + model + "_full.csv"));
      } else {
        const predicted = document.createElement("p");
        predicted.append(document.createTextNode("Predicted "), chip(result.pred));
        block.append(predicted, probabilityBars(result));
      }
      body.appendChild(block);
    });
    const evidenceTitle = document.createElement("h4");
    evidenceTitle.textContent = "Evidence";
    body.appendChild(evidenceTitle);
    const parts = (claim.evidence || "").split(" [SEP] ").filter(Boolean);
    if (!parts.length) {
      const empty = document.createElement("p");
      empty.textContent = "No evidence text was stored for this claim.";
      body.appendChild(empty);
    }
    parts.forEach(function (part) {
      const paragraph = document.createElement("p");
      paragraph.className = "pair";
      paragraph.textContent = part;
      body.appendChild(paragraph);
    });
    if (claim.evidence_shortened) {
      const note = document.createElement("p");
      note.className = "detail";
      note.textContent = "Evidence text is shortened.";
      body.appendChild(note);
    }
    drawer.hidden = false;
    document.body.classList.add("drawer-open");
    document.getElementById("drawer-close").focus();
  }

  function closeDrawer() {
    const drawer = document.getElementById("drawer");
    drawer.hidden = true;
    document.body.classList.remove("drawer-open");
    if (claimState.lastFocus) claimState.lastFocus.focus();
  }

  const errorState = { model: "", reason: "" };

  function errorRows() {
    const rows = [];
    predictionModels.forEach(function (model) {
      const sample = data.errors && data.errors[model];
      if (!sample) return;
      sample.forEach(function (row) { rows.push(row); });
    });
    return rows;
  }

  function reasonLabel(reason) {
    return String(reason || "unknown").replaceAll("_", " ");
  }

  function renderErrors() {
    const host = document.getElementById("errors-mount");
    clear(host);
    predictionModels.forEach(function (model) {
      const sample = data.errors && data.errors[model];
      if (!sample) host.appendChild(pendingCard((files.errors && files.errors[model]) || model));
    });
    const rows = errorRows();
    if (!rows.length) return;
    const reasons = Array.from(new Set(rows.map(function (row) { return row.reason; }))).sort();
    const presentModels = predictionModels.filter(function (model) { return data.errors && data.errors[model]; });
    mountChart("errors", host, {
      type: "bar",
      data: {
        labels: reasons.map(reasonLabel),
        datasets: presentModels.map(function (model) {
          const sample = data.errors[model];
          return {
            label: names[model] || model,
            data: reasons.map(function (reason) {
              return sample.filter(function (row) { return row.reason === reason; }).length;
            }),
            backgroundColor: seriesColor(model),
            borderWidth: 0,
          };
        }),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0, color: cssVar("--muted") }, grid: { color: cssVar("--line") } },
          x: { ticks: { color: cssVar("--muted") }, grid: { display: false } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    }, "Bar chart of automatic error reasons in the saved sample");

    const filters = document.createElement("div");
    filters.className = "filters";
    const modelLabel = document.createElement("label");
    modelLabel.textContent = "Model";
    const modelSelect = document.createElement("select");
    const allModels = document.createElement("option");
    allModels.value = "";
    allModels.textContent = "Both";
    modelSelect.appendChild(allModels);
    presentModels.forEach(function (model) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = names[model] || model;
      modelSelect.appendChild(option);
    });
    modelSelect.value = errorState.model;
    modelSelect.addEventListener("change", function () {
      errorState.model = modelSelect.value;
      renderErrorTable();
    });
    modelLabel.appendChild(modelSelect);
    const reasonLabelNode = document.createElement("label");
    reasonLabelNode.textContent = "Reason";
    const reasonSelect = document.createElement("select");
    const allReasons = document.createElement("option");
    allReasons.value = "";
    allReasons.textContent = "Any";
    reasonSelect.appendChild(allReasons);
    reasons.forEach(function (reason) {
      const option = document.createElement("option");
      option.value = reason;
      option.textContent = reasonLabel(reason);
      reasonSelect.appendChild(option);
    });
    reasonSelect.value = errorState.reason;
    reasonSelect.addEventListener("change", function () {
      errorState.reason = reasonSelect.value;
      renderErrorTable();
    });
    reasonLabelNode.appendChild(reasonSelect);
    filters.append(modelLabel, reasonLabelNode);
    const tableHost = document.createElement("div");
    tableHost.id = "error-table-host";
    host.append(filters, tableHost);
    renderErrorTable();
  }

  function renderErrorTable() {
    const host = document.getElementById("error-table-host");
    if (!host) return;
    clear(host);
    const rows = errorRows().filter(function (row) {
      if (errorState.model && row.model !== errorState.model) return false;
      if (errorState.reason && row.reason !== errorState.reason) return false;
      return true;
    });
    if (!rows.length) {
      const empty = document.createElement("p");
      empty.textContent = "No sampled errors match these filters.";
      host.appendChild(empty);
      return;
    }
    const body = rows.map(function (row) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Evidence";
      const paragraph = document.createElement("p");
      paragraph.textContent = row.evidence || "No evidence text was stored.";
      details.append(summary, paragraph);
      return [
        row.claim,
        chip(row.true),
        names[row.model] || row.model,
        reasonLabel(row.reason),
        chip(row.pred_tfidf),
        row.pred_sbert ? chip(row.pred_sbert) : document.createTextNode("Pending"),
        details,
      ];
    });
    host.appendChild(dataTable(
      "Misclassified claims in the saved sample.",
      ["Claim", "True", "Sample", "Reason", "TF-IDF", "SBERT", "Evidence"],
      body
    ));
  }

  function pipeline(title, stages) {
    const section = document.createElement("div");
    section.className = "pipeline";
    const heading = document.createElement("h3");
    heading.textContent = title;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 980 150");
    svg.setAttribute("role", "img");
    const svgTitle = document.createElementNS("http://www.w3.org/2000/svg", "title");
    svgTitle.textContent = title;
    svg.appendChild(svgTitle);
    const width = 150;
    const gap = 188;
    stages.forEach(function (stage, index) {
      const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
      group.setAttribute("tabindex", "0");
      group.setAttribute("role", "button");
      group.setAttribute("aria-label", stage.title + ". " + stage.note);
      const x = 16 + index * gap;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", String(x));
      rect.setAttribute("y", "36");
      rect.setAttribute("width", String(width));
      rect.setAttribute("height", "72");
      rect.setAttribute("rx", "14");
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", String(x + width / 2));
      text.setAttribute("y", "78");
      text.setAttribute("text-anchor", "middle");
      text.textContent = stage.title;
      group.append(rect, text);
      function show() {
        document.getElementById("method-note").textContent = stage.note;
      }
      group.addEventListener("mouseenter", show);
      group.addEventListener("focus", show);
      svg.appendChild(group);
      if (index < stages.length - 1) {
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", String(x + width + 6));
        line.setAttribute("x2", String(x + gap - 8));
        line.setAttribute("y1", "72");
        line.setAttribute("y2", "72");
        svg.appendChild(line);
      }
    });
    section.append(heading, svg);
    return section;
  }

  function renderMethods() {
    const host = document.getElementById("pipelines");
    clear(host);
    host.appendChild(pipeline("TF-IDF, then logistic regression", [
      { title: "Claim + evidence", note: "The claim and the question-answer evidence are kept as two text fields." },
      { title: "Word TF-IDF", note: "Word and bigram counts become TF-IDF features, fit only on the training fold." },
      { title: "Logistic regression", note: "One linear classifier, class-balanced, chooses a verdict." },
      { title: "Verdict", note: "Supported, Refuted, Not Enough Evidence, or Conflicting Evidence." },
    ]));
    host.appendChild(pipeline("Sentence-BERT, then logistic regression", [
      { title: "Claim + evidence", note: "The claim is one string. Evidence pairs are encoded, then averaged." },
      { title: "Frozen encoder", note: "A frozen MiniLM encoder writes one vector for the claim and one for the evidence." },
      { title: "u, v, |u−v|, u·v", note: "The classifier sees the two vectors, their absolute difference, and their product." },
      { title: "Logistic regression", note: "The same classifier family and the same C grid as TF-IDF. The encoder is not trained." },
      { title: "Verdict", note: "The same four labels, scored with the same macro-F1." },
    ]));
    const protocol = data.protocol || {};
    const grid = (protocol.c_grid || []).map(function (value) { return formatNumber(value, 2); }).join(", ");
    document.getElementById("protocol").textContent =
      "Shared protocol: stratified " + protocol.cv_folds + "-fold cross-validation, seed " + protocol.seed +
      ", class-balanced logistic regression, and one C grid (" + grid + "). The selection score is macro-F1. " +
      "The bootstrap draws " + protocol.n_bootstrap + " resamples. The encoder is " + protocol.sbert_model + " and stays frozen.";
  }

  function barChart(key, host, labelsText, datasets, description) {
    mountChart(key, host, {
      type: "bar",
      data: { labels: labelsText, datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0, color: cssVar("--muted") }, grid: { color: cssVar("--line") } },
          x: { ticks: { color: cssVar("--muted"), maxRotation: 0, autoSkip: true }, grid: { display: false } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    }, description);
  }

  function renderData() {
    const host = document.getElementById("data-mount");
    clear(host);
    if (!data.preprocess) {
      host.appendChild(pendingCard(files.preprocess || "data/processed/preprocess_report.json"));
    } else {
      const report = data.preprocess;
      const summary = document.createElement("div");
      summary.className = "split";
      ["train", "test"].forEach(function (split) {
        const block = report[split];
        if (!block) return;
        const card = document.createElement("article");
        card.className = "card";
        const heading = document.createElement("h3");
        heading.textContent = split === "train" ? "Train" : "Dev, used as test";
        const line = document.createElement("p");
        line.textContent = "Loaded " + block.loaded + ". Kept " + block.final + ".";
        card.append(heading, line);
        if (block.class_counts) {
          const list = document.createElement("ul");
          labels.forEach(function (label) {
            if (block.class_counts[label] === undefined) return;
            const item = document.createElement("li");
            item.textContent = short(label) + ": " + block.class_counts[label];
            list.appendChild(item);
          });
          card.appendChild(list);
        }
        summary.appendChild(card);
      });
      host.appendChild(summary);
      const counts = labels.map(function (label) {
        return {
          label: short(label),
          train: report.train && report.train.class_counts ? report.train.class_counts[label] : 0,
          test: report.test && report.test.class_counts ? report.test.class_counts[label] : 0,
        };
      });
      const classHost = document.createElement("div");
      barChart("classes", classHost, counts.map(function (row) { return row.label; }), [
        { label: "Train", data: counts.map(function (row) { return row.train; }), backgroundColor: cssVar("--series-tfidf"), borderWidth: 0 },
        { label: "Dev", data: counts.map(function (row) { return row.test; }), backgroundColor: cssVar("--series-sbert"), borderWidth: 0 },
      ], "Class counts in the train split and the dev split");
      host.appendChild(classHost);
    }
    if (!data.lengths || (!data.lengths.claim_len && !data.lengths.evidence_len)) {
      host.appendChild(pendingCard(files.lengths || "data/processed/train.parquet"));
    } else {
    [["claim_len", "Claim length, in words", "claim-length"], ["evidence_len", "Evidence length, in words", "evidence-length"]].forEach(function (spec) {
      const histogram = data.lengths[spec[0]];
      const panel = document.createElement("div");
      const heading = document.createElement("h3");
      heading.textContent = spec[1];
      panel.appendChild(heading);
      if (!histogram) {
        panel.appendChild(pendingCard(files.lengths || "data/processed/train.parquet"));
      } else {
        barChart(spec[2], panel, histogram.labels, [
          { label: "Train", data: histogram.train, backgroundColor: cssVar("--series-tfidf"), borderWidth: 0 },
          { label: "Dev", data: histogram.test, backgroundColor: cssVar("--series-sbert"), borderWidth: 0 },
        ], spec[1] + " histogram");
      }
      host.appendChild(panel);
    });
    }
    if (data.eda) {
      const rows = data.eda.map(function (row) {
        return [
          row.split,
          short(row.label),
          String(row.count),
          formatNumber(row.share, 3),
          formatNumber(row.mean_claim_len, 1),
          formatNumber(row.mean_evidence_len, 1),
        ];
      });
      host.appendChild(dataTable(
        "Mean lengths from the exploratory summary.",
        ["Split", "Label", "Count", "Share", "Mean claim length", "Mean evidence length"],
        rows
      ));
    }
  }

  function refreshCharts() {
    renderComparison();
    renderAblation();
    renderConfusion();
    renderErrors();
    renderData();
  }

  function setupTheme() {
    const button = document.getElementById("theme-toggle");
    const stored = localStorage.getItem("truthtrace-theme");
    function systemDark() {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function apply(theme) {
      if (theme === "light" || theme === "dark") document.documentElement.setAttribute("data-theme", theme);
      else document.documentElement.removeAttribute("data-theme");
      const dark = theme === "dark" || (theme !== "light" && systemDark());
      button.setAttribute("aria-pressed", dark ? "true" : "false");
      button.textContent = dark ? "Light mode" : "Dark mode";
    }
    apply(stored || "");
    button.addEventListener("click", function () {
      const current = document.documentElement.getAttribute("data-theme");
      const dark = current === "dark" || (current !== "light" && systemDark());
      const next = dark ? "light" : "dark";
      localStorage.setItem("truthtrace-theme", next);
      apply(next);
      refreshCharts();
    });
  }

  function setupChrome() {
    const progress = document.getElementById("progress");
    function onScroll() {
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const ratio = max > 0 ? window.scrollY / max : 0;
      progress.style.transform = "scaleX(" + ratio + ")";
    }
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const toggle = document.getElementById("nav-toggle");
    toggle.addEventListener("click", function () {
      const open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    const links = Array.from(document.querySelectorAll("#nav-links a"));
    links.forEach(function (link) {
      link.addEventListener("click", function () {
        document.body.classList.remove("nav-open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
    const spy = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (link) {
          const current = link.getAttribute("href") === "#" + entry.target.id;
          if (current) link.setAttribute("aria-current", "true");
          else link.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-45% 0px -45% 0px", threshold: 0 });
    links.forEach(function (link) {
      const section = document.querySelector(link.getAttribute("href"));
      if (section) spy.observe(section);
    });
    document.querySelectorAll(".reveal").forEach(function (node) {
      if (reduceMotion || !("IntersectionObserver" in window)) {
        node.classList.add("is-in");
        return;
      }
      const watcher = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-in");
            watcher.unobserve(entry.target);
          }
        });
      }, { threshold: 0.12 });
      watcher.observe(node);
    });
  }

  function setupControls() {
    document.getElementById("tab-overall").addEventListener("click", function () {
      comparisonView = "overall";
      document.getElementById("tab-overall").setAttribute("aria-selected", "true");
      document.getElementById("tab-class").setAttribute("aria-selected", "false");
      renderComparison();
    });
    document.getElementById("tab-class").addEventListener("click", function () {
      comparisonView = "class";
      document.getElementById("tab-overall").setAttribute("aria-selected", "false");
      document.getElementById("tab-class").setAttribute("aria-selected", "true");
      renderComparison();
    });
    document.getElementById("cm-raw").addEventListener("click", function () {
      confusionNormalized = false;
      document.getElementById("cm-raw").setAttribute("aria-selected", "true");
      document.getElementById("cm-share").setAttribute("aria-selected", "false");
      renderConfusion();
    });
    document.getElementById("cm-share").addEventListener("click", function () {
      confusionNormalized = true;
      document.getElementById("cm-raw").setAttribute("aria-selected", "false");
      document.getElementById("cm-share").setAttribute("aria-selected", "true");
      renderConfusion();
    });
    document.getElementById("drawer-close").addEventListener("click", closeDrawer);
    document.getElementById("drawer").addEventListener("click", function (event) {
      if (event.target.id === "drawer") closeDrawer();
    });
    document.addEventListener("keydown", function (event) {
      const drawer = document.getElementById("drawer");
      if (drawer.hidden) return;
      if (event.key === "Escape") closeDrawer();
      if (event.key !== "Tab") return;
      const focusable = Array.from(drawer.querySelectorAll("button, a, summary, input, select")).filter(function (node) {
        return !node.disabled;
      });
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });
  }

  fillCopy();
  setupTheme();
  setupChrome();
  setupControls();
  renderHero();
  renderComparison();
  renderAblation();
  renderConfusion();
  renderStatistics();
  renderClaims();
  renderErrors();
  renderMethods();
  renderData();
})();
