(function () {
  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");

  function currentTheme() {
    var explicit = root.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") return explicit;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function paintToggle() {
    if (!toggle) return;
    var next = currentTheme() === "dark" ? "light" : "dark";
    toggle.textContent = next === "dark" ? "Dark mode" : "Light mode";
  }

  paintToggle();
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("truthtrace-theme", next); } catch (error) {}
      paintToggle();
    });
  }

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduceMotion) {
    document.querySelectorAll(".metric").forEach(function (node) {
      var match = node.textContent.match(/(\d+\.\d+)/);
      if (!match) return;
      var target = Number(match[1]);
      if (!isFinite(target)) return;
      var prefix = node.textContent.slice(0, match.index);
      var suffix = node.textContent.slice(match.index + match[1].length);
      var digits = (match[1].split(".")[1] || "").length;
      var started = performance.now();
      var duration = 700;
      function frame(now) {
        var progress = Math.min(1, (now - started) / duration);
        var value = target * (1 - Math.pow(1 - progress, 3));
        node.textContent = prefix + value.toFixed(digits) + suffix;
        if (progress < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  var lab = document.getElementById("claim-lab");
  if (!lab) return;
  var cards = Array.prototype.slice.call(lab.querySelectorAll(".example"));
  var progress = lab.querySelector(".lab-progress");
  var score = lab.querySelector(".lab-score");
  var empty = lab.querySelector(".lab-empty");
  var index = 0;
  var filter = "all";
  var seen = {};

  function list() {
    return cards.filter(function (card) {
      return filter === "all" || card.getAttribute("data-disagree") === "yes";
    });
  }

  function fillBars(card) {
    card.querySelectorAll(".probs li").forEach(function (item) {
      var fill = item.querySelector(".p-fill");
      var value = Number(item.getAttribute("data-p"));
      if (!fill || !isFinite(value)) return;
      fill.style.width = Math.max(0, Math.min(100, value * 100)) + "%";
    });
  }

  function updateScore() {
    var correct = 0;
    var attempts = 0;
    Object.keys(seen).forEach(function (key) {
      attempts += 1;
      if (seen[key]) correct += 1;
    });
    if (score) {
      score.textContent = attempts ? "Matched " + correct + " / " + attempts : "";
    }
  }

  function show() {
    var shown = list();
    cards.forEach(function (card) { card.hidden = true; });
    if (!shown.length) {
      if (empty) empty.hidden = false;
      if (progress) progress.textContent = "0 claims";
      return;
    }
    if (empty) empty.hidden = true;
    if (index >= shown.length) index = 0;
    if (index < 0) index = shown.length - 1;
    shown[index].hidden = false;
    if (progress) progress.textContent = "Claim " + (index + 1) + " of " + shown.length;
  }

  lab.classList.add("is-ready");
  show();

  lab.addEventListener("click", function (event) {
    var filterButton = event.target.closest("[data-filter]");
    if (filterButton && lab.contains(filterButton)) {
      filter = filterButton.getAttribute("data-filter");
      lab.querySelectorAll("[data-filter]").forEach(function (button) {
        button.setAttribute("aria-pressed", button === filterButton ? "true" : "false");
      });
      index = 0;
      show();
      return;
    }
    var guess = event.target.closest("[data-guess]");
    if (!guess) return;
    var card = guess.closest(".example");
    if (!card || card.classList.contains("is-revealed")) return;
    var truth = card.getAttribute("data-true");
    var picked = guess.getAttribute("data-guess");
    var matched = picked === truth;
    card.classList.add("is-revealed");
    card.querySelectorAll("[data-guess]").forEach(function (button) {
      button.disabled = true;
      if (button.getAttribute("data-guess") === truth) button.classList.add("is-truth");
      if (button === guess) button.classList.add(matched ? "is-match" : "is-miss");
    });
    var feedback = card.querySelector(".guess-feedback");
    if (feedback) {
      feedback.hidden = false;
      feedback.textContent = matched
        ? "That matches the dev label."
        : "The dev label is " + truth + ".";
    }
    seen[card.querySelector(".claim-kicker").textContent] = matched;
    fillBars(card);
    updateScore();
  });

  var previous = document.getElementById("claim-prev");
  var next = document.getElementById("claim-next");
  if (previous) previous.addEventListener("click", function () { index -= 1; show(); });
  if (next) next.addEventListener("click", function () { index += 1; show(); });
})();
