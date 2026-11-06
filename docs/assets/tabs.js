document.querySelectorAll("[data-tabs]").forEach(function (root) {
  var buttons = root.querySelectorAll("[data-tab]");
  var panels = root.querySelectorAll("[data-panel]");

  function activate(id) {
    buttons.forEach(function (button) {
      button.setAttribute("aria-selected", button.getAttribute("data-tab") === id ? "true" : "false");
    });
    panels.forEach(function (panel) {
      panel.hidden = panel.getAttribute("data-panel") !== id;
    });
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      activate(button.getAttribute("data-tab"));
    });
  });

  if (buttons.length) {
    activate(buttons[0].getAttribute("data-tab"));
  }
});
