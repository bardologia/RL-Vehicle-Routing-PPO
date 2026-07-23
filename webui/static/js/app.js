"use strict";

window.apiGet = async function (url) {
  try {
    const res = await fetch(url);
    return await res.json();
  } catch (e) {
    return { error: "backend unreachable" };
  }
};

window.apiPost = async function (url, body) {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    return await res.json();
  } catch (e) {
    return { ok: false, error: "backend unreachable" };
  }
};

let _toastTimer = null;
window.toast = function (message, kind) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = "toast is-show" + (kind ? ` is-${kind}` : "");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => {
    el.className = "toast";
  }, 3400);
};

window.fmtWhen = function (epoch) {
  const delta = Date.now() / 1000 - epoch;
  if (delta < 90) return "just now";
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return `${Math.round(delta / 86400)}d ago`;
};

window.fmtBytes = function (size) {
  if (size > 1e9) return (size / 1e9).toFixed(2) + " GB";
  if (size > 1e6) return (size / 1e6).toFixed(1) + " MB";
  if (size > 1e3) return (size / 1e3).toFixed(1) + " KB";
  return size + " B";
};

window.fmtNum = function (value, digits) {
  if (value == null) return "–";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value) && Math.abs(value) < 1e7) return String(value);
  return value.toFixed(digits == null ? 3 : digits);
};

window.escapeHtml = function (text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
};

window.openLightbox = function (src, caption) {
  const box = document.getElementById("lightbox");
  box.querySelector("img").src = src;
  box.querySelector(".lightbox__cap").textContent = caption || "";
  box.classList.add("is-open");
};

class App {
  constructor() {
    this.views = {};
    this.active = null;
  }

  async init() {
    document.getElementById("lightbox").addEventListener("click", (e) => {
      e.currentTarget.classList.remove("is-open");
    });

    this.views = {
      home: new HomeView(),
      launch: new LaunchView(),
      console: new ConsoleView(),
      results: new ResultsView(),
      scenario: new ScenarioView(),
    };

    this.router = new Router((page, param) => this._onRoute(page, param));
    window.router = this.router;

    this._health();
    setInterval(() => this._health(), 30000);

    this.router.start();
  }

  async _health() {
    const h = await apiGet("/api/health");
    for (const svc of ["osrm", "vroom"]) {
      const el = document.getElementById(`svc-${svc}`);
      el.classList.toggle("is-up", h[svc] === true);
      el.classList.toggle("is-down", h[svc] === false);
      el.title = h[`${svc}_url`] || "";
    }
  }

  _onRoute(page, param) {
    if (this.active && this.views[this.active] && this.views[this.active].leave) {
      this.views[this.active].leave();
    }
    this.active = page;
    const view = this.views[page];
    if (view && view.enter) view.enter(param);
  }
}

document.addEventListener("DOMContentLoaded", () => new App().init());
