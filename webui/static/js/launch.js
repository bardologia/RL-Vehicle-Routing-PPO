"use strict";

class LaunchView {
  constructor() {
    this.rail = document.getElementById("launch-rail");
    this.main = document.getElementById("launch-main");
    this.scripts = [];
    this.form = null;
    this.dirty = {};
    this.activeKey = null;
  }

  async enter(param) {
    if (!this.scripts.length) {
      const res = await apiGet("/api/scripts");
      this.scripts = res.scripts || [];
      this._renderRail();
    }
    const key = param || this.activeKey || (this.scripts[0] && this.scripts[0].key);
    if (key) this._select(key);
  }

  leave() {}

  _renderRail() {
    const groups = {};
    for (const s of this.scripts) (groups[s.group] = groups[s.group] || []).push(s);

    this.rail.innerHTML = Object.entries(groups).map(([group, items]) => `
      <div class="script-group">${escapeHtml(group)}</div>
      ${items.map((s) => `
        <button class="script-card" data-key="${s.key}">
          <div class="script-card__title">${escapeHtml(s.title)}</div>
          <div class="script-card__sum">${escapeHtml(s.summary)}</div>
        </button>`).join("")}`).join("");

    this.rail.querySelectorAll(".script-card").forEach((el) => {
      el.addEventListener("click", () => { window.location.hash = `#/launch/${el.dataset.key}`; });
    });
  }

  async _select(key) {
    this.activeKey = key;
    this.dirty = {};
    this.rail.querySelectorAll(".script-card").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.key === key);
    });

    this.main.innerHTML = `<p class="scn-hint">loading…</p>`;
    const form = await apiGet(`/api/scripts/${key}/config`);
    if (form.error) {
      this.main.innerHTML = `<p class="scn-hint">${escapeHtml(form.error)}</p>`;
      return;
    }
    this.form = form;
    this._renderForm();
  }

  _renderForm() {
    const f = this.form;
    this.main.innerHTML = `
      <div class="launch__actions">
        <button class="btn btn--primary" id="lc-launch">Launch</button>
        <button class="btn" id="lc-queue">Queue</button>
        <button class="btn btn--ghost" id="lc-reset">Reset overrides</button>
        <span class="scn-hint" id="lc-count"></span>
      </div>
      <div class="cmd-preview" id="lc-cmd"></div>
      ${f.sections.map((section, si) => `
        <div class="card form-section">
          <h3 class="card__title">${escapeHtml(section.title)}</h3>
          <div class="form-grid">
            ${section.fields.map((field, fi) => this._fieldHtml(field, si, fi)).join("")}
          </div>
        </div>`).join("")}`;

    this.main.querySelectorAll("[data-path]").forEach((el) => {
      const handler = () => this._onEdit(el);
      el.addEventListener("input", handler);
      el.addEventListener("change", handler);
    });

    document.getElementById("lc-launch").addEventListener("click", () => this._launch(false));
    document.getElementById("lc-queue").addEventListener("click", () => this._launch(true));
    document.getElementById("lc-reset").addEventListener("click", () => this._select(this.activeKey));

    this._syncPreview();
  }

  _fieldHtml(field, si, fi) {
    const id = `fld-${si}-${fi}`;
    const label = `<span>${escapeHtml(field.label)}</span><span class="field__path">${escapeHtml(field.path)}</span>`;
    const help = field.help ? `<span class="field__help">${escapeHtml(field.help)}</span>` : "";

    if (field.choices) {
      const opts = field.choices.map((c) => `<option value="${escapeHtml(c)}" ${c === field.default ? "selected" : ""}>${escapeHtml(c)}</option>`).join("");
      return `<label class="field" data-field="${field.path}"><span class="field__label">${label}</span><select id="${id}" data-path="${field.path}">${opts}</select>${help}</label>`;
    }

    if (field.kind === "bool") {
      return `<div class="field" data-field="${field.path}">
        <span class="field__label">${label}</span>
        <label class="switch"><input type="checkbox" id="${id}" data-path="${field.path}" data-kind="bool" ${field.default ? "checked" : ""}><i></i><span>${field.default ? "true" : "false"}</span></label>${help}</div>`;
    }

    let value = field.default;
    if (value == null) value = "";
    if (Array.isArray(value)) value = value.join(", ");

    const type = field.kind === "int" || field.kind === "float" ? "number" : "text";
    const step = field.kind === "float" ? `step="any"` : field.kind === "int" ? `step="1"` : "";
    const placeholder = field.nullable ? `placeholder="none"` : "";

    return `<label class="field" data-field="${field.path}">
      <span class="field__label">${label}</span>
      <input type="${type}" ${step} ${placeholder} id="${id}" value="${escapeHtml(String(value))}" data-path="${field.path}" data-kind="${field.kind}" data-default="${escapeHtml(String(value))}">${help}</label>`;
  }

  _onEdit(el) {
    const path = el.dataset.path;
    let value;

    if (el.dataset.kind === "bool") {
      value = el.checked ? "true" : "false";
      el.parentElement.querySelector("span").textContent = value;
      const original = this._fieldByPath(path).default ? "true" : "false";
      if (value === original) delete this.dirty[path];
      else this.dirty[path] = value;
    } else {
      value = el.value.trim();
      const original = el.dataset.default != null ? el.dataset.default : String(this._fieldByPath(path).default);
      if (value === original || (value === "" && this._fieldByPath(path).nullable)) delete this.dirty[path];
      else this.dirty[path] = value === "" ? "none" : value;
    }

    const wrap = this.main.querySelector(`[data-field="${CSS.escape(path)}"]`);
    if (wrap) wrap.classList.toggle("is-dirty", path in this.dirty);
    this._syncPreview();
  }

  _fieldByPath(path) {
    for (const section of this.form.sections) {
      for (const field of section.fields) {
        if (field.path === path) return field;
      }
    }
    return {};
  }

  _syncPreview() {
    const parts = [`python -u main/${this.form.key}.py`];
    for (const [path, value] of Object.entries(this.dirty)) {
      parts.push(`  <span class="flag">--${escapeHtml(path)}</span> <span class="val">${escapeHtml(value)}</span>`);
    }
    document.getElementById("lc-cmd").innerHTML = parts.join("\n");
    document.getElementById("lc-count").textContent = Object.keys(this.dirty).length
      ? `${Object.keys(this.dirty).length} override(s)`
      : "all defaults";
  }

  async _launch(queue) {
    const res = await apiPost("/api/run", { script: this.form.key, overrides: this.dirty, queue });
    if (!res.ok) {
      toast(res.error || "launch failed", "error");
      return;
    }
    toast(queue && res.queued ? "job queued" : "job launched", "ok");
    window.location.hash = "#/console";
  }
}

window.LaunchView = LaunchView;
