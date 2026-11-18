import { api } from "../api.js";
import { el, esc, plural } from "../format.js";
import { errorBlock, skeleton } from "../ui/status.js";

export class LaunchView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.scripts = null;
    this.form = null;
    this.dirty = {};
    this.invalid = {};
    this.activeKey = null;

    root.appendChild(el(`
      <div class="launch">
        <aside class="launch__rail" aria-label="Scripts"></aside>
        <div class="launch__main"></div>
      </div>`));
    this.rail = root.querySelector(".launch__rail");
    this.main = root.querySelector(".launch__main");

    this.main.addEventListener("input", (event) => this.onEdit(event));
    this.main.addEventListener("change", (event) => this.onEdit(event));
    this.main.addEventListener("click", (event) => this.onClick(event));
  }

  async enter(parts, query) {
    if (!this.scripts) {
      try {
        this.scripts = (await api.get("/api/scripts")).scripts;
      } catch (error) {
        this.main.innerHTML = errorBlock("The script list could not be loaded.", error.message);
        return;
      }
      this.renderRail();
    }
    const key = parts[0] || this.activeKey || this.scripts[0].key;
    const hasQuery = Object.keys(query).length > 0;
    if (key !== this.activeKey || hasQuery || !this.form) await this.select(key, query);
  }

  leave() {}

  renderRail() {
    const groups = new Map();
    for (const script of this.scripts) {
      if (!groups.has(script.group)) groups.set(script.group, []);
      groups.get(script.group).push(script);
    }
    this.rail.innerHTML = [...groups.entries()].map(([group, items]) => `
      <h4 class="launch__group">${esc(group)}</h4>
      ${items.map((script) => `
        <a class="launch__script" href="#/launch/${esc(script.key)}" data-key="${esc(script.key)}">
          <span class="launch__scripttitle">${esc(script.title)}</span>
          <span class="launch__scriptsum">${esc(script.summary)}</span>
        </a>`).join("")}`).join("");
  }

  async select(key, query) {
    this.activeKey = key;
    this.dirty = {};
    this.invalid = {};
    this.launchError = null;
    for (const link of this.rail.querySelectorAll(".launch__script")) {
      if (link.dataset.key === key) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    }

    this.main.innerHTML = skeleton(4);
    try {
      this.form = await api.get(`/api/scripts/${encodeURIComponent(key)}/config`);
    } catch (error) {
      this.form = null;
      this.main.innerHTML = errorBlock(`The form for "${key}" could not be loaded.`, error.message);
      return;
    }

    const known = new Set(this.fields().map((field) => field.path));
    for (const [path, value] of Object.entries(query || {})) {
      if (known.has(path)) this.dirty[path] = value;
    }
    this.renderForm();
  }

  fields() {
    return this.form.sections.flatMap((section) => section.fields);
  }

  field(path) {
    return this.fields().find((field) => field.path === path);
  }

  defaultText(field) {
    if (field.default === null || field.default === undefined) return "";
    if (Array.isArray(field.default)) return field.default.join(", ");
    return String(field.default);
  }

  fieldHtml(field) {
    const dirty = field.path in this.dirty;
    const current = dirty ? this.dirty[field.path] : this.defaultText(field);
    const help = field.help ? `<span class="field__help">${esc(field.help)}</span>` : "";
    const head = `<span class="field__label">${esc(field.label)}</span><code class="field__path">${esc(field.path)}</code>`;
    let control;

    if (field.choices) {
      control = `<select class="select" data-path="${esc(field.path)}">${field.choices.map((choice) => `<option value="${esc(choice)}"${String(choice) === String(current) ? " selected" : ""}>${esc(choice)}</option>`).join("")}</select>`;
    } else if (field.kind === "bool") {
      const on = String(current).toLowerCase() === "true";
      control = `<label class="check"><input type="checkbox" data-path="${esc(field.path)}" data-kind="bool"${on ? " checked" : ""}><span>${on ? "true" : "false"}</span></label>`;
    } else {
      const shown = current === "none" ? "" : current;
      control = `<input class="input" type="text" inputmode="${field.kind === "int" || field.kind === "float" ? "decimal" : "text"}" spellcheck="false" data-path="${esc(field.path)}" data-kind="${esc(field.kind)}" value="${esc(shown)}"${field.nullable ? ' placeholder="not set"' : ""}>`;
    }

    return `
      <div class="field${dirty ? " is-dirty" : ""}" data-field="${esc(field.path)}">
        <div class="field__head">${head}</div>
        ${control}
        <span class="field__error" role="alert"></span>
        ${help}
      </div>`;
  }

  renderForm() {
    const form = this.form;
    const essentials = form.essentials.map((path) => this.field(path)).filter((field) => field);
    const essentialPaths = new Set(essentials.map((field) => field.path));

    const sections = form.sections.map((section, index) => {
      const rest = section.fields.filter((field) => !essentialPaths.has(field.path));
      if (!rest.length) return "";
      return `
        <details class="launch__section" data-section="${index}">
          <summary><span class="launch__sectiontitle">${esc(section.title)}</span><span class="muted">${plural(rest.length, "field")}</span><span class="launch__changed" data-changed="${index}"></span></summary>
          <div class="launch__grid">${rest.map((field) => this.fieldHtml(field)).join("")}</div>
        </details>`;
    }).join("");

    this.main.innerHTML = `
      <header class="launch__head">
        <div>
          <h2>${esc(form.title)}</h2>
          <p class="muted">${esc(form.summary)}</p>
        </div>
        <div class="launch__actions">
          <span class="muted" data-role="count"></span>
          <button type="button" class="btn btn--quiet" data-action="reset">Reset</button>
          <button type="button" class="btn" data-action="queue">Queue</button>
          <button type="button" class="btn btn--primary" data-action="launch">Launch</button>
        </div>
      </header>
      <div data-role="error"></div>
      <pre class="launch__cmd" data-role="cmd" aria-label="Command preview"></pre>
      <section class="panel">
        <header class="panel__head"><h3 class="panel__title">Essentials</h3><span class="panel__help">The fields most runs change. Everything else keeps its default unless edited below.</span></header>
        <div class="panel__body"><div class="launch__grid">${essentials.map((field) => this.fieldHtml(field)).join("")}</div></div>
      </section>
      <div class="launch__sections">${sections}</div>`;

    for (const path of Object.keys(this.dirty)) {
      const wrap = this.main.querySelector(`[data-field="${CSS.escape(path)}"]`);
      const section = wrap.closest("details");
      if (section) section.open = true;
    }
    this.syncPreview();
  }

  validate(field, text) {
    if (text === "") return field.nullable ? null : "A value is required.";
    if (field.kind === "int" && !/^-?\d+$/.test(text)) return "Expected a whole number.";
    if (field.kind === "float" && !/^-?(\d+\.?\d*|\.\d+)(e-?\d+)?$/i.test(text)) return "Expected a number.";
    if (field.kind === "tuple" && !text.split(",").every((part) => part.trim().length)) return "Expected comma-separated values.";
    return null;
  }

  onEdit(event) {
    const input = event.target;
    if (!(input instanceof HTMLElement) || !input.dataset.path) return;
    const path = input.dataset.path;
    const field = this.field(path);
    const wrap = input.closest(".field");
    let value;
    let problem = null;

    if (input.dataset.kind === "bool") {
      value = input.checked ? "true" : "false";
      input.parentElement.querySelector("span").textContent = value;
      if (value === String(Boolean(field.default))) delete this.dirty[path];
      else this.dirty[path] = value;
    } else {
      value = input.value.trim();
      problem = field.choices ? null : this.validate(field, value);
      if (value === this.defaultText(field)) delete this.dirty[path];
      else this.dirty[path] = value === "" ? "none" : value;
    }

    if (problem) this.invalid[path] = problem;
    else delete this.invalid[path];
    if (input.classList.contains("input")) input.setAttribute("aria-invalid", String(Boolean(problem)));
    wrap.querySelector(".field__error").textContent = problem || "";
    wrap.classList.toggle("is-dirty", path in this.dirty);
    this.syncPreview();
  }

  syncPreview() {
    const parts = [`python -u main/${esc(this.form.key)}.py`];
    for (const [path, value] of Object.entries(this.dirty)) parts.push(`    <b>--${esc(path)}</b> ${esc(value)}`);
    this.main.querySelector("[data-role=cmd]").innerHTML = parts.join(" \\\n");

    const count = Object.keys(this.dirty).length;
    const bad = Object.keys(this.invalid).length;
    this.main.querySelector("[data-role=count]").textContent = bad ? `${plural(bad, "field")} invalid` : count ? `${plural(count, "override")}` : "All defaults";
    for (const button of this.main.querySelectorAll("[data-action=launch], [data-action=queue]")) button.disabled = bad > 0;

    this.form.sections.forEach((section, index) => {
      const node = this.main.querySelector(`[data-changed="${index}"]`);
      if (!node) return;
      const changed = section.fields.filter((field) => field.path in this.dirty && !this.form.essentials.includes(field.path)).length;
      node.textContent = changed ? `${changed} changed` : "";
    });
  }

  onClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    if (button.dataset.action === "reset") this.select(this.activeKey, {});
    if (button.dataset.action === "launch") this.launch(false);
    if (button.dataset.action === "queue") this.launch(true);
  }

  async launch(queue) {
    const errorNode = this.main.querySelector("[data-role=error]");
    errorNode.innerHTML = "";
    let result;
    try {
      result = await api.post("/api/run", { script: this.form.key, overrides: this.dirty, queue });
    } catch (error) {
      errorNode.innerHTML = errorBlock("The job was not started.", error.message);
      return;
    }
    this.app.toast.confirm(queue && result.queued ? "Job queued" : "Job launched");
    this.app.pollJobs();
    this.app.router.go(`#/console/${result.job_id}`);
  }
}
