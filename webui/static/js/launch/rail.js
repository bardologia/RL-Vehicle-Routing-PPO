import { api } from "../api.js";
import { esc, icon } from "../format.js";
import { Orb } from "../fx/orb.js";
import { ConfigForm } from "./config_form.js";

export class EntryFormControl {
  constructor(view) {
    this.view = view;
  }

  entryKey() {
    return this.view.detail ? this.view.detail.key : null;
  }

  entryTitle() {
    return this.view.detail ? this.view.detail.title : "";
  }

  formNotReady() {
    if (!this.view.config) return "the configuration is still loading";
    return "";
  }
}

export class RunConfigCopy extends EntryFormControl {
  constructor(view, runsRoot) {
    super(view);
    this.runsRoot = runsRoot;
    this.base = null;
    this.run = "";
    this.loadSeq = 0;
    this.timer = null;
  }

  build() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `
      <span class="rail-block__label">Copy from run</span>
      <input class="input rail-input" type="text" spellcheck="false" placeholder="runs directory" aria-label="Runs directory" data-role="base">
      <select class="select rail-input" aria-label="Run to copy from" data-role="run"></select>
      <span class="rail-note" data-role="note"></span>
      <button type="button" class="btn btn--sm rail-wide" data-role="apply" disabled title="Replace every field with the selected run's recorded settings, so you only change what should differ; the run's own name is kept as it is">Copy settings</button>`;

    const base = block.querySelector("[data-role=base]");
    const select = block.querySelector("[data-role=run]");
    const note = block.querySelector("[data-role=note]");
    const apply = block.querySelector("[data-role=apply]");
    base.value = this.base !== null ? this.base : this.runsRoot;

    base.addEventListener("input", () => {
      this.base = base.value;
      clearTimeout(this.timer);
      this.timer = setTimeout(() => this.load(select, note, apply), 350);
    });
    select.addEventListener("change", () => {
      this.run = select.value;
      apply.disabled = !select.value;
    });
    apply.addEventListener("click", () => this.apply(select.value, apply));

    this.load(select, note, apply);
    return block;
  }

  async load(select, note, apply) {
    const base = this.base !== null ? this.base : this.runsRoot;
    const seq = ++this.loadSeq;

    note.innerHTML = `${Orb.html("searching", 20)}<span>listing</span>`;
    let res;
    try {
      res = await api.get(`/api/run-config/runs?base=${encodeURIComponent(base)}`);
    } catch (error) {
      if (seq !== this.loadSeq) return;
      select.innerHTML = `<option value="">no runs found</option>`;
      note.textContent = error.message;
      apply.disabled = true;
      return;
    }
    if (seq !== this.loadSeq) return;

    const runs = res.runs || [];
    note.textContent = runs.length ? `${runs.length} run${runs.length > 1 ? "s" : ""} with an evaluation report` : "no evaluated runs here";
    select.innerHTML = `<option value="">${runs.length ? "select a run" : "no runs found"}</option>${runs.map((run) => `<option value="${esc(run.path)}"${run.path === this.run ? " selected" : ""}>${esc(run.name)}  ·  ${esc(run.source)}</option>`).join("")}`;
    apply.disabled = !select.value;
  }

  async apply(path, button) {
    if (!path) return;
    const notReady = this.formNotReady();
    if (notReady) {
      this.view.app.toast.show(`Nothing copied: ${notReady}`, "error");
      return;
    }

    button.disabled = true;
    try {
      const res = await api.get(`/api/run-config?path=${encodeURIComponent(path)}`);
      const outcome = this.view.applyRunConfig(res.config);
      const parts = [`Copied ${res.run}: ${outcome.applied} override${outcome.applied === 1 ? "" : "s"} set`];
      if (res.identity.length) parts.push(`${res.identity.length} identity field${res.identity.length === 1 ? "" : "s"} kept: ${res.identity.join(", ")}`);
      parts.push(...ConfigForm.outcomeParts(outcome));
      this.view.app.toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
    } catch (error) {
      this.view.app.toast.show(error.message || "Could not read the run's settings", "error");
    } finally {
      button.disabled = false;
    }
  }
}

export class ConfigTemplates extends EntryFormControl {
  constructor(view) {
    super(view);
    this.entries = [];
    this.selected = "";
    this.loadSeq = 0;
  }

  build() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `
      <span class="rail-block__label">Config templates</span>
      <select class="select rail-input" aria-label="Config template" data-role="select"></select>
      <span class="rail-note" data-role="note"></span>
      <div class="rail-row">
        <button type="button" class="btn btn--sm" data-role="import" title="Set the form to this template's overrides; every field the template does not carry goes back to its default">Import</button>
        <button type="button" class="btn btn--sm" data-role="save" title="Store the overrides currently set on this form under a name, to import on any launch page later">Save as template</button>
        <button type="button" class="tbtn tbtn--sm" data-role="delete" title="Delete the selected template" aria-label="Delete the selected template">${icon("trash")}</button>
      </div>`;

    this.selectEl = block.querySelector("[data-role=select]");
    this.noteEl = block.querySelector("[data-role=note]");
    this.importBtn = block.querySelector("[data-role=import]");
    this.saveBtn = block.querySelector("[data-role=save]");
    this.deleteBtn = block.querySelector("[data-role=delete]");

    this.selectEl.addEventListener("change", () => {
      this.selected = this.selectEl.value;
      this.paintActions();
    });
    this.importBtn.addEventListener("click", () => this.importSelected());
    this.saveBtn.addEventListener("click", () => this.save());
    this.deleteBtn.addEventListener("click", () => this.remove());

    this.load();
    return block;
  }

  async load() {
    const seq = ++this.loadSeq;
    this.noteEl.innerHTML = `${Orb.html("searching", 20)}<span>loading</span>`;
    try {
      const res = await api.get("/api/config-templates");
      if (seq !== this.loadSeq) return;
      this.entries = res.templates;
    } catch (error) {
      if (seq !== this.loadSeq) return;
      this.entries = [];
      this.noteEl.textContent = error.message;
    }
    this.paintOptions();
  }

  paintOptions() {
    const key = this.entryKey();
    const mine = this.entries.filter((entry) => entry.script === key);
    const rest = this.entries.filter((entry) => entry.script !== key);
    const option = (entry, tagged) => {
      const count = Object.keys(entry.overrides).length;
      return `<option value="${esc(entry.template_id)}">${esc(entry.name)}  ·  ${count} override${count === 1 ? "" : "s"}${tagged ? `  ·  ${esc(entry.script)}` : ""}</option>`;
    };

    this.selectEl.innerHTML = `<option value="">${this.entries.length ? "select a template" : "no templates saved yet"}</option>`
      + (mine.length ? `<optgroup label="this entry point">${mine.map((entry) => option(entry, false)).join("")}</optgroup>` : "")
      + (rest.length ? `<optgroup label="other entry points">${rest.map((entry) => option(entry, true)).join("")}</optgroup>` : "");

    this.selectEl.value = this.entries.some((entry) => entry.template_id === this.selected) ? this.selected : "";
    this.selected = this.selectEl.value;
    this.noteEl.textContent = this.entries.length ? `${mine.length} for this entry point, ${rest.length} other` : "save the current overrides to start one";
    this.paintActions();
  }

  paintActions() {
    const entry = this.current();
    this.importBtn.disabled = !entry;
    this.deleteBtn.disabled = !entry;
  }

  current() {
    return this.entries.find((entry) => entry.template_id === this.selected) || null;
  }

  importSelected() {
    const entry = this.current();
    if (!entry) return;
    const notReady = this.formNotReady();
    if (notReady) {
      this.view.app.toast.show(`Nothing imported: ${notReady}`, "error");
      return;
    }

    const outcome = this.view.applyRunConfig(entry.overrides);
    const parts = [`Imported "${entry.name}": ${outcome.applied} override${outcome.applied === 1 ? "" : "s"} set`];
    if (entry.script !== this.entryKey()) parts.push(`saved on ${entry.script}`);
    parts.push(...ConfigForm.outcomeParts(outcome));
    this.view.app.toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
  }

  async save() {
    const key = this.entryKey();
    if (!key || !this.view.config) {
      this.view.app.toast.show("The configuration is still loading", "error");
      return;
    }

    const raw = window.prompt("Name this config template:", "");
    if (raw === null) return;
    const name = raw.trim();
    if (!name) {
      this.view.app.toast.show("A template needs a name", "error");
      return;
    }

    const clash = this.entries.find((entry) => entry.script === key && entry.name === name);
    if (clash && !window.confirm(`A template named "${name}" already exists for ${key}. Replace it?`)) return;

    const overrides = { ...this.view.dirty };
    const count = Object.keys(overrides).length;
    this.saveBtn.disabled = true;
    try {
      const res = await api.post("/api/config-templates", { script_key: key, title: this.entryTitle(), name, overrides, replace: Boolean(clash) });
      this.selected = res.entry.template_id;
      await this.load();
      this.view.app.toast.show(`${res.replaced ? "Replaced" : "Saved"} template "${name}" with ${count} override${count === 1 ? "" : "s"}`, "ok");
    } catch (error) {
      this.view.app.toast.show(error.message || "Save failed", "error");
    } finally {
      this.saveBtn.disabled = false;
    }
  }

  async remove() {
    const entry = this.current();
    if (!entry || !window.confirm(`Delete template "${entry.name}"?`)) return;
    try {
      await api.post(`/api/config-templates/${entry.template_id}/delete`, {});
      this.selected = "";
      await this.load();
      this.view.app.toast.show(`Deleted template "${entry.name}"`, "ok");
    } catch (error) {
      this.view.app.toast.show(error.message || "Delete failed", "error");
    }
  }
}
