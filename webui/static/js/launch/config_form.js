import { esc, icon } from "../format.js";
import { MultiValueField, NumberField, RunPicker } from "./form_widgets.js";

export class ConfigForm {
  static TYPE_REFRESH_MS = 120;

  constructor() {
    this.resetForm();
  }

  resetForm() {
    clearTimeout(this.refreshTimer);

    this.dirty = {};
    this.controls = {};
    this.states = [];
    this.gates = [];
    this.sections = [];
    this.byPath = new Map();
    this.activeSection = null;
    this.query = "";
    this.currentSection = null;
    this.config = null;
    this.layoutEl = null;
    this.navHost = null;
    this.nomatchEl = null;
    this.countEl = null;
    this.refreshTimer = null;
    this.shellCache = new Map();
  }

  buildToolbar(cfg) {
    const bar = document.createElement("div");
    bar.className = "cfg-toolbar";
    bar.innerHTML = `
      <input class="input cfg-toolbar__search" type="search" placeholder="Filter ${cfg.leaves.length} fields" aria-label="Filter configuration fields" spellcheck="false" data-role="filter">
      <span class="cfg-toolbar__count" data-role="count"></span>
      <button type="button" class="btn btn--sm" data-role="reset-all">Reset all</button>`;

    bar.querySelector("[data-role=filter]").addEventListener("input", (event) => {
      this.query = event.target.value.trim().toLowerCase();
      this.applyVisibility();
    });
    bar.querySelector("[data-role=reset-all]").addEventListener("click", () => this.resetAll());
    this.countEl = bar.querySelector("[data-role=count]");
    return bar;
  }

  applyRunConfig(values) {
    const changed = [];
    const skipped = [];
    const locked = [];

    this.dirty = {};

    Object.entries(values).forEach(([path, value]) => {
      const leaf = this.byPath.get(path);
      if (!leaf) {
        skipped.push(path);
        return;
      }
      if (String(value) === leaf.value) return;
      if (!leaf.editable) {
        locked.push(path);
        return;
      }
      this.dirty[path] = String(value);
      changed.push(path);
    });

    Object.values(this.controls).forEach((control) => control.reset());
    this.refresh();

    return { applied: changed.length, skipped, locked };
  }

  static outcomeParts(outcome) {
    const parts = [];
    if (outcome.locked.length) parts.push(`${outcome.locked.length} locked field${outcome.locked.length === 1 ? "" : "s"} kept`);
    if (outcome.skipped.length) parts.push(`${outcome.skipped.length} unknown key${outcome.skipped.length === 1 ? "" : "s"} skipped: ${outcome.skipped.slice(0, 4).join(", ")}${outcome.skipped.length > 4 ? ", ..." : ""}`);
    return parts;
  }

  static outcomeKind(outcome) {
    return outcome.skipped.length || outcome.locked.length ? "warn" : "ok";
  }

  buildPins(pinned) {
    const panel = document.createElement("section");
    panel.className = "cfg-panel cfg-panel--pins";
    panel.innerHTML = `<header class="cfg-panel__head"><h4 class="cfg-panel__name">Run essentials</h4><span class="cfg-panel__hint">check these before every launch</span></header>`;

    const grid = document.createElement("div");
    grid.className = "cfg-pins";
    pinned.forEach((leaf) => grid.appendChild(this.buildRow(leaf, "essentials", true)));
    panel.appendChild(grid);
    return panel;
  }

  renderLayout(host, cfg) {
    const layout = cfg.layout;
    this.byPath = new Map(cfg.leaves.map((leaf) => [leaf.path, leaf]));
    this.shellCache.clear();

    const wrap = document.createElement("div");
    wrap.className = `cfg-layout${layout.mode === "single" ? " cfg-layout--single" : ""}`;
    this.layoutEl = wrap;

    const main = document.createElement("div");
    main.className = "cfg-main";

    const nav = document.createElement("nav");
    nav.className = "secnav";
    nav.setAttribute("aria-label", "Configuration sections");
    this.navHost = nav;

    const declared = [];
    if (layout.essentials.length) declared.push({ key: "essentials", title: "Essentials", panels: null });
    layout.sections.forEach((section) => declared.push(section));

    declared.forEach((section) => {
      this.currentSection = section.key;

      const el = document.createElement("section");
      el.className = "cfg-section";
      el.dataset.section = section.key;
      el.innerHTML = `<h3 class="cfg-section__title">${esc(section.title)}</h3>`;

      const body = document.createElement("div");
      body.className = "cfg-section__body";
      el.appendChild(body);

      if (section.panels === null) {
        body.appendChild(this.buildPins(layout.essentials.map((entry) => this.byPath.get(entry.path))));
      } else {
        section.panels.forEach((panel) => {
          if (panel.kind === "hidden") return;
          body.appendChild(this.buildFieldsPanel(panel));
        });
      }

      const record = { key: section.key, title: section.title, when: section.when || null, el, navBtn: null, badge: null };

      if (layout.mode === "sections") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secnav__item";
        button.innerHTML = `<span class="secnav__name">${esc(section.title)}</span><span class="secnav__badge" hidden></span>`;
        button.addEventListener("click", () => this.navigate(record.key));
        record.navBtn = button;
        record.badge = button.querySelector(".secnav__badge");
        nav.appendChild(button);
      }

      main.appendChild(el);
      this.sections.push(record);
    });

    const empty = document.createElement("p");
    empty.className = "cfg-note cfg-nomatch";
    empty.textContent = "No fields match this filter.";
    empty.hidden = true;
    this.nomatchEl = empty;
    main.appendChild(empty);

    wrap.appendChild(main);
    if (layout.mode === "sections") wrap.appendChild(nav);
    host.appendChild(wrap);
    this.setActiveSection(this.activeSection || this.sections[0].key);
  }

  buildFieldsPanel(panel) {
    const el = document.createElement("section");
    el.className = "cfg-panel";
    el.dataset.cols = String(Math.min(panel.groups.length, 4));
    if (panel.title) el.insertAdjacentHTML("beforeend", `<header class="cfg-panel__head"><h4 class="cfg-panel__name">${esc(panel.title)}</h4></header>`);
    if (panel.note) el.insertAdjacentHTML("beforeend", `<p class="cfg-panel__note">${esc(panel.note)}</p>`);
    el.appendChild(this.buildGroups(panel.groups));
    return el;
  }

  buildGroups(groups) {
    const body = document.createElement("div");
    body.className = "cfg-panel__groups";

    groups.forEach((group) => {
      const groupEl = document.createElement("div");
      groupEl.className = "field-group";
      if (group.title) groupEl.insertAdjacentHTML("beforeend", `<div class="field-group__title">${esc(group.title)}</div>`);
      const inner = document.createElement("div");
      inner.className = "field-group__grid";
      group.fields.forEach((entry) => this.buildEntry(entry, inner));
      groupEl.appendChild(inner);
      body.appendChild(groupEl);
    });

    return body;
  }

  buildEntry(entry, host) {
    if (!entry.gate) {
      host.appendChild(this.buildRow(this.byPath.get(entry.path), this.currentSection));
      return;
    }

    const lead = this.byPath.get(entry.gate);
    const cell = document.createElement("div");
    cell.className = "band-block";
    cell.appendChild(this.buildRow(lead, this.currentSection));

    const gated = [];
    entry.fields.forEach((sub) => {
      const row = this.buildRow(this.byPath.get(sub.path), this.currentSection);
      row.classList.add("cfg-row--dependent");
      cell.appendChild(row);
      gated.push(this.states[this.states.length - 1]);
    });

    this.gates.push({ leaf: lead, states: gated });
    host.appendChild(cell);
  }

  shortName(leaf) {
    return leaf.section ? leaf.path.slice(leaf.section.length + 1) : leaf.path;
  }

  describe(leaf) {
    const text = this.config && this.config.descriptions ? this.config.descriptions[leaf.path] : "";
    return text || "";
  }

  buildRow(leaf, sectionKey, pinned = false) {
    const short = pinned ? leaf.path : this.shortName(leaf);
    const description = this.describe(leaf);

    const row = document.createElement("div");
    row.className = "cfg-row";
    row.dataset.path = leaf.path;

    const name = document.createElement("div");
    name.className = "cfg-row__name";
    name.innerHTML = `<span class="cfg-row__label">${esc(short)}</span>`;
    name.dataset.hintTitle = `--${leaf.path} · ${leaf.type}`;
    name.dataset.hint = description || "No description.";

    const reset = document.createElement("button");
    reset.type = "button";
    reset.className = "tbtn tbtn--sm cfg-row__reset";
    reset.title = `Reset ${leaf.path} to ${leaf.value === "" ? "none" : leaf.value}`;
    reset.setAttribute("aria-label", `Reset ${leaf.path}`);
    reset.innerHTML = icon("refresh");
    reset.addEventListener("click", () => this.resetField(leaf.path));
    name.appendChild(reset);
    row.appendChild(name);

    const spec = leaf.editable ? this.widgetSpec(leaf) : null;
    if (spec && spec.required) {
      row.classList.add("cfg-row--required");
      name.insertAdjacentHTML("beforeend", `<small class="cfg-row__required">required</small>`);
      name.dataset.hint = `${name.dataset.hint} Required: the launch stays disabled until it is set.`;
    }
    const kind = spec ? spec.kind : null;
    const choices = Array.isArray(leaf.choices) && leaf.choices.length ? leaf.choices : kind === "choice" ? spec.options : null;
    let control;

    if (kind === "multi") {
      control = new MultiValueField(this, leaf, spec).build();
      row.classList.add("cfg-row--wide");
    } else if (kind === "run") {
      control = new RunPicker(this, leaf).build();
    } else if (choices) {
      control = this.choiceControl(leaf, choices);
    } else if (!leaf.editable) {
      control = this.textControl(leaf);
      control.input.disabled = true;
      control.input.title = "not overridable from the command line";
    } else if (leaf.type === "bool") {
      control = this.switchControl(leaf);
    } else if (leaf.type === "int" || leaf.type === "float" || kind === "number") {
      control = new NumberField(this, leaf, short, kind === "number" ? spec : null).build();
      if (kind === "number") row.classList.add("cfg-row--presets");
    } else {
      control = this.textControl(leaf);
      if (leaf.type === "tuple" || /url|dir|path/.test(leaf.path)) row.classList.add("cfg-row--long");
    }

    row.appendChild(control.el);
    this.controls[leaf.path] = { leaf, reset: control.reset, input: control.input };
    this.states.push({ leaf, row, sectionKey: sectionKey !== undefined ? sectionKey : this.currentSection, pinned });
    return row;
  }

  widgetSpec(leaf) {
    if (!this.config || !this.config.layout) return null;
    return this.config.layout.widgets[leaf.path] || null;
  }

  effective(leaf) {
    return this.dirty[leaf.path] !== undefined ? this.dirty[leaf.path] : leaf.value;
  }

  choiceControl(leaf, choices) {
    const select = document.createElement("select");
    select.className = "select cfg-input";
    select.setAttribute("aria-label", leaf.path);

    const current = String(leaf.value);
    [...new Set([current, ...choices])].forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });

    const paint = () => {
      select.value = String(this.effective(leaf));
      select.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    select.addEventListener("change", () => {
      this.setValue(leaf, select.value);
      paint();
    });
    paint();
    return { el: select, input: select, reset: paint };
  }

  textControl(leaf) {
    const input = document.createElement("input");
    input.className = "cfg-input";
    input.type = "text";
    input.spellcheck = false;
    input.placeholder = leaf.nullable ? "none" : "";
    input.setAttribute("aria-label", leaf.path);

    const paint = () => {
      input.value = this.effective(leaf);
      input.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
    };
    input.addEventListener("input", () => {
      input.classList.toggle("is-dirty", input.value !== leaf.value);
      this.setValue(leaf, input.value, true);
    });
    paint();
    return { el: input, input, reset: paint };
  }

  switchControl(leaf) {
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "switch";
    toggle.setAttribute("role", "switch");
    toggle.setAttribute("aria-label", leaf.path);
    toggle.innerHTML = `<span class="switch__knob"></span><span class="switch__word"></span>`;

    const paint = () => {
      const on = this.effective(leaf) === "True";
      toggle.classList.toggle("is-on", on);
      toggle.classList.toggle("is-dirty", this.dirty[leaf.path] !== undefined);
      toggle.setAttribute("aria-checked", String(on));
      toggle.querySelector(".switch__word").textContent = on ? "on" : "off";
    };
    toggle.addEventListener("click", () => {
      this.setValue(leaf, this.effective(leaf) === "True" ? "False" : "True");
      paint();
    });
    paint();
    return { el: toggle, input: toggle, reset: paint };
  }

  setValue(leaf, value, defer) {
    if (value !== leaf.value) this.dirty[leaf.path] = value;
    else delete this.dirty[leaf.path];
    this.runRefresh(defer);
  }

  unsetValue(leaf, defer) {
    delete this.dirty[leaf.path];
    this.runRefresh(defer);
  }

  runRefresh(defer) {
    clearTimeout(this.refreshTimer);
    if (!defer) {
      this.refreshTimer = null;
      this.refresh();
      return;
    }
    this.refreshTimer = setTimeout(() => this.refresh(), ConfigForm.TYPE_REFRESH_MS);
  }

  static shellToken(value) {
    const raw = String(value);
    return /^[\w@%+=:,./-]+$/.test(raw) ? raw : `'${raw.replace(/'/g, `'\\''`)}'`;
  }

  commandLine(base, join, detach) {
    let text = base;
    Object.entries(this.dirty).forEach(([path, value]) => {
      text += `${join}--${path} ${ConfigForm.shellToken(value)}`;
    });
    if (detach) text += `${join}--detach`;
    return text;
  }

  resetField(path) {
    const control = this.controls[path];
    delete this.dirty[path];
    if (control) control.reset();
    this.refresh();
  }

  resetAll() {
    this.dirty = {};
    Object.values(this.controls).forEach((control) => control.reset());
    this.refresh();
  }

  navigate(key) {
    this.setActiveSection(key);
  }

  sectionHidden(section) {
    if (!section.when) return false;
    const conditions = Array.isArray(section.when) ? section.when : [section.when];
    return conditions.some((condition) => {
      const leaf = this.byPath.get(condition.field);
      return !leaf || !condition.in.map(String).includes(String(this.effective(leaf)));
    });
  }

  setActiveSection(key) {
    const target = this.sections.find((section) => section.key === key && !this.sectionHidden(section));
    const fallback = this.sections.find((section) => !this.sectionHidden(section));
    this.activeSection = (target || fallback || this.sections[0]).key;

    this.sections.forEach((section) => {
      if (section.navBtn) section.navBtn.setAttribute("aria-current", String(section.key === this.activeSection));
    });
    this.applyVisibility();
  }

  refreshGates() {
    this.states.forEach(({ row }) => delete row.dataset.gated);
    this.gates.forEach((gate) => {
      if (this.effective(gate.leaf) !== "True") gate.states.forEach(({ row }) => (row.dataset.gated = "1"));
    });
    this.applyVisibility();
  }

  applyVisibility() {
    const searching = Boolean(this.query);
    if (this.layoutEl) this.layoutEl.classList.toggle("is-searching", searching);

    this.states.forEach(({ leaf, row }) => {
      const matches = !searching || leaf.path.toLowerCase().includes(this.query) || this.describe(leaf).toLowerCase().includes(this.query);
      row.hidden = !matches || row.dataset.gated === "1";
    });

    this.sweepShells();

    let anyVisible = false;
    const single = this.config && this.config.layout && this.config.layout.mode === "single";
    this.sections.forEach((section) => {
      const whenHidden = this.sectionHidden(section);
      const hasRows = this.states.some(({ row, sectionKey }) => sectionKey === section.key && !row.hidden);
      if (section.navBtn) {
        section.navBtn.hidden = whenHidden;
        section.navBtn.classList.toggle("is-empty", searching && !hasRows);
      }
      const show = !whenHidden && (searching ? hasRows : single || section.key === this.activeSection);
      section.el.hidden = !show;
      anyVisible = anyVisible || (show && (!searching || hasRows));
    });

    if (this.nomatchEl) this.nomatchEl.hidden = !searching || anyVisible;
  }

  shellsFor(row) {
    const cached = this.shellCache.get(row);
    if (cached) return cached;
    const shells = [];
    for (let node = row.parentElement; node && node !== this.layoutEl; node = node.parentElement) {
      if (node.classList.contains("band-block") || node.classList.contains("field-group") || node.classList.contains("cfg-panel")) shells.push(node);
    }
    this.shellCache.set(row, shells);
    return shells;
  }

  sweepShells() {
    if (!this.layoutEl) return;
    const filled = new Map();
    this.states.forEach(({ row }) => {
      this.shellsFor(row).forEach((shell) => filled.set(shell, (filled.get(shell) || false) || !row.hidden));
    });
    filled.forEach((visible, shell) => {
      shell.style.display = visible ? "" : "none";
    });
  }

  refreshBadges() {
    const counts = new Map();
    this.states.forEach(({ leaf, sectionKey, row }) => {
      const dirty = this.dirty[leaf.path] !== undefined;
      row.classList.toggle("is-dirty", dirty);
      if (dirty) counts.set(sectionKey, (counts.get(sectionKey) || 0) + 1);
    });

    this.sections.forEach((section) => {
      if (!section.badge) return;
      const n = counts.get(section.key) || 0;
      section.badge.hidden = n === 0;
      section.badge.textContent = n ? String(n) : "";
    });
  }

  refresh() {
    this.refreshBadges();
    this.refreshGates();
  }
}
