import { api } from "../api.js";
import { esc } from "../format.js";

export class NumberField {
  constructor(view, leaf, short, spec = null) {
    this.view = view;
    this.leaf = leaf;
    this.short = short || leaf.path.split(".").pop();
    this.integer = leaf.type === "int" || Boolean(spec && spec.int);
    this.unset = leaf.value === "";
    this.default = Number.isFinite(Number(leaf.value)) ? Number(leaf.value) : 0;
    this.log = false;
    this.range = this.resolve(spec);
    this.input = null;
    this.chips = new Map();
    this.reset = () => this.paint();
  }

  build() {
    const el = document.createElement("div");
    el.className = "numfield";

    const input = document.createElement("input");
    input.className = "cfg-input numfield__input";
    input.type = "text";
    input.inputMode = this.integer ? "numeric" : "decimal";
    input.spellcheck = false;
    input.setAttribute("aria-label", this.leaf.path);
    this.input = input;

    input.addEventListener("input", () => {
      const raw = input.value.trim();
      const invalid = raw !== "" && (this.integer ? !/^-?\d+$/.test(raw) : !Number.isFinite(Number(raw)));

      input.classList.toggle("is-invalid", invalid);
      input.title = invalid ? (this.integer ? "not a whole number; this field takes an integer" : "not a number") : "";

      if (raw === "" || invalid) this.view.unsetValue(this.leaf, true);
      else this.view.setValue(this.leaf, !this.unset && Number(raw) === this.default ? this.leaf.value : raw, true);
      this.mark();
    });
    el.appendChild(input);

    if (this.range.presets.length) {
      const presets = document.createElement("div");
      presets.className = "segmented numfield__presets";
      presets.setAttribute("role", "group");
      presets.setAttribute("aria-label", `Presets for ${this.short}`);
      this.range.presets.forEach((value) => presets.appendChild(this.chip(value)));
      el.appendChild(presets);
    }

    this.paint();
    return { el, input, reset: this.reset };
  }

  resolve(spec) {
    const range = spec ? { min: spec.min, max: spec.max, step: spec.step || 1, log: Boolean(spec.log), presets: spec.presets ? spec.presets.slice() : this.spanPresets(spec) } : { min: -Infinity, max: Infinity, step: 1, log: false, presets: [] };

    this.log = range.log;
    if (spec && !this.unset) {
      range.min = Math.min(range.min, this.default);
      range.max = Math.max(range.max, this.default);
      range.presets.push(this.default);
    }
    range.presets = this.cleanPresets(range.presets, range);
    return range;
  }

  spanPresets(spec) {
    const step = spec.step || 1;
    const span = spec.max - spec.min;
    const snap = (x) => spec.min + Math.round((x - spec.min) / step) * step;
    return [0, 0.25, 0.5, 0.75, 1].map((f) => snap(spec.min + span * f));
  }

  cleanPresets(list, range) {
    const within = list.filter((x) => Number.isFinite(x) && x >= range.min - 1e-12 && x <= range.max + 1e-12);
    const seen = new Map();
    within.forEach((x) => {
      const key = this.format(x);
      if (!seen.has(key)) seen.set(key, x);
    });
    return [...seen.values()].sort((a, b) => a - b).slice(0, 8);
  }

  format(value) {
    if (this.integer) return String(Math.round(value));
    if (value === 0) return "0";
    return String(Number(value.toPrecision(this.log ? 2 : 6)));
  }

  chip(value) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.textContent = this.format(value);
    chip.title = `set ${this.short} = ${this.format(value)}`;
    chip.addEventListener("click", () => {
      this.input.value = this.format(value);
      this.input.classList.remove("is-invalid");
      const out = !this.unset && Math.abs(value - this.default) <= Math.abs(this.default) * 1e-12 ? this.leaf.value : this.format(value);
      this.view.setValue(this.leaf, out);
      this.mark();
    });
    this.chips.set(value, chip);
    return chip;
  }

  mark() {
    const current = Number(this.view.effective(this.leaf));
    const dirty = this.view.dirty[this.leaf.path] !== undefined;
    this.input.classList.toggle("is-dirty", dirty);
    this.chips.forEach((chip, key) => {
      const tolerance = this.integer ? 0.5 : Math.max(1e-15, Math.abs(current) * 1e-6);
      chip.setAttribute("aria-pressed", String(Number.isFinite(current) && this.view.effective(this.leaf) !== "" && Math.abs(key - current) < tolerance));
    });
  }

  paint() {
    this.input.value = this.view.effective(this.leaf);
    this.input.classList.remove("is-invalid");
    this.input.title = "";
    this.mark();
  }
}

export class MultiValueField {
  constructor(view, leaf, spec) {
    this.view = view;
    this.leaf = leaf;
    this.spec = spec;
    this.el = null;
    this.chips = null;
    this.count = null;
    this.reset = () => this.paint();
  }

  build() {
    this.el = document.createElement("div");
    this.el.className = "multivalue";

    this.chips = document.createElement("div");
    this.chips.className = "chips multivalue__chips";
    this.el.appendChild(this.chips);

    this.count = document.createElement("span");
    this.count.className = "cfg-note multivalue__note";
    this.el.appendChild(this.count);

    this.paint();
    return { el: this.el, input: this.chips, reset: this.reset };
  }

  values() {
    return this.view.effective(this.leaf).split(",").map((part) => part.trim()).filter(Boolean);
  }

  emit(values) {
    this.view.setValue(this.leaf, values.join(", "));
    this.paint();
  }

  toggle(value) {
    const values = this.values();
    const index = values.indexOf(value);
    if (index >= 0) values.splice(index, 1);
    else values.push(value);
    this.emit(this.spec.choices.map(([choice]) => choice).filter((choice) => values.includes(choice)));
  }

  paint() {
    const values = this.values();
    this.chips.innerHTML = "";

    this.spec.choices.forEach(([value, label]) => {
      const on = values.includes(value);
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = label;
      chip.title = `--${this.leaf.path} · ${value}`;
      chip.setAttribute("aria-pressed", String(on));
      chip.addEventListener("click", () => this.toggle(value));
      this.chips.appendChild(chip);
    });

    this.count.textContent = values.length ? `${values.length} of ${this.spec.choices.length}` : this.spec.empty || "select at least one value";
    this.count.classList.toggle("is-warn", !values.length);
    this.el.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
  }
}

export class RunPicker {
  static runs = null;
  static loading = null;

  static async list() {
    if (RunPicker.runs) return RunPicker.runs;
    if (!RunPicker.loading) {
      RunPicker.loading = api.get("/api/runs").then((data) => {
        RunPicker.runs = data.runs;
        return RunPicker.runs;
      }, () => {
        RunPicker.loading = null;
        return [];
      });
    }
    return RunPicker.loading;
  }

  static forget() {
    RunPicker.runs = null;
    RunPicker.loading = null;
  }

  constructor(view, leaf) {
    this.view = view;
    this.leaf = leaf;
    this.input = null;
    this.note = null;
    this.reset = () => this.paint();
  }

  build() {
    const el = document.createElement("div");
    el.className = "runpicker";
    const listId = `runs-${this.leaf.path.replace(/\W/g, "-")}-${Math.random().toString(36).slice(2, 8)}`;

    this.input = document.createElement("input");
    this.input.className = "cfg-input runpicker__input";
    this.input.type = "text";
    this.input.spellcheck = false;
    this.input.placeholder = this.leaf.nullable ? "none" : "";
    this.input.setAttribute("list", listId);
    this.input.setAttribute("aria-label", this.leaf.path);

    const list = document.createElement("datalist");
    list.id = listId;

    this.note = document.createElement("span");
    this.note.className = "cfg-note runpicker__note";

    this.input.addEventListener("input", () => {
      this.view.setValue(this.leaf, this.input.value.trim(), true);
      this.describe();
    });

    el.append(this.input, list, this.note);
    RunPicker.list().then((runs) => {
      this.runs = runs;
      list.innerHTML = runs.map((run) => `<option value="${esc(run.name)}">${esc(RunPicker.label(run))}</option>`).join("");
      this.describe();
    });

    this.paint();
    return { el, input: this.input, reset: this.reset };
  }

  describe() {
    const value = this.view.effective(this.leaf);
    if (!this.runs || !value) {
      this.note.textContent = "";
      return;
    }
    const run = this.runs.find((entry) => entry.name === value);
    this.note.textContent = run ? RunPicker.label(run) : "no run of that name under runs/";
    this.note.classList.toggle("is-warn", !run || (run.has_checkpoint && !run.loadable));
  }

  static label(run) {
    const checkpoint = !run.has_checkpoint ? "no checkpoint" : run.loadable ? "loadable" : "incompatible";
    const reports = run.reports.length ? `${run.reports.length} report${run.reports.length === 1 ? "" : "s"}` : "no report";
    return [run.kind, checkpoint, reports].join(" · ");
  }

  paint() {
    this.input.value = this.view.effective(this.leaf);
    this.input.classList.toggle("is-dirty", this.view.dirty[this.leaf.path] !== undefined);
    this.describe();
  }
}
