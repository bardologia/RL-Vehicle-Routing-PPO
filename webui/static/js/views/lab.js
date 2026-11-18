import { api } from "../api.js";
import { el, esc, fmtAge, icon, plural, Store } from "../format.js";
import { keys } from "../keys.js";
import { AgentPicker } from "../episode/agent_picker.js";
import { Vehicles } from "../episode/encoding.js";
import { KpiStrip } from "../episode/kpi_strip.js";
import { MapView } from "../episode/map_view.js";
import { errorBlock, statusChip } from "../ui/status.js";
import { DataTable } from "../ui/table.js";

const TOOLS = [
  { key: "select", label: "Select", iconName: "cursor", hint: "Click a job or a vehicle to edit it. Drag any marker to move it." },
  { key: "job", label: "Add job", iconName: "job", hint: "Click the map to place a job." },
  { key: "vehicle", label: "Add vehicle", iconName: "vehicle", hint: "Click the map to place a vehicle." },
  { key: "depot", label: "Move depot", iconName: "depot", hint: "Click the map to move the depot." },
];

const CHECK_TEXT = {
  job_on_vehicle: (check) => `job ${check.job_id} ends on vehicle ${check.vehicle_id}`,
  job_assigned: (check) => `job ${check.job_id} is assigned`,
  job_unassigned: (check) => `job ${check.job_id} stays unassigned`,
  job_not_on_vehicle: (check) => `job ${check.job_id} is not on vehicle ${check.vehicle_id}`,
  max_edits: (check) => `at most ${plural(check.count, "edit")} in the first round`,
};

export class LabView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.templates = null;
    this.saved = [];
    this.checkpoints = [];
    this.scenario = Store.read("lab.scenario", null);
    this.plan = null;
    this.solveState = { status: "idle", error: null };
    this.solveTimer = null;
    this.solveToken = 0;
    this.selected = null;
    this.tool = "select";
    this.built = false;
    this.notice = null;

    keys.bind("lab", "Escape", "Back to the select tool", () => this.setTool("select"));
    app.onHealth(() => {
      if (this.built) this.renderSummary();
    });
    app.theme.onChange(() => {
      if (this.map) this.map.restyle();
    });
  }

  build() {
    this.root.innerHTML = "";
    this.root.appendChild(el(`
      <div class="lab">
        <aside class="lab__left">
          <section class="lab__block" data-role="source"></section>
          <section class="lab__block lab__block--grow">
            <header class="lab__blockhead"><h3>Jobs</h3><span class="muted" data-role="job-count"></span></header>
            <div data-role="jobs" class="lab__table"></div>
          </section>
          <section class="lab__block lab__block--grow">
            <header class="lab__blockhead"><h3>Vehicles</h3><span class="muted" data-role="vehicle-count"></span></header>
            <div data-role="vehicles" class="lab__table"></div>
          </section>
        </aside>
        <div class="lab__map">
          <div class="lab__tools" role="toolbar" aria-label="Map tools"></div>
        </div>
        <aside class="lab__right">
          <section class="lab__block" data-role="editor"></section>
          <section class="lab__block" data-role="summary"></section>
          <section class="lab__block" data-role="run"></section>
        </aside>
      </div>`));
    const find = (role) => this.root.querySelector(`[data-role=${role}]`);
    this.sourceNode = find("source");
    this.editorNode = find("editor");
    this.summaryNode = find("summary");
    this.runNode = find("run");
    this.toolsNode = this.root.querySelector(".lab__tools");

    this.map = new MapView(this.root.querySelector(".lab__map"), {
      controls: ["fit", "buildings", "crossings"],
      onSelect: (entity) => this.select(entity.kind, entity.id),
      onIsolate: () => {},
      onMapClick: (lonLat) => this.onMapClick(lonLat),
      onDrag: (kind, id, lonLat) => this.onDrag(kind, id, lonLat),
    });
    this.map.setEditable(true);
    this.map.root.appendChild(this.toolsNode);

    this.jobTable = new DataTable(find("jobs"), {
      rowKey: (row) => row.id,
      emptyText: "No jobs. Use Add job on the map.",
      onSelect: (row) => this.select("job", row.id),
      onHover: (row) => this.map.highlight(row ? { jobId: row.id } : null),
      columns: [
        { key: "id", label: "Job", numeric: true },
        { key: "kind", label: "Kind" },
        { key: "priority", label: "Priority", numeric: true },
        { key: "vehicle", label: "Vehicle", sortValue: (row) => row.vehicle, render: (row) => (row.vehicle === null ? `<span class="muted">unassigned</span>` : Vehicles.swatch(row.vehicle)) },
      ],
    });
    this.vehicleTable = new DataTable(find("vehicles"), {
      rowKey: (row) => row.id,
      emptyText: "No vehicles. Use Add vehicle on the map.",
      onSelect: (row) => this.select("vehicle", row.id),
      onHover: (row) => this.map.highlight(row ? { vehicleId: row.id } : null),
      columns: [
        { key: "id", label: "Vehicle", render: (row) => Vehicles.swatch(row.id, row.stops === 0) },
        { key: "capacity", label: "Capacity", numeric: true },
        { key: "onboard", label: "Onboard", numeric: true },
        { key: "stops", label: "Stops", numeric: true },
      ],
    });

    this.sourceNode.addEventListener("change", (event) => this.onSourceChange(event));
    this.sourceNode.addEventListener("click", (event) => this.onSourceClick(event));
    this.editorNode.addEventListener("change", (event) => this.onEditorChange(event));
    this.editorNode.addEventListener("click", (event) => this.onEditorClick(event));
    this.summaryNode.addEventListener("click", (event) => {
      if (event.target.closest("[data-action=free]")) this.resolveFreely();
    });
    this.runNode.addEventListener("click", (event) => {
      const button = event.target.closest("[data-action=run]");
      if (button && !button.disabled) this.run();
    });
    this.toolsNode.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tool]");
      if (button) this.setTool(button.dataset.tool);
    });
    this.built = true;
  }

  async enter(parts, query) {
    if (!this.built) this.build();
    try {
      if (!this.templates) {
        const [templates, checkpoints] = await Promise.all([api.get("/api/scenario/templates"), api.get("/api/checkpoints")]);
        this.templates = templates.templates;
        this.checkpoints = checkpoints.checkpoints;
      }
      this.saved = (await api.get("/api/scenarios")).scenarios;
      if (query.episode) {
        const stored = await api.get(`/api/episodes/${encodeURIComponent(query.episode)}/scenario`);
        this.adopt({ ...stored, pristine: Boolean(stored.template) }, true);
        this.app.router.silent("#/lab");
      } else if (parts[0]) {
        const stored = await api.get(`/api/scenarios/${encodeURIComponent(parts[0])}`);
        this.adopt({ ...stored, pristine: Boolean(stored.template) }, true);
      }
    } catch (error) {
      this.notice = { cause: "The Lab could not load its data.", details: error.message };
    }
    if (!this.scenario) this.scenario = { name: null, template: null, pristine: false, jobs: [], vehicles: [], depot: [-46.6333, -23.5505], assignment: null };
    this.renderAll();
    this.map.map.invalidateSize({ animate: false });
    if (!this.plan) this.solveNow(true);
    else this.drawMap(true);
  }

  leave() {
    clearTimeout(this.solveTimer);
  }

  adopt(scenario, fit) {
    this.scenario = {
      name: scenario.name || null,
      template: scenario.template || null,
      pristine: Boolean(scenario.pristine),
      jobs: scenario.jobs,
      vehicles: scenario.vehicles,
      depot: scenario.depot,
      assignment: scenario.assignment || null,
    };
    this.selected = null;
    this.plan = null;
    this.persist();
    if (fit && this.map) this.map.resetFit();
  }

  persist() {
    Store.write("lab.scenario", this.scenario);
  }

  template() {
    return this.scenario.template ? this.templates.find((entry) => entry.key === this.scenario.template) || null : null;
  }

  touch() {
    this.scenario.pristine = false;
    this.persist();
  }

  renderAll() {
    this.renderTools();
    this.renderSource();
    this.renderTables();
    this.renderEditor();
    this.renderSummary();
    this.renderRun();
  }

  renderTools() {
    this.toolsNode.innerHTML = TOOLS.map((tool) => `<button type="button" class="btn btn--sm" data-tool="${tool.key}" aria-pressed="${this.tool === tool.key}">${icon(tool.iconName)}<span>${tool.label}</span></button>`).join("");
    this.map.setCaption(esc(TOOLS.find((tool) => tool.key === this.tool).hint));
    this.map.setCursor(this.tool === "select" ? "" : "place");
  }

  setTool(tool) {
    this.tool = tool;
    this.renderTools();
  }

  renderSource() {
    const scenario = this.scenario;
    const template = this.template();
    const templateOptions = (this.templates || []).map((entry) => `<option value="${esc(entry.key)}"${scenario.template === entry.key ? " selected" : ""}>${esc(entry.title)}</option>`).join("");
    const savedOptions = this.saved.map((entry) => `<option value="${esc(entry.name)}"${scenario.name === entry.name ? " selected" : ""}>${esc(entry.name)} (${plural(entry.jobs, "job")}, ${plural(entry.vehicles, "vehicle")}, ${fmtAge(entry.mtime)})</option>`).join("");
    const sampler = Store.read("lab.sampler", { jobs: 12, vehicles: 3, seed: 0 });
    const checks = template ? template.expectation.map((check) => CHECK_TEXT[check.kind](check)).join("; ") : "";

    this.sourceNode.innerHTML = `
      ${this.notice ? errorBlock(this.notice.cause, this.notice.details) : ""}
      <header class="lab__blockhead"><h3>Scenario</h3><span class="muted">${scenario.name ? `saved as ${esc(scenario.name)}` : template ? (scenario.pristine ? "template, unedited" : "template, edited") : "not saved"}</span></header>
      <label class="fieldlabel"><span>Template</span>
        <select class="select" data-role="template"><option value="">Choose a template</option>${templateOptions}</select>
      </label>
      ${template ? `<p class="lab__desc">${esc(template.description)}</p><p class="lab__desc"><b>Expected:</b> ${esc(template.expected)}</p><p class="lab__desc muted">Checked after a run: ${esc(checks)}.${scenario.pristine ? "" : " The scenario was edited, so the check may no longer apply."}</p>` : ""}
      <label class="fieldlabel"><span>Saved scenarios</span>
        <span class="fieldrow"><select class="select lab__grow" data-role="saved"><option value="">${this.saved.length ? "Choose a saved scenario" : "None saved yet"}</option>${savedOptions}</select>
        <button type="button" class="btn btn--sm" data-action="delete"${scenario.name ? "" : " disabled"} title="Delete the loaded saved scenario">${icon("trash")}<span>Delete</span></button></span>
      </label>
      <span class="fieldrow"><input class="input lab__grow" data-role="save-name" placeholder="Name, letters digits _ -" value="${esc(scenario.name || "")}" aria-label="Scenario name"><button type="button" class="btn btn--sm" data-action="save">${icon("save")}<span>Save as</span></button></span>
      <div data-role="source-error"></div>
      <div class="lab__sampler">
        <span class="fieldlabel"><span>Jobs</span><input class="input input--num" data-role="sample-jobs" inputmode="numeric" value="${sampler.jobs}"></span>
        <span class="fieldlabel"><span>Vehicles</span><input class="input input--num" data-role="sample-vehicles" inputmode="numeric" value="${sampler.vehicles}"></span>
        <span class="fieldlabel"><span>Seed</span><input class="input input--num" data-role="sample-seed" inputmode="numeric" value="${sampler.seed}"></span>
        <button type="button" class="btn btn--sm" data-action="sample">${icon("refresh")}<span>Sample</span></button>
      </div>`;
  }

  sourceError(cause, details) {
    this.sourceNode.querySelector("[data-role=source-error]").innerHTML = cause ? errorBlock(cause, details) : "";
  }

  async onSourceChange(event) {
    const role = event.target.dataset.role;
    if (role === "template" && event.target.value) {
      const template = this.templates.find((entry) => entry.key === event.target.value);
      this.notice = null;
      this.adopt({ name: null, template: template.key, pristine: true, jobs: structuredClone(template.jobs), vehicles: structuredClone(template.vehicles), depot: [...template.depot], assignment: structuredClone(template.assignment) }, true);
      this.renderAll();
      this.solveNow(true);
    }
    if (role === "saved" && event.target.value) {
      try {
        const stored = await api.get(`/api/scenarios/${encodeURIComponent(event.target.value)}`);
        this.notice = null;
        this.adopt({ ...stored, pristine: Boolean(stored.template) }, true);
        this.renderAll();
        this.solveNow(true);
      } catch (error) {
        this.sourceError("The saved scenario could not be loaded.", error.message);
      }
    }
  }

  async onSourceClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button || button.disabled) return;
    const value = (role) => this.sourceNode.querySelector(`[data-role=${role}]`).value.trim();
    this.sourceError(null);

    if (button.dataset.action === "sample") {
      const numbers = { jobs: Number(value("sample-jobs")), vehicles: Number(value("sample-vehicles")), seed: Number(value("sample-seed")) };
      if (![numbers.jobs, numbers.vehicles, numbers.seed].every((number) => Number.isInteger(number) && number >= 0) || numbers.jobs < 1 || numbers.vehicles < 1) {
        this.sourceError("Jobs, vehicles and seed must be whole numbers, with at least one job and one vehicle.");
        return;
      }
      Store.write("lab.sampler", numbers);
      try {
        const sampled = await api.post("/api/scenario/sample", { num_jobs: numbers.jobs, num_vehicles: numbers.vehicles, seed: numbers.seed });
        this.adopt({ name: null, template: null, pristine: false, ...sampled, assignment: null }, true);
        this.renderAll();
        this.solveNow(true);
      } catch (error) {
        this.sourceError("The sampler failed.", error.message);
      }
    }

    if (button.dataset.action === "save") {
      const name = value("save-name");
      if (!/^[A-Za-z0-9_-]{1,48}$/.test(name)) {
        this.sourceError("A name has 1 to 48 letters, digits, underscores or hyphens.");
        return;
      }
      const scenario = this.scenario;
      try {
        await api.put(`/api/scenarios/${encodeURIComponent(name)}`, { name, jobs: scenario.jobs, vehicles: scenario.vehicles, depot: scenario.depot, assignment: scenario.assignment, template: scenario.pristine ? scenario.template : null });
        this.saved = (await api.get("/api/scenarios")).scenarios;
      } catch (error) {
        this.sourceError("The scenario was not saved.", error.message);
        return;
      }
      scenario.name = name;
      this.persist();
      this.app.toast.confirm(`Scenario saved as ${name}`);
      this.renderSource();
    }

    if (button.dataset.action === "delete") {
      const name = this.scenario.name;
      try {
        await api.del(`/api/scenarios/${encodeURIComponent(name)}`);
        this.saved = (await api.get("/api/scenarios")).scenarios;
      } catch (error) {
        this.sourceError("The scenario was not deleted.", error.message);
        return;
      }
      this.scenario.name = null;
      this.persist();
      this.app.toast.confirm(`Scenario ${name} deleted`);
      this.renderSource();
    }
  }

  assignmentOfPlan() {
    const out = {};
    for (const vehicle of this.scenario.vehicles) out[vehicle.id] = [];
    if (!this.plan) return out;
    for (const route of this.plan.routes) {
      const seen = new Set();
      for (const stop of route.stops) {
        if (!seen.has(stop.job_id)) {
          seen.add(stop.job_id);
          out[route.vehicle_id].push(stop.job_id);
        }
      }
    }
    return out;
  }

  renderTables() {
    const assignment = this.assignmentOfPlan();
    const vehicleOf = new Map();
    for (const [vehicleId, jobIds] of Object.entries(assignment)) for (const jobId of jobIds) vehicleOf.set(jobId, Number(vehicleId));
    this.jobTable.setRows(this.scenario.jobs.map((job) => ({ id: job.id, kind: job.kind, priority: job.priority, vehicle: vehicleOf.has(job.id) ? vehicleOf.get(job.id) : null })));
    this.vehicleTable.setRows(this.scenario.vehicles.map((vehicle) => ({ id: vehicle.id, capacity: vehicle.capacity, onboard: vehicle.onboard, stops: (assignment[vehicle.id] || []).length })));
    this.root.querySelector("[data-role=job-count]").textContent = String(this.scenario.jobs.length);
    this.root.querySelector("[data-role=vehicle-count]").textContent = String(this.scenario.vehicles.length);
    if (this.selected) (this.selected.kind === "job" ? this.jobTable : this.vehicleTable).setSelected(this.selected.id);
  }

  entity() {
    if (!this.selected) return null;
    const list = this.selected.kind === "job" ? this.scenario.jobs : this.scenario.vehicles;
    return list.find((entry) => entry.id === this.selected.id) || null;
  }

  select(kind, id) {
    this.selected = { kind, id };
    this.jobTable.setSelected(kind === "job" ? id : null);
    this.vehicleTable.setSelected(kind === "vehicle" ? id : null);
    this.map.highlight(kind === "job" ? { jobId: id } : { vehicleId: id });
    this.renderEditor();
  }

  renderEditor() {
    const entity = this.entity();
    if (!entity) {
      this.editorNode.innerHTML = `<header class="lab__blockhead"><h3>Editor</h3></header><p class="muted">Select a job or a vehicle on the map or in the lists to edit every field.</p>`;
      return;
    }
    const number = (field, label, value, unit, half) => `<label class="fieldlabel${half ? " lab__half" : ""}"><span>${label}${unit ? ` (${unit})` : ""}</span><input class="input input--num" data-field="${field}" inputmode="decimal" value="${esc(value)}"></label>`;

    if (this.selected.kind === "job") {
      this.editorNode.innerHTML = `
        <header class="lab__blockhead"><h3>Job ${entity.id}</h3><button type="button" class="btn btn--sm" data-action="delete">${icon("trash")}<span>Delete</span></button></header>
        <div class="lab__fields">
          <label class="fieldlabel"><span>Kind</span><select class="select" data-field="kind"><option value="support"${entity.kind === "support" ? " selected" : ""}>support</option><option value="repossession"${entity.kind === "repossession" ? " selected" : ""}>repossession</option></select></label>
          <label class="fieldlabel"><span>Priority (1 to 5)</span><select class="select" data-field="priority">${[1, 2, 3, 4, 5].map((value) => `<option${entity.priority === value ? " selected" : ""}>${value}</option>`).join("")}</select></label>
          ${number("service", "Service", entity.service, "s")}
          ${number("setup", "Setup", entity.setup, "s")}
          ${number("amount", "Amount", entity.amount, "bikes")}
        </div>
        <p class="muted">Location ${entity.location[0].toFixed(4)}, ${entity.location[1].toFixed(4)}. Drag the marker to move it.</p>
        <div data-role="editor-error"></div>`;
      return;
    }

    const assigned = new Set(this.assignmentOfPlan()[entity.id] || []);
    this.editorNode.innerHTML = `
      <header class="lab__blockhead"><h3>Vehicle ${Vehicles.swatch(entity.id)}</h3><button type="button" class="btn btn--sm" data-action="delete">${icon("trash")}<span>Delete</span></button></header>
      <div class="lab__fields lab__fields--vehicle">
        ${number("capacity", "Capacity", entity.capacity, "bikes")}
        ${number("onboard", "Onboard", entity.onboard, "bikes")}
        ${number("speed_factor", "Speed factor", entity.speed_factor, "")}
        ${number("shift_start", "Shift start", entity.time_window[0], "s of day", true)}
        ${number("shift_end", "Shift end", entity.time_window[1], "s of day", true)}
      </div>
      <fieldset class="lab__assign"><legend>Assigned jobs (changing this pins the assignment)</legend>
        ${this.scenario.jobs.length ? this.scenario.jobs.map((job) => `<label class="check"><input type="checkbox" data-assign="${job.id}"${assigned.has(job.id) ? " checked" : ""}><span>${job.id}</span></label>`).join("") : `<span class="muted">no jobs</span>`}
      </fieldset>
      <div data-role="editor-error"></div>`;
  }

  onEditorChange(event) {
    const entity = this.entity();
    if (!entity) return;
    const target = event.target;
    const errorNode = this.editorNode.querySelector("[data-role=editor-error]");
    errorNode.innerHTML = "";

    if (target.dataset.assign !== undefined) {
      const jobId = Number(target.dataset.assign);
      const assignment = this.assignmentOfPlan();
      for (const key of Object.keys(assignment)) assignment[key] = assignment[key].filter((id) => id !== jobId);
      if (target.checked) assignment[entity.id].push(jobId);
      this.scenario.assignment = assignment;
      this.touch();
      this.scheduleSolve();
      return;
    }

    const field = target.dataset.field;
    if (!field) return;
    if (field === "kind") {
      entity.kind = target.value;
      entity.amount = target.value === "repossession" ? Math.max(1, entity.amount) : 0;
    } else {
      const value = Number(target.value);
      const whole = field !== "speed_factor";
      if (!Number.isFinite(value) || value < 0 || (whole && !Number.isInteger(value)) || (field === "speed_factor" && value <= 0)) {
        target.setAttribute("aria-invalid", "true");
        errorNode.innerHTML = errorBlock(whole ? "This field takes a whole number of zero or more." : "The speed factor is a positive number.");
        return;
      }
      target.setAttribute("aria-invalid", "false");
      if (field === "shift_start") entity.time_window = [value, entity.time_window[1]];
      else if (field === "shift_end") entity.time_window = [entity.time_window[0], value];
      else entity[field] = value;
    }
    this.touch();
    this.renderEditor();
    this.scheduleSolve();
  }

  onEditorClick(event) {
    if (!event.target.closest("[data-action=delete]")) return;
    const { kind, id } = this.selected;
    if (kind === "job") {
      this.scenario.jobs = this.scenario.jobs.filter((job) => job.id !== id);
      if (this.scenario.assignment) for (const key of Object.keys(this.scenario.assignment)) this.scenario.assignment[key] = this.scenario.assignment[key].filter((jobId) => jobId !== id);
    } else {
      this.scenario.vehicles = this.scenario.vehicles.filter((vehicle) => vehicle.id !== id);
      if (this.scenario.assignment) delete this.scenario.assignment[id];
    }
    this.selected = null;
    this.touch();
    this.renderEditor();
    this.scheduleSolve();
  }

  nextId(list) {
    return list.length ? Math.max(...list.map((entry) => entry.id)) + 1 : 0;
  }

  onMapClick(lonLat) {
    if (this.tool === "select") return;
    if (this.tool === "job") {
      const id = this.nextId(this.scenario.jobs);
      this.scenario.jobs.push({ id, location: lonLat, kind: "support", service: 600, setup: 0, amount: 0, priority: 3, description: `Job ${id}` });
      this.selected = { kind: "job", id };
    } else if (this.tool === "vehicle") {
      const id = this.nextId(this.scenario.vehicles);
      this.scenario.vehicles.push({ id, start: lonLat, capacity: 4, onboard: 0, speed_factor: 1.0, time_window: [28800, 72000], progress_credit: 0, description: `Vehicle ${id}` });
      if (this.scenario.assignment) this.scenario.assignment[id] = [];
      this.selected = { kind: "vehicle", id };
    } else {
      this.scenario.depot = lonLat;
      this.setTool("select");
    }
    this.touch();
    this.renderEditor();
    this.scheduleSolve();
  }

  onDrag(kind, id, lonLat) {
    if (kind === "depot") this.scenario.depot = lonLat;
    else if (kind === "job") this.scenario.jobs.find((job) => job.id === id).location = lonLat;
    else this.scenario.vehicles.find((vehicle) => vehicle.id === id).start = lonLat;
    this.touch();
    if (this.selected && this.selected.kind === kind && this.selected.id === id) this.renderEditor();
    this.scheduleSolve();
  }

  resolveFreely() {
    this.scenario.assignment = null;
    this.touch();
    this.solveNow(false);
  }

  scheduleSolve() {
    clearTimeout(this.solveTimer);
    this.solveState = { status: "solving", error: null };
    this.renderSummary();
    this.renderSource();
    this.solveTimer = setTimeout(() => this.solveNow(false), 400);
  }

  emptyPlan() {
    return { routes: [], idle: [], idle_vehicle_ids: this.scenario.vehicles.map((vehicle) => vehicle.id), unassigned: this.scenario.jobs.map((job) => ({ job_id: job.id, priority: job.priority, kind: job.kind, waiting_s: 0 })), kpis: null };
  }

  async solveNow(fit) {
    clearTimeout(this.solveTimer);
    const scenario = this.scenario;
    this.solveToken += 1;
    const token = this.solveToken;

    if (!scenario.jobs.length || !scenario.vehicles.length) {
      this.plan = null;
      this.solveState = { status: "incomplete", error: null };
      this.afterSolve(fit);
      return;
    }
    if (!this.app.servicesUp && this.app.health) {
      this.plan = null;
      this.solveState = { status: "down", error: null };
      this.afterSolve(fit);
      return;
    }

    this.solveState = { status: "solving", error: null };
    this.renderSummary();
    try {
      const solved = await api.post("/api/scenario/solve", { jobs: scenario.jobs, vehicles: scenario.vehicles, assignment: scenario.assignment, depot: scenario.depot });
      if (token !== this.solveToken) return;
      this.plan = solved.plan;
      this.solveState = { status: "solved", error: null };
    } catch (error) {
      if (token !== this.solveToken) return;
      this.plan = null;
      this.solveState = { status: "error", error: error.message };
    }
    this.afterSolve(fit);
  }

  afterSolve(fit) {
    this.renderSource();
    this.renderTables();
    this.renderSummary();
    this.renderRun();
    if (this.selected && this.selected.kind === "vehicle") this.renderEditor();
    this.drawMap(fit);
  }

  drawMap(fit) {
    const scenario = this.scenario;
    const jobs = {};
    for (const job of scenario.jobs) jobs[job.id] = job;
    const vehicles = {};
    for (const vehicle of scenario.vehicles) vehicles[vehicle.id] = { start: vehicle.start, onboard: vehicle.onboard };
    if (fit) this.map.resetFit();
    this.map.setScene({ depot: scenario.depot, jobs, vehicles, plan: this.plan || this.emptyPlan(), pathOf: (route) => route.path || [] });
    if (this.selected) this.map.highlight(this.selected.kind === "job" ? { jobId: this.selected.id } : { vehicleId: this.selected.id });
  }

  renderSummary() {
    const state = this.solveState;
    const pinned = Boolean(this.scenario && this.scenario.assignment);
    let body;
    if (state.status === "solving") body = `<p class="muted" role="status">solving</p>`;
    else if (state.status === "error") body = errorBlock("VROOM could not solve this scenario.", state.error);
    else if (state.status === "down") body = `<div class="noteblock">${statusChip("DOWN", "routing services")}<span>Solving needs OSRM and VROOM. Start them from the repository root:</span><code>sg docker -c "docker compose up -d"</code></div>`;
    else if (state.status === "incomplete") body = `<p class="muted">Add at least one job and one vehicle to get a plan.</p>`;
    else if (this.plan) body = KpiStrip.planHtml(this.plan.kpis);
    else body = "";

    this.summaryNode.innerHTML = `
      <header class="lab__blockhead"><h3>Plan</h3><span class="muted">${pinned ? "assignment pinned" : "assigned by VROOM"}</span></header>
      ${body}
      ${pinned ? `<button type="button" class="btn btn--sm" data-action="free">${icon("refresh")}<span>Re-solve freely</span></button>` : ""}`;
  }

  renderRun() {
    if (!this.runNode.querySelector("[data-role=picker]")) {
      const saved = Store.read("lab.run", { rounds: 2, edits: 8, seed: 0, disruption: false });
      this.runNode.innerHTML = `
        <header class="lab__blockhead"><h3>Run agents</h3><span class="muted">opens Episode</span></header>
        <div data-role="picker"></div>
        <div class="lab__fields">
          <label class="fieldlabel"><span>Rounds</span><input class="input input--num" data-run="rounds" inputmode="numeric" value="${saved.rounds}"></label>
          <label class="fieldlabel"><span>Edits per round</span><input class="input input--num" data-run="edits" inputmode="numeric" value="${saved.edits}"></label>
          <label class="fieldlabel"><span>Seed</span><input class="input input--num" data-run="seed" inputmode="numeric" value="${saved.seed}"></label>
        </div>
        <label class="check"><input type="checkbox" data-run="disruption"${saved.disruption ? " checked" : ""}><span>Apply an initial disruption</span></label>
        <div class="lab__runrow"><button type="button" class="btn btn--primary" data-action="run">${icon("play")}<span>Run agents</span></button><span class="muted" data-role="run-note"></span></div>
        <div data-role="run-error"></div>`;
      this.picker = new AgentPicker(this.runNode.querySelector("[data-role=picker]"), { checkpoints: this.checkpoints, preferredRun: this.app.context.run, storeKey: "lab.agents", teacherDefault: true });
    }
    const ready = this.solveState.status === "solved";
    this.runNode.querySelector("[data-action=run]").disabled = !ready;
    this.runNode.querySelector("[data-role=run-note]").textContent = ready ? "" : "Needs a solved plan.";
  }

  async run() {
    const errorNode = this.runNode.querySelector("[data-role=run-error]");
    errorNode.innerHTML = "";
    const read = (name) => this.runNode.querySelector(`[data-run=${name}]`);
    const numbers = { rounds: Number(read("rounds").value), edits: Number(read("edits").value), seed: Number(read("seed").value) };
    if (!Number.isInteger(numbers.rounds) || numbers.rounds < 1 || numbers.rounds > 10 || !Number.isInteger(numbers.edits) || numbers.edits < 1 || numbers.edits > 8 || !Number.isInteger(numbers.seed) || numbers.seed < 0) {
      errorNode.innerHTML = errorBlock("Rounds are 1 to 10, edits per round 1 to 8, and the seed a whole number of zero or more.");
      return;
    }
    const disruption = read("disruption").checked;
    Store.write("lab.run", { ...numbers, disruption });
    const agents = this.picker.agents();
    if (!agents.length) {
      errorNode.innerHTML = errorBlock("Select at least one agent.");
      return;
    }
    const scenario = this.scenario;
    let created;
    try {
      created = await api.post("/api/episodes", {
        source: { kind: "scenario", name: scenario.name, template: scenario.pristine ? scenario.template : null, jobs: scenario.jobs, vehicles: scenario.vehicles, depot: scenario.depot, assignment: scenario.assignment, seed: numbers.seed, initial_disruption: disruption },
        caps: { rounds: numbers.rounds, edits: numbers.edits },
        agents,
      });
    } catch (error) {
      errorNode.innerHTML = errorBlock("The agents were not started.", error.message);
      return;
    }
    this.app.router.go(`#/episode/ep/${created.episode_id}/${encodeURIComponent(created.agent_keys[0])}/0`);
  }
}
