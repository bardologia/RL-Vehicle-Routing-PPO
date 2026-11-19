import { api } from "../api.js";
import { el, esc, fmtAge, icon, plural, Store } from "../format.js";
import { keys } from "../keys.js";
import { AgentPicker } from "../episode/agent_picker.js";
import { Vehicles } from "../episode/encoding.js";
import { KpiStrip } from "../episode/kpi_strip.js";
import { MapView } from "../episode/map_view.js";
import { infoText } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { errorBlock } from "../ui/status.js";
import { FigureStates } from "../ui/figure_states.js";
import { DataTable } from "../ui/table.js";

const TOOLS = [
  { key: "select", label: "Select", iconName: "cursor", hint: "Click a job or a vehicle to edit it. Drag any marker to move it. Key Escape." },
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
          <div class="segmented lab__tools" role="radiogroup" aria-label="Map tool"></div>
        </div>
        <aside class="lab__right">
          <div class="fig-band lab__band">${HairlineFigure.html("fleet")}</div>
          <section class="lab__block" data-role="editor"></section>
          <section class="lab__block" data-role="summary"></section>
          <section class="lab__block" data-role="run"></section>
          <footer class="lab__foot" data-role="runfoot">
            <button type="button" class="btn btn--primary" data-action="run" data-hint-title="Run agents" data-hint="Records one episode of this scenario with every selected agent and opens it in Episode.">${icon("play")}<span>Run agents</span></button>
            <span class="lab__footnote" data-role="run-note"></span>
          </footer>
        </aside>
      </div>`));
    const find = (role) => this.root.querySelector(`[data-role=${role}]`);
    this.sourceNode = find("source");
    this.editorNode = find("editor");
    this.summaryNode = find("summary");
    this.runNode = find("run");
    this.footNode = find("runfoot");
    this.toolsNode = this.root.querySelector(".lab__tools");
    this.fleet = this.root.querySelector("hairline-figure[name=fleet]");

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
      emptyText: "No jobs yet.",
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
      emptyText: "No vehicles yet.",
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
      const more = event.target.closest("[data-action=plan-more]");
      if (more && this.plan) Popover.toggle(more, KpiStrip.planMoreHtml(this.plan.kpis), { label: "Plan measures" });
    });
    this.runNode.addEventListener("click", (event) => {
      const option = event.target.closest("button[data-disruption]");
      if (!option) return;
      for (const button of this.runNode.querySelectorAll("button[data-disruption]")) button.setAttribute("aria-checked", String(button === option));
    });
    this.footNode.addEventListener("click", (event) => {
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
    this.toolsNode.innerHTML = TOOLS.map((tool) => `<button type="button" role="radio" data-tool="${tool.key}" aria-checked="${this.tool === tool.key}" data-hint-title="${esc(tool.label)}" data-hint="${esc(tool.hint)}">${icon(tool.iconName)}<span>${tool.label}</span></button>`).join("");
    this.map.setCursor(this.tool === "select" ? "" : "place");
  }

  setTool(tool) {
    this.tool = tool;
    this.renderTools();
  }

  scenarioNote() {
    const scenario = this.scenario;
    if (scenario.name) return `saved as ${scenario.name}`;
    if (this.template()) return scenario.pristine ? "template, unedited" : "template, edited";
    return "not saved";
  }

  renderSource() {
    const scenario = this.scenario;
    const template = this.template();
    const templateOptions = (this.templates || []).map((entry) => `<option value="${esc(entry.key)}"${scenario.template === entry.key ? " selected" : ""}>${esc(entry.title)}</option>`).join("");
    const checks = template ? template.expectation.map((check) => CHECK_TEXT[check.kind](check)).join("; ") : "";
    const info = template ? Popover.infoButton(template.title, infoText(template.description, `Checked after a run: ${checks}.`, scenario.pristine ? null : "The scenario was edited, so the check may no longer apply.")) : "";

    this.sourceNode.innerHTML = `
      ${this.notice ? errorBlock(this.notice.cause, this.notice.details) : ""}
      <header class="lab__blockhead"><h3>Scenario</h3><span class="muted">${esc(this.scenarioNote())}</span><span class="lab__info">${info}</span></header>
      <label class="fieldlabel"><span>Template</span>
        <select class="select" data-role="template"><option value="">Choose a template</option>${templateOptions}</select>
      </label>
      ${template ? `<p class="lab__expect">Expected: ${esc(template.expected)}</p>` : ""}
      <div class="lab__srcbtns">
        <button type="button" class="btn btn--sm" data-action="saved" aria-haspopup="dialog" aria-expanded="false">Saved<span class="tabs__count">${this.saved.length}</span></button>
        <button type="button" class="btn btn--sm" data-action="sampler" aria-haspopup="dialog" aria-expanded="false">Sample</button>
      </div>
      <div data-role="source-error"></div>`;
  }

  savedHtml() {
    const rows = this.saved.map((entry) => `
      <li class="lab__saved${this.scenario.name === entry.name ? " is-current" : ""}">
        <button type="button" class="lab__savedname" data-open="${esc(entry.name)}">${esc(entry.name)}</button>
        <span class="lab__savedmeta">${plural(entry.jobs, "job")}, ${plural(entry.vehicles, "vehicle")}, ${esc(fmtAge(entry.mtime))}</span>
        <button type="button" class="tbtn tbtn--sm" data-delete="${esc(entry.name)}" aria-label="Delete ${esc(entry.name)}" title="Delete ${esc(entry.name)}">${icon("trash")}</button>
      </li>`).join("");
    return `
      <header class="popover__head"><h3>Saved scenarios</h3></header>
      ${this.saved.length ? `<ul class="lab__savedlist">${rows}</ul>` : `<p class="muted">None saved yet.</p>`}
      <div class="lab__saveas"><input class="input" data-role="save-name" placeholder="name, letters digits _ -" value="${esc(this.scenario.name || "")}" aria-label="Scenario name"><button type="button" class="btn btn--sm" data-action="save">Save as</button></div>
      <div data-role="popover-error"></div>`;
  }

  samplerHtml() {
    const sampler = Store.read("lab.sampler", { jobs: 12, vehicles: 3, seed: 0 });
    const field = (role, label, value, ch) => `<label class="ctl"><span class="ctl__label">${label}</span><input class="input input--num" data-role="${role}" inputmode="numeric" style="--ch:${ch}" value="${value}"></label>`;
    return `
      <header class="popover__head"><h3>Sample a scenario</h3></header>
      <div class="ctls">${field("sample-jobs", "Jobs", sampler.jobs, 3)}${field("sample-vehicles", "Vehicles", sampler.vehicles, 3)}${field("sample-seed", "Seed", sampler.seed, 6)}</div>
      <div class="popover__foot"><button type="button" class="btn btn--sm" data-action="sample">Sample</button></div>
      <div data-role="popover-error"></div>`;
  }

  sourceError(cause, details) {
    const target = Popover.current && this.sourceNode.contains(Popover.current.anchor) ? Popover.current.node.querySelector("[data-role=popover-error]") : this.sourceNode.querySelector("[data-role=source-error]");
    target.innerHTML = cause ? errorBlock(cause, details) : "";
    Popover.reflow();
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
  }

  async openSaved(name) {
    try {
      const stored = await api.get(`/api/scenarios/${encodeURIComponent(name)}`);
      this.notice = null;
      Popover.close();
      this.adopt({ ...stored, pristine: Boolean(stored.template) }, true);
      this.renderAll();
      this.solveNow(true);
    } catch (error) {
      this.sourceError("The saved scenario could not be loaded.", error.message);
    }
  }

  async onPopoverClick(event) {
    const value = (role) => Popover.current.node.querySelector(`[data-role=${role}]`).value.trim();
    const open = event.target.closest("[data-open]");
    if (open) {
      this.openSaved(open.dataset.open);
      return;
    }
    const remove = event.target.closest("[data-delete]");
    if (remove) {
      const name = remove.dataset.delete;
      try {
        await api.del(`/api/scenarios/${encodeURIComponent(name)}`);
        this.saved = (await api.get("/api/scenarios")).scenarios;
      } catch (error) {
        this.sourceError("The scenario was not deleted.", error.message);
        return;
      }
      if (this.scenario.name === name) {
        this.scenario.name = null;
        this.persist();
      }
      this.app.toast.confirm(`Scenario ${name} deleted`);
      this.reopenSource("saved");
      return;
    }
    const button = event.target.closest("button[data-action]");
    if (!button) return;

    if (button.dataset.action === "sample") {
      const numbers = { jobs: Number(value("sample-jobs")), vehicles: Number(value("sample-vehicles")), seed: Number(value("sample-seed")) };
      if (![numbers.jobs, numbers.vehicles, numbers.seed].every((number) => Number.isInteger(number) && number >= 0) || numbers.jobs < 1 || numbers.vehicles < 1) {
        this.sourceError("Jobs, vehicles and seed must be whole numbers, with at least one job and one vehicle.");
        return;
      }
      Store.write("lab.sampler", numbers);
      try {
        const sampled = await api.post("/api/scenario/sample", { num_jobs: numbers.jobs, num_vehicles: numbers.vehicles, seed: numbers.seed });
        Popover.close();
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
      Popover.close();
      this.renderSource();
    }
  }

  reopenSource(action) {
    Popover.close();
    this.renderSource();
    this.toggleSourcePopover(this.sourceNode.querySelector(`[data-action=${action}]`));
  }

  toggleSourcePopover(button) {
    const saved = button.dataset.action === "saved";
    const node = Popover.toggle(button, saved ? this.savedHtml() : this.samplerHtml(), { label: saved ? "Saved scenarios" : "Sample a scenario", className: saved ? "popover--wide" : "" });
    if (node) node.addEventListener("click", (event) => this.onPopoverClick(event));
  }

  onSourceClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button || button.disabled) return;
    this.sourceError(null);
    if (button.dataset.action === "saved" || button.dataset.action === "sampler") this.toggleSourcePopover(button);
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
      this.editorNode.innerHTML = `<p class="lab__hint">Select a job or a vehicle to edit it; drag a marker to move it.</p>`;
      return;
    }
    const field = (name, label, value, unit, span) => `<label class="fieldlabel lab__field" style="--span:${span}"><span>${label}${unit ? `<small>${unit}</small>` : ""}</span><input class="input input--num" data-field="${name}" inputmode="decimal" value="${esc(value)}"></label>`;
    const remove = `<button type="button" class="tbtn tbtn--sm" data-action="delete" aria-label="Delete" title="Delete">${icon("trash")}</button>`;

    if (this.selected.kind === "job") {
      this.editorNode.innerHTML = `
        <header class="lab__blockhead"><h3>Job ${entity.id}</h3><span class="muted">${entity.location[0].toFixed(4)}, ${entity.location[1].toFixed(4)}</span>${remove}</header>
        <div class="lab__fields">
          <label class="fieldlabel lab__field" style="--span:3"><span>Kind</span><select class="select" data-field="kind"><option value="support"${entity.kind === "support" ? " selected" : ""}>support</option><option value="repossession"${entity.kind === "repossession" ? " selected" : ""}>repossession</option></select></label>
          <label class="fieldlabel lab__field" style="--span:3" title="1 to 5"><span>Priority</span><select class="select" data-field="priority">${[1, 2, 3, 4, 5].map((value) => `<option${entity.priority === value ? " selected" : ""}>${value}</option>`).join("")}</select></label>
          ${field("service", "Service", entity.service, "s", 2)}
          ${field("setup", "Setup", entity.setup, "s", 2)}
          ${field("amount", "Bikes", entity.amount, "", 2)}
        </div>
        <div data-role="editor-error"></div>`;
      return;
    }

    const assigned = new Set(this.assignmentOfPlan()[entity.id] || []);
    this.editorNode.innerHTML = `
      <header class="lab__blockhead"><h3>Vehicle ${Vehicles.swatch(entity.id)}</h3><span class="muted">${entity.start[0].toFixed(4)}, ${entity.start[1].toFixed(4)}</span>${remove}</header>
      <div class="lab__fields">
        ${field("capacity", "Capacity", entity.capacity, "", 2)}
        ${field("onboard", "Onboard", entity.onboard, "", 2)}
        ${field("speed_factor", "Speed", entity.speed_factor, "", 2)}
        ${field("shift_start", "Shift start", entity.time_window[0], "s", 3)}
        ${field("shift_end", "Shift end", entity.time_window[1], "s", 3)}
      </div>
      <div class="lab__assign" role="group" aria-label="Assigned jobs">
        <span class="ctl__label" data-hint-title="Assigned jobs" data-hint="Changing the assignment pins it; Re-solve freely hands it back to VROOM.">Assigned jobs</span>
        <span class="chips">${this.scenario.jobs.length ? this.scenario.jobs.map((job) => `<button type="button" class="chip" data-assign="${job.id}" aria-pressed="${assigned.has(job.id)}">${job.id}</button>`).join("") : `<span class="lab__hint">no jobs</span>`}</span>
      </div>
      <div data-role="editor-error"></div>`;
  }

  toggleAssign(jobId, on) {
    const entity = this.entity();
    const assignment = this.assignmentOfPlan();
    for (const key of Object.keys(assignment)) assignment[key] = assignment[key].filter((id) => id !== jobId);
    if (on) assignment[entity.id].push(jobId);
    this.scenario.assignment = assignment;
    this.touch();
    this.scheduleSolve();
  }

  onEditorChange(event) {
    const entity = this.entity();
    if (!entity) return;
    const target = event.target;
    const errorNode = this.editorNode.querySelector("[data-role=editor-error]");
    errorNode.innerHTML = "";

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
    const chip = event.target.closest("button[data-assign]");
    if (chip) {
      const on = chip.getAttribute("aria-pressed") !== "true";
      chip.setAttribute("aria-pressed", String(on));
      this.toggleAssign(Number(chip.dataset.assign), on);
      return;
    }
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
    FigureStates.write(this.fleet, FigureStates.fleet(this.plan, this.scenario, state.status === "solving"));
    const pinned = Boolean(this.scenario && this.scenario.assignment);
    const reopen = Popover.current && this.summaryNode.contains(Popover.current.anchor);
    let body;
    if (state.status === "solving") body = `<p class="lab__hint" role="status">solving</p>`;
    else if (state.status === "error") body = errorBlock("VROOM could not solve this scenario.", state.error);
    else if (state.status === "down") body = `<div class="noteblock"><span>Solving needs OSRM and VROOM. Start them from the repository root: <code>sg docker -c "docker compose up -d"</code></span></div>`;
    else if (state.status === "incomplete") body = `<p class="lab__hint">Add at least one job and one vehicle to get a plan.</p>`;
    else if (this.plan) body = KpiStrip.planHtml(this.plan.kpis);
    else body = "";
    const more = this.plan && state.status === "solved" ? `<button type="button" class="btn btn--sm" data-action="plan-more" aria-haspopup="dialog" aria-expanded="false">More</button>` : "";

    this.summaryNode.innerHTML = `
      <header class="lab__blockhead"><h3 data-hint-title="Plan" data-hint="${pinned ? "Assignment pinned by hand; VROOM orders the stops." : "Assigned and ordered by VROOM."}">Plan</h3>${pinned ? `<span class="muted">pinned</span>` : ""}</header>
      ${body}
      ${more || pinned ? `<div class="lab__planactions">${more}${pinned ? `<button type="button" class="btn btn--sm" data-action="free">${icon("refresh")}<span>Re-solve freely</span></button>` : ""}</div>` : ""}`;
    if (reopen) Popover.close();
  }

  renderRun() {
    if (!this.runNode.querySelector("[data-role=picker]")) {
      const saved = Store.read("lab.run", { rounds: 2, edits: 8, seed: 0, disruption: false });
      const field = (name, label, value, ch) => `<label class="ctl"><span class="ctl__label">${label}</span><input class="input input--num" data-run="${name}" inputmode="numeric" style="--ch:${ch}" value="${value}"></label>`;
      const option = (value, label) => `<button type="button" role="radio" data-disruption="${value}" aria-checked="${String(saved.disruption) === value}">${label}</button>`;
      this.runNode.innerHTML = `
        <header class="lab__blockhead"><h3>Run agents</h3></header>
        <div data-role="picker"></div>
        <div class="ctls lab__runfields">
          ${field("rounds", "Rounds", saved.rounds, 2)}${field("edits", "Edits", saved.edits, 2)}${field("seed", "Seed", saved.seed, 6)}
          <span class="ctl" role="group" aria-label="Initial disruption"><span class="ctl__label" data-hint-title="Initial disruption" data-hint="Apply a disruption before the first round.">Disruption</span><span class="segmented" role="radiogroup" aria-label="Initial disruption">${option("false", "off")}${option("true", "on")}</span></span>
        </div>
        <div data-role="run-error"></div>`;
      this.picker = new AgentPicker(this.runNode.querySelector("[data-role=picker]"), { checkpoints: this.checkpoints, preferredRun: this.app.context.run, storeKey: "lab.agents", teacherDefault: true });
    }
    const ready = this.solveState.status === "solved";
    this.footNode.querySelector("[data-action=run]").disabled = !ready;
    this.footNode.querySelector("[data-role=run-note]").textContent = ready ? "" : "needs a solved plan";
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
    const disruption = this.runNode.querySelector("button[data-disruption][aria-checked=true]").dataset.disruption === "true";
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
