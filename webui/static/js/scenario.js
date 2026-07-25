"use strict";

const SCN_EVENT_LABELS = {
  new_job:        ["new job", "new jobs"],
  remove_job:     ["job cancelled", "jobs cancelled"],
  new_vehicle:    ["vehicle arrived", "vehicles arrived"],
  remove_vehicle: ["vehicle broke down", "vehicles broke down"],
};

class ScenarioView {
  constructor() {
    this.toolbar = document.getElementById("scn-toolbar");
    this.rail = document.getElementById("scn-rail");

    this.map = null;
    this.routeLayer = null;
    this.markerLayer = null;
    this.fxLayer = null;
    this.banner = null;

    this.jobs = [];
    this.vehicles = [];
    this.mode = "pan";
    this.selected = null;
    this.solveState = null;
    this.runResult = null;
    this.stepIndex = 0;
    this.railTab = "build";
    this.checkpoints = [];
    this.runBusy = false;
    this.templates = [];
    this.templateKey = "";
    this.activeTemplate = null;
    this.assignment = null;
    this.depot = null;

    this.playing = false;
    this.animating = false;
    this.focusMode = true;
    this._animToken = 0;

    this.runOptions = { agent: "model", run_name: null, greedy: true, max_steps: 8, seed: 0, disruptions: 3 };

    this._onKey = this._onKey.bind(this);
  }

  async enter() {
    if (!this.map) this._initMap();
    setTimeout(() => this.map.invalidateSize(), 60);

    const res = await apiGet("/api/scenario/checkpoints");
    this.checkpoints = res.checkpoints || [];
    if (!this.runOptions.run_name && this.checkpoints.length) this.runOptions.run_name = this.checkpoints[0].run;

    if (!this.templates.length) {
      const tpl = await apiGet("/api/scenario/templates");
      this.templates = tpl.templates || [];
      if (this.templates.length) this.templateKey = this.templates[0].key;
    }

    document.addEventListener("keydown", this._onKey);
    this._renderToolbar();
    this._renderRail();
    this._draw();
  }

  leave() {
    this.playing = false;
    document.removeEventListener("keydown", this._onKey);
  }

  _initMap() {
    this.map = L.map("scn-map", { zoomControl: true }).setView([-23.55, -46.63], 12);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(this.map);

    this.routeLayer = L.layerGroup().addTo(this.map);
    this.markerLayer = L.layerGroup().addTo(this.map);
    this.fxLayer = L.layerGroup().addTo(this.map);

    this.banner = document.createElement("div");
    this.banner.className = "scn-banner";
    document.querySelector(".scenario__map-wrap").appendChild(this.banner);

    this.map.on("click", (e) => this._onMapClick(e));
  }

  _color(vehicleId) {
    return LineChart.palette(vehicleId);
  }

  _currentStep() {
    return this.runResult ? this.runResult.steps[this.stepIndex] : null;
  }

  _onMapClick(e) {
    if (this.runResult) {
      toast("finish or discard the run to edit the scenario", "error");
      return;
    }
    if (this.mode === "job") this._addJob([e.latlng.lng, e.latlng.lat]);
    else if (this.mode === "vehicle") this._addVehicle([e.latlng.lng, e.latlng.lat]);
    else if (this.mode === "depot") {
      this.depot = [e.latlng.lng, e.latlng.lat];
      this.solveState = null;
      this.runResult = null;
      this._renderRail();
      this._draw();
    }
  }

  _nextId(list) {
    return list.length ? Math.max(...list.map((x) => x.id)) + 1 : 0;
  }

  _addJob(location) {
    const id = this._nextId(this.jobs);
    this.jobs.push({ id, location, kind: "support", service: 600, setup: 0, amount: 0, priority: 3, description: `Job ${id}` });
    this.selected = { type: "job", id };
    this._invalidate();
  }

  _addVehicle(location) {
    const id = this._nextId(this.vehicles);
    this.vehicles.push({ id, start: location, capacity: 4, onboard: 0, speed_factor: 1.0, time_window: [28800, 72000], progress_credit: 0, description: `Vehicle ${id}` });
    this.selected = { type: "vehicle", id };
    this._invalidate();
  }

  _invalidate() {
    this.solveState = null;
    this.runResult = null;
    this.stepIndex = 0;
    this.playing = false;
    this.activeTemplate = null;
    this.assignment = null;
    this._renderRail();
    this._draw();
  }

  async _loadTemplate(key) {
    const template = this.templates.find((t) => t.key === key);
    if (!template) return;

    this.jobs = structuredClone(template.jobs);
    this.vehicles = structuredClone(template.vehicles);
    this.selected = null;
    this._invalidate();
    this.activeTemplate = template;
    this.assignment = template.assignment ? structuredClone(template.assignment) : null;
    this.depot = structuredClone(template.depot);

    const pts = [...this.jobs.map((j) => [j.location[1], j.location[0]]), ...this.vehicles.map((v) => [v.start[1], v.start[0]])];
    if (pts.length) this.map.fitBounds(L.latLngBounds(pts).pad(0.2));

    await this._solve();
    this._renderRail();
  }

  _renderToolbar() {
    this.toolbar.innerHTML = `
      <button class="btn btn--sm ${this.mode === "pan" ? "is-on" : ""}" data-mode="pan">Pan</button>
      <button class="btn btn--sm ${this.mode === "job" ? "is-on" : ""}" data-mode="job">+ Job</button>
      <button class="btn btn--sm ${this.mode === "vehicle" ? "is-on" : ""}" data-mode="vehicle">+ Vehicle</button>
      <button class="btn btn--sm ${this.mode === "depot" ? "is-on" : ""}" data-mode="depot">Depot</button>
      <button class="btn btn--sm btn--teal" data-act="solve">Solve</button>
      <button class="btn btn--sm" data-act="clear">Clear</button>`;

    this.toolbar.querySelectorAll("[data-mode]").forEach((el) => {
      el.addEventListener("click", () => {
        this.mode = el.dataset.mode;
        this._renderToolbar();
      });
    });
    this.toolbar.querySelector("[data-act=solve]").addEventListener("click", () => this._solve());
    this.toolbar.querySelector("[data-act=clear]").addEventListener("click", () => {
      this.jobs = [];
      this.vehicles = [];
      this.selected = null;
      this._invalidate();
    });
  }

  _draw(overrideStep, opts) {
    if (!this.map) return;
    opts = opts || {};
    this.routeLayer.clearLayers();
    this.markerLayer.clearLayers();

    const step = overrideStep || this._currentStep();
    const state = step ? step.state : this.solveState;
    const jobs = step ? step.jobs : this.jobs;
    const vehicles = step ? step.vehicles : this.vehicles;

    const assignment = {};
    if (state) {
      for (const route of state.routes) {
        for (const stop of route.stops) assignment[stop.job_id] = route.vehicle_id;
      }
    }

    if (state) {
      for (const route of state.routes) {
        const color = this._color(route.vehicle_id);
        let latlngs = null;
        if (route.path && route.path.length > 1) {
          latlngs = route.path;
        } else {
          const pts = [];
          if (route.start) pts.push([route.start[1], route.start[0]]);
          for (const stop of route.stops) pts.push([stop.location[1], stop.location[0]]);
          if (route.end) pts.push([route.end[1], route.end[0]]);
          latlngs = pts;
        }
        L.polyline(latlngs, { color, weight: 3.5, opacity: 0.85 }).addTo(this.routeLayer);
      }
    }

    if (this.depot) {
      const depotIcon = L.divIcon({
        className: "",
        html: `<div class="veh-icon" style="width:24px;height:24px;background:#1a1a1a;border-radius:4px">D</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });
      L.marker([this.depot[1], this.depot[0]], { icon: depotIcon }).addTo(this.markerLayer).bindTooltip("Depot (drop point)");
    }

    if (!opts.hideVehicles) {
      for (const vehicle of vehicles) {
        const hasRoute = state && state.routes.some((r) => r.vehicle_id === vehicle.id);
        const color = hasRoute ? this._color(vehicle.id) : "#5b636a";
        const onboard = vehicle.onboard ? `<span class="veh-icon__load">${vehicle.onboard}</span>` : "";
        const icon = L.divIcon({
          className: "",
          html: `<div class="veh-icon" style="width:22px;height:22px;background:${color}">${vehicle.id}${onboard}</div>`,
          iconSize: [22, 22],
          iconAnchor: [11, 11],
        });
        const marker = L.marker([vehicle.start[1], vehicle.start[0]], { icon }).addTo(this.markerLayer);
        marker.on("click", () => this._selectEntity("vehicle", vehicle.id));
      }
    }

    for (const job of jobs) {
      const assignedTo = assignment[job.id];
      const isUnassigned = state ? assignedTo == null : false;
      const color = assignedTo != null ? this._color(assignedTo) : isUnassigned ? "#b91c1c" : "#5b636a";
      const isRepossession = job.kind === "repossession";
      const marker = L.circleMarker([job.location[1], job.location[0]], {
        radius: 5 + job.priority * 0.8,
        color: isUnassigned ? "#b91c1c" : isRepossession ? "#1a1a1a" : color,
        fillColor: color,
        fillOpacity: assignedTo != null ? 0.85 : 0.35,
        weight: isRepossession ? 2.5 : isUnassigned ? 2.5 : 1.5,
        dashArray: isUnassigned ? "3 3" : null,
      }).addTo(this.markerLayer);
      marker.bindTooltip(`Job ${job.id} · ${isRepossession ? "repossession" : "support"} · p${job.priority}${assignedTo != null ? ` · veh ${assignedTo}` : isUnassigned ? " · unassigned" : ""}`);
      marker.on("click", () => this._selectEntity("job", job.id));
    }
  }

  _selectEntity(type, id) {
    if (this.runResult) return;
    this.selected = { type, id };
    this.railTab = "build";
    this._renderRail();
  }

  _renderRail() {
    this.rail.innerHTML = `
      <div class="tabs" style="margin-bottom:0">
        <button class="tab ${this.railTab === "build" ? "is-active" : ""}" data-tab="build">Build</button>
        <button class="tab ${this.railTab === "run" ? "is-active" : ""}" data-tab="run">Run</button>
      </div>
      <div id="scn-panel"></div>`;

    this.rail.querySelectorAll(".tab").forEach((el) => {
      el.addEventListener("click", () => {
        this.railTab = el.dataset.tab;
        this._renderRail();
      });
    });

    if (this.railTab === "build") this._renderBuildPanel();
    else this._renderRunPanel();
  }

  _renderBuildPanel() {
    const panel = document.getElementById("scn-panel");
    const state = this.solveState;

    const templateOptions = this.templates.map((t) => `<option value="${escapeHtml(t.key)}" ${t.key === this.templateKey ? "selected" : ""}>${escapeHtml(t.title)}</option>`).join("");
    const active = this.activeTemplate;

    panel.innerHTML = `
      <div class="card">
        <h3 class="card__title">Situation templates</h3>
        <div class="editor-grid">
          <label class="field"><span class="field__label"><span>Template</span></span>
            <select id="scn-template">${templateOptions || `<option value="">none available</option>`}</select></label>
          <div class="field" style="justify-content:flex-end"><button class="btn" id="scn-load-template">Load</button></div>
        </div>
        ${active ? `
        <div class="tpl-brief">
          <p class="scn-hint" style="margin:10px 0 0">${escapeHtml(active.description)}</p>
          <p class="scn-hint" style="margin:6px 0 0"><b>Expected:</b> ${escapeHtml(active.expected)}</p>
        </div>` : `
        <p class="scn-hint" style="margin:10px 0 0">Curated situations with a pinned starting plan, each staging one decision: cheap vs expensive insertions, relocations, capacity swaps, outlier abandonment, idle activation, tangled routes. Load one, then compare agents on the Run tab.</p>`}
      </div>

      <div class="card">
        <h3 class="card__title">Random scenario</h3>
        <div class="editor-grid">
          <label class="field"><span class="field__label"><span>Jobs</span></span><input type="number" id="scn-njobs" value="12" min="1" step="1"></label>
          <label class="field"><span class="field__label"><span>Vehicles</span></span><input type="number" id="scn-nveh" value="3" min="1" step="1"></label>
          <label class="field"><span class="field__label"><span>Seed</span></span><input type="number" id="scn-sseed" value="0" step="1"></label>
          <div class="field" style="justify-content:flex-end"><button class="btn" id="scn-sample">Sample</button></div>
        </div>
      </div>

      <div class="card">
        <h3 class="card__title">Jobs (${this.jobs.length})</h3>
        <div class="ent-list" id="scn-job-list"></div>
      </div>

      <div class="card">
        <h3 class="card__title">Vehicles (${this.vehicles.length})</h3>
        <div class="ent-list" id="scn-veh-list"></div>
      </div>

      <div id="scn-editor"></div>

      ${state ? `
      <div class="card">
        <h3 class="card__title">Current plan</h3>
        <div class="summary-strip">
          <div class="cell"><b>${state.num_routes}</b><span>routes</span></div>
          <div class="cell"><b>${state.num_unassigned}</b><span>unassigned</span></div>
          <div class="cell"><b>${fmtNum(state.cost, 0)}</b><span>cost</span></div>
          <div class="cell"><b>${(state.distance / 1000).toFixed(1)} km</b><span>distance</span></div>
        </div>
      </div>` : ""}

      <p class="scn-hint">Pick <b>+ Job</b> or <b>+ Vehicle</b> and click the map to place entities, or sample a random scenario. <b>Solve</b> asks VROOM for the initial plan; then switch to <b>Run</b> to watch an agent react.</p>`;

    document.getElementById("scn-sample").addEventListener("click", () => this._sample());
    const templateSelect = document.getElementById("scn-template");
    templateSelect.addEventListener("change", () => { this.templateKey = templateSelect.value; });
    document.getElementById("scn-load-template").addEventListener("click", () => this._loadTemplate(this.templateKey));
    this._renderEntityLists();
    this._renderEditor();
  }

  _renderEntityLists() {
    const state = this.solveState;
    const assignment = {};
    if (state) {
      for (const route of state.routes) {
        for (const stop of route.stops) assignment[stop.job_id] = route.vehicle_id;
      }
    }

    const jobHost = document.getElementById("scn-job-list");
    jobHost.innerHTML = this.jobs.map((job) => {
      const assignedTo = assignment[job.id];
      const color = assignedTo != null ? this._color(assignedTo) : "#9aa196";
      const sel = this.selected && this.selected.type === "job" && this.selected.id === job.id;
      return `<div class="ent-row ${sel ? "is-selected" : ""}" data-type="job" data-id="${job.id}">
        <span class="ent-dot" style="background:${color}"></span>
        <span class="grow">Job ${job.id} · p${job.priority}</span>
        <span class="dim">${assignedTo != null ? "veh " + assignedTo : state ? "unassigned" : ""}</span>
      </div>`;
    }).join("") || `<p class="scn-hint">none yet</p>`;

    const vehHost = document.getElementById("scn-veh-list");
    vehHost.innerHTML = this.vehicles.map((vehicle) => {
      const sel = this.selected && this.selected.type === "vehicle" && this.selected.id === vehicle.id;
      return `<div class="ent-row ${sel ? "is-selected" : ""}" data-type="vehicle" data-id="${vehicle.id}">
        <span class="ent-sq" style="background:${this._color(vehicle.id)}"></span>
        <span class="grow">Vehicle ${vehicle.id} · cap ${vehicle.capacity}</span>
        <span class="dim">x${vehicle.speed_factor}</span>
      </div>`;
    }).join("") || `<p class="scn-hint">none yet</p>`;

    this.rail.querySelectorAll(".ent-row").forEach((el) => {
      el.addEventListener("click", () => {
        this.selected = { type: el.dataset.type, id: Number(el.dataset.id) };
        this._renderRail();
        this._draw();
      });
    });
  }

  _renderEditor() {
    const host = document.getElementById("scn-editor");
    if (!this.selected) { host.innerHTML = ""; return; }

    const { type, id } = this.selected;
    const entity = (type === "job" ? this.jobs : this.vehicles).find((x) => x.id === id);
    if (!entity) { host.innerHTML = ""; this.selected = null; return; }

    if (type === "job") {
      host.innerHTML = `
        <div class="card">
          <h3 class="card__title">Job ${id}</h3>
          <div class="editor-grid">
            <label class="field"><span class="field__label"><span>Kind</span></span>
              <select data-k="kind">
                <option value="support" ${entity.kind === "support" ? "selected" : ""}>support (no capacity)</option>
                <option value="repossession" ${entity.kind === "repossession" ? "selected" : ""}>repossession (carries bike)</option>
              </select></label>
            <label class="field"><span class="field__label"><span>Priority (1-5)</span></span><input type="number" data-k="priority" value="${entity.priority}" min="1" max="5" step="1"></label>
            <label class="field"><span class="field__label"><span>Service (s)</span></span><input type="number" data-k="service" value="${entity.service}" min="0" step="60"></label>
            <label class="field"><span class="field__label"><span>Setup (s)</span></span><input type="number" data-k="setup" value="${entity.setup}" min="0" step="60"></label>
          </div>
          <div style="margin-top:12px"><button class="btn btn--sm btn--danger" id="scn-del">Delete job</button></div>
        </div>`;
    } else {
      host.innerHTML = `
        <div class="card">
          <h3 class="card__title">Vehicle ${id}</h3>
          <div class="editor-grid">
            <label class="field"><span class="field__label"><span>Capacity</span></span><input type="number" data-k="capacity" value="${entity.capacity}" min="1" step="1"></label>
            <label class="field"><span class="field__label"><span>Speed factor</span></span><input type="number" data-k="speed_factor" value="${entity.speed_factor}" min="0.5" max="2" step="0.1"></label>
            <label class="field"><span class="field__label"><span>Shift start (h)</span></span><input type="number" data-k="tw0" value="${entity.time_window[0] / 3600}" min="0" max="24" step="0.5"></label>
            <label class="field"><span class="field__label"><span>Shift end (h)</span></span><input type="number" data-k="tw1" value="${entity.time_window[1] / 3600}" min="0" max="24" step="0.5"></label>
          </div>
          <div style="margin-top:12px"><button class="btn btn--sm btn--danger" id="scn-del">Delete vehicle</button></div>
        </div>`;
    }

    host.querySelectorAll("[data-k]").forEach((el) => {
      el.addEventListener("change", () => {
        const k = el.dataset.k;
        if (k === "kind") {
          entity.kind = el.value;
          entity.amount = el.value === "repossession" ? 1 : 0;
        }
        else if (k === "tw0") entity.time_window = [Math.round(Number(el.value) * 3600), entity.time_window[1]];
        else if (k === "tw1") entity.time_window = [entity.time_window[0], Math.round(Number(el.value) * 3600)];
        else if (k === "speed_factor") entity[k] = Number(el.value);
        else entity[k] = Math.round(Number(el.value));
        this.solveState = null;
        this.runResult = null;
        this._renderRail();
        this._draw();
      });
    });

    host.querySelector("#scn-del").addEventListener("click", () => {
      if (type === "job") this.jobs = this.jobs.filter((x) => x.id !== id);
      else this.vehicles = this.vehicles.filter((x) => x.id !== id);
      this.selected = null;
      this._invalidate();
    });
  }

  async _sample() {
    const num_jobs = Number(document.getElementById("scn-njobs").value) || 12;
    const num_vehicles = Number(document.getElementById("scn-nveh").value) || 3;
    const seed = Number(document.getElementById("scn-sseed").value) || 0;

    const res = await apiPost("/api/scenario/sample", { num_jobs, num_vehicles, seed });
    if (res.error) { toast(res.error, "error"); return; }

    this.jobs = res.jobs;
    this.vehicles = res.vehicles;
    this.selected = null;
    this._invalidate();
    this.depot = res.depot;

    const pts = [...this.jobs.map((j) => [j.location[1], j.location[0]]), ...this.vehicles.map((v) => [v.start[1], v.start[0]])];
    if (pts.length) this.map.fitBounds(L.latLngBounds(pts).pad(0.2));
    toast(`sampled ${this.jobs.length} jobs, ${this.vehicles.length} vehicles`, "ok");
  }

  async _solve() {
    if (!this.jobs.length || !this.vehicles.length) {
      toast("place at least one job and one vehicle", "error");
      return;
    }
    if (!this.depot) {
      toast("place a depot first (Depot mode, click the map)", "error");
      return;
    }
    this.runResult = null;
    toast("solving…");
    const res = await apiPost("/api/scenario/solve", { jobs: this.jobs, vehicles: this.vehicles, assignment: this.assignment, depot: this.depot });
    if (res.error) { toast(res.error, "error"); return; }

    this.solveState = res.state;
    this._renderRail();
    this._draw();
    toast(`plan: ${res.state.num_routes} routes, ${res.state.num_unassigned} unassigned, cost ${res.state.cost}`, "ok");
  }

  _renderRunPanel() {
    const panel = document.getElementById("scn-panel");
    const o = this.runOptions;

    const ckptOptions = this.checkpoints.map((c) => `<option value="${escapeHtml(c.run)}" ${c.run === o.run_name ? "selected" : ""}>${escapeHtml(c.run)}</option>`).join("");

    panel.innerHTML = `
      <div class="card">
        <h3 class="card__title">Agent</h3>
        <div class="editor-grid">
          <label class="field"><span class="field__label"><span>Agent</span></span>
            <select id="scn-agent">
              ${["model", "teacher", "insertion_only", "do_nothing"].map((a) => `<option value="${a}" ${a === o.agent ? "selected" : ""}>${a}</option>`).join("")}
            </select></label>
          <label class="field" id="scn-ckpt-wrap" style="${o.agent === "model" ? "" : "display:none"}"><span class="field__label"><span>Checkpoint</span></span>
            <select id="scn-ckpt">${ckptOptions || `<option value="">no checkpoints</option>`}</select></label>
          <label class="switch" id="scn-greedy-wrap" style="align-self:end;${o.agent === "model" ? "" : "display:none"}"><input type="checkbox" id="scn-greedy" ${o.greedy ? "checked" : ""}><i></i><span>greedy (argmax)</span></label>
          <label class="field"><span class="field__label"><span>Disruptions</span></span><input type="number" id="scn-disruptions" value="${o.disruptions}" min="1" max="20" step="1"></label>
          <label class="field"><span class="field__label"><span>Max steps</span></span><input type="number" id="scn-steps" value="${o.max_steps}" min="1" max="50" step="1"></label>
          <label class="field"><span class="field__label"><span>Seed</span></span><input type="number" id="scn-seed" value="${o.seed}" step="1"></label>
          <div class="field" style="grid-column:1/-1;justify-content:flex-end"><button class="btn btn--primary" id="scn-run" ${this.runBusy ? "disabled" : ""}>${this.runBusy ? "Running…" : "Run agent"}</button></div>
        </div>
        <p class="scn-hint" style="margin:10px 0 0">Each <b>disruption round</b> is a guaranteed event (new/removed jobs or vehicles) followed by the agent's <b>response burst</b> at frozen time; a random wait then <b>advances</b> the vehicles before the next disruption. Round one starts from the scenario as built. Max steps caps total edits.</p>
        ${this.activeTemplate ? `<p class="scn-hint" style="margin:6px 0 0"><b>${escapeHtml(this.activeTemplate.title)}</b> — ${escapeHtml(this.activeTemplate.expected)}</p>` : ""}
      </div>
      <div id="scn-run-result"></div>`;

    document.getElementById("scn-agent").addEventListener("change", (e) => {
      o.agent = e.target.value;
      document.getElementById("scn-ckpt-wrap").style.display = o.agent === "model" ? "" : "none";
      document.getElementById("scn-greedy-wrap").style.display = o.agent === "model" ? "" : "none";
    });
    document.getElementById("scn-run").addEventListener("click", () => this._run());

    this._renderRunResult();
  }

  async _run() {
    if (!this.jobs.length || !this.vehicles.length) {
      toast("build a scenario first (Build tab)", "error");
      return;
    }
    if (!this.depot) {
      toast("place a depot first (Build tab, Depot mode)", "error");
      return;
    }

    const o = this.runOptions;
    o.run_name = (document.getElementById("scn-ckpt") || {}).value || null;
    o.greedy = !!(document.getElementById("scn-greedy") || {}).checked;
    o.max_steps = Number(document.getElementById("scn-steps").value) || 8;
    o.seed = Number(document.getElementById("scn-seed").value) || 0;
    o.disruptions = Number(document.getElementById("scn-disruptions").value) || 1;

    this.runBusy = true;
    this.playing = false;
    this._renderRunPanel();

    const res = await apiPost("/api/scenario/run", {
      jobs: this.jobs,
      vehicles: this.vehicles,
      assignment: this.assignment,
      depot: this.depot,
      agent: o.agent,
      run_name: o.run_name,
      greedy: o.greedy,
      max_steps: o.max_steps,
      seed: o.seed,
      disruptions: o.disruptions,
    });

    this.runBusy = false;
    if (res.error) {
      this._renderRunPanel();
      toast(res.error, "error");
      return;
    }

    this.runResult = res;
    this.stepIndex = 0;
    this._renderRunPanel();
    this._draw();
    toast(`${res.summary.total_rounds} rounds · ${res.summary.total_steps} edits · ${res.summary.stopped_reason}`, "ok");
  }

  _renderRunResult() {
    const host = document.getElementById("scn-run-result");
    if (!this.runResult) {
      host.innerHTML = this.solveState ? "" : `<p class="scn-hint">Solve the scenario first so the agent starts from the VROOM plan.</p>`;
      return;
    }

    const { steps, summary } = this.runResult;
    const rewardClass = summary.cumulative_reward > 0 ? "pos" : summary.cumulative_reward < 0 ? "neg" : "";

    host.innerHTML = `
      <div class="card">
        <h3 class="card__title">Run — ${escapeHtml(summary.agent)}${summary.run_name ? ` · ${escapeHtml(summary.run_name)}` : ""}</h3>
        <div class="summary-strip">
          <div class="cell"><b>${summary.total_rounds}</b><span>rounds</span></div>
          <div class="cell"><b>${summary.total_steps}</b><span>edits</span></div>
          <div class="cell"><b class="${rewardClass}">${fmtNum(summary.cumulative_reward, 3)}</b><span>total reward</span></div>
          <div class="cell"><b>${summary.initial_unassigned} → ${summary.final_unassigned}</b><span>unassigned</span></div>
        </div>
        <p class="scn-hint" style="margin:8px 0 0">cost ${fmtNum(summary.initial_cost, 0)} → ${fmtNum(summary.final_cost, 0)} · stopped: ${escapeHtml(summary.stopped_reason)}</p>
      </div>

      <div class="card">
        <div class="playbar">
          <button class="btn btn--sm" id="scn-prev" title="previous (←)">&larr;</button>
          <button class="btn btn--sm btn--primary" id="scn-play" title="play / pause (space)">▶</button>
          <button class="btn btn--sm" id="scn-next" title="next (→)">&rarr;</button>
          <input type="range" id="scn-slider" min="0" max="${steps.length - 1}" value="${this.stepIndex}">
          <span class="playback__step" id="scn-counter"></span>
          <button class="btn btn--sm ${this.focusMode ? "is-on" : ""}" id="scn-focus" title="zoom into each vehicle in turn">◎ Focus</button>
        </div>
        <div class="reward-spark" id="scn-spark"></div>
        <div class="tl" id="scn-timeline"></div>
        <div class="step-info" id="scn-info" style="margin-top:14px"></div>
      </div>

      <button class="btn btn--ghost" id="scn-discard">Discard run &amp; edit scenario</button>`;

    const slider = document.getElementById("scn-slider");
    slider.addEventListener("input", () => { this.playing = false; this._goTo(Number(slider.value), false); });
    document.getElementById("scn-prev").addEventListener("click", () => { this.playing = false; this._goTo(this.stepIndex - 1, false); });
    document.getElementById("scn-next").addEventListener("click", () => { this.playing = false; this._goTo(this.stepIndex + 1, true); });
    document.getElementById("scn-play").addEventListener("click", () => this._play());
    document.getElementById("scn-focus").addEventListener("click", (e) => {
      this.focusMode = !this.focusMode;
      e.currentTarget.classList.toggle("is-on", this.focusMode);
    });
    document.getElementById("scn-discard").addEventListener("click", () => {
      this.playing = false;
      this.runResult = null;
      this.stepIndex = 0;
      this.fxLayer.clearLayers();
      this._renderRail();
      this._draw();
    });

    document.getElementById("scn-timeline").innerHTML = this._timelineHtml(steps);
    document.querySelectorAll("#scn-timeline .tl-node").forEach((el) => {
      el.addEventListener("click", () => { this.playing = false; this._goTo(Number(el.dataset.seq), false); });
    });

    this._syncPlayback();
  }

  _syncPlayback() {
    if (!this.runResult) return;
    const steps = this.runResult.steps;
    const step = steps[this.stepIndex];

    const slider = document.getElementById("scn-slider");
    if (slider) slider.value = this.stepIndex;

    const counter = document.getElementById("scn-counter");
    if (counter) counter.textContent = `${this.stepIndex}/${steps.length - 1}`;

    const play = document.getElementById("scn-play");
    if (play) play.textContent = this.playing ? "❚❚" : "▶";

    const info = document.getElementById("scn-info");
    if (info) info.innerHTML = this._stepInfoHtml(step);

    const spark = document.getElementById("scn-spark");
    if (spark) spark.innerHTML = this._sparklineSvg(steps, this.stepIndex);

    document.querySelectorAll("#scn-timeline .tl-node").forEach((el) => {
      el.classList.toggle("is-current", Number(el.dataset.seq) === this.stepIndex);
      el.classList.toggle("is-seen", Number(el.dataset.seq) <= this.stepIndex);
    });
  }

  _onKey(e) {
    if (!this.runResult) return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    if (e.key === "ArrowRight") { e.preventDefault(); this.playing = false; this._goTo(this.stepIndex + 1, true); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); this.playing = false; this._goTo(this.stepIndex - 1, false); }
    else if (e.key === " ") { e.preventDefault(); this._play(); }
  }

  async _play() {
    if (this.playing) { this.playing = false; this._syncPlayback(); return; }
    const last = this.runResult.steps.length - 1;
    if (this.stepIndex >= last) { this.stepIndex = 0; this._draw(); this._syncPlayback(); await this._sleep(200); }
    this.playing = true;
    this._syncPlayback();

    while (this.playing && this.stepIndex < last) {
      await this._goTo(this.stepIndex + 1, true);
      if (!this.playing) break;
      await this._sleep(this.focusMode ? 250 : 650);
    }
    this.playing = false;
    this._syncPlayback();
  }

  async _goTo(target, animate) {
    const steps = this.runResult.steps;
    target = Math.max(0, Math.min(steps.length - 1, target));
    const from = this.stepIndex;

    if (!animate || target !== from + 1) {
      this._animToken++;
      this.animating = false;
      this.stepIndex = target;
      this.fxLayer.clearLayers();
      this._draw();
      this._syncPlayback();
      return;
    }

    if (this.animating) return;
    this.animating = true;
    const token = ++this._animToken;
    const alive = () => this._animToken === token && this.animating;

    const fromStep = steps[from];
    const toStep = steps[target];
    this.stepIndex = target;
    this._syncPlayback();

    try {
      await this._animatePhase(fromStep, toStep, alive);
    } catch (err) {
      this.playing = false;
    }

    if (this._animToken === token) {
      this.fxLayer.clearLayers();
      this._draw(toStep);
      this._syncPlayback();
      this.animating = false;
    }
  }

  async _animatePhase(fromStep, toStep, alive) {
    if (this.focusMode && toStep.phase === "advance") {
      await this._focusAdvance(fromStep, toStep, alive);
    } else if (this.focusMode && toStep.phase === "edit") {
      await this._focusEdit(fromStep, toStep, alive);
    } else if (toStep.phase === "advance") {
      await this._fxAdvance(fromStep, toStep);
    } else if (toStep.phase === "commit") {
      await this._fxCommit(toStep);
    } else {
      await this._fxAction(fromStep, toStep);
    }
  }

  async _focusAdvance(fromStep, toStep, alive) {
    const gapMin = Math.round((toStep.gap || 0) / 60);
    const movers = fromStep.state.routes.map((route) => route.vehicle_id).sort((a, b) => a - b);
    const served = (toStep.executed && toStep.executed.served) || [];
    const failed = (toStep.executed && toStep.executed.failed) || [];
    const dropped = (toStep.executed && toStep.executed.dropped) || 0;
    const byVehicle = this._servedByVehicle(fromStep, served, failed);

    this._showBanner(`⏱ +${gapMin} min — touring ${movers.length} vehicle${movers.length === 1 ? "" : "s"}`, "advance");
    this._draw(fromStep, { hideVehicles: true });

    for (const vehicleId of movers) {
      if (!alive()) return;
      await this._focusVehicleAdvance(fromStep, toStep, vehicleId, byVehicle[vehicleId], gapMin, alive);
    }

    if (dropped > 0 && this.depot && alive()) {
      await this._flyToBounds(L.latLng(this.depot[1], this.depot[0]).toBounds(1400));
      this._popMark([this.depot[1], this.depot[0]], `−${dropped} 🏍 depot`, "fx-dropped");
      await this._sleep(700);
    }

    if (alive()) await this._focusDisruption(fromStep, toStep, served, failed, alive);
    if (alive()) await this._flyToStep(toStep);
  }

  async _focusVehicleAdvance(fromStep, toStep, vehicleId, service, gapMin, alive) {
    const fromVehicle = fromStep.vehicles.find((v) => v.id === vehicleId);
    const toVehicle = toStep.vehicles.find((v) => v.id === vehicleId);
    if (!fromVehicle) return;

    const jobLocs = this._jobLocIndex(fromStep);
    const from = [fromVehicle.start[1], fromVehicle.start[0]];
    const to = toVehicle ? [toVehicle.start[1], toVehicle.start[0]] : from;
    const route = fromStep.state.routes.find((r) => r.vehicle_id === vehicleId);
    const path = route && route.path && route.path.length > 1 ? route.path : [from, to];
    const endFrac = this._nearestFrac(path, to);

    const served = (service && service.served) || [];
    const failed = (service && service.failed) || [];
    const framePts = path.concat(served.map((id) => jobLocs[id]), failed.map((id) => jobLocs[id])).filter(Boolean);
    await this._flyToBounds(L.latLngBounds(framePts).pad(0.45));
    if (!alive()) return;

    const parts = [`Vehicle ${vehicleId}`, `drove ${gapMin} min`];
    if (served.length) parts.push(`served ${served.join(", ")}`);
    if (failed.length) parts.push(`failed ${failed.join(", ")}`);
    if (!served.length && !failed.length) parts.push("en route");
    this._showBanner(parts.join(" · "), "advance");

    const ghost = this._ghostVehicle(fromVehicle);
    this._pulseRing(from, this._color(vehicleId));
    await this._tween(950, (t) => ghost.setLatLng(this._pointAlong(path, this._ease(t) * endFrac)));

    for (const id of served) if (jobLocs[id]) this._popMark(jobLocs[id], "✓", "fx-served");
    for (const id of failed) if (jobLocs[id]) this._popMark(jobLocs[id], "✕", "fx-failed");
    await this._sleep(served.length || failed.length ? 750 : 400);
  }

  async _focusDisruption(fromStep, toStep, served, failed, alive) {
    const fromJobs = this._jobLocIndex(fromStep);
    const toJobs = this._jobLocIndex(toStep);
    const points = [];

    for (const id of Object.keys(toJobs)) if (!(id in fromJobs)) points.push(toJobs[id]);
    const executed = new Set([...served, ...failed].map(Number));
    for (const id of Object.keys(fromJobs)) if (!(id in toJobs) && !executed.has(Number(id))) points.push(fromJobs[id]);

    const fromVeh = new Set(fromStep.vehicles.map((v) => v.id));
    const toVeh = new Set(toStep.vehicles.map((v) => v.id));
    for (const v of toStep.vehicles) if (!fromVeh.has(v.id)) points.push([v.start[1], v.start[0]]);
    for (const v of fromStep.vehicles) if (!toVeh.has(v.id)) points.push([v.start[1], v.start[0]]);

    if (!points.length) return;
    await this._flyToBounds(L.latLngBounds(points).pad(0.5));
    if (!alive()) return;
    await this._fxDisruption(fromStep, toStep, served, failed);
  }

  async _focusEdit(fromStep, toStep, alive) {
    const a = toStep.action;
    if (!a || a.operator_name === "DO_NOTHING") { await this._fxAction(fromStep, toStep); return; }

    const jobLocs = this._jobLocIndex(fromStep, toStep);
    const veh = fromStep.vehicles.find((v) => v.id === a.vehicle_id) || toStep.vehicles.find((v) => v.id === a.vehicle_id);
    const job = jobLocs[a.job_id];
    if (job && veh) {
      const route = fromStep.state.routes.find((r) => r.vehicle_id === a.vehicle_id);
      const pts = [job, [veh.start[1], veh.start[0]]].concat(route && route.path ? route.path : []);
      await this._flyToBounds(L.latLngBounds(pts).pad(0.5));
    }
    if (!alive()) return;
    await this._fxAction(fromStep, toStep);
  }

  _servedByVehicle(fromStep, served, failed) {
    const owner = {};
    for (const route of fromStep.state.routes) {
      for (const stop of route.stops) owner[stop.job_id] = route.vehicle_id;
    }

    const map = {};
    for (const id of served) {
      const vid = owner[id];
      if (vid == null) continue;
      (map[vid] = map[vid] || { served: [], failed: [] }).served.push(id);
    }
    for (const id of failed) {
      const vid = owner[id];
      if (vid == null) continue;
      (map[vid] = map[vid] || { served: [], failed: [] }).failed.push(id);
    }
    return map;
  }

  _flyToStep(step) {
    const pts = step.vehicles.map((v) => [v.start[1], v.start[0]]).concat(step.jobs.map((j) => [j.location[1], j.location[0]]));
    if (this.depot) pts.push([this.depot[1], this.depot[0]]);
    if (!pts.length) return Promise.resolve();
    return this._flyToBounds(L.latLngBounds(pts).pad(0.2));
  }

  _flyToBounds(bounds) {
    return new Promise((resolve) => {
      let done = false;
      const finish = () => { if (!done) { done = true; resolve(); } };
      this.map.once("moveend", finish);
      try {
        this.map.flyToBounds(bounds, { duration: 0.7, padding: [50, 50], maxZoom: 16 });
      } catch (err) {
        finish();
        return;
      }
      setTimeout(finish, 1600);
    });
  }

  async _fxAdvance(fromStep, toStep) {
    const gapMin = Math.round((toStep.gap || 0) / 60);
    this._showBanner(`⏱ +${gapMin} min — vehicles advancing`, "advance");

    this._draw(fromStep, { hideVehicles: true });

    const ghosts = [];
    const toById = {};
    for (const v of toStep.vehicles) toById[v.id] = v;

    for (const v of fromStep.vehicles) {
      const from = [v.start[1], v.start[0]];
      const target = toById[v.id] ? [toById[v.id].start[1], toById[v.id].start[0]] : from;
      const route = fromStep.state.routes.find((r) => r.vehicle_id === v.id);
      const path = route && route.path && route.path.length > 1 ? route.path : [from, target];
      const endFrac = this._nearestFrac(path, target);
      const marker = this._ghostVehicle(v);
      ghosts.push({ marker, path, endFrac });
    }

    const served = (toStep.executed && toStep.executed.served) || [];
    const failed = (toStep.executed && toStep.executed.failed) || [];
    const dropped = (toStep.executed && toStep.executed.dropped) || 0;
    const jobLoc = this._jobLocIndex(fromStep);

    await this._tween(1100, (t) => {
      const e = this._ease(t);
      for (const g of ghosts) g.marker.setLatLng(this._pointAlong(g.path, e * g.endFrac));
    });

    for (const id of served) if (jobLoc[id]) this._popMark(jobLoc[id], "✓", "fx-served");
    for (const id of failed) if (jobLoc[id]) this._popMark(jobLoc[id], "✕", "fx-failed");
    if (dropped > 0 && this.depot) this._popMark([this.depot[1], this.depot[0]], `−${dropped} 🏍`, "fx-dropped");
    if (served.length || failed.length || dropped > 0) await this._sleep(500);

    ghosts.forEach((g) => this.fxLayer.removeLayer(g.marker));
    await this._fxDisruption(fromStep, toStep, served, failed);
  }

  async _fxDisruption(fromStep, toStep, served, failed) {
    const fromJobs = this._jobLocIndex(fromStep);
    const toJobs = this._jobLocIndex(toStep);
    const executed = new Set([...served, ...failed].map(Number));

    const addedJobs = Object.keys(toJobs).filter((id) => !(id in fromJobs)).map(Number);
    const cancelledJobs = Object.keys(fromJobs).filter((id) => !(id in toJobs) && !executed.has(Number(id))).map(Number);

    const fromVeh = new Set(fromStep.vehicles.map((v) => v.id));
    const toVeh = new Set(toStep.vehicles.map((v) => v.id));
    const addedVeh = toStep.vehicles.filter((v) => !fromVeh.has(v.id));
    const removedVeh = fromStep.vehicles.filter((v) => !toVeh.has(v.id));

    const event = (toStep.events && toStep.events[0]) || null;
    if (event) {
      const [one, many] = SCN_EVENT_LABELS[event.type] || [event.type, event.type];
      this._showBanner(`⚡ ${event.count} ${event.count === 1 ? one : many}`, "disrupt");
    }

    for (const id of addedJobs) this._ripple(toJobs[id], "fx-add-job");
    for (const id of cancelledJobs) this._popMark(fromJobs[id], "✕", "fx-cancel");
    for (const v of addedVeh) this._ripple([v.start[1], v.start[0]], "fx-add-veh");
    for (const v of removedVeh) this._popMark([v.start[1], v.start[0]], "⚠", "fx-cancel");

    await this._sleep(850);
  }

  async _fxAction(fromStep, toStep) {
    const a = toStep.action;
    this._draw(fromStep);

    if (!a || a.operator_name === "DO_NOTHING") {
      await this._sleep(220);
      return;
    }

    const jobLoc = this._jobLocIndex(fromStep, toStep);
    const veh = (fromStep.vehicles.find((v) => v.id === a.vehicle_id)) || (toStep.vehicles.find((v) => v.id === a.vehicle_id));
    const job = jobLoc[a.job_id];
    if (!job || !veh) { await this._sleep(200); return; }

    const vehLL = [veh.start[1], veh.start[0]];
    const insert = a.operator_name === "INSERT";
    const color = this._color(a.vehicle_id);

    this._showBanner(`${insert ? "＋ INSERT" : "－ REMOVE"} job ${a.job_id} ${insert ? "→" : "⇠"} veh ${a.vehicle_id}`, insert ? "insert" : "remove");
    this._pulseRing(job, insert ? color : "#b91c1c");
    this._pulseRing(vehLL, color);

    const arcPts = this._arcPoints(insert ? job : vehLL, insert ? vehLL : job, 0.22);
    const arc = L.polyline([], { color: insert ? color : "#b91c1c", weight: 3, opacity: 0.9, dashArray: "6 6" }).addTo(this.fxLayer);
    const dot = L.circleMarker(arcPts[0], { radius: 5, color: "#fff", weight: 2, fillColor: insert ? color : "#b91c1c", fillOpacity: 1 }).addTo(this.fxLayer);

    await this._tween(760, (t) => {
      const e = this._ease(t);
      const k = Math.max(1, Math.round(e * (arcPts.length - 1)));
      arc.setLatLngs(arcPts.slice(0, k + 1));
      dot.setLatLng(arcPts[k]);
    });

    if (toStep.total_reward != null) this._rewardPop(job, toStep.total_reward);
    await this._sleep(280);
  }

  async _fxCommit(toStep) {
    this._showBanner("✓ round committed — plan locked in", "commit");
    this._draw(toStep);
    for (const route of toStep.state.routes) {
      if (route.path && route.path.length > 1) {
        L.polyline(route.path, { color: "#15803d", weight: 5, opacity: 0.5, className: "fx-commit-line" }).addTo(this.fxLayer);
      }
    }
    if (toStep.total_reward != null && toStep.total_reward !== 0) {
      const first = toStep.state.routes[0];
      if (first && first.start) this._rewardPop([first.start[1], first.start[0]], toStep.total_reward);
    }
    await this._sleep(620);
  }

  _ghostVehicle(v) {
    const icon = L.divIcon({
      className: "",
      html: `<div class="veh-icon veh-icon--moving" style="width:22px;height:22px;background:${this._color(v.id)}">${v.id}</div>`,
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    });
    return L.marker([v.start[1], v.start[0]], { icon, interactive: false }).addTo(this.fxLayer);
  }

  _ripple(latlng, cls) {
    const icon = L.divIcon({ className: "", html: `<div class="fx-ripple ${cls}"></div>`, iconSize: [1, 1], iconAnchor: [0, 0] });
    const marker = L.marker(latlng, { icon, interactive: false }).addTo(this.fxLayer);
    setTimeout(() => { if (this.fxLayer) this.fxLayer.removeLayer(marker); }, 1400);
  }

  _popMark(latlng, text, cls) {
    const icon = L.divIcon({ className: "", html: `<div class="fx-pop ${cls}">${text}</div>`, iconSize: [1, 1], iconAnchor: [0, 0] });
    const marker = L.marker(latlng, { icon, interactive: false }).addTo(this.fxLayer);
    setTimeout(() => { if (this.fxLayer) this.fxLayer.removeLayer(marker); }, 1400);
  }

  _pulseRing(latlng, color) {
    const icon = L.divIcon({ className: "", html: `<div class="fx-pulse" style="border-color:${color}"></div>`, iconSize: [1, 1], iconAnchor: [0, 0] });
    const marker = L.marker(latlng, { icon, interactive: false }).addTo(this.fxLayer);
    setTimeout(() => { if (this.fxLayer) this.fxLayer.removeLayer(marker); }, 1200);
  }

  _rewardPop(latlng, value) {
    const cls = value > 0 ? "pos" : value < 0 ? "neg" : "";
    const text = `${value > 0 ? "+" : ""}${fmtNum(value, 2)}`;
    const icon = L.divIcon({ className: "", html: `<div class="fx-reward ${cls}">${text}</div>`, iconSize: [1, 1], iconAnchor: [0, -6] });
    const marker = L.marker(latlng, { icon, interactive: false }).addTo(this.fxLayer);
    setTimeout(() => { if (this.fxLayer) this.fxLayer.removeLayer(marker); }, 1300);
  }

  _showBanner(html, kind) {
    if (!this.banner) return;
    this.banner.className = `scn-banner scn-banner--${kind}`;
    this.banner.innerHTML = html;
    void this.banner.offsetWidth;
    this.banner.classList.add("is-on");
    clearTimeout(this._bannerTimer);
    this._bannerTimer = setTimeout(() => this.banner.classList.remove("is-on"), 1900);
  }

  _jobLocIndex() {
    const index = {};
    for (const step of arguments) {
      if (!step || !step.jobs) continue;
      for (const job of step.jobs) if (!(job.id in index)) index[job.id] = [job.location[1], job.location[0]];
    }
    return index;
  }

  _pointAlong(path, frac) {
    if (path.length === 1) return path[0];
    const total = this._pathLength(path);
    let target = frac * total;
    for (let i = 1; i < path.length; i++) {
      const seg = this._segLen(path[i - 1], path[i]);
      if (target <= seg || i === path.length - 1) {
        const r = seg === 0 ? 0 : target / seg;
        return [path[i - 1][0] + (path[i][0] - path[i - 1][0]) * r, path[i - 1][1] + (path[i][1] - path[i - 1][1]) * r];
      }
      target -= seg;
    }
    return path[path.length - 1];
  }

  _nearestFrac(path, target) {
    if (path.length < 2) return 1;
    const total = this._pathLength(path);
    let acc = 0, best = 0, bestD = Infinity;
    for (let i = 0; i < path.length; i++) {
      if (i > 0) acc += this._segLen(path[i - 1], path[i]);
      const d = this._segLen(path[i], target);
      if (d < bestD) { bestD = d; best = acc; }
    }
    return total === 0 ? 1 : best / total;
  }

  _pathLength(path) {
    let s = 0;
    for (let i = 1; i < path.length; i++) s += this._segLen(path[i - 1], path[i]);
    return s;
  }

  _segLen(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  _arcPoints(a, b, bend) {
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const ctrl = [mid[0] - dy * bend, mid[1] + dx * bend];
    const pts = [];
    for (let i = 0; i <= 24; i++) {
      const t = i / 24, u = 1 - t;
      pts.push([
        u * u * a[0] + 2 * u * t * ctrl[0] + t * t * b[0],
        u * u * a[1] + 2 * u * t * ctrl[1] + t * t * b[1],
      ]);
    }
    return pts;
  }

  _tween(duration, onFrame) {
    return new Promise((resolve) => {
      const start = performance.now();
      const tick = (now) => {
        const raw = Math.min(1, (now - start) / duration);
        onFrame(raw);
        if (raw < 1 && this.animating !== false) requestAnimationFrame(tick);
        else resolve();
      };
      requestAnimationFrame(tick);
    });
  }

  _ease(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  _clockLabel(clock) {
    if (clock == null) return "";
    const hours = Math.floor(clock / 3600);
    const minutes = Math.floor((clock % 3600) / 60);
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}`;
  }

  _timelineHtml(steps) {
    const rounds = [];
    for (const step of steps) {
      if (step.phase === "initial") { rounds.push({ round: 0, nodes: [step] }); continue; }
      let group = rounds.find((g) => g.round === step.round);
      if (!group) { group = { round: step.round, nodes: [] }; rounds.push(group); }
      group.nodes.push(step);
    }

    return rounds.map((g) => {
      const nodes = g.nodes.map((s) => this._timelineNode(s)).join("");
      return `<div class="tl-round"><span class="tl-round__lab">R${g.round + 1}</span><div class="tl-round__nodes">${nodes}</div></div>`;
    }).join("");
  }

  _timelineNode(step) {
    let cls = "tl-node--edit", glyph = "•", title = "";
    if (step.phase === "initial") { cls = "tl-node--start"; glyph = "▸"; title = "initial VROOM plan"; }
    else if (step.phase === "advance") { cls = "tl-node--advance"; glyph = "⏱"; title = this._advanceTitle(step); }
    else if (step.phase === "commit") { cls = "tl-node--commit"; glyph = "✓"; title = "commit — advance the world"; }
    else if (step.action && step.action.operator_name === "INSERT") { cls = "tl-node--insert"; glyph = "＋"; title = `insert job ${step.action.job_id} → veh ${step.action.vehicle_id}`; }
    else if (step.action && step.action.operator_name === "REMOVE") { cls = "tl-node--remove"; glyph = "－"; title = `remove job ${step.action.job_id} from veh ${step.action.vehicle_id}`; }

    return `<button class="tl-node ${cls}" data-seq="${step.seq}" title="${escapeHtml(title)}">${glyph}</button>`;
  }

  _advanceTitle(step) {
    const ev = (step.events && step.events[0]) || null;
    const gap = Math.round((step.gap || 0) / 60);
    const label = ev ? `${ev.count} ${(SCN_EVENT_LABELS[ev.type] || [ev.type, ev.type])[ev.count === 1 ? 0 : 1]}` : "disruption";
    return `advance +${gap} min · ${label}`;
  }

  _stepInfoHtml(step) {
    if (step.phase === "initial") {
      return `<div class="step-action is-noop">Initial plan at ${this._clockLabel(step.clock)} · cost ${fmtNum(step.state.cost, 0)} · ${step.state.num_unassigned} unassigned</div>`;
    }

    if (step.phase === "advance") return this._advanceInfoHtml(step);
    return this._actionInfoHtml(step);
  }

  _advanceInfoHtml(step) {
    const parts = [];
    const gap = Math.round((step.gap || 0) / 60);
    parts.push(`<div class="step-phase step-phase--advance">⏱ world advanced +${gap} min · clock ${this._clockLabel(step.clock)}</div>`);

    const ex = step.executed;
    if (ex) {
      if (ex.served.length) parts.push(`<div class="step-event step-event--ok">✓ served job(s) ${ex.served.join(", ")}</div>`);
      if (ex.failed.length) parts.push(`<div class="step-event step-event--bad">✕ repossession(s) ${ex.failed.join(", ")} failed (bike not found)</div>`);
      if (ex.dropped > 0) parts.push(`<div class="step-event">🏍 dropped ${ex.dropped} bike(s) at the depot</div>`);
    }

    for (const event of step.events || []) {
      const [one, many] = SCN_EVENT_LABELS[event.type] || [event.type, event.type];
      parts.push(`<div class="step-event step-event--disrupt">⚡ ${event.count} ${event.count === 1 ? one : many}</div>`);
    }

    parts.push(`<dl class="kv">
      <dt>plan cost</dt><dd>${fmtNum(step.state.cost, 0)}</dd>
      <dt>unassigned</dt><dd>${step.state.num_unassigned}</dd>
      <dt>cumulative</dt><dd>${fmtNum(step.cumulative_reward, 3)}</dd>
    </dl>
    <p class="scn-hint" style="margin:0">Advancing carries no reward — execution and the new disruption are exogenous. The agent responds below.</p>`);

    return parts.join("");
  }

  _actionInfoHtml(step) {
    const parts = [];
    const a = step.action;
    const cls = { INSERT: "", REMOVE: "is-remove", DO_NOTHING: "is-noop" }[a.operator_name] || "";
    let text = a.operator_name;
    if (a.operator_name === "INSERT") text = `＋ INSERT job ${a.job_id} → vehicle ${a.vehicle_id}`;
    if (a.operator_name === "REMOVE") text = `－ REMOVE job ${a.job_id} from vehicle ${a.vehicle_id}`;
    if (a.operator_name === "DO_NOTHING") text = `✓ DO_NOTHING — commit round ${step.round + 1}`;
    parts.push(`<div class="step-action ${cls}">${text}</div>`);

    const r = step.rewards || {};
    const c = step.costs || {};
    const rows = [
      ["distance", r.distance_reward],
      ["unassigned", r.unassigned_reward],
      ["idle", r.idle_reward],
      ["priority", r.priority_reward],
      ["action", r.action_reward],
    ];
    parts.push(`<dl class="kv">
      ${rows.map(([k, v]) => `<dt>${k}</dt><dd class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${fmtNum(v, 3)}</dd>`).join("")}
      <dt><b>step total</b></dt><dd class="${step.total_reward > 0 ? "pos" : step.total_reward < 0 ? "neg" : ""}"><b>${fmtNum(step.total_reward, 3)}</b></dd>
      <dt>cumulative</dt><dd>${fmtNum(step.cumulative_reward, 3)}</dd>
      ${c.disruption ? `<dt>disrupted jobs</dt><dd class="neg">${c.disruption}</dd>` : ""}
      <dt>plan cost</dt><dd>${fmtNum(step.state.cost, 0)}</dd>
      <dt>unassigned</dt><dd>${step.state.num_unassigned}</dd>
    </dl>`);

    return parts.join("");
  }

  _sparklineSvg(steps, currentSeq) {
    const values = steps.map((s) => s.cumulative_reward);
    const min = Math.min(0, ...values);
    const max = Math.max(0, ...values);
    const span = max - min || 1;
    const n = steps.length;
    const W = 100, H = 30, pad = 2;

    const x = (i) => pad + (n <= 1 ? 0 : (i / (n - 1)) * (W - 2 * pad));
    const y = (v) => pad + (1 - (v - min) / span) * (H - 2 * pad);
    const zeroY = y(0);

    const line = values.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const area = `${line} L${x(n - 1).toFixed(1)},${zeroY.toFixed(1)} L${x(0).toFixed(1)},${zeroY.toFixed(1)} Z`;
    const cx = x(currentSeq), cy = y(values[currentSeq]);
    const cur = values[currentSeq];
    const curCls = cur > 0 ? "pos" : cur < 0 ? "neg" : "";

    return `
      <div class="reward-spark__head"><span>cumulative reward</span><span class="reward-spark__val ${curCls}">${fmtNum(cur, 3)}</span></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" class="reward-spark__svg">
        <path d="${area}" class="rs-area"/>
        <line x1="0" y1="${zeroY.toFixed(1)}" x2="${W}" y2="${zeroY.toFixed(1)}" class="rs-zero"/>
        <path d="${line}" class="rs-line"/>
        <circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="2.2" class="rs-dot ${curCls}"/>
      </svg>`;
  }
}

window.ScenarioView = ScenarioView;
