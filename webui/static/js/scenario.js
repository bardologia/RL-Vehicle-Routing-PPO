"use strict";

class ScenarioView {
  constructor() {
    this.toolbar = document.getElementById("scn-toolbar");
    this.rail = document.getElementById("scn-rail");

    this.map = null;
    this.routeLayer = null;
    this.markerLayer = null;

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

    this.runOptions = { agent: "model", run_name: null, max_steps: 8, seed: 0, event_probability: 0.0 };
  }

  async enter() {
    if (!this.map) this._initMap();
    setTimeout(() => this.map.invalidateSize(), 60);

    const res = await apiGet("/api/scenario/checkpoints");
    this.checkpoints = res.checkpoints || [];
    if (!this.runOptions.run_name && this.checkpoints.length) this.runOptions.run_name = this.checkpoints[0].run;

    this._renderToolbar();
    this._renderRail();
    this._draw();
  }

  leave() {}

  _initMap() {
    this.map = L.map("scn-map", { zoomControl: true }).setView([-23.55, -46.63], 12);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors',
    }).addTo(this.map);

    this.routeLayer = L.layerGroup().addTo(this.map);
    this.markerLayer = L.layerGroup().addTo(this.map);

    this.map.on("click", (e) => this._onMapClick(e));
  }

  _color(vehicleId) {
    const order = this._displayVehicles().findIndex((v) => v.id === vehicleId);
    return LineChart.palette(order < 0 ? 0 : order);
  }

  _displayJobs() {
    if (this.runResult) return this.runResult.steps[this.stepIndex].jobs;
    return this.jobs;
  }

  _displayVehicles() {
    if (this.runResult) return this.runResult.steps[this.stepIndex].vehicles;
    return this.vehicles;
  }

  _displayState() {
    if (this.runResult) return this.runResult.steps[this.stepIndex].state;
    return this.solveState;
  }

  _onMapClick(e) {
    if (this.runResult) {
      toast("finish or discard the run to edit the scenario", "error");
      return;
    }
    if (this.mode === "job") this._addJob([e.latlng.lng, e.latlng.lat]);
    else if (this.mode === "vehicle") this._addVehicle([e.latlng.lng, e.latlng.lat]);
  }

  _nextId(list) {
    return list.length ? Math.max(...list.map((x) => x.id)) + 1 : 0;
  }

  _addJob(location) {
    const id = this._nextId(this.jobs);
    this.jobs.push({ id, location, service: 300, setup: 0, amount: 1, priority: 3, description: `Job ${id}` });
    this.selected = { type: "job", id };
    this._invalidate();
  }

  _addVehicle(location) {
    const id = this._nextId(this.vehicles);
    this.vehicles.push({ id, start: location, capacity: 4, speed_factor: 1.0, time_window: [28800, 72000], return_to_depot: false, description: `Vehicle ${id}` });
    this.selected = { type: "vehicle", id };
    this._invalidate();
  }

  _invalidate() {
    this.solveState = null;
    this.runResult = null;
    this.stepIndex = 0;
    this._renderRail();
    this._draw();
  }

  _renderToolbar() {
    this.toolbar.innerHTML = `
      <button class="btn btn--sm ${this.mode === "pan" ? "is-on" : ""}" data-mode="pan">Pan</button>
      <button class="btn btn--sm ${this.mode === "job" ? "is-on" : ""}" data-mode="job">+ Job</button>
      <button class="btn btn--sm ${this.mode === "vehicle" ? "is-on" : ""}" data-mode="vehicle">+ Vehicle</button>
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

  _draw() {
    if (!this.map) return;
    this.routeLayer.clearLayers();
    this.markerLayer.clearLayers();

    const state = this._displayState();
    const jobs = this._displayJobs();
    const vehicles = this._displayVehicles();

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

    for (const vehicle of vehicles) {
      const hasRoute = state && state.routes.some((r) => r.vehicle_id === vehicle.id);
      const color = hasRoute ? this._color(vehicle.id) : "#5b636a";
      const icon = L.divIcon({
        className: "",
        html: `<div class="veh-icon" style="width:22px;height:22px;background:${color}">${vehicle.id}</div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });
      const marker = L.marker([vehicle.start[1], vehicle.start[0]], { icon }).addTo(this.markerLayer);
      marker.on("click", () => this._selectEntity("vehicle", vehicle.id));
    }

    for (const job of jobs) {
      const assignedTo = assignment[job.id];
      const isUnassigned = state ? assignedTo == null : false;
      const color = assignedTo != null ? this._color(assignedTo) : isUnassigned ? "#b91c1c" : "#5b636a";
      const marker = L.circleMarker([job.location[1], job.location[0]], {
        radius: 5 + job.priority * 0.8,
        color: isUnassigned ? "#b91c1c" : color,
        fillColor: color,
        fillOpacity: assignedTo != null ? 0.85 : 0.35,
        weight: isUnassigned ? 2.5 : 1.5,
        dashArray: isUnassigned ? "3 3" : null,
      }).addTo(this.markerLayer);
      marker.bindTooltip(`Job ${job.id} · p${job.priority}${assignedTo != null ? ` · veh ${assignedTo}` : isUnassigned ? " · unassigned" : ""}`);
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

    panel.innerHTML = `
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
            <label class="field"><span class="field__label"><span>Priority (1-5)</span></span><input type="number" data-k="priority" value="${entity.priority}" min="1" max="5" step="1"></label>
            <label class="field"><span class="field__label"><span>Amount</span></span><input type="number" data-k="amount" value="${entity.amount}" min="1" step="1"></label>
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
          <label class="switch" style="margin-top:8px"><input type="checkbox" data-k="return_to_depot" ${entity.return_to_depot ? "checked" : ""}><i></i><span>return to depot</span></label>
          <div style="margin-top:12px"><button class="btn btn--sm btn--danger" id="scn-del">Delete vehicle</button></div>
        </div>`;
    }

    host.querySelectorAll("[data-k]").forEach((el) => {
      el.addEventListener("change", () => {
        const k = el.dataset.k;
        if (k === "return_to_depot") entity.return_to_depot = el.checked;
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

    const pts = [...this.jobs.map((j) => [j.location[1], j.location[0]]), ...this.vehicles.map((v) => [v.start[1], v.start[0]])];
    if (pts.length) this.map.fitBounds(L.latLngBounds(pts).pad(0.2));
    toast(`sampled ${this.jobs.length} jobs, ${this.vehicles.length} vehicles`, "ok");
  }

  async _solve() {
    if (!this.jobs.length || !this.vehicles.length) {
      toast("place at least one job and one vehicle", "error");
      return;
    }
    this.runResult = null;
    toast("solving…");
    const res = await apiPost("/api/scenario/solve", { jobs: this.jobs, vehicles: this.vehicles });
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
              ${["model", "teacher", "insertion_only", "always_reoptimize", "do_nothing"].map((a) => `<option value="${a}" ${a === o.agent ? "selected" : ""}>${a}</option>`).join("")}
            </select></label>
          <label class="field" id="scn-ckpt-wrap" style="${o.agent === "model" ? "" : "display:none"}"><span class="field__label"><span>Checkpoint</span></span>
            <select id="scn-ckpt">${ckptOptions || `<option value="">no checkpoints</option>`}</select></label>
          <label class="field"><span class="field__label"><span>Max steps</span></span><input type="number" id="scn-steps" value="${o.max_steps}" min="1" max="50" step="1"></label>
          <label class="field"><span class="field__label"><span>Seed</span></span><input type="number" id="scn-seed" value="${o.seed}" step="1"></label>
          <label class="field"><span class="field__label"><span>Event prob.</span></span><input type="number" id="scn-events" value="${o.event_probability}" min="0" max="1" step="0.1"></label>
          <div class="field" style="justify-content:flex-end"><button class="btn btn--primary" id="scn-run" ${this.runBusy ? "disabled" : ""}>${this.runBusy ? "Running…" : "Run agent"}</button></div>
        </div>
        <p class="scn-hint" style="margin:10px 0 0">With event probability &gt; 0, random disruptions (new/removed jobs and vehicles) fire between steps, like in training.</p>
      </div>
      <div id="scn-run-result"></div>`;

    document.getElementById("scn-agent").addEventListener("change", (e) => {
      o.agent = e.target.value;
      document.getElementById("scn-ckpt-wrap").style.display = o.agent === "model" ? "" : "none";
    });
    document.getElementById("scn-run").addEventListener("click", () => this._run());

    this._renderRunResult();
  }

  async _run() {
    if (!this.jobs.length || !this.vehicles.length) {
      toast("build a scenario first (Build tab)", "error");
      return;
    }

    const o = this.runOptions;
    o.run_name = (document.getElementById("scn-ckpt") || {}).value || null;
    o.max_steps = Number(document.getElementById("scn-steps").value) || 8;
    o.seed = Number(document.getElementById("scn-seed").value) || 0;
    o.event_probability = Number(document.getElementById("scn-events").value) || 0;

    this.runBusy = true;
    this._renderRunPanel();

    const res = await apiPost("/api/scenario/run", {
      jobs: this.jobs,
      vehicles: this.vehicles,
      agent: o.agent,
      run_name: o.run_name,
      max_steps: o.max_steps,
      seed: o.seed,
      event_probability: o.event_probability,
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
    toast(`${res.summary.total_steps} steps · ${res.summary.stopped_reason}`, "ok");
  }

  _renderRunResult() {
    const host = document.getElementById("scn-run-result");
    if (!this.runResult) {
      host.innerHTML = this.solveState ? "" : `<p class="scn-hint">Solve the scenario first so the agent starts from the VROOM plan.</p>`;
      return;
    }

    const { steps, summary } = this.runResult;
    const step = steps[this.stepIndex];

    host.innerHTML = `
      <div class="card">
        <h3 class="card__title">Run — ${escapeHtml(summary.agent)}${summary.run_name ? ` · ${escapeHtml(summary.run_name)}` : ""}</h3>
        <div class="summary-strip">
          <div class="cell"><b>${summary.total_steps}</b><span>steps</span></div>
          <div class="cell"><b>${fmtNum(summary.cumulative_reward, 3)}</b><span>total reward</span></div>
          <div class="cell"><b>${fmtNum(summary.initial_cost, 0)} → ${fmtNum(summary.final_cost, 0)}</b><span>cost</span></div>
          <div class="cell"><b>${summary.initial_unassigned} → ${summary.final_unassigned}</b><span>unassigned</span></div>
        </div>
        <p class="scn-hint" style="margin:8px 0 0">stopped: ${escapeHtml(summary.stopped_reason)}</p>
      </div>

      <div class="card">
        <h3 class="card__title">Playback</h3>
        <div class="playback">
          <button class="btn btn--sm" id="scn-prev">&larr;</button>
          <input type="range" id="scn-slider" min="0" max="${steps.length - 1}" value="${this.stepIndex}">
          <button class="btn btn--sm" id="scn-next">&rarr;</button>
          <span class="playback__step">${this.stepIndex}/${steps.length - 1}</span>
        </div>
        <div class="step-info" style="margin-top:12px">${this._stepInfoHtml(step)}</div>
      </div>

      <button class="btn btn--ghost" id="scn-discard">Discard run &amp; edit scenario</button>`;

    const slider = document.getElementById("scn-slider");
    slider.addEventListener("input", () => {
      this.stepIndex = Number(slider.value);
      this._renderRunResult();
      this._draw();
    });
    document.getElementById("scn-prev").addEventListener("click", () => this._step(-1));
    document.getElementById("scn-next").addEventListener("click", () => this._step(1));
    document.getElementById("scn-discard").addEventListener("click", () => {
      this.runResult = null;
      this.stepIndex = 0;
      this._renderRail();
      this._draw();
    });
  }

  _step(delta) {
    const max = this.runResult.steps.length - 1;
    this.stepIndex = Math.min(max, Math.max(0, this.stepIndex + delta));
    this._renderRunResult();
    this._draw();
  }

  _stepInfoHtml(step) {
    if (step.index === 0) {
      return `<div class="step-action is-noop">Initial VROOM plan · cost ${fmtNum(step.state.cost, 0)} · ${step.state.num_unassigned} unassigned</div>`;
    }

    const parts = [];

    for (const event of step.events || []) {
      const label = { new_job: "new job(s) appeared", remove_job: "job(s) cancelled", new_vehicle: "vehicle(s) arrived", remove_vehicle: "vehicle(s) broke down" }[event.type] || event.type;
      parts.push(`<div class="step-event">⚡ ${event.count} ${label}</div>`);
    }

    const a = step.action;
    const cls = { INSERT: "", REMOVE: "is-remove", DO_NOTHING: "is-noop", REOPTIMIZE: "is-reopt" }[a.operator_name] || "";
    let text = a.operator_name;
    if (a.operator_name === "INSERT") text = `INSERT job ${a.job_id} → vehicle ${a.vehicle_id}`;
    if (a.operator_name === "REMOVE") text = `REMOVE job ${a.job_id} from vehicle ${a.vehicle_id}`;
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
}

window.ScenarioView = ScenarioView;
