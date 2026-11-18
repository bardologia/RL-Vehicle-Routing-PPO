import { api } from "../api.js";
import { el, esc, fmtMin, fmtSigned, icon, isNum, NONE, plural } from "../format.js";
import { keys } from "../keys.js";
import { Router } from "../router.js";
import { AgentPicker } from "../episode/agent_picker.js";
import { Compare } from "../episode/compare.js";
import { DecisionPanel } from "../episode/decision_panel.js";
import { Agents, Sentences, Vehicles } from "../episode/encoding.js";
import { Inputs } from "../episode/inputs.js";
import { KpiStrip } from "../episode/kpi_strip.js";
import { Geometry, MapView } from "../episode/map_view.js";
import { RouteTable } from "../episode/route_table.js";
import { StartPage } from "../episode/start_page.js";
import { Timeline } from "../episode/timeline.js";
import { Position, TraceModel } from "../episode/trace_model.js";
import { Drawer } from "../ui/drawer.js";
import { errorBlock, skeleton, statusChip } from "../ui/status.js";
import { DataTable } from "../ui/table.js";

export class EpisodeView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.ref = null;
    this.status = null;
    this.lanes = [];
    this.selected = { lane: null, seq: null };
    this.pending = null;
    this.isolated = null;
    this.entity = null;
    this.verdicts = new Map();
    this.pollTimer = null;
    this.playTimer = null;
    this.popover = null;
    this.token = 0;
    this.built = false;

    this.startNode = el(`<div class="episode-start"></div>`);
    this.recordNode = el(`<div class="ep hidden"></div>`);
    root.append(this.startNode, this.recordNode);
    this.start = new StartPage(app, this.startNode);

    this.bindKeys();
    app.theme.onChange(() => {
      if (this.map) this.map.restyle();
    });
    app.onHealth(() => {
      if (this.built && this.ref) this.renderBar();
    });
  }

  bindKeys() {
    const bind = (key, description, handler) => keys.bind("episode", key, description, () => (this.active() ? handler() : false));
    bind("ArrowRight", "Next frame", () => this.step(1));
    bind("ArrowLeft", "Previous frame", () => this.step(-1));
    bind("Shift+ArrowRight", "Next round", () => this.stepRound(1));
    bind("Shift+ArrowLeft", "Previous round", () => this.stepRound(-1));
    bind("Home", "First frame", () => this.select(this.selected.lane, 0));
    bind("End", "Last frame", () => this.select(this.selected.lane, this.lane().trace.length - 1));
    bind("Space", "Play or pause", () => this.togglePlay());
    bind("ArrowUp", "Previous lane", () => this.stepLane(-1));
    bind("ArrowDown", "Next lane", () => this.stepLane(1));
    bind("v", "Cycle vehicle isolation", () => this.cycleIsolation());
    bind("0", "Clear isolation", () => this.isolate(null));
    bind("Escape", "Close popovers and clear isolation", () => {
      this.closePopover();
      this.isolate(null);
    });
    bind("t", "Ask the teacher (insertions)", () => this.askTeacher("insertions"));
    bind("b", "Toggle buildings", () => this.map.control("buildings"));
    bind("g", "Toggle the ghost lane", () => this.map.control("ghost"));
    bind("f", "Fit the plan", () => this.map.control("fit"));
    bind("l", "Toggle the legend", () => this.map.toggleLegend());
    bind("]", "Next finding", () => this.stepFinding(1));
    bind("[", "Previous finding", () => this.stepFinding(-1));
  }

  active() {
    return Boolean(this.built && this.ref && this.lane() && this.selected.seq !== null);
  }

  lane(key) {
    return this.lanes.find((lane) => lane.key === (key === undefined ? this.selected.lane : key)) || null;
  }

  enter(parts) {
    const [kind] = parts;
    if (kind === "ep" && parts[1]) return this.open({ kind: "ep", episodeId: parts[1] }, parts[2], parts[3]);
    if (kind === "diag" && parts.length >= 4) return this.open({ kind: "diag", run: parts[1], seed: Number(parts[2]), index: Number(parts[3]) }, parts[4], parts[5]);
    if (kind === "seed" && parts.length >= 4) return this.recordFromReport(parts[1], Number(parts[2]), Number(parts[3]));
    return this.showStart();
  }

  leave() {
    this.stopPolling();
    this.stopPlay();
    this.closePopover();
  }

  showStart() {
    this.stopPolling();
    this.stopPlay();
    this.ref = null;
    this.recordNode.classList.add("hidden");
    this.startNode.classList.remove("hidden");
    this.start.show();
  }

  showRecording() {
    this.startNode.classList.add("hidden");
    this.recordNode.classList.remove("hidden");
  }

  fail(cause, details) {
    this.stopPolling();
    this.stopPlay();
    this.built = false;
    this.map = null;
    this.ref = null;
    this.showRecording();
    this.recordNode.innerHTML = `<div class="view__pad">${errorBlock(cause, details)}<p><a class="btn" href="#/episode">Back to the episode start page</a></p></div>`;
  }

  async recordFromReport(run, seed, index) {
    this.showRecording();
    this.built = false;
    this.map = null;
    this.recordNode.innerHTML = `<div class="view__pad"><p class="muted">Recording episode ${index} of ${esc(run)}, report seed ${seed}.</p>${skeleton(3)}</div>`;
    let created;
    try {
      created = await api.post("/api/episodes", { source: { kind: "evaluation", run, report_seed: seed, index }, agents: [{ agent: "model", run_name: run, greedy: true }, { agent: "insertion_only", run_name: null, greedy: true }, { agent: "do_nothing", run_name: null, greedy: true }] });
    } catch (error) {
      this.fail(`Episode ${index} of ${run} could not be recorded.`, error.message);
      return;
    }
    this.app.router.replace(`#/episode/ep/${created.episode_id}/${encodeURIComponent(created.agent_keys[0])}/0`);
  }

  sameRef(ref) {
    return this.ref && JSON.stringify(this.ref) === JSON.stringify(ref);
  }

  async open(ref, laneKey, positionText) {
    if (this.sameRef(ref) && this.built) {
      const lane = this.lane(laneKey) || this.lane();
      if (lane && lane.trace.length) {
        const seq = lane.trace.resolve(Position.parse(positionText));
        this.select(lane.key, seq === null ? 0 : seq);
      }
      if (this.isRunning() && !this.pollTimer) this.poll();
      return;
    }

    this.stopPolling();
    this.stopPlay();
    this.token += 1;
    const token = this.token;
    this.ref = ref;
    this.status = null;
    this.lanes = [];
    this.selected = { lane: null, seq: null };
    this.pending = { lane: laneKey || null, position: positionText || null };
    this.isolated = null;
    this.entity = null;
    this.verdicts.clear();
    this.showRecording();
    this.build();

    try {
      if (ref.kind === "ep") this.applyStatus(await api.get(`/api/episodes/${encodeURIComponent(ref.episodeId)}`));
      else this.applyStatus(await this.diagnosticsStatus(ref));
    } catch (error) {
      if (token === this.token) this.fail(ref.kind === "ep" ? `Recording ${ref.episodeId} could not be opened.` : `The stored trace of episode ${ref.index} of ${ref.run} could not be opened.`, error.message);
      return;
    }
    if (token !== this.token) return;
    this.renderBar();
    this.renderTimeline();
    await this.loadTraces();
    if (token !== this.token) return;
    this.settleSelection();
    this.renderAll();
    if (this.isRunning()) this.poll();
  }

  async diagnosticsStatus(ref) {
    const index = await api.get(`/api/runs/${encodeURIComponent(ref.run)}/diagnostics/${ref.seed}`);
    const episode = index.episodes.find((entry) => entry.index === ref.index);
    if (!episode) throw new Error(`The diagnostics index of ${ref.run} has no episode ${ref.index}.`);
    const lanes = {};
    for (const [key, summary] of Object.entries(episode.agents)) {
      lanes[key] = { status: "finished", round: index.meta.caps.rounds - 1, decisions: summary.decisions, frames: 1, elapsed_s: 0, error: null, traceback: null, total_reward: summary.total_reward };
    }
    return {
      episode_id: null,
      status: "finished",
      source: { kind: "diagnostics", run: ref.run, report_seed: ref.seed, index: ref.index, seed: episode.seed },
      caps: index.meta.caps,
      lanes,
      shared: null,
      divergence: episode.divergent_rounds.map((round) => ({ round, kind: "event_ids", agents: Object.keys(episode.agents) })),
    };
  }

  traceUrl(key) {
    if (this.ref.kind === "ep") return `/api/traces/ep/${encodeURIComponent(this.ref.episodeId)}/${encodeURIComponent(key)}`;
    return `/api/traces/diag/${encodeURIComponent(this.ref.run)}/${this.ref.seed}/${this.ref.index}/${encodeURIComponent(key)}`;
  }

  applyStatus(status) {
    this.status = status;
    const ordered = Object.keys(status.lanes).sort((a, b) => Agents.order(a) - Agents.order(b) || a.localeCompare(b));
    this.lanes = ordered.map((key) => {
      const existing = this.lanes.find((lane) => lane.key === key);
      if (existing) {
        existing.info = status.lanes[key];
        return existing;
      }
      return { key, info: status.lanes[key], trace: new TraceModel(key), loadError: null, final: false };
    });
  }

  isRunning() {
    return this.lanes.some((lane) => ["queued", "running"].includes(lane.info.status));
  }

  async loadTraces() {
    let changed = false;
    await Promise.all(this.lanes.map(async (lane) => {
      const running = lane.info.status === "running";
      const done = ["finished", "failed", "cancelled"].includes(lane.info.status);
      if (lane.final || lane.info.frames === 0) return;
      if (running && lane.info.frames <= lane.trace.length) return;
      if (!running && !done) return;
      try {
        if (running) {
          lane.trace.merge(await api.get(`${this.traceUrl(lane.key)}?from=${lane.trace.length}`));
        } else {
          const fresh = new TraceModel(lane.key);
          fresh.merge(await api.get(this.traceUrl(lane.key)));
          lane.trace = fresh;
          lane.final = true;
        }
        lane.loadError = null;
        changed = true;
      } catch (error) {
        if (done) {
          lane.final = true;
          lane.loadError = error.message;
          changed = true;
        }
      }
    }));
    return changed;
  }

  settleSelection() {
    const withFrames = this.lanes.filter((lane) => lane.trace.length);
    if (!withFrames.length) return;
    if (this.selected.lane && this.lane() && this.lane().trace.frame(this.selected.seq)) return;
    const wanted = this.pending && this.pending.lane ? withFrames.find((lane) => lane.key === this.pending.lane) : null;
    const lane = wanted || withFrames[0];
    const seq = this.pending ? lane.trace.resolve(Position.parse(this.pending.position)) : null;
    this.selected = { lane: lane.key, seq: seq === null ? 0 : seq };
    if (wanted || !this.pending || !this.pending.lane) this.pending = null;
  }

  poll() {
    clearTimeout(this.pollTimer);
    const token = this.token;
    this.pollTimer = setTimeout(async () => {
      this.pollTimer = null;
      if (token !== this.token || this.ref.kind !== "ep") return;
      let status;
      try {
        status = await api.get(`/api/episodes/${encodeURIComponent(this.ref.episodeId)}`);
      } catch (error) {
        if (token === this.token) this.fail("The recording stopped answering.", error.message);
        return;
      }
      if (token !== this.token) return;
      const before = this.selected.lane;
      this.applyStatus(status);
      const changed = await this.loadTraces();
      if (token !== this.token) return;
      const hadSelection = this.selected.seq !== null;
      this.settleSelection();
      if (!hadSelection && this.selected.seq !== null) this.renderAll();
      else {
        this.renderBar();
        this.renderTimeline();
        this.renderAlerts();
        if (changed) {
          if (this.lane(before) && this.lane(before).final) this.renderFrame(false);
          this.renderDrawer();
        }
      }
      if (this.isRunning()) this.poll();
    }, 500);
  }

  stopPolling() {
    clearTimeout(this.pollTimer);
    this.pollTimer = null;
  }

  build() {
    if (this.map) this.map.destroy();
    if (this.timeline) this.timeline.observer.disconnect();
    this.recordNode.innerHTML = `
      <div class="ep__bar"></div>
      <div class="ep__alerts"></div>
      <div class="ep__timeline"></div>
      <div class="ep__main">
        <div class="ep__left">
          <div class="ep__kpis"></div>
          <div class="ep__map"></div>
        </div>
        <aside class="ep__decision" aria-label="Decision"></aside>
      </div>`;
    this.barNode = this.recordNode.querySelector(".ep__bar");
    this.alertsNode = this.recordNode.querySelector(".ep__alerts");
    this.kpiNode = this.recordNode.querySelector(".ep__kpis");
    const left = this.recordNode.querySelector(".ep__left");

    this.timeline = new Timeline(this.recordNode.querySelector(".ep__timeline"), { onSelect: (laneKey, seq) => this.select(laneKey, seq) });
    this.map = new MapView(this.recordNode.querySelector(".ep__map"), {
      onSelect: (entity) => this.selectEntity(entity),
      onIsolate: (vehicleId) => this.isolate(this.isolated === vehicleId ? null : vehicleId),
      onToggle: () => this.renderOverlays(),
    });
    this.decision = new DecisionPanel(this.recordNode.querySelector(".ep__decision"), {
      onHighlight: (target) => this.map.highlight(target),
      onIsolate: (vehicleId) => this.isolate(this.isolated === vehicleId ? null : vehicleId),
      onJump: (seq) => this.select(this.selected.lane, seq),
      onVerdict: (scope) => this.askTeacher(scope),
    });
    this.drawer = new Drawer(left, {
      storeKey: "episode.drawer",
      startCollapsed: window.innerHeight < 900,
      tabs: [{ value: "routes", label: "Routes" }, { value: "unassigned", label: "Unassigned" }, { value: "compare", label: "Compare" }, { value: "inputs", label: "Inputs" }],
      onTab: () => this.renderDrawer(),
    });
    this.drawer.body.innerHTML = `<div data-pane="routes"></div><div data-pane="unassigned"></div><div data-pane="compare"></div><div data-pane="inputs"></div>`;
    this.routeTable = new RouteTable(this.drawer.body.querySelector("[data-pane=routes]"), {
      onIsolate: (vehicleId) => this.isolate(this.isolated === vehicleId ? null : vehicleId),
      onHover: (vehicleId) => this.map.highlight(vehicleId === null ? null : { vehicleId }),
    });
    this.unassignedTable = new DataTable(this.drawer.body.querySelector("[data-pane=unassigned]"), {
      rowKey: (row) => row.job_id,
      emptyText: "Every job is on a route in this frame.",
      onSelect: (row) => this.selectEntity({ kind: "job", id: row.job_id }),
      onHover: (row) => this.map.highlight(row ? { jobId: row.job_id } : null),
      sort: { key: "priority", descending: true },
      columns: [
        { key: "job_id", label: "Job", render: (row) => `job ${row.job_id}` },
        { key: "kind", label: "Kind" },
        { key: "priority", label: "Priority", numeric: true },
        { key: "waiting_s", label: "Waiting", unit: "min", numeric: true, format: (value) => fmtMin(value) },
        { key: "best_reward", label: "Best insertion", unit: "pts", numeric: true, render: (row) => (row.verdict ? (isNum(row.best_reward) ? fmtSigned(row.best_reward, 2) : `<span class="muted">infeasible</span>`) : `<span class="faint">ask the teacher</span>`) },
        { key: "best_vehicle", label: "Into vehicle", render: (row) => (isNum(row.best_vehicle) ? Vehicles.swatch(row.best_vehicle) : NONE) },
      ],
    });

    this.barNode.addEventListener("click", (event) => this.onBarClick(event));
    this.built = true;
  }

  sourceText() {
    const source = this.status.source;
    const caps = `${plural(this.status.caps.rounds, "round")}, up to ${this.status.caps.edits} edits each`;
    if (source.kind === "diagnostics") return `Stored trace: episode ${source.index} of ${source.run}, report seed ${source.report_seed}, ${caps}`;
    return `${StartPage.sourceText(source)}, ${caps}`;
  }

  flaggedSeqs() {
    const lane = this.lane();
    if (!lane) return [];
    const seqs = new Set();
    for (const finding of lane.trace.findings) {
      if (finding.status === "WARN" || finding.status === "FAIL") for (const seq of finding.seqs) seqs.add(seq);
    }
    return [...seqs].sort((a, b) => a - b);
  }

  renderBar() {
    if (!this.status) {
      this.barNode.innerHTML = `<span class="muted">Opening the recording.</span>`;
      return;
    }
    const lane = this.lane();
    const running = this.isRunning();
    const isEpisode = this.ref.kind === "ep";
    const frames = lane ? lane.trace.length : 0;
    const position = lane && this.selected.seq !== null ? `frame ${this.selected.seq + 1} of ${frames}` : "no frame yet";
    const flagged = lane ? lane.trace.findings.filter((finding) => finding.status === "WARN" || finding.status === "FAIL").length : 0;
    const expectations = this.lanes.filter((entry) => entry.trace.expectation).map((entry) => statusChip(entry.trace.expectation.status, Agents.label(entry.key)));
    const hasTeacher = this.lanes.some((entry) => entry.key === "teacher");
    const up = this.app.servicesUp;
    const playing = Boolean(this.playTimer);

    this.barNode.innerHTML = `
      <a class="btn btn--quiet btn--sm" href="#/episode" title="Back to the episode start page">${icon("chevron-right", "icon--flip")}<span>Episodes</span></a>
      <span class="ep__source" title="${esc(this.sourceText())}">${esc(this.sourceText())}</span>
      <span class="ep__transport" role="group" aria-label="Playback">
        <button type="button" class="btn btn--sm" data-action="back" title="Previous frame (Left)" aria-label="Previous frame">${icon("step-back")}</button>
        <button type="button" class="btn btn--sm" data-action="play" title="Play or pause (Space)" aria-pressed="${playing}">${icon(playing ? "pause" : "play")}<span>${playing ? "Pause" : "Play"}</span></button>
        <button type="button" class="btn btn--sm" data-action="forward" title="Next frame (Right)" aria-label="Next frame">${icon("step-forward")}</button>
        <span class="ep__position muted">${position}</span>
      </span>
      ${expectations.length ? `<span class="ep__expect"><span class="muted">Expectation:</span>${expectations.join("")}</span>` : ""}
      <span class="ep__spacer"></span>
      ${!up && isEpisode ? `<span class="ep__down">${statusChip("DOWN", "routing services")}<code>sg docker -c "docker compose up -d"</code></span>` : ""}
      <button type="button" class="btn btn--sm" data-action="findings" aria-haspopup="dialog">${icon("flag")}<span>Findings</span><span class="tabs__count">${flagged}</span></button>
      ${this.status.source.kind === "scenario" ? `<a class="btn btn--sm" href="#/lab?episode=${esc(this.ref.episodeId)}">${icon("edit")}<span>Edit scenario</span></a>` : ""}
      ${isEpisode ? `<button type="button" class="btn btn--sm" data-action="add" aria-haspopup="dialog"${up ? "" : " disabled"}>${icon("plus")}<span>Add agent</span></button>` : ""}
      ${isEpisode && !hasTeacher ? `<button type="button" class="btn btn--sm" data-action="teacher" title="Adds a teacher lane: slow, about 30 to 150 s"${up ? "" : " disabled"}>${icon("plus")}<span>Add teacher</span></button>` : ""}
      ${isEpisode && running ? `<button type="button" class="btn btn--sm" data-action="cancel">${icon("stop")}<span>Cancel</span></button>` : ""}
      ${!isEpisode ? `<a class="btn btn--sm" href="#/episode/seed/${encodeURIComponent(this.ref.run)}/${this.ref.seed}/${this.ref.index}" title="Record this episode again as an interactive recording, to add agents">${icon("record")}<span>Record interactively</span></a>` : ""}
      <div class="ep__error" data-role="bar-error"></div>`;
  }

  renderAlerts() {
    const failed = this.lanes.filter((lane) => lane.info.status === "failed");
    const broken = this.lanes.filter((lane) => lane.loadError && lane.info.status !== "failed");
    this.alertsNode.innerHTML = [
      ...failed.map((lane) => errorBlock(`${Agents.label(lane.key)}: ${lane.info.error || "the lane failed"}`, lane.info.traceback)),
      ...broken.map((lane) => errorBlock(`${Agents.label(lane.key)}: the trace could not be read.`, lane.loadError)),
    ].join("");
  }

  renderTimeline() {
    if (!this.status) return;
    const frame = this.frame();
    this.timeline.render({ lanes: this.lanes, selected: { lane: this.selected.lane, seq: this.selected.seq, round: frame ? frame.round : null }, divergence: this.status.divergence, caps: this.status.caps });
  }

  frame() {
    const lane = this.lane();
    return lane && this.selected.seq !== null ? lane.trace.frame(this.selected.seq) : null;
  }

  verdictKey() {
    return `${this.selected.lane}:${this.selected.seq}`;
  }

  verdictState() {
    return this.verdicts.get(this.verdictKey()) || { status: "idle" };
  }

  renderAll() {
    this.renderBar();
    this.renderAlerts();
    this.renderTimeline();
    this.renderFrame(false);
  }

  renderFrame(animate) {
    const lane = this.lane();
    const frame = this.frame();
    if (!lane || !frame) {
      this.kpiNode.innerHTML = "";
      this.map.setScene(null);
      this.map.setCaption(this.isRunning() ? "Recording. The first frame arrives after the opening solve." : "No frames were recorded.");
      this.decision.container.innerHTML = this.isRunning() ? `<div class="decision__head">${skeleton(4)}</div>` : `<p class="decision__rule muted">No lane of this recording has frames.</p>`;
      this.renderDrawer();
      return;
    }

    const trace = lane.trace;
    this.app.router.silent(Router.build(["episode", ...this.refParts(), lane.key, Position.of(frame)]));
    this.kpiNode.innerHTML = KpiStrip.html(trace, frame.seq);
    this.map.setScene({ depot: trace.depot, jobs: trace.entities.jobs, vehicles: frame.plan.vehicles, plan: frame.plan, pathOf: (route) => trace.path(route) });
    this.map.isolate(this.isolated);
    this.map.setCaption(`<span class="lanekey lanekey--${Agents.style(lane.key)}"><i></i>${esc(Agents.label(lane.key))}</span><b>${esc(Sentences.frame(frame, trace.meta.caps))}</b>${frame.decision && frame.decision.wasted ? statusChip("WASTED") : ""}`);
    this.map.showChange(this.change(trace, frame, animate));
    this.renderOverlays();
    this.decision.render({ trace, seq: frame.seq, verdictState: this.verdictState() });
    this.renderDrawer();
  }

  refParts() {
    if (this.ref.kind === "ep") return ["ep", this.ref.episodeId];
    return ["diag", this.ref.run, this.ref.seed, this.ref.index];
  }

  change(trace, frame, animate) {
    const previous = frame.seq > 0 ? trace.frame(frame.seq - 1) : null;
    if (frame.kind === "decision") {
      const action = frame.decision.action;
      if (action.operator === 0) return { inserted: { jobId: action.job_id, vehicleId: action.vehicle_id, wasted: frame.decision.wasted } };
      if (action.operator === 1) {
        const oldRoute = previous ? previous.plan.routes.find((route) => route.vehicle_id === action.vehicle_id) : null;
        return { removed: { jobId: action.job_id, vehicleId: action.vehicle_id, oldPath: oldRoute ? trace.path(oldRoute) : null } };
      }
      return null;
    }

    const open = frame.round_open;
    const change = {
      outcomes: [...open.executed.served.map((entry) => ({ jobId: entry.job_id, word: "served" })), ...open.executed.failed.map((entry) => ({ jobId: entry.job_id, word: "failed" }))],
      arrived: open.event ? open.event.added_job_ids : [],
      moves: [],
    };
    if (animate && previous) {
      for (const route of previous.plan.routes) {
        const now = frame.plan.vehicles[route.vehicle_id];
        const path = trace.path(route);
        if (!now || path.length < 2) continue;
        const end = Geometry.nearestIndex(path, now.start);
        if (end > 0) change.moves.push({ vehicleId: route.vehicle_id, path: path.slice(0, end + 1) });
      }
    }
    return change;
  }

  ghostLane() {
    const others = this.lanes.filter((lane) => lane.key !== this.selected.lane && lane.trace.length);
    return others.length ? others[0] : null;
  }

  renderOverlays() {
    const frame = this.frame();
    if (!frame) return;
    const ghost = this.ghostLane();
    const verdict = frame.decision ? frame.decision.verdict : null;
    const alternative = verdict && verdict.teacher_action.operator === 0 ? verdict.teacher_action : verdict && verdict.best ? verdict.best : null;
    this.map.setAvailable({ ghost: Boolean(ghost), teacher: Boolean(alternative) });
    if (ghost) {
      const seq = ghost.trace.nearest(frame.round, frame.kind === "round_open" ? null : 99);
      const other = seq === null ? null : ghost.trace.frame(seq);
      this.map.setGhost(other ? other.plan.routes : null, (route) => ghost.trace.path(route));
    } else {
      this.map.setGhost(null);
    }
    this.map.setTeacherAlternative(alternative);
  }

  renderDrawer() {
    const frame = this.frame();
    const tab = this.drawer.tab;
    for (const pane of this.drawer.body.querySelectorAll("[data-pane]")) pane.classList.toggle("hidden", pane.dataset.pane !== tab);
    this.drawer.tabs.setCount("unassigned", frame ? frame.plan.unassigned.length : null);
    if (tab === "compare") {
      this.drawer.body.querySelector("[data-pane=compare]").innerHTML = this.status && this.status.shared ? Compare.html(this.lanes, this.status) : Compare.html(this.lanes, this.derivedShared());
      return;
    }
    if (!frame) return;
    if (tab === "routes") this.routeTable.render(frame.plan, this.isolated);
    if (tab === "unassigned") this.unassignedTable.setRows(this.unassignedRows(frame));
    if (tab === "inputs") this.drawer.body.querySelector("[data-pane=inputs]").innerHTML = Inputs.html(frame, this.entity);
  }

  derivedShared() {
    const reference = this.lanes.find((lane) => lane.trace.length);
    const rounds = reference ? reference.trace.rounds() : [];
    return {
      shared: {
        gaps: rounds.map((round) => (round && round.open ? round.open.round_open.gap_s : 0)),
        events: rounds.map((round) => (round && round.open && round.open.round_open.event ? { type: round.open.round_open.event.type, count: round.open.round_open.event.count } : null)),
      },
      divergence: this.status ? this.status.divergence : [],
    };
  }

  unassignedRows(frame) {
    const verdict = frame.decision ? frame.decision.verdict : null;
    return frame.plan.unassigned.map((entry) => {
      const row = { ...entry, verdict: Boolean(verdict), best_reward: null, best_vehicle: null };
      if (verdict) {
        const options = verdict.insertions.filter((option) => option.job_id === entry.job_id);
        if (options.length) {
          const best = options.reduce((a, b) => (b.reward > a.reward ? b : a));
          row.best_reward = best.reward;
          row.best_vehicle = best.vehicle_id;
        }
      }
      return row;
    });
  }

  select(laneKey, seq) {
    const lane = this.lane(laneKey);
    if (!lane || !lane.trace.length) return;
    let target = seq;
    if (target === null || target === undefined) {
      const current = this.frame();
      target = current ? lane.trace.nearest(current.round, current.kind === "round_open" ? null : current.edit) : 0;
    }
    target = Math.max(0, Math.min(lane.trace.length - 1, target));
    if (!lane.trace.frame(target)) return;
    const animate = laneKey === this.selected.lane && target === this.selected.seq + 1;
    if (laneKey === this.selected.lane && target === this.selected.seq) return;
    this.selected = { lane: laneKey, seq: target };
    this.closePopover();
    this.renderBar();
    this.renderTimeline();
    this.renderFrame(animate);
  }

  step(delta) {
    this.select(this.selected.lane, this.selected.seq + delta);
  }

  stepRound(delta) {
    const lane = this.lane();
    const frame = this.frame();
    const target = frame.kind === "decision" && delta < 0 ? frame.round : frame.round + delta;
    const open = lane.trace.roundOpen(target);
    if (open) this.select(lane.key, open.seq);
  }

  stepLane(delta) {
    const usable = this.lanes.filter((lane) => lane.trace.length);
    const index = usable.findIndex((lane) => lane.key === this.selected.lane);
    const next = usable[index + delta];
    if (next) this.select(next.key, null);
  }

  stepFinding(delta) {
    const seqs = this.flaggedSeqs();
    const next = delta > 0 ? seqs.find((seq) => seq > this.selected.seq) : [...seqs].reverse().find((seq) => seq < this.selected.seq);
    if (next !== undefined) this.select(this.selected.lane, next);
  }

  togglePlay() {
    if (this.playTimer) this.stopPlay();
    else {
      if (this.selected.seq >= this.lane().trace.length - 1) this.select(this.selected.lane, 0);
      this.map.playing = true;
      this.playTimer = setInterval(() => {
        if (this.selected.seq >= this.lane().trace.length - 1) this.stopPlay();
        else this.step(1);
      }, 900);
    }
    this.renderBar();
  }

  stopPlay() {
    clearInterval(this.playTimer);
    this.playTimer = null;
    if (this.map) this.map.playing = false;
    if (this.built && this.status) this.renderBar();
  }

  isolate(vehicleId) {
    this.isolated = vehicleId;
    this.map.isolate(vehicleId);
    this.routeTable.setIsolated(vehicleId);
  }

  cycleIsolation() {
    const frame = this.frame();
    const ids = frame.plan.routes.map((route) => route.vehicle_id).sort((a, b) => a - b);
    if (!ids.length) return;
    const index = ids.indexOf(this.isolated);
    this.isolate(index === ids.length - 1 ? null : ids[index + 1]);
  }

  selectEntity(entity) {
    this.entity = entity;
    this.map.highlight(entity.kind === "job" ? { jobId: entity.id } : { vehicleId: entity.id });
    if (this.drawer.tab === "inputs") this.renderDrawer();
  }

  async askTeacher(scope) {
    const frame = this.frame();
    const lane = this.lane();
    if (!frame || frame.kind !== "decision") return;
    const key = this.verdictKey();
    if (this.verdictState().status === "running") return;
    const stored = frame.decision.verdict;
    if (stored && (stored.scope === "full" || stored.scope === scope)) return;

    this.verdicts.set(key, { status: "running", scope });
    this.decision.render({ trace: lane.trace, seq: frame.seq, verdictState: this.verdictState() });
    try {
      frame.decision.verdict = await api.post(`${this.traceUrl(lane.key)}/verdict`, { seq: frame.seq, scope });
      this.verdicts.set(key, { status: "idle" });
      const refreshed = await api.get(`${this.traceUrl(lane.key)}?from=${lane.trace.length - 1}`);
      lane.trace.findings = refreshed.findings;
      lane.trace.expectation = refreshed.expectation;
    } catch (error) {
      this.verdicts.set(key, { status: "error", scope, error: error.message });
    }
    if (this.verdictKey() === key) {
      this.decision.render({ trace: lane.trace, seq: frame.seq, verdictState: this.verdictState() });
      this.renderOverlays();
      this.renderDrawer();
      this.renderBar();
    }
  }

  onBarClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button || button.disabled) return;
    const action = button.dataset.action;
    if (action === "back" && this.active()) this.step(-1);
    if (action === "forward" && this.active()) this.step(1);
    if (action === "play" && this.active()) this.togglePlay();
    if (action === "findings") this.togglePopover("findings", button);
    if (action === "add") this.togglePopover("add", button);
    if (action === "teacher") this.addAgents([{ agent: "teacher", run_name: null, greedy: true }]);
    if (action === "cancel") this.cancel();
  }

  barError(cause, details) {
    this.barNode.querySelector("[data-role=bar-error]").innerHTML = errorBlock(cause, details);
  }

  async addAgents(agents) {
    try {
      await api.post(`/api/episodes/${encodeURIComponent(this.ref.episodeId)}/agents`, { agents });
    } catch (error) {
      this.barError("The lane was not added.", error.message);
      return;
    }
    this.closePopover();
    this.applyStatus(await api.get(`/api/episodes/${encodeURIComponent(this.ref.episodeId)}`));
    this.renderBar();
    this.renderTimeline();
    this.poll();
  }

  async cancel() {
    try {
      await api.post(`/api/episodes/${encodeURIComponent(this.ref.episodeId)}/cancel`, {});
    } catch (error) {
      this.barError("The recording was not cancelled.", error.message);
    }
  }

  closePopover() {
    if (this.popover) this.popover.remove();
    this.popover = null;
    this.popoverKind = null;
  }

  togglePopover(kind, anchor) {
    const same = this.popoverKind === kind;
    this.closePopover();
    if (same) return;
    const box = anchor.getBoundingClientRect();
    const node = el(`<div class="popover" role="dialog" aria-label="${kind === "findings" ? "Episode findings" : "Add an agent"}"></div>`);
    node.style.top = `${box.bottom + 6}px`;
    node.style.right = `${Math.max(8, window.innerWidth - box.right)}px`;
    document.body.appendChild(node);
    this.popover = node;
    this.popoverKind = kind;
    if (kind === "findings") this.fillFindings(node);
    else this.fillAdd(node);
  }

  fillFindings(node) {
    const lane = this.lane();
    const findings = lane ? lane.trace.findings : [];
    const order = { FAIL: 0, WARN: 1, PASS: 2, "NO DATA": 3 };
    const rows = [...findings].sort((a, b) => order[a.status] - order[b.status]).map((finding) => {
      const jumps = finding.seqs.slice(0, 14).map((seq) => `<button type="button" class="btn btn--sm" data-jump="${seq}">${seq + 1}</button>`).join("");
      return `
        <li class="finding">
          <div class="finding__head">${finding.status === "NO DATA" ? `<span class="chip chip--plain"><span class="chip__word">NO DATA</span></span>` : statusChip(finding.status)}<b>${esc(finding.id)} ${esc(finding.title)}</b></div>
          <p>${esc(finding.text)}</p>
          ${jumps ? `<div class="finding__jumps"><span class="muted">frames</span>${jumps}${finding.seqs.length > 14 ? `<span class="muted">and ${finding.seqs.length - 14} more</span>` : ""}</div>` : ""}
        </li>`;
    }).join("");
    node.innerHTML = `
      <header class="popover__head"><h3>Findings for ${esc(lane ? Agents.label(lane.key) : "")}</h3><span class="muted">keys [ and ] step through flagged frames</span></header>
      ${rows ? `<ul class="findings">${rows}</ul>` : `<p class="muted">${lane && !lane.trace.complete ? "Findings are computed when the lane has finished." : "No findings for this lane."}</p>`}`;
    node.addEventListener("click", (event) => {
      const jump = event.target.closest("[data-jump]");
      if (jump) this.select(this.selected.lane, Number(jump.dataset.jump));
    });
  }

  async fillAdd(node) {
    node.innerHTML = skeleton(3);
    let checkpoints;
    try {
      checkpoints = (await api.get("/api/checkpoints")).checkpoints;
    } catch (error) {
      node.innerHTML = errorBlock("The checkpoint list could not be loaded.", error.message);
      return;
    }
    node.innerHTML = `<header class="popover__head"><h3>Add lanes to this episode</h3></header><div data-role="picker"></div><div class="popover__foot"><button type="button" class="btn btn--primary" data-action="confirm-add">Add</button><span class="muted">Lanes already recorded are skipped.</span></div><div data-role="add-error"></div>`;
    const picker = new AgentPicker(node.querySelector("[data-role=picker]"), { checkpoints, preferredRun: null, storeKey: "episode.add" });
    node.querySelector("[data-action=confirm-add]").addEventListener("click", () => {
      const existing = new Set(this.lanes.map((lane) => lane.key));
      const agents = picker.agents().filter((agent) => !existing.has(agent.agent === "model" ? `model~${agent.run_name}~${agent.greedy ? "greedy" : "sampled"}` : agent.agent));
      if (!agents.length) {
        node.querySelector("[data-role=add-error]").innerHTML = errorBlock("Every selected agent already has a lane.");
        return;
      }
      this.addAgents(agents);
    });
  }
}
