import { api } from "../api.js";
import { esc, icon } from "../format.js";
import { Orb } from "../fx/orb.js";
import { statusChip } from "../ui/status.js";
import { ConsoleTile, TerminalGauge } from "./console_tile.js";

export class RunConsole {
  static LIST_POLL_MS = 5000;
  static ORB_STATES = { running: "working", queued: "breathing", scheduled: "breathing", pending: "breathing" };

  constructor(app, refs) {
    this.app = app;
    this.listEl = refs.list;
    this.tilesEl = refs.tiles;
    this.hintEl = refs.hint;
    this.countEl = refs.count;
    this.onJobs = refs.onJobs;
    this.jobs = [];
    this.listError = null;
    this.tiles = new Map();
    this.dismissed = new Set();
    this.runNames = new Set();
    this.gaugeTerm = null;
    this.fitTimer = null;
    this.fitFrame = 0;
    this.visible = false;

    window.addEventListener("resize", () => this.queueFit());
    document.fonts.ready.then(() => this.queueFit());
    setInterval(() => {
      if (document.hidden || !this.visible) return;
      if (this.listError || this.jobs.some((job) => RunConsole.isLive(job)) || this.hasLiveTile()) this.refresh();
    }, RunConsole.LIST_POLL_MS);
  }

  static isLive(job) {
    return ConsoleTile.LIVE_STATUSES.has(job.status);
  }

  hasLiveTile() {
    return [...this.tiles.values()].some((tile) => RunConsole.isLive(tile.job));
  }

  async refresh() {
    let data;
    try {
      data = await api.get("/api/jobs");
    } catch (error) {
      this.listError = error.status === 0 ? "backend unreachable" : error.message;
      this.renderList();
      return;
    }

    this.listError = null;
    this.jobs = data.jobs;
    await this.loadRunNames();
    this.syncTiles();
    this.jobs.filter((job) => RunConsole.isLive(job) && !this.dismissed.has(job.job_id) && !this.tiles.has(job.job_id)).forEach((job) => this.open(job.job_id));
    this.renderList();
    if (this.onJobs) this.onJobs(this.jobs);
  }

  async loadRunNames() {
    const wanted = this.jobs.some((job) => job.status === "finished" && job.overrides && job.overrides["io.run_name"] && !this.runNames.has(job.overrides["io.run_name"]));
    if (!wanted) return;
    try {
      this.runNames = new Set((await api.get("/api/runs")).runs.map((run) => run.name));
    } catch (error) {
      this.runNames = new Set();
    }
  }

  syncTiles() {
    const listed = new Map(this.jobs.map((job) => [job.job_id, job]));
    this.tiles.forEach((tile, jobId) => {
      const job = listed.get(jobId);
      if (job) tile.syncFromList(job);
      else tile.markUntracked();
    });
  }

  open(jobId) {
    if (this.tiles.has(jobId)) return;
    const job = this.jobs.find((entry) => entry.job_id === jobId);
    if (!job) return;
    this.dismissed.delete(jobId);
    this.tiles.set(jobId, new ConsoleTile({ ...job }, this, this.tilesEl));
    this.layout();
    this.renderList();
  }

  close(jobId) {
    const tile = this.tiles.get(jobId);
    if (!tile) return;
    tile.dispose();
    this.tiles.delete(jobId);
    this.dismissed.add(jobId);
    this.layout();
    this.renderList();
  }

  toggle(jobId) {
    if (this.tiles.has(jobId)) this.close(jobId);
    else this.open(jobId);
  }

  retry(job) {
    const launch = this.app.views.launch;
    launch.pendingImport = {
      name: `${job.script} · ${job.started.replace("T", " ")}`,
      script: job.script,
      overrides: job.request.overrides,
      launch: { interpreter: job.interpreter, detach: job.detach, follow_up: job.request.follow_up },
    };
    this.app.router.go(`#/launch/${job.script}`);
  }

  async stop(jobId) {
    try {
      await api.post(`/api/jobs/${jobId}/stop`, {});
      this.app.toast.show("Stop signal sent", "ok");
    } catch (error) {
      this.app.toast.show(error.message || "Could not stop", "error");
    }
    setTimeout(() => this.refresh(), 1500);
  }

  show() {
    this.visible = true;
    this.queueFit();
  }

  hide() {
    this.visible = false;
  }

  layout() {
    const n = this.tiles.size;
    this.hintEl.hidden = n > 0;
    this.tilesEl.hidden = n === 0;
    const cols = n <= 1 ? 1 : n <= 4 ? 2 : 3;
    const rows = Math.max(1, Math.ceil(n / cols));
    this.tilesEl.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
    this.tilesEl.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    this.queueFit();
  }

  get gauge() {
    if (!this.gaugeTerm) this.gaugeTerm = new TerminalGauge(ConsoleTile.terminalOptions());
    return this.gaugeTerm;
  }

  queueFit() {
    clearTimeout(this.fitTimer);
    cancelAnimationFrame(this.fitFrame);
    this.fitTimer = setTimeout(() => this.fitInTurns([...this.tiles.keys()]), 120);
  }

  fitInTurns(pending) {
    const tile = this.tiles.get(pending.shift());
    if (tile) tile.fit();
    if (pending.length) this.fitFrame = requestAnimationFrame(() => this.fitInTurns(pending));
  }

  energy(wh) {
    if (!wh) return "";
    return wh < 10 ? `${wh.toFixed(2)} Wh` : `${Math.round(wh)} Wh`;
  }

  itemHtml(job) {
    const orb = RunConsole.ORB_STATES[job.status] ? Orb.html(RunConsole.ORB_STATES[job.status], 20, job.status) : "";
    const word = job.status === "failed" && job.stopped ? "stopped" : job.status;
    const meta = [job.started.replace("T", " ").slice(5, 16), job.pid ? `pid ${job.pid}` : "", this.energy(job.energy_wh)].filter(Boolean).join(" · ");
    return `
      <div class="job-item__top"><span class="job-item__name">${esc(job.title || job.script)}</span><span class="job-item__live">${orb}${statusChip(word)}</span></div>
      ${job.description ? `<div class="job-item__desc" title="${esc(job.description)}">${esc(job.description)}</div>` : ""}
      <div class="job-item__meta" title="${esc(job.started.replace("T", " "))}">${esc(meta)}</div>`;
  }

  renderList() {
    this.listEl.innerHTML = "";
    if (this.countEl) this.countEl.textContent = this.jobs.length ? String(this.jobs.length) : "";

    if (this.listError) this.listEl.insertAdjacentHTML("beforeend", `<li class="job-list__error">${esc(`job list unavailable, ${this.listError}`)}</li>`);

    if (!this.jobs.length) {
      this.listEl.insertAdjacentHTML("beforeend", `<li class="job-list__empty">No jobs yet.</li>`);
      return;
    }

    const followers = new Map();
    this.jobs.forEach((job) => {
      if (job.follow_of) followers.set(job.follow_of, job);
    });

    this.jobs.forEach((job) => {
      if (job.follow_of) return;
      const item = document.createElement("li");
      item.className = `job-item${this.tiles.has(job.job_id) ? " is-active" : ""}`;
      item.dataset.job = job.job_id;
      item.tabIndex = 0;
      item.setAttribute("aria-pressed", String(this.tiles.has(job.job_id)));
      item.innerHTML = this.itemHtml(job);
      item.addEventListener("click", () => this.toggle(job.job_id));
      item.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        this.toggle(job.job_id);
      });

      const next = followers.get(job.job_id);
      if (next) {
        const sub = document.createElement("div");
        sub.className = `job-item__follow${this.tiles.has(next.job_id) ? " is-active" : ""}`;
        sub.title = next.description || `${next.script} runs once ${job.script} exits cleanly`;
        sub.innerHTML = `${icon("then", "job-item__arrow")}<span class="job-item__name">${esc(next.title || next.script)}</span>${statusChip(next.status)}`;
        sub.addEventListener("click", (event) => {
          event.stopPropagation();
          this.toggle(next.job_id);
        });
        item.appendChild(sub);
      }

      this.listEl.appendChild(item);
    });
  }
}
