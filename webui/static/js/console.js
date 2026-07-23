"use strict";

class ConsoleView {
  constructor() {
    this.body = document.getElementById("console-body");
    this.timer = null;
    this.sources = {};
    this.rendered = {};
  }

  enter() {
    this._load();
    this.timer = setInterval(() => this._load(), 3000);
  }

  leave() {
    clearInterval(this.timer);
    this.timer = null;
  }

  async _load() {
    const res = await apiGet("/api/jobs");
    const jobs = res.jobs || [];

    if (!jobs.length) {
      this.body.innerHTML = `<div class="console-empty">No jobs this session. Launch one from the Launch page.</div>`;
      this.rendered = {};
      return;
    }

    if (this.body.querySelector(".console-empty")) this.body.innerHTML = "";

    for (const job of jobs) {
      if (!this.rendered[job.job_id]) this._addTile(job);
      else this._updateTile(job);
    }
  }

  _addTile(job) {
    const tile = document.createElement("div");
    tile.className = "job-tile";
    tile.dataset.jobId = job.job_id;
    tile.innerHTML = `
      <div class="job-tile__head">
        <span class="job-tile__title">${escapeHtml(job.title)}</span>
        <span class="badge" data-role="status"></span>
        <span class="job-tile__meta"><span data-role="meta"></span></span>
        <span class="job-tile__spacer"></span>
        <button class="btn btn--sm btn--danger" data-role="stop">Stop</button>
      </div>
      <div class="job-tile__cmd">${escapeHtml(job.command)}</div>
      <pre class="job-log" data-role="log"></pre>`;

    tile.querySelector(".job-tile__head").addEventListener("click", (e) => {
      if (e.target.dataset.role === "stop") return;
      tile.classList.toggle("is-collapsed");
    });

    tile.querySelector("[data-role=stop]").addEventListener("click", async () => {
      const res = await apiPost(`/api/jobs/${job.job_id}/stop`, {});
      toast(res.ok ? "stop signal sent" : res.error || "stop failed", res.ok ? "ok" : "error");
    });

    this.body.prepend(tile);
    this.rendered[job.job_id] = tile;
    this._updateTile(job);
    this._openStream(job.job_id, tile);
  }

  _updateTile(job) {
    const tile = this.rendered[job.job_id];
    if (!tile) return;

    const badge = tile.querySelector("[data-role=status]");
    badge.className = `badge badge--${job.status}`;
    badge.innerHTML = job.status === "running" ? `<i class="pulse"></i>running` : job.status;

    const bits = [];
    if (job.pid) bits.push(`pid ${job.pid}`);
    bits.push(job.started.replace("T", " "));
    if (job.exit_code != null) bits.push(`exit ${job.exit_code}`);
    tile.querySelector("[data-role=meta]").textContent = bits.join("  ·  ");

    const stoppable = job.status === "running" || job.status === "queued" || job.status === "pending";
    tile.querySelector("[data-role=stop]").style.display = stoppable ? "" : "none";
  }

  _openStream(jobId, tile) {
    if (this.sources[jobId]) return;

    const log = tile.querySelector("[data-role=log]");
    const source = new EventSource(`/api/jobs/${jobId}/stream`);
    this.sources[jobId] = source;

    source.onmessage = (e) => {
      const event = JSON.parse(e.data);

      if (event.type === "chunk") {
        const clean = event.data.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r(?!\n)/g, "\n");
        log.textContent += clean;
        if (log.textContent.length > 400000) log.textContent = log.textContent.slice(-300000);
        log.scrollTop = log.scrollHeight;
      } else if (event.type === "status") {
        this._load();
      } else if (event.type === "end") {
        source.close();
        delete this.sources[jobId];
        this._load();
      }
    };

    source.onerror = () => {
      source.close();
      delete this.sources[jobId];
    };
  }
}

window.ConsoleView = ConsoleView;
