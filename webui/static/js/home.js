"use strict";

class HomeView {
  constructor() {
    this.body = document.getElementById("home-body");
    this.timer = null;
  }

  enter() {
    this._load();
    this.timer = setInterval(() => this._load(), 15000);
  }

  leave() {
    clearInterval(this.timer);
    this.timer = null;
  }

  async _load() {
    const [health, jobs, runs, datasets] = await Promise.all([
      apiGet("/api/health"),
      apiGet("/api/jobs"),
      apiGet("/api/runs"),
      apiGet("/api/datasets"),
    ]);

    const jobList = jobs.jobs || [];
    const runList = runs.runs || [];
    const dsList = datasets.datasets || [];
    const running = jobList.filter((j) => j.status === "running" || j.status === "queued");

    this.body.innerHTML = `
      <div class="grid grid--4">
        <div class="card"><div class="stat"><span class="stat__num">${runList.length}</span><span class="stat__label">runs</span></div></div>
        <div class="card"><div class="stat"><span class="stat__num">${dsList.length}</span><span class="stat__label">datasets</span></div></div>
        <div class="card"><div class="stat"><span class="stat__num">${running.length}</span><span class="stat__label">active jobs</span></div></div>
        <div class="card"><div class="stat"><span class="stat__num">${(health.osrm ? 1 : 0) + (health.vroom ? 1 : 0)}/2</span><span class="stat__label">services up</span></div></div>
      </div>

      <div class="home-row">
        <div class="card">
          <h3 class="card__title">Recent jobs</h3>
          ${this._jobsHtml(jobList.slice(0, 6))}
        </div>
        <div class="card">
          <h3 class="card__title">Recent runs</h3>
          ${this._runsHtml(runList.slice(0, 6))}
        </div>
      </div>

      <div class="home-row">
        <div class="card">
          <h3 class="card__title">Services</h3>
          <ul class="list-plain">
            <li><span class="badge ${health.osrm ? "badge--finished" : "badge--failed"}">${health.osrm ? "UP" : "DOWN"}</span><span class="grow">OSRM routing</span><span class="dim">${escapeHtml(health.osrm_url || "")}</span></li>
            <li><span class="badge ${health.vroom ? "badge--finished" : "badge--failed"}">${health.vroom ? "UP" : "DOWN"}</span><span class="grow">VROOM solver</span><span class="dim">${escapeHtml(health.vroom_url || "")}</span></li>
          </ul>
          <p class="scn-hint" style="margin: 12px 0 0">Both services must be up for training, dataset generation, and the Scenario Lab. Start them with <code>sg docker -c "docker compose up -d"</code> in the repo root.</p>
        </div>
        <div class="card">
          <h3 class="card__title">Datasets</h3>
          ${this._datasetsHtml(dsList)}
        </div>
      </div>`;

    this.body.querySelectorAll("[data-goto]").forEach((el) => {
      el.addEventListener("click", () => { window.location.hash = el.dataset.goto; });
    });
  }

  _jobsHtml(jobs) {
    if (!jobs.length) return `<p class="scn-hint">No jobs yet. Head to <a href="#/launch" style="color: var(--accent)">Launch</a> to start one.</p>`;
    return `<ul class="list-plain">${jobs.map((j) => `
      <li data-goto="#/console" style="cursor:pointer">
        <span class="badge badge--${j.status}">${j.status}</span>
        <span class="grow">${escapeHtml(j.title)}</span>
        <span class="dim">${escapeHtml(j.started.replace("T", " "))}</span>
      </li>`).join("")}</ul>`;
  }

  _runsHtml(runs) {
    if (!runs.length) return `<p class="scn-hint">No runs yet.</p>`;
    return `<ul class="list-plain">${runs.map((r) => `
      <li data-goto="#/results/${encodeURIComponent(r.name)}" style="cursor:pointer">
        <span class="badge badge--${r.kind}">${r.kind}</span>
        <span class="grow">${escapeHtml(r.name)}</span>
        <span class="dim">${r.evaluation && r.evaluation.model != null ? "model " + r.evaluation.model : ""}</span>
        <span class="dim">${fmtWhen(r.mtime)}</span>
      </li>`).join("")}</ul>`;
  }

  _datasetsHtml(datasets) {
    if (!datasets.length) return `<p class="scn-hint">No datasets yet. Generate one from the Launch page.</p>`;
    return `<ul class="list-plain">${datasets.map((d) => `
      <li>
        <span class="grow" style="font-family: var(--mono); font-size: 12.5px">${escapeHtml(d.rel)}</span>
        <span class="dim">${d.chunks} chunks</span>
        <span class="dim">${d.size_mb} MB</span>
      </li>`).join("")}</ul>`;
  }
}

window.HomeView = HomeView;
