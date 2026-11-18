import { api } from "../api.js";
import { el, esc } from "../format.js";
import { errorBlock, statusChip } from "../ui/status.js";

export class ConsoleView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.timer = null;
    this.sources = {};
    this.tiles = {};
    this.focusJob = null;

    root.appendChild(el(`
      <div class="view__pad">
        <header class="pagehead"><h2>Console</h2><p class="muted">Jobs started from Launch in this server session, newest first, with their live logs.</p></header>
        <div data-role="error"></div>
        <div class="jobs" data-role="jobs"></div>
      </div>`));
    this.list = root.querySelector("[data-role=jobs]");
    this.errorNode = root.querySelector("[data-role=error]");
    this.list.addEventListener("click", (event) => this.onClick(event));
  }

  enter(parts) {
    this.focusJob = parts[0] || null;
    this.load();
    if (!this.timer) this.timer = setInterval(() => this.load(), 3000);
  }

  leave() {
    clearInterval(this.timer);
    this.timer = null;
  }

  async load() {
    let jobs;
    try {
      jobs = (await api.get("/api/jobs")).jobs;
    } catch (error) {
      this.errorNode.innerHTML = errorBlock("The job list could not be loaded.", error.message);
      return;
    }
    this.errorNode.innerHTML = "";

    if (!jobs.length) {
      this.list.innerHTML = `<div class="emptyblock"><p>No jobs in this server session.</p><a class="btn" href="#/launch">Open Launch</a></div>`;
      this.tiles = {};
      return;
    }
    if (this.list.querySelector(".emptyblock")) this.list.innerHTML = "";

    for (const job of jobs) {
      if (!this.tiles[job.job_id]) this.addTile(job);
      this.updateTile(job);
    }

    if (this.focusJob && this.tiles[this.focusJob]) {
      this.tiles[this.focusJob].scrollIntoView({ block: "start" });
      this.tiles[this.focusJob].classList.add("is-focus");
      this.focusJob = null;
    }
  }

  addTile(job) {
    const tile = el(`
      <article class="job" data-job="${esc(job.job_id)}">
        <header class="job__head">
          <h3 class="job__title">${esc(job.title)}</h3>
          <span data-role="status"></span>
          <span class="job__meta muted" data-role="meta"></span>
          <span class="job__spacer"></span>
          <label class="check"><input type="checkbox" data-role="follow" checked><span>Follow</span></label>
          <button type="button" class="btn btn--sm" data-action="collapse" aria-expanded="true">Collapse</button>
          <button type="button" class="btn btn--sm" data-action="stop">Stop</button>
        </header>
        <div data-role="error"></div>
        <pre class="job__cmd">${esc(job.command)}</pre>
        <pre class="job__log" data-role="log" tabindex="0" aria-label="Log of ${esc(job.title)}"></pre>
      </article>`);
    this.list.prepend(tile);
    this.tiles[job.job_id] = tile;
    this.openStream(job.job_id, tile);
  }

  updateTile(job) {
    const tile = this.tiles[job.job_id];
    tile.querySelector("[data-role=status]").innerHTML = statusChip(job.status);
    const bits = [];
    if (job.pid) bits.push(`pid ${job.pid}`);
    bits.push(`started ${job.started.replace("T", " ")}`);
    if (job.exit_code !== null && job.exit_code !== undefined) bits.push(`exit code ${job.exit_code}`);
    tile.querySelector("[data-role=meta]").textContent = bits.join(", ");
    const stoppable = ["running", "queued", "pending"].includes(job.status);
    tile.querySelector("[data-action=stop]").classList.toggle("hidden", !stoppable);
  }

  async onClick(event) {
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    const tile = button.closest(".job");
    if (button.dataset.action === "collapse") {
      const collapsed = tile.classList.toggle("is-collapsed");
      button.textContent = collapsed ? "Expand" : "Collapse";
      button.setAttribute("aria-expanded", String(!collapsed));
      return;
    }
    if (button.dataset.action === "stop") {
      const errorNode = tile.querySelector("[data-role=error]");
      errorNode.innerHTML = "";
      try {
        await api.post(`/api/jobs/${tile.dataset.job}/stop`, {});
        this.app.toast.confirm("Stop signal sent");
      } catch (error) {
        errorNode.innerHTML = errorBlock("The job could not be stopped.", error.message);
      }
    }
  }

  openStream(jobId, tile) {
    const log = tile.querySelector("[data-role=log]");
    const follow = tile.querySelector("[data-role=follow]");
    const source = new EventSource(`/api/jobs/${jobId}/stream`);
    this.sources[jobId] = source;

    source.onmessage = (message) => {
      const event = JSON.parse(message.data);
      if (event.type === "chunk") {
        const clean = event.data.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "").replace(/\r(?!\n)/g, "\n");
        log.textContent += clean;
        if (log.textContent.length > 400000) log.textContent = log.textContent.slice(-300000);
        if (follow.checked) log.scrollTop = log.scrollHeight;
      } else if (event.type === "status") {
        this.load();
        this.app.pollJobs();
      } else if (event.type === "end") {
        source.close();
        delete this.sources[jobId];
        this.load();
        this.app.pollJobs();
      }
    };

    source.onerror = () => {
      source.close();
      delete this.sources[jobId];
    };
  }
}
