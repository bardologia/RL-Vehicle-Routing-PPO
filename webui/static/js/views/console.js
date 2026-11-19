import { api } from "../api.js";
import { el, esc } from "../format.js";
import { LogText } from "../ui/logtext.js";
import { Figure } from "../fx/figure.js";
import { PageHead } from "../ui/page_head.js";
import { emptyBlock, errorBlock, statusChip } from "../ui/status.js";

export class ConsoleView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.timer = null;
    this.sources = {};
    this.tiles = {};
    this.focusJob = null;
    this.texts = {};
    this.runNames = new Set();

    root.appendChild(el(`
      <div class="view__pad">
        ${PageHead.html({ title: "Console", form: "narrow", figure: Figure.html("terminal") })}
        <div data-role="error"></div>
        <div class="jobs" data-role="jobs"></div>
      </div>`));
    this.list = root.querySelector("[data-role=jobs]");
    this.errorNode = root.querySelector("[data-role=error]");
    this.dateline = root.querySelector("[data-role=dateline]");
    this.list.addEventListener("click", (event) => this.onClick(event));
    this.list.addEventListener("input", (event) => this.onSearch(event));
  }

  focusSearch() {
    const input = this.list.querySelector("input[type=search]");
    if (input) input.focus();
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
    const count = (...states) => jobs.filter((job) => states.includes(job.status)).length;
    this.dateline.textContent = jobs.length ? `${count("running")} running, ${count("queued", "pending")} queued, ${count("finished")} finished, ${count("failed", "cancelled")} failed` : "";
    if (jobs.some((job) => !["running", "queued", "pending"].includes(job.status) && job.overrides["io.run_name"] && !this.runNames.has(job.overrides["io.run_name"]))) {
      try {
        this.runNames = new Set((await api.get("/api/runs")).runs.map((run) => run.name));
      } catch (error) {
        this.runNames = new Set();
      }
    }

    if (!jobs.length) {
      this.list.innerHTML = emptyBlock("No jobs in this server session.", `<a class="btn" href="#/launch">Open Launch</a>`);
      this.tiles = {};
      return;
    }
    if (this.list.querySelector(".figure-empty")) this.list.innerHTML = "";

    for (const job of [...jobs].reverse()) {
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
          <a class="btn btn--sm hidden" data-role="open-run" href="#/runs">Open run</a>
          <label class="job__search"><input class="input" type="search" placeholder="Search in log" aria-label="Search in the log of ${esc(job.title)}" data-role="search"><span class="muted" data-role="matches"></span></label>
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
    this.texts[job.job_id] = "";
    this.openStream(job.job_id, tile);
  }

  updateTile(job) {
    const tile = this.tiles[job.job_id];
    tile.querySelector("[data-role=status]").innerHTML = statusChip(job.status === "failed" && job.exit_code === -15 ? "stopped" : job.status);
    const bits = [];
    if (job.pid) bits.push(`pid ${job.pid}`);
    bits.push(`started ${job.started.replace("T", " ")}`);
    if (job.exit_code !== null && job.exit_code !== undefined) bits.push(`exit code ${job.exit_code}`);
    tile.querySelector("[data-role=meta]").textContent = bits.join(", ");
    const stoppable = ["running", "queued", "pending"].includes(job.status);
    tile.querySelector("[data-action=stop]").classList.toggle("hidden", !stoppable);
    const runName = job.overrides["io.run_name"];
    const openRun = tile.querySelector("[data-role=open-run]");
    const showRun = job.status === "finished" && Boolean(runName) && this.runNames.has(runName);
    openRun.classList.toggle("hidden", !showRun);
    if (showRun) openRun.href = `#/runs/${encodeURIComponent(runName)}/summary`;
  }

  paint(jobId) {
    const tile = this.tiles[jobId];
    const log = tile.querySelector("[data-role=log]");
    const query = tile.querySelector("[data-role=search]").value.trim();
    const count = LogText.fill(log, this.texts[jobId], query);
    tile.querySelector("[data-role=matches]").textContent = query ? `${count} ${count === 1 ? "match" : "matches"}` : "";
    return log;
  }

  onSearch(event) {
    const input = event.target.closest("input[data-role=search]");
    if (!input) return;
    const log = this.paint(input.closest(".job").dataset.job);
    const mark = log.querySelector("mark");
    if (mark) mark.scrollIntoView({ block: "nearest" });
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
        this.texts[jobId] += clean;
        if (this.texts[jobId].length > 400000) this.texts[jobId] = this.texts[jobId].slice(-300000);
        this.paint(jobId);
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
