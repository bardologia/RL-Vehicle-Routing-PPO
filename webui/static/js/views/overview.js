import { api } from "../api.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { Agents } from "../episode/encoding.js";
import { el, esc, fmtAge, fmtSigned, NONE, plural } from "../format.js";
import { AgentTable } from "../diagnose/agent_table.js";
import { kpiRow } from "../ui/kpi.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { errorBlock, failView, skeleton, statusChip } from "../ui/status.js";
import { FigureStates } from "../ui/figure_states.js";

const DOCKER = 'sg docker -c "docker compose up -d"';

export class OverviewView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.timer = null;
    this.token = 0;
    this.selected = null;
    this.report = null;

    root.appendChild(el(`
      <div class="view__pad over">
        ${PageHead.html({
          title: "Overview",
          form: "hero",
          controls: `${PageHead.control("Run", `<select class="select" data-role="run" aria-label="Run"></select>`)}<span class="ctl" data-role="status-actions"></span>`,
          after: `<section class="panel over__board" data-region="status"><div data-role="status"></div></section>`,
          figure: HairlineFigure.html("plans"),
        })}
        <div data-role="error"></div>
        <section class="panel" data-region="findings">
          ${panelHead("Findings", { info: infoText("Rules flagged WARN or FAIL for the selected run. FAIL rules state their finding; open a row for its numbers, its rule id and its links. The title opens the evidence.") })}
          <div class="panel__body" data-role="findings"></div>
        </section>
        <div class="grid-12 over__pair">
          <section class="panel span-7" data-region="checkpoints">
            ${panelHead("Checkpoints", { info: infoText("Every run directory that holds a policy file, checked against the current architecture.", "Reward is the model mean reward of the newest report, in points. A report seed opens that report in Diagnose.") })}
            <div class="panel__body figcard"><div class="figcard__fig">${HairlineFigure.html("checkpoint")}</div><div class="figcard__body" data-role="checkpoints"></div></div>
          </section>
          <section class="panel span-5" data-region="activity">
            ${panelHead("Activity", { info: infoText("Jobs of this server session and the latest recordings.") })}
            <div class="panel__body figcard"><div class="figcard__fig">${HairlineFigure.html("queue")}</div><div class="figcard__body" data-role="activity"></div></div>
          </section>
        </div>
      </div>`));
    this.nodes = {};
    for (const node of root.querySelectorAll("[data-role]")) this.nodes[node.dataset.role] = node;
    this.figures = {};
    for (const node of root.querySelectorAll("hairline-figure")) this.figures[node.getAttribute("name")] = node;
    this.slotNames = [];
    this.figures.checkpoint.addEventListener("figure-pick", (event) => this.pickSlot(event.detail.slot));
    this.nodes.run.addEventListener("change", () => {
      this.selected = this.nodes.run.value;
      this.renderCheckpoints();
      this.loadRun();
    });
    this.nodes.status.addEventListener("click", (event) => this.onRecord(event));
    this.nodes.kpis.addEventListener("click", (event) => this.onKpi(event));
    this.nodes.kpis.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") this.onKpi(event);
    });
    this.nodes["status-actions"].addEventListener("click", (event) => this.onRecord(event));
    app.onHealth(() => this.syncRecord());

    let timer = null;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (this.app.active === "overview" && this.report) this.drawBoard();
      }, 150);
    });
  }

  focusSearch() {
    this.nodes.run.focus();
  }

  onKpi(event) {
    const tile = event.target.closest(".kpi[data-kpi=findings]");
    if (!tile) return;
    event.preventDefault();
    const panel = this.root.querySelector("[data-region=findings]");
    panel.scrollIntoView({ block: "start", behavior: "auto" });
    panel.classList.add("is-target");
    setTimeout(() => panel.classList.remove("is-target"), 2400);
  }

  pickSlot(slot) {
    const name = this.slotNames[slot];
    const run = this.runs ? this.runs.find((entry) => entry.name === name) : null;
    if (!run) return;
    if (!run.has_checkpoint) {
      this.app.router.go(Links.runs(run.name));
      return;
    }
    this.selected = run.name;
    this.nodes.run.value = run.name;
    this.renderCheckpoints();
    this.loadRun();
  }

  enter() {
    this.load();
    if (!this.timer) this.timer = setInterval(() => this.loadActivity(), 15000);
  }

  leave() {
    clearInterval(this.timer);
    this.timer = null;
  }

  syncRecord() {
    const up = this.app.servicesUp;
    for (const link of this.root.querySelectorAll('a[href="#/episode"].btn')) {
      link.classList.toggle("is-disabled", !up);
      link.setAttribute("aria-disabled", String(!up));
      link.title = up ? "" : `Recording needs OSRM and VROOM: ${DOCKER}`;
    }
  }

  onRecord(event) {
    const link = event.target.closest('a[href="#/episode"]');
    if (link && link.classList.contains("is-disabled")) {
      event.preventDefault();
      return;
    }

    if (event.target.closest('a[href="#/episode"]') && this.run) this.app.setContext(this.run.name, this.run.reports.length ? this.run.reports[0].seed : null);
  }

  async load() {
    const token = (this.token += 1);
    this.nodes.error.innerHTML = "";
    let runs;
    let checkpoints;
    try {
      [runs, checkpoints] = await Promise.all([api.get("/api/runs"), api.get("/api/checkpoints")]);
    } catch (error) {
      this.nodes.error.innerHTML = failView("The runs and checkpoints could not be loaded.", error.message, error);
      return;
    }
    if (token !== this.token) return;
    this.runs = runs.runs;
    this.checkpoints = checkpoints.checkpoints;
    this.loadActivity();

    const context = this.app.context;
    const loadable = this.runs.filter((run) => run.loadable);
    const remembered = this.runs.find((entry) => entry.name === context.run && entry.reports.length);
    const preferred = this.selected || (remembered ? remembered.name : null);
    const run = this.runs.find((entry) => entry.name === preferred && entry.has_checkpoint) || loadable.find((entry) => entry.reports.length) || loadable[0] || null;
    this.selected = run ? run.name : null;
    this.renderCheckpoints();
    const options = this.runs.filter((entry) => entry.has_checkpoint);
    this.nodes.run.innerHTML = options.length ? options.map((entry) => `<option value="${esc(entry.name)}"${entry.name === this.selected ? " selected" : ""}>${esc(entry.name)}${entry.loadable ? "" : " (incompatible)"}</option>`).join("") : `<option>none</option>`;
    this.nodes.run.disabled = !options.length;
    await this.loadRun();
  }

  async loadRun() {
    const token = (this.token += 1);
    this.report = null;
    this.statusTiles = [];
    this.findingCounts = null;
    this.nodes.kpis.innerHTML = "";
    FigureStates.write(this.figures.plans, FigureStates.plans(null));
    const run = this.runs.find((entry) => entry.name === this.selected) || null;
    this.run = run;
    if (!run) {
      this.nodes.dateline.textContent = "";
      this.nodes["status-actions"].innerHTML = "";
      this.nodes.status.innerHTML = OverviewView.boardNote("No checkpoint loads on the current architecture. A pretrain run produces the first usable policy.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`);
      this.nodes.findings.innerHTML = `<p class="muted">Findings need a run.</p>`;
      return;
    }

    const seed = run.reports.length ? run.reports[0].seed : null;
    this.nodes.dateline.textContent = `${run.kind}, ${fmtAge(run.mtime)}${seed === null ? "" : `, report seed ${seed}`}`;
    this.nodes.status.innerHTML = skeleton(3);
    this.nodes.findings.innerHTML = skeleton(3);
    const base = `/api/runs/${encodeURIComponent(run.name)}`;
    let report = null;
    let findings;
    try {
      [report, findings] = await Promise.all([seed === null ? Promise.resolve(null) : api.get(`${base}/evaluation/${seed}`), api.get(`${base}/findings${seed === null ? "" : `?seed=${seed}`}`)]);
    } catch (error) {
      if (token !== this.token) return;
      this.nodes.status.innerHTML = errorBlock(`The report of "${run.name}" could not be loaded.`, error.message);
      this.nodes.findings.innerHTML = "";
      return;
    }
    if (token !== this.token) return;
    this.report = report;
    this.renderStatus(run, seed);
    this.renderFindings(run, seed, findings.findings);
  }

  static boardNote(sentence, actionHtml) {
    return `<div class="over__note"><p>${esc(sentence)}</p>${actionHtml || ""}</div>`;
  }

  renderStatus(run, seed) {
    const record = run.loadable ? `<a class="btn btn--sm btn--quiet" href="#/episode">Record episode</a>` : "";
    if (!run.loadable) {
      this.nodes["status-actions"].innerHTML = `<a class="btn" href="${esc(Links.runs(run.name))}">Open run</a>`;
      this.nodes.status.innerHTML = errorBlock(`${run.name} does not load on the current architecture: ${run.reason || "no reason given"}.`);
      return;
    }
    if (!this.report) {
      this.nodes["status-actions"].innerHTML = record;
      this.syncRecord();
      this.nodes.status.innerHTML = OverviewView.boardNote(`${run.name} loads but has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds.`, `<a class="btn btn--primary" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`);
      return;
    }

    const report = this.report;
    FigureStates.write(this.figures.plans, FigureStates.plans(report));
    this.nodes.dateline.textContent = `${run.kind}, ${fmtAge(run.mtime)}, report seed ${seed}, n = ${report.meta.episodes}, ${report.meta.greedy ? "greedy" : "sampled"}`;
    this.nodes["status-actions"].innerHTML = `<a class="btn btn--primary" href="${esc(Links.diagnose(run.name, seed))}">Open in Diagnose</a>${record}`;
    const model = report.agents.model;
    const teacher = report.paired.teacher;
    const separable = teacher && (teacher.ci95[0] > 0 || teacher.ci95[1] < 0);
    const tiles = [];
    if (model) {
      tiles.push({ key: "reward", label: "Model reward", full: "Model mean reward", value: fmtSigned(model.mean_reward), unit: "pts", title: `95% interval ${fmtSigned(model.ci95[0])} to ${fmtSigned(model.ci95[1])} points.${model.identical_to ? ` Identical to ${model.identical_to}: same reward in every episode.` : ""}` });
      if (teacher) {
        tiles.push({ key: "paired", label: "Model minus teacher", value: fmtSigned(teacher.mean_diff), unit: "pts", tone: separable ? (teacher.mean_diff > 0 ? "gain" : "loss") : "", subs: [separable ? "separable" : "not separable"], title: `Paired difference of the episode reward. 95% interval ${fmtSigned(teacher.ci95[0])} to ${fmtSigned(teacher.ci95[1])} points${separable ? "" : ", not separable at this n"}.` });
        tiles.push({ key: "record", label: "Won / level / lost", value: `${teacher.wins} / ${teacher.ties} / ${teacher.losses}`, subs: ["against the teacher"], title: "Episodes the model won, tied and lost against the teacher." });
      }
    }
    this.statusTiles = tiles;
    this.renderHead();
    this.drawBoard();
    this.syncRecord();
  }

  renderHead() {
    const tiles = [...(this.statusTiles || [])];
    if (this.findingCounts) tiles.push({ key: "findings", label: "Findings", detail: "fail / warn", value: `${this.findingCounts.FAIL} / ${this.findingCounts.WARN}`, subs: [FindingsList.restLabel(this.findingCounts)], title: "Rules flagged FAIL and WARN for this run. Click to open the list." });
    this.nodes.kpis.innerHTML = tiles.length ? kpiRow(tiles, { columns: 4 }) : "";
    const link = this.nodes.kpis.querySelector(".kpi[data-kpi=findings]");
    if (link) {
      link.tabIndex = 0;
      link.setAttribute("role", "link");
      link.classList.add("kpi--link");
    }
  }

  drawBoard() {
    if (!this.report) return;
    const info = Popover.infoButton("Agents", infoText(`Mean episode reward of every agent in the report of the selected run, and the paired difference between the model and each other agent, both with a 95% interval, in points, n = ${this.report.meta.episodes}.`, "Hover a row for its interval. Diagnose holds the full agent table."));
    AgentTable.render(this.nodes.status, this.report, { set: "brief", tail: info });
  }

  renderFindings(run, seed, findings) {
    const counts = FindingsList.counts(findings);
    this.findingCounts = counts;
    this.renderHead();
    const trace = `<a class="btn btn--sm" href="${esc(Links.recordTraces(run.name, seed, this.report ? this.report.meta.episodes : null))}">Record traces</a>`;
    const actions = { report: `<a class="btn btn--sm" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`, traces: trace, verdicts: trace };
    this.nodes.findings.innerHTML = FindingsList.list(findings, { actions, columns: true, emptyText: `No rule is flagged for ${run.name}.` });
  }

  renderCheckpoints() {
    const shelf = FigureStates.checkpoints(this.runs, this.selected);
    this.slotNames = shelf.names;
    FigureStates.write(this.figures.checkpoint, shelf.state);
    const byRun = new Map(this.runs.map((run) => [run.name, run]));
    const dash = `<span class="faint">${NONE}</span>`;
    const rows = this.checkpoints.map((checkpoint) => {
      const run = byRun.get(checkpoint.run);
      const reward = run && run.reports.length && run.reports[0].agents.model ? fmtSigned(run.reports[0].agents.model.mean_reward) : dash;
      const reports = checkpoint.reports.length ? checkpoint.reports.map((seed) => `<a href="${esc(Links.diagnose(checkpoint.run, seed))}" data-hint="Open the report in Diagnose">${seed}</a>`).join(", ") : dash;
      const traces = checkpoint.diagnostics.length ? checkpoint.diagnostics.map((seed) => esc(seed)).join(", ") : dash;
      const word = checkpoint.loadable ? "LOADABLE" : "INCOMPATIBLE";
      const hint = checkpoint.reason ? ` data-hint="${esc(checkpoint.reason)}" data-hint-title="${word.toLowerCase()}"` : "";
      return `
        <tr${checkpoint.run === this.selected ? ' aria-selected="true"' : ""}>
          <td><a href="${esc(Links.runs(checkpoint.run))}">${esc(checkpoint.run)}</a></td>
          <td>${esc(checkpoint.kind)}</td>
          <td class="num">${esc(fmtAge(checkpoint.mtime).replace(/ ago$/, ""))}</td>
          <td${hint}>${statusChip(word)}</td>
          <td class="num">${reward}</td>
          <td class="num">${reports}</td>
          <td class="num">${traces}</td>
        </tr>`;
    }).join("");
    this.nodes.checkpoints.innerHTML = this.checkpoints.length ? `
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr><th>Run</th><th>Kind</th><th class="num" data-hint="Time since the checkpoint was written">Age</th><th>State</th><th class="num" data-hint="Model mean reward of the newest report">Reward<small>pts</small></th><th class="num" data-hint="Seeds with an evaluation report">Report</th><th class="num" data-hint="Seeds with recorded traces">Traces</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>` : `<p class="muted">No run holds a checkpoint yet. <a href="#/launch/pretrain">Launch pretrain</a></p>`;
  }

  async loadActivity() {
    let jobs;
    let episodes;
    try {
      [jobs, episodes] = await Promise.all([api.get("/api/jobs"), api.get("/api/episodes")]);
    } catch (error) {
      this.nodes.activity.innerHTML = errorBlock("Jobs and recordings could not be loaded.", error.message);
      return;
    }
    FigureStates.write(this.figures.queue, FigureStates.queue(jobs.jobs));
    const jobRows = jobs.jobs.slice(0, 6).map((job) => `<tr><td><a href="#/console/${esc(job.job_id)}">${esc(job.title)}</a></td><td>${statusChip(job.status === "failed" && job.stopped ? "stopped" : job.status)}</td><td class="muted">${esc(job.overrides && job.overrides["io.run_name"] ? job.overrides["io.run_name"] : "")}</td><td class="num muted">${esc(job.started.replace("T", " ").slice(5, 16))}</td></tr>`).join("");
    const episodeRows = episodes.episodes.slice(0, 6).map((episode) => {
      const keys = Object.keys(episode.agents).sort((a, b) => Agents.order(a) - Agents.order(b));
      const first = keys[0];
      const source = episode.source.kind === "scenario" ? `scenario ${episode.source.template || episode.source.name || "custom"}` : episode.source.kind === "evaluation" ? `${episode.source.run} episode ${episode.source.index}` : `seed ${episode.source.seed}`;
      return `<tr><td><a href="#/episode/ep/${esc(episode.episode_id)}/${encodeURIComponent(first)}/0">${esc(source)}</a></td><td>${statusChip(episode.status)}</td><td class="muted">${plural(keys.length, "agent")}</td><td class="num muted">${esc(fmtAge(episode.created))}</td></tr>`;
    }).join("");
    this.nodes.activity.innerHTML = `
      <h4 class="ctl__label">Jobs</h4>
      ${jobRows ? `<div class="dtable dtable--dense dtable--static"><table><tbody>${jobRows}</tbody></table></div>` : `<p class="muted">No job in this server session. <a href="#/launch">Open Launch</a></p>`}
      <h4 class="ctl__label over__sub">Recordings</h4>
      ${episodeRows ? `<div class="dtable dtable--dense dtable--static"><table><tbody>${episodeRows}</tbody></table></div>` : `<p class="muted">No recording yet. <a href="#/episode">Open Episode</a></p>`}`;
  }
}
