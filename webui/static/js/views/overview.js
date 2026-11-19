import { api } from "../api.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { Agents } from "../episode/encoding.js";
import { el, esc, fmtAge, fmtSigned, plural } from "../format.js";
import { Figure } from "../fx/figure.js";
import { IntervalPlot } from "../ui/charts.js";
import { kpiRow } from "../ui/kpi.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead } from "../ui/panel.js";
import { emptyBlock, errorBlock, skeleton, statusChip } from "../ui/status.js";

const AGENTS = ["model", "teacher", "insertion_only", "do_nothing"];
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
          figure: Figure.html("router"),
        })}
        <div data-role="error"></div>
        <section class="panel" data-region="status">
          ${panelHead("Model status", { info: infoText("Mean episode reward of every agent in the report of the selected run, and the paired difference between the model and each other agent, both with a 95% interval.") })}
          <div class="panel__body" data-role="status"></div>
        </section>
        <section class="panel" data-region="findings">
          ${panelHead("Findings", { info: infoText("Rules flagged WARN or FAIL for the selected run, each with a link to its evidence.") })}
          <div class="panel__body" data-role="findings"></div>
        </section>
        <div class="grid-12">
          <section class="panel panel--flush span-7" data-region="checkpoints">
            ${panelHead("Checkpoints", { info: infoText("Every run directory that holds a policy file, checked against the current architecture.") })}
            <div class="panel__body" data-role="checkpoints"></div>
          </section>
          <section class="panel span-5" data-region="activity">
            ${panelHead("Activity", { info: infoText("Jobs of this server session and the latest recordings.") })}
            <div class="panel__body" data-role="activity"></div>
          </section>
        </div>
        <section class="panel" data-region="services">
          ${panelHead("Services", { info: infoText("Recording, solving and teacher verdicts need both routing services.") })}
          <div class="panel__body" data-role="services"></div>
        </section>
      </div>`));
    this.nodes = {};
    for (const node of root.querySelectorAll("[data-role]")) this.nodes[node.dataset.role] = node;
    this.nodes.run.addEventListener("change", () => {
      this.selected = this.nodes.run.value;
      this.loadRun();
    });
    this.nodes.findingsNote = root.querySelector("[data-region=findings] [data-role=panel-note]");
    this.nodes.status.addEventListener("click", (event) => this.onRecord(event));
    this.nodes["status-actions"].addEventListener("click", (event) => this.onRecord(event));
    app.onHealth((health) => this.renderServices(health));

    let timer = null;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (this.app.active === "overview" && this.report) this.drawIntervals();
      }, 150);
    });
  }

  focusSearch() {
    this.nodes.run.focus();
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
      this.nodes.error.innerHTML = errorBlock("The runs and checkpoints could not be loaded.", error.message);
      return;
    }
    if (token !== this.token) return;
    this.runs = runs.runs;
    this.checkpoints = checkpoints.checkpoints;
    this.renderCheckpoints();
    this.loadActivity();

    const context = this.app.context;
    const loadable = this.runs.filter((run) => run.loadable);
    const remembered = this.runs.find((entry) => entry.name === context.run && entry.reports.length);
    const preferred = this.selected || (remembered ? remembered.name : null);
    const run = this.runs.find((entry) => entry.name === preferred && entry.has_checkpoint) || loadable.find((entry) => entry.reports.length) || loadable[0] || null;
    this.selected = run ? run.name : null;
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
    const run = this.runs.find((entry) => entry.name === this.selected) || null;
    this.run = run;
    if (!run) {
      this.nodes.dateline.textContent = "";
      this.nodes["status-actions"].innerHTML = "";
      this.nodes.status.innerHTML = emptyBlock("No checkpoint loads on the current architecture. A pretrain run produces the first usable policy.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`);
      this.nodes.findings.innerHTML = `<p class="muted">Findings need a run.</p>`;
      this.nodes.findingsNote.textContent = "";
      return;
    }

    const seed = run.reports.length ? run.reports[0].seed : null;
    this.nodes.dateline.textContent = `${run.kind}, ${fmtAge(run.mtime)}${seed === null ? "" : `, report seed ${seed}`}`;
    this.nodes.status.innerHTML = skeleton(4);
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
      this.nodes.status.innerHTML = emptyBlock(`${run.name} loads but has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds.`, `<a class="btn btn--primary" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`);
      return;
    }

    const report = this.report;
    this.nodes.dateline.textContent = `${run.kind}, ${fmtAge(run.mtime)}, report seed ${seed}, n = ${report.meta.episodes}, ${report.meta.greedy ? "greedy" : "sampled"}`;
    this.nodes["status-actions"].innerHTML = `<a class="btn btn--primary" href="${esc(Links.diagnose(run.name, seed))}">Open in Diagnose</a>${record}`;
    const model = report.agents.model;
    const teacher = report.paired.teacher;
    const separable = teacher && (teacher.ci95[0] > 0 || teacher.ci95[1] < 0);
    const tiles = [];
    if (model) {
      const lanes = ["teacher", "do_nothing"].filter((key) => report.agents[key]).map((key) => ({ html: `<span><span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span> ${fmtSigned(report.agents[key].mean_reward)}</span>` }));
      tiles.push({ key: "reward", label: "Model reward", full: "Model mean reward", value: fmtSigned(model.mean_reward), unit: "pts", subs: lanes, title: `95% interval ${fmtSigned(model.ci95[0])} to ${fmtSigned(model.ci95[1])} points.${model.identical_to ? ` Identical to ${model.identical_to}: same reward in every episode.` : ""}` });
      if (teacher) {
        tiles.push({ key: "paired", label: "Model minus teacher", value: fmtSigned(teacher.mean_diff), unit: "pts", tone: separable ? (teacher.mean_diff > 0 ? "gain" : "loss") : "", subs: [separable ? "separable" : "not separable"], title: `Paired difference of the episode reward. 95% interval ${fmtSigned(teacher.ci95[0])} to ${fmtSigned(teacher.ci95[1])} points${separable ? "" : ", not separable at this n"}.` });
        tiles.push({ key: "record", label: "Won / level / lost", value: `${teacher.wins} / ${teacher.ties} / ${teacher.losses}`, subs: [`of ${plural(report.meta.episodes, "episode")}`], title: "Episodes the model won, tied and lost against the teacher." });
      }
    }
    this.statusTiles = tiles;
    this.renderHead();
    this.nodes.status.innerHTML = `
      <div class="over__plots">
        <div><p class="over__plotnote">reward, points, 95% interval</p><div data-role="rewards"></div></div>
        <div><p class="over__plotnote">model minus agent, points, 95% interval</p><div data-role="paired"></div></div>
      </div>
      ${report.missing_agents.length ? `<p class="muted">Missing from this report: ${report.missing_agents.map((key) => esc(Agents.label(key))).join(", ")}.</p>` : ""}`;
    this.drawIntervals();
    this.syncRecord();
  }

  renderHead() {
    const tiles = [...(this.statusTiles || [])];
    if (this.findingCounts) tiles.push({ key: "findings", label: "Findings", detail: "fail / warn", value: `${this.findingCounts.FAIL} / ${this.findingCounts.WARN}`, subs: [FindingsList.restLabel(this.findingCounts)], title: "Rules flagged FAIL and WARN for this run. The list is below." });
    this.nodes.kpis.innerHTML = tiles.length ? kpiRow(tiles, { columns: 4 }) : "";
  }

  drawIntervals() {
    const report = this.report;
    const present = AGENTS.filter((key) => report.agents[key]);
    const rewards = present.map((key) => ({ label: Agents.label(key), style: Agents.style(key), mean: report.agents[key].mean_reward, lo: report.agents[key].ci95[0], hi: report.agents[key].ci95[1], n: report.agents[key].episodes }));
    const paired = present.filter((key) => report.paired[key]).map((key) => ({ label: `minus ${Agents.label(key)}`, style: Agents.style(key), mean: report.paired[key].mean_diff, lo: report.paired[key].ci95[0], hi: report.paired[key].ci95[1], n: report.meta.episodes }));
    const rewardNode = this.nodes.status.querySelector("[data-role=rewards]");
    const pairedNode = this.nodes.status.querySelector("[data-role=paired]");
    if (!rewardNode) return;
    IntervalPlot.render(rewardNode, rewards, { xLabel: `episode reward (points), n = ${report.meta.episodes}`, zeroLine: true });
    if (paired.length) IntervalPlot.render(pairedNode, paired, { xLabel: `reward difference (points), n = ${report.meta.episodes}`, zeroLine: true });
    else pairedNode.innerHTML = `<p class="muted">The report holds no agent to compare the model with.</p>`;
  }

  renderFindings(run, seed, findings) {
    const counts = FindingsList.counts(findings);
    this.findingCounts = counts;
    this.renderHead();
    this.nodes.findingsNote.textContent = [counts.FAIL ? `${counts.FAIL} fail` : null, counts.WARN ? `${counts.WARN} warn` : null].filter((part) => part).join(", ");
    const trace = `<a class="btn btn--sm" href="${esc(Links.recordTraces(run.name, seed, this.report ? this.report.meta.episodes : null))}">Record traces</a>`;
    const actions = { report: `<a class="btn btn--sm" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`, traces: trace, verdicts: trace };
    this.nodes.findings.innerHTML = FindingsList.list(findings, { actions, columns: true, emptyText: `No rule is flagged for ${run.name}.` });
  }

  renderCheckpoints() {
    const byRun = new Map(this.runs.map((run) => [run.name, run]));
    const rows = this.checkpoints.map((checkpoint) => {
      const run = byRun.get(checkpoint.run);
      const reward = run && run.reports.length && run.reports[0].agents.model ? fmtSigned(run.reports[0].agents.model.mean_reward) : "";
      const reports = checkpoint.reports.length ? checkpoint.reports.map((seed) => `<a href="${esc(Links.diagnose(checkpoint.run, seed))}" title="Open the report in Diagnose">${seed}</a>`).join(", ") : `<span class="faint">none</span>`;
      const traces = checkpoint.diagnostics.length ? checkpoint.diagnostics.map((seed) => esc(seed)).join(", ") : `<span class="faint">none</span>`;
      return `
        <tr>
          <td><a href="${esc(Links.runs(checkpoint.run))}">${esc(checkpoint.run)}</a></td>
          <td>${esc(checkpoint.kind)}</td>
          <td class="num">${esc(fmtAge(checkpoint.mtime))}</td>
          <td${checkpoint.reason ? ` data-hint="${esc(checkpoint.reason)}"` : ""}>${statusChip(checkpoint.loadable ? "LOADABLE" : "INCOMPATIBLE")}</td>
          <td class="num">${reward}</td>
          <td class="num">${reports}</td>
          <td class="num">${traces}</td>
        </tr>`;
    }).join("");
    this.nodes.checkpoints.innerHTML = this.checkpoints.length ? `
      <div class="dtable dtable--static"><table>
        <thead><tr><th>Run</th><th>Kind</th><th class="num">Age</th><th>Checkpoint</th><th class="num" title="Model mean reward of the newest report">Reward<small>points</small></th><th class="num">Report seeds</th><th class="num">Trace seeds</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>` : `<div class="over__pad">${emptyBlock("No run holds a checkpoint yet.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`)}</div>`;
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
    const jobRows = jobs.jobs.slice(0, 6).map((job) => `<tr><td><a href="#/console/${esc(job.job_id)}">${esc(job.title)}</a></td><td>${statusChip(job.status === "failed" && job.exit_code === -15 ? "stopped" : job.status)}</td><td class="muted">${esc(job.overrides && job.overrides["io.run_name"] ? job.overrides["io.run_name"] : "")}</td><td class="num muted">${esc(job.started.replace("T", " ").slice(5, 16))}</td></tr>`).join("");
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

  renderServices(health) {
    const down = !health.osrm || !health.vroom;
    this.nodes.services.innerHTML = `
      <div class="over__services">
        <span>${statusChip(health.osrm ? "UP" : "DOWN", health.osrm_url, "OSRM")}</span>
        <span>${statusChip(health.vroom ? "UP" : "DOWN", health.vroom_url, "VROOM")}</span>
        ${health.unreachable ? `<span class="muted">The console backend did not answer.</span>` : ""}
        ${down ? `<span class="muted">Recording is disabled. Start the services from the repository root:</span><code class="over__cmd">${esc(DOCKER)}</code>` : `<span class="muted">Both routing services answer.</span>`}
      </div>`;
    this.syncRecord();
  }
}
