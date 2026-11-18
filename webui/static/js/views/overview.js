import { api } from "../api.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { Agents } from "../episode/encoding.js";
import { el, esc, fmtAge, fmtSigned, plural } from "../format.js";
import { IntervalPlot } from "../ui/charts.js";
import { emptyBlock, errorBlock, plainChip, skeleton, statusChip } from "../ui/status.js";

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
        <header class="diag__bar">
          <h2>Overview</h2>
          <label class="fieldlabel"><span>Run</span><select class="select" data-role="run" aria-label="Run"></select></label>
          <span class="muted" data-role="runfacts"></span>
        </header>
        <div data-role="error"></div>
        <section class="panel" data-region="status">
          <header class="panel__head"><h3 class="panel__title">Model status</h3><span class="panel__help" data-role="status-help"></span><span class="panel__actions" data-role="status-actions"></span></header>
          <div class="panel__body" data-role="status"></div>
        </section>
        <section class="panel" data-region="findings">
          <header class="panel__head"><h3 class="panel__title">Findings</h3><span class="panel__help">Rules flagged WARN or FAIL for the selected run, each with a link to its evidence.</span><span class="panel__actions" data-role="findings-chips"></span></header>
          <div class="panel__body" data-role="findings"></div>
        </section>
        <div class="over__split">
          <section class="panel panel--flush" data-region="checkpoints">
            <header class="panel__head"><h3 class="panel__title">Checkpoints</h3><span class="panel__help">Every run directory that holds a policy file, checked against the current architecture.</span></header>
            <div class="panel__body" data-role="checkpoints"></div>
          </section>
          <section class="panel" data-region="activity">
            <header class="panel__head"><h3 class="panel__title">Activity</h3><span class="panel__help">Jobs of this server session and the latest recordings.</span></header>
            <div class="panel__body" data-role="activity"></div>
          </section>
        </div>
        <section class="panel" data-region="services">
          <header class="panel__head"><h3 class="panel__title">Services</h3><span class="panel__help">Recording, solving and teacher verdicts need both routing services.</span></header>
          <div class="panel__body" data-role="services"></div>
        </section>
      </div>`));
    this.nodes = {};
    for (const node of root.querySelectorAll("[data-role]")) this.nodes[node.dataset.role] = node;
    this.nodes.run.addEventListener("change", () => {
      this.selected = this.nodes.run.value;
      this.loadRun();
    });
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
    const run = this.runs.find((entry) => entry.name === this.selected) || null;
    this.run = run;
    if (!run) {
      this.nodes.runfacts.textContent = "";
      this.nodes["status-help"].textContent = "";
      this.nodes["status-actions"].innerHTML = "";
      this.nodes.status.innerHTML = emptyBlock("No checkpoint loads on the current architecture. A pretrain run produces the first usable policy.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`);
      this.nodes.findings.innerHTML = `<p class="muted">Findings need a run.</p>`;
      this.nodes["findings-chips"].innerHTML = "";
      return;
    }

    const seed = run.reports.length ? run.reports[0].seed : null;
    this.nodes.runfacts.textContent = `${run.kind}, ${fmtAge(run.mtime)}${seed === null ? "" : `, report seed ${seed}`}`;
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
    const record = run.loadable ? `<a class="btn" href="#/episode">Record episode</a>` : "";
    if (!run.loadable) {
      this.nodes["status-help"].textContent = "";
      this.nodes["status-actions"].innerHTML = `<a class="btn" href="${esc(Links.runs(run.name))}">Open run</a>`;
      this.nodes.status.innerHTML = `<div class="errorblock" role="alert">${statusChip("INCOMPATIBLE")}<span class="errorblock__cause">${esc(run.name)} does not load on the current architecture: ${esc(run.reason || "no reason given")}.</span></div>`;
      return;
    }
    if (!this.report) {
      this.nodes["status-help"].textContent = "";
      this.nodes["status-actions"].innerHTML = record;
      this.syncRecord();
      this.nodes.status.innerHTML = emptyBlock(`${run.name} loads but has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds.`, `<a class="btn btn--primary" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`);
      return;
    }

    const report = this.report;
    this.nodes["status-help"].textContent = `${run.name}, report seed ${seed}: n = ${plural(report.meta.episodes, "episode")}, ${report.meta.greedy ? "greedy" : "sampled"} policy.`;
    this.nodes["status-actions"].innerHTML = `<a class="btn btn--primary" href="${esc(Links.diagnose(run.name, seed))}">Open in Diagnose</a>${record}`;
    const model = report.agents.model;
    const teacher = report.paired.teacher;
    const separable = teacher && (teacher.ci95[0] > 0 || teacher.ci95[1] < 0);
    const headline = model ? `
      <div class="over__headline">
        <div class="over__figure"><span>Model mean reward</span><b>${fmtSigned(model.mean_reward)}</b><small>points, 95% interval ${fmtSigned(model.ci95[0])} to ${fmtSigned(model.ci95[1])}</small></div>
        ${teacher ? `<div class="over__figure"><span>Model minus teacher, paired</span><b class="${separable ? (teacher.mean_diff > 0 ? "gain" : "loss") : ""}">${fmtSigned(teacher.mean_diff)}</b><small>points, ${fmtSigned(teacher.ci95[0])} to ${fmtSigned(teacher.ci95[1])}${separable ? "" : ", not separable at this n"}</small></div>
        <div class="over__figure"><span>Episodes won / level / lost against teacher</span><b>${teacher.wins} / ${teacher.ties} / ${teacher.losses}</b><small>of ${report.meta.episodes}</small></div>` : ""}
        ${model.identical_to ? `<div class="over__figure"><span>Behaviour</span><b class="over__word">${plainChip(`identical to ${model.identical_to}`)}</b><small>same reward in every episode</small></div>` : ""}
      </div>` : "";
    this.nodes.status.innerHTML = `${headline}
      <div class="over__plots">
        <div><h4 class="subhead">Mean reward with 95% interval</h4><div data-role="rewards"></div></div>
        <div><h4 class="subhead">Model minus agent, paired difference with 95% interval</h4><div data-role="paired"></div></div>
      </div>
      ${report.missing_agents.length ? `<p class="muted">Missing from this report: ${report.missing_agents.map((key) => esc(Agents.label(key))).join(", ")}.</p>` : ""}`;
    this.drawIntervals();
    this.syncRecord();
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
    const flagged = FindingsList.flagged(findings).sort((a, b) => FindingsList.ORDER[a.status] - FindingsList.ORDER[b.status]);
    const counts = FindingsList.counts(findings);
    this.nodes["findings-chips"].innerHTML = FindingsList.summaryChips(findings);
    const trace = `<a class="btn btn--sm" href="${esc(Links.recordTraces(run.name, seed, this.report ? this.report.meta.episodes : null))}">Record traces</a>`;
    const actions = { report: `<a class="btn btn--sm" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`, traces: trace, verdicts: trace };
    const rows = flagged.length ? `<ul class="findings findings--cols">${flagged.map((finding) => FindingsList.row(finding, actions)).join("")}</ul>` : `<p class="muted">No rule is flagged WARN or FAIL for ${esc(run.name)}.</p>`;
    const rest = [counts.PASS ? `${plural(counts.PASS, "rule")} at PASS` : null, counts["NO DATA"] ? `${plural(counts["NO DATA"], "rule")} without data` : null].filter((part) => part).join(", ");
    const more = seed === null ? Links.runs(run.name) : Links.diagnose(run.name, seed, "findings");
    this.nodes.findings.innerHTML = `${rows}${rest ? `<p class="over__rest"><span class="muted">${rest}.</span> <a href="${esc(more)}">See all rules</a></p>` : ""}`;
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
          <td class="over__reason">${statusChip(checkpoint.loadable ? "LOADABLE" : "INCOMPATIBLE")}${checkpoint.reason ? ` <span class="muted">${esc(checkpoint.reason)}</span>` : ""}</td>
          <td class="num">${reward}</td>
          <td class="num">${reports}</td>
          <td class="num">${traces}</td>
        </tr>`;
    }).join("");
    this.nodes.checkpoints.innerHTML = this.checkpoints.length ? `
      <div class="dtable dtable--static"><table>
        <thead><tr><th>Run</th><th>Kind</th><th class="num">Age</th><th>Checkpoint</th><th class="num">Model mean reward<small>points</small></th><th class="num">Report seeds</th><th class="num">Trace seeds</th></tr></thead>
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
      <h4 class="subhead">Jobs</h4>
      ${jobRows ? `<div class="dtable dtable--dense dtable--static"><table><tbody>${jobRows}</tbody></table></div>` : `<p class="muted">No job in this server session. <a href="#/launch">Open Launch</a></p>`}
      <h4 class="subhead">Recordings</h4>
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
