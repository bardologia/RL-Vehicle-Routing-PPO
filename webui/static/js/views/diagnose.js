import { api } from "../api.js";
import { AgentTable } from "../diagnose/agent_table.js";
import { BehaviourPanels } from "../diagnose/behaviour.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { Agents } from "../episode/encoding.js";
import { el, esc, fmtPct, fmtSigned, fmtSignedInt, isNum, plural } from "../format.js";
import { Figure } from "../fx/figure.js";
import { Scatter } from "../ui/charts.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead } from "../ui/panel.js";
import { emptyBlock, errorBlock, skeleton } from "../ui/status.js";
import { DataTable, needsTraces } from "../ui/table.js";
import { Segmented } from "../ui/tabs.js";

const REFERENCES = [
  { value: "teacher", label: "teacher" },
  { value: "insertion_only", label: "insertion only" },
  { value: "do_nothing", label: "do nothing" },
];

export class DiagnoseView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.runs = [];
    this.run = null;
    this.seed = null;
    this.report = null;
    this.index = null;
    this.reference = "teacher";
    this.token = 0;

    root.appendChild(el(`
      <div class="view__pad diag">
        ${PageHead.html({
          title: "Diagnose",
          controls: `${PageHead.control("Run", `<select class="select" data-role="run" aria-label="Run"></select>`)}<label class="ctl" data-role="seedwrap"><span class="ctl__label">Report seed</span><select class="select" data-role="seed" aria-label="Report"></select></label><a class="btn btn--sm btn--quiet" data-role="record" href="#/launch/diagnose">Record traces</a>`,
          extra: `<div data-role="flags"></div>`,
          figure: Figure.html("loupe"),
        })}
        <div data-role="error"></div>
        <div data-role="content" class="diag__content"></div>
      </div>`));
    this.runSelect = root.querySelector("[data-role=run]");
    this.seedSelect = root.querySelector("[data-role=seed]");
    this.facts = root.querySelector("[data-role=dateline]");
    this.seedWrap = root.querySelector("[data-role=seedwrap]");
    this.flags = root.querySelector("[data-role=flags]");
    this.recordLink = root.querySelector("[data-role=record]");
    this.errorNode = root.querySelector("[data-role=error]");
    this.content = root.querySelector("[data-role=content]");

    this.runSelect.addEventListener("change", () => this.app.router.go(Links.diagnose(this.runSelect.value, null)));
    this.seedSelect.addEventListener("change", () => this.app.router.go(Links.diagnose(this.run, this.seedSelect.value)));

    app.keys.bind("diagnose", "j", "Next row of the regret table", () => this.moveRow(1));
    app.keys.bind("diagnose", "k", "Previous row of the regret table", () => this.moveRow(-1));
    app.keys.bind("diagnose", "Enter", "Open the selected episode", () => this.openSelected());

    let timer = null;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (this.app.active === "diagnose" && this.report) this.drawCharts();
      }, 150);
    });
  }

  focusSearch() {
    this.runSelect.focus();
  }

  leave() {}

  async enter(parts, query) {
    const token = (this.token += 1);
    this.errorNode.innerHTML = "";
    try {
      this.runs = (await api.get("/api/runs")).runs;
    } catch (error) {
      if (token !== this.token) return;
      this.runs = [];
      this.clearReport();
      this.renderSelectors(null, null);
      this.content.innerHTML = "";
      this.errorNode.innerHTML = errorBlock("The run list could not be loaded.", error.message);
      return;
    }
    if (token !== this.token) return;

    const context = this.app.context;
    const withReports = this.runs.filter((run) => run.reports.length);
    const wanted = parts[0] || (withReports.some((run) => run.name === context.run) ? context.run : null) || (withReports.find((run) => run.loadable) || withReports[0] || {}).name;
    if (!wanted) {
      this.clearReport();
      this.renderSelectors(null, null);
      this.content.innerHTML = `<section class="panel"><div class="panel__body">${emptyBlock("No run has an evaluation report yet. An evaluation of 10 episodes takes about 5 minutes at 3 rounds; the teacher dominates at about 30 s per episode.", `<a class="btn" href="#/launch/evaluate">Open Launch: evaluate</a>`)}</div></section>`;
      return;
    }
    const run = this.runs.find((entry) => entry.name === wanted);
    if (!run) {
      this.clearReport();
      this.renderSelectors(null, wanted);
      this.content.innerHTML = `<section class="panel"><div class="panel__body">${errorBlock(`Run "${wanted}" does not exist.`, "The hash names a run that is not in the runs directory.")}</div></section>`;
      return;
    }
    const seeds = run.reports.map((report) => report.seed);
    const seed = parts[1] !== undefined && seeds.includes(Number(parts[1])) ? Number(parts[1]) : parts[1] === undefined && context.run === run.name && seeds.includes(context.seed) ? context.seed : seeds[0];
    if (seed !== undefined && (parts[0] !== run.name || Number(parts[1]) !== seed)) {
      this.app.router.replace(Links.diagnose(run.name, seed, query.panel));
      return;
    }

    this.clearReport();
    this.run = run.name;
    this.seed = seed === undefined ? null : seed;
    this.runInfo = run;
    this.renderSelectors(run, null);
    this.recordLink.href = Links.recordTraces(run.name, this.seed, null);

    if (this.seed === null) {
      this.content.innerHTML = `<section class="panel"><div class="panel__body">${emptyBlock(`Run "${run.name}" has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds, about 150 s at 5.`, `<a class="btn btn--primary" href="${esc(Links.evaluate(run.name))}">Evaluate this run</a>`)}</div></section>`;
      return;
    }

    this.app.setContext(this.run, this.seed);
    this.content.innerHTML = `<section class="panel"><div class="panel__body">${skeleton(5)}</div></section>`;
    let report;
    let findings;
    let index = null;
    try {
      [report, findings, index] = await Promise.all([
        api.get(`/api/runs/${encodeURIComponent(this.run)}/evaluation/${this.seed}`),
        api.get(`/api/runs/${encodeURIComponent(this.run)}/findings?seed=${this.seed}`),
        run.diagnostics.includes(this.seed) ? api.get(`/api/runs/${encodeURIComponent(this.run)}/diagnostics/${this.seed}`) : Promise.resolve(null),
      ]);
    } catch (error) {
      if (token !== this.token) return;
      this.content.innerHTML = "";
      this.errorNode.innerHTML = errorBlock(`The report of "${this.run}", seed ${this.seed}, could not be loaded.`, error.message);
      return;
    }
    if (token !== this.token) return;
    this.report = report;
    this.findings = findings.findings;
    this.index = index;
    this.render();
    if (query.panel) this.focusPanel(query.panel);
  }

  clearReport() {
    this.run = null;
    this.seed = null;
    this.runInfo = null;
    this.report = null;
    this.findings = [];
    this.index = null;
    this.facts.innerHTML = "";
    this.flags.innerHTML = "";
    this.recordLink.href = "#/launch/diagnose";
  }

  renderSelectors(run, missing) {
    const ordered = [...this.runs].sort((a, b) => Number(Boolean(b.loadable && b.reports.length)) - Number(Boolean(a.loadable && a.reports.length)) || Number(b.reports.length > 0) - Number(a.reports.length > 0));
    const placeholder = run ? "" : `<option value="" selected disabled>${missing ? `${esc(missing)} (not found)` : "no run selected"}</option>`;
    this.runSelect.innerHTML = placeholder + ordered.map((entry) => `<option value="${esc(entry.name)}"${run && entry.name === run.name ? " selected" : ""}>${esc(entry.name)}${entry.reports.length ? "" : " (no report)"}${entry.loadable === false ? " (incompatible)" : ""}</option>`).join("");
    this.runSelect.disabled = !this.runs.length;
    const seeds = run ? run.reports.map((report) => report.seed) : [];
    this.seedSelect.innerHTML = seeds.length ? seeds.map((seed) => `<option value="${seed}"${seed === this.seed ? " selected" : ""}>${seed}</option>`).join("") : `<option>none</option>`;
    this.seedSelect.disabled = seeds.length < 2;
    this.seedWrap.hidden = seeds.length < 2;
  }

  get recordHash() {
    return Links.recordTraces(this.run, this.seed, this.report.meta.episodes);
  }

  estimate() {
    const episodes = this.report.meta.episodes;
    return `Recording model, insertion only and do nothing with insertion verdicts takes about ${Math.max(1, Math.round(episodes * 0.15))} to ${Math.max(2, Math.round(episodes * 0.25))} minutes for ${plural(episodes, "episode")}; a teacher lane adds about 30 to 150 s per episode.`;
  }

  renderFacts() {
    const meta = this.report.meta;
    const facts = [`${plural(meta.episodes, "episode")}`, `seed ${meta.seed}`, meta.greedy ? "greedy" : "sampled", `${plural(meta.caps.max_disruptions_per_episode, "round")}`, `${plural(meta.caps.max_edits_per_round, "edit")} per round`];
    const order = ["model", "model_sampled", "teacher", "insertion_only", "do_nothing"];
    const coverage = order.map((key) => {
      const done = this.report.episodes.filter((episode) => episode.has_trace[key]).length;
      return done || key !== "model_sampled" ? `${Agents.label(key)} ${done} of ${meta.episodes}` : null;
    }).filter((part) => part);
    this.facts.textContent = facts.join(", ");
    const withoutData = FindingsList.counts(this.findings)["NO DATA"];
    this.flags.innerHTML = FindingsList.strip(this.findings, { actions: this.findingActions(), tail: `<button type="button" class="btn btn--sm btn--quiet" data-info aria-haspopup="dialog" aria-expanded="false" aria-label="Trace coverage">${withoutData ? `${withoutData} without data` : "Traces"}</button><template data-info-body><p>Traces: ${coverage.map(esc).join(", ")}</p></template>` });
    this.recordLink.href = this.recordHash;
  }

  render() {
    this.renderFacts();
    this.content.innerHTML = `
      <section class="panel" data-panel="agent-table">
        ${panelHead("Agents", { info: AgentTable.info(this.report) })}
        <div class="panel__body" data-role="agents"></div>
      </section>
      <div class="grid-12">
        <section class="panel span-5" data-panel="paired-scatter">
          ${panelHead("Model against reference", { note: "points", info: infoText("One point per episode: model reward against the reward of the reference agent.", "Points above the diagonal are episodes the model won. Click a point to open the episode.") })}
          <div class="panel__body"><div class="ctls diag__against"><span class="ctl"><span class="ctl__label">Against</span><span data-role="reference"></span></span></div><div data-role="scatter"></div></div>
        </section>
        <section class="panel panel--flush span-7" data-panel="regret-table">
          ${panelHead("Episodes by regret", { info: infoText("Regret = reference reward minus model reward, in points.", "J and K move, Enter opens the episode.") })}
          <div class="panel__body"><div data-role="regret" class="diag__regret"></div></div>
        </section>
      </div>
      <div class="sechead diag__section"><h2>Behaviour <span class="muted">from recorded traces</span></h2></div>
      <div data-role="behaviour-note"></div>
      <div data-role="behaviour"></div>`;

    const available = REFERENCES.filter((option) => this.report.agents[option.value]);
    if (!available.some((option) => option.value === this.reference)) this.reference = available.length ? available[0].value : null;
    if (available.length) {
      new Segmented(this.content.querySelector("[data-role=reference]"), available, this.reference, (value) => {
        this.reference = value;
        this.drawScatter();
      }, "Reference agent");
    }

    this.buildRegret();
    this.behaviour = new BehaviourPanels(this.content.querySelector("[data-role=behaviour]"), { run: this.run, seed: this.seed, recordHash: this.recordHash, onNavigate: (hash) => this.app.router.go(hash) });
    this.drawCharts();
  }

  findingActions() {
    return {
      traces: `<a class="btn btn--sm" href="${esc(this.recordHash)}">Record traces</a>`,
      verdicts: `<a class="btn btn--sm" href="${esc(this.recordHash)}">Record traces</a>`,
      curves: `<a class="btn btn--sm" href="${esc(Links.runs(this.run, "health"))}">Training health</a>`,
    };
  }

  drawCharts() {
    AgentTable.render(this.content.querySelector("[data-role=agents]"), this.report);
    this.drawScatter();
    if (this.index) this.behaviour.render(this.index);
    else {
      this.behaviour.needsTraces();
      this.content.querySelector("[data-role=behaviour-note]").innerHTML = `<p class="noteblock"><span>No traces are recorded for this report, so the eight behaviour panels and the trace columns of the regret table are empty. ${esc(this.estimate())}</span><a class="btn btn--primary" href="${esc(this.recordHash)}">Record traces</a></p>`;
    }
  }

  drawScatter() {
    const node = this.content.querySelector("[data-role=scatter]");
    if (!this.reference || !this.report.agents.model) {
      node.innerHTML = `<p class="muted">The report holds no ${this.report.agents.model ? "reference agent" : "model agent"}.</p>`;
      return;
    }
    const points = this.report.episodes.map((episode) => ({ x: episode.reward[this.reference], y: episode.reward.model, title: `Episode ${episode.index}, seed ${episode.seed}`, episode }));
    Scatter.render(node, points, { xLabel: `${Agents.label(this.reference)} reward`, yLabel: "model reward", height: 320, onClick: (point) => this.openEpisode(point.episode) });
  }

  episodeHash(episode) {
    return Links.episode(this.run, this.seed, episode.index, episode.has_trace.model);
  }

  openEpisode(episode) {
    this.app.router.go(this.episodeHash(episode));
  }

  static eventWords(events) {
    const counts = new Map();
    for (const word of events) counts.set(word, (counts.get(word) || 0) + 1);
    return [...counts].map(([word, count]) => (count > 1 ? `${count} ${word}` : word)).join(", ") || "none";
  }

  buildRegret() {
    const traced = new Map((this.index ? this.index.episodes : []).map((episode) => [episode.index, episode]));
    const model = (row) => (row.trace && row.trace.agents.model ? row.trace.agents.model : null);
    const fromTrace = (pick, format) => ({
      numeric: true,
      hideable: true,
      sortValue: (row) => (model(row) ? pick(model(row)) : null),
      render: (row) => (model(row) ? esc(format ? format(pick(model(row))) : pick(model(row))) : needsTraces()),
    });
    const signed = (value) => `<span class="${isNum(value) && Math.abs(value) > 5e-7 ? (value > 0 ? "loss" : "gain") : ""}">${fmtSigned(value)}</span>`;
    const maxRegret = Math.max(1e-9, ...this.report.episodes.map((episode) => Math.abs(episode.regret_teacher === null ? episode.regret_best_baseline : episode.regret_teacher)));

    const columns = [
      { key: "index", label: "Episode", numeric: true, render: (row) => `<a href="${esc(this.episodeHash(row))}" title="${row.has_trace.model ? "Open the stored trace" : "Record this episode now"}">${row.index}</a>` },
      { key: "seed", label: "Seed", numeric: true },
      { key: "regret_teacher", label: "Regret", full: "Regret against the teacher: teacher reward minus model reward", unit: "pts", numeric: true, render: (row) => signed(row.regret_teacher), bar: { value: (row) => Math.max(0, row.regret_teacher), max: () => maxRegret } },
      { key: "regret_best_baseline", label: "Vs best", full: "Regret against the best baseline", unit: "pts", numeric: true, render: (row) => signed(row.regret_best_baseline) },
      { key: "unassigned_delta", label: "Unassigned", unit: "jobs", numeric: true, title: "Final unassigned jobs, model minus teacher", format: fmtSignedInt },
      { key: "served_delta", label: "Served", unit: "jobs", numeric: true, title: "Served jobs, model minus teacher", format: fmtSignedInt },
      { key: "jobs", label: "Jobs", numeric: true, hideable: true, sortValue: (row) => (row.trace ? row.trace.jobs : null), render: (row) => (row.trace ? row.trace.jobs : needsTraces()) },
      { key: "vehicles", label: "Vehicles", numeric: true, hideable: true, sortValue: (row) => (row.trace ? row.trace.vehicles : null), render: (row) => (row.trace ? row.trace.vehicles : needsTraces()) },
      { key: "events", label: "Events", hideable: true, sortable: false, title: "Disruption of each round, in round order", render: (row) => (row.trace ? `<span title="${esc(`In order: ${row.trace.events.join(", ") || "none"}`)}">${esc(DiagnoseView.eventWords(row.trace.events))}</span>` : needsTraces()) },
      { key: "holding_share", label: "Holding share", ...fromTrace((agent) => agent.holding_share, (value) => fmtPct(value)) },
      { key: "decisions", label: "Decisions", ...fromTrace((agent) => agent.decisions) },
      { key: "inserts", label: "INSERT", ...fromTrace((agent) => agent.operator_counts[0]) },
      { key: "wasted_edits", label: "Wasted", full: "Wasted edits", ...fromTrace((agent) => agent.wasted_edits) },
      { key: "abandonments", label: "Abandonments", ...fromTrace((agent) => agent.abandonments) },
      { key: "profitable_left", label: "Left", title: "Profitable insertions left at round ends, from teacher verdicts", ...fromTrace((agent) => agent.profitable_left, (value) => (value === null ? "needs verdicts" : value)) },
    ];

    this.regret = new DataTable(this.content.querySelector("[data-role=regret]"), {
      columns,
      rowKey: (row) => row.index,
      storeKey: "diagnose.regret",
      menuHost: this.content.querySelector("[data-panel=regret-table] [data-role=panel-actions]"),
      sort: { key: this.report.agents.teacher ? "regret_teacher" : "regret_best_baseline", descending: true },
      onOpen: (row) => this.openEpisode(row),
      emptyText: "The report holds no episode.",
    });
    const rows = this.report.episodes.map((episode) => ({ ...episode, trace: traced.get(episode.index) || null }));
    this.regret.setRows(rows);
    const first = this.regret.sortedRows()[0];
    if (first) this.regret.setSelected(first.index);
  }

  moveRow(step) {
    if (!this.regret || !this.report) return false;
    const rows = this.regret.sortedRows();
    const at = rows.findIndex((row) => String(row.index) === this.regret.selected);
    const next = rows[Math.max(0, Math.min(rows.length - 1, at + step))];
    if (next) this.regret.setSelected(next.index, true);
    return true;
  }

  openSelected() {
    if (!this.regret || !this.report || this.regret.selected === null) return false;
    const focus = document.activeElement;
    if (focus && focus !== document.body && !this.regret.container.contains(focus)) return false;
    this.openEpisode(this.regret.find(this.regret.selected));
    return true;
  }

  focusPanel(key) {
    const panel = this.content.querySelector(`[data-panel="${CSS.escape(key)}"]`);
    if (!panel) return;
    panel.scrollIntoView({ block: "start", behavior: "auto" });
    panel.classList.add("is-target");
    setTimeout(() => panel.classList.remove("is-target"), 2400);
  }
}
