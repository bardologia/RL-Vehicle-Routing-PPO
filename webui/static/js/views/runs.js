import { api, ApiError } from "../api.js";
import { AgentTable } from "../diagnose/agent_table.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { el, esc, fmtAge, fmtBytes, fmtFixed, fmtPct, fmtSigned, icon, isNum, plural, Store } from "../format.js";
import { LineChart } from "../ui/charts.js";
import { kpiRow } from "../ui/kpi.js";
import { LogText } from "../ui/logtext.js";
import { MarkdownTable } from "../ui/markdown_table.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead, panelHtml } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { EMPTY_FIGURES, emptyBlock, errorBlock, failView, skeleton, statusChip, statusTone } from "../ui/status.js";
import { FigureStates } from "../ui/figure_states.js";
import { kvTable } from "../ui/table.js";
import { Segmented, Tabs } from "../ui/tabs.js";

const SECTIONS = [
  { value: "summary", label: "Summary" },
  { value: "health", label: "Training health" },
  { value: "curves", label: "Curves" },
  { value: "logs", label: "Logs" },
  { value: "files", label: "Files" },
  { value: "gallery", label: "Gallery" },
];

const SECTION_FIGURES = { summary: "checkpoint", health: "training", curves: "training", logs: "terminal", files: "cabinet", gallery: "riffle" };
const STYLES = ["solid", "dashed", "dotted"];
const PAGE = 24;

export class RunStatus {
  static word(run) {
    if (!run.has_checkpoint) return "NO CHECKPOINT";
    return run.loadable ? "LOADABLE" : "INCOMPATIBLE";
  }

  static chip(run) {
    const hint = run.loadable === false && run.reason ? ` data-hint="${esc(run.reason)}"` : "";
    return `<span${hint}>${statusChip(RunStatus.word(run))}</span>`;
  }

  static dot(run) {
    const word = RunStatus.word(run);
    const hint = [word.toLowerCase(), run.loadable === false && run.reason ? run.reason : null].filter((part) => part).join(": ");
    return `<span class="state state--${statusTone(word)}" data-hint="${esc(hint)}"><i></i><span class="sr-only">${esc(word.toLowerCase())}</span></span>`;
  }

  static rewardLine(run) {
    const report = run.reports[0];
    if (!report || !report.agents.model) return "no report";
    const model = report.agents.model.mean_reward;
    const teacher = report.agents.teacher ? report.agents.teacher.mean_reward : null;
    return `model ${fmtSigned(model)}${teacher === null ? "" : `, ${fmtSigned(model - teacher)} to teacher`}`;
  }
}

export class CurveSet {
  static group(tag) {
    return tag.includes("/") ? tag.split("/")[0] : "other";
  }

  static build(tags) {
    const names = Object.keys(tags).sort();
    const used = new Set();
    const entries = [];
    for (const tag of names) {
      if (used.has(tag)) continue;
      const cut = tag.lastIndexOf("/") + 1;
      const leaf = tag.slice(cut);
      if (leaf.startsWith("old_") || leaf.startsWith("new_")) {
        const stem = leaf.slice(4);
        const oldTag = `${tag.slice(0, cut)}old_${stem}`;
        const newTag = `${tag.slice(0, cut)}new_${stem}`;
        if (tags[oldTag] && tags[newTag]) {
          used.add(oldTag);
          used.add(newTag);
          entries.push({ key: `${tag.slice(0, cut)}${stem}`, title: `${tag.slice(0, cut)}${stem} (old and new)`, series: [{ label: "old", tag: oldTag }, { label: "new", tag: newTag }] });
          continue;
        }
      }
      used.add(tag);
      entries.push({ key: tag, title: tag, series: [{ label: leaf, tag }] });
    }
    return entries;
  }
}

export class RunsView {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.runs = [];
    this.run = null;
    this.detail = null;
    this.section = "summary";
    this.filter = "";
    this.token = 0;
    this.cache = {};
    this.charts = [];
    this.smoothing = Store.read("runs.smoothing", 0.6);
    this.curveState = { search: "", page: 0, log: false, overlay: "" };
    this.logState = { file: null, search: "", follow: false };
    this.followTimer = null;
    this.shell = null;
    this.slotNames = [];

    root.appendChild(el(`
      <div class="runs">
        <aside class="runs__rail" aria-label="Runs">
          <label class="runs__search">${icon("search")}<input class="input" type="search" placeholder="Search runs" aria-label="Search runs" data-role="search"></label>
          <div class="runs__list" data-role="list"></div>
        </aside>
        <div class="runs__main" data-role="main"></div>
      </div>`));
    this.list = root.querySelector("[data-role=list]");
    this.main = root.querySelector("[data-role=main]");
    this.search = root.querySelector("[data-role=search]");
    this.search.addEventListener("input", () => {
      this.filter = this.search.value.trim().toLowerCase();
      this.renderRail();
    });
    this.main.addEventListener("click", (event) => this.onClick(event));
    this.main.addEventListener("figure-pick", (event) => {
      const name = this.slotNames[event.detail.slot];
      if (name && name !== this.run) this.app.router.go(Links.runs(name, this.section));
    });
    app.onJobs(() => this.writeTraining());
    this.main.addEventListener("input", (event) => this.onInput(event));
    this.main.addEventListener("change", (event) => this.onInput(event));

    let timer = null;
    window.addEventListener("resize", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        if (this.app.active !== "runs") return;
        for (const chart of this.charts) chart.render(chart.data);
        this.drawAgentTables();
      }, 150);
    });
  }

  focusSearch() {
    const inner = this.main.querySelector("input[type=search]");
    (inner || this.search).focus();
  }

  leave() {
    clearInterval(this.followTimer);
    this.followTimer = null;
  }

  async enter(parts) {
    const token = (this.token += 1);
    try {
      this.runs = (await api.get("/api/runs")).runs;
    } catch (error) {
      this.shell = null;
      this.main.innerHTML = `<div class="view__pad">${failView("The run list could not be loaded.", error.message, error)}</div>`;
      return;
    }
    if (token !== this.token) return;

    const context = this.app.context;
    const name = parts[0] || (this.runs.some((run) => run.name === context.run) ? context.run : null) || (this.runs[0] ? this.runs[0].name : null);
    const section = SECTIONS.some((entry) => entry.value === parts[1]) ? parts[1] : "summary";
    if (!name) {
      this.renderRail();
      this.shell = null;
      this.main.innerHTML = `<div class="view__pad">${emptyBlock("The runs directory is empty. A pretrain run is the usual first step.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`, EMPTY_FIGURES.checkpoints)}</div>`;
      return;
    }
    if (parts[0] !== name || parts[1] !== section) {
      this.app.router.replace(Links.runs(name, section));
      return;
    }

    const changed = name !== this.run;
    this.run = name;
    this.section = section;
    this.renderRail();
    if (changed) {
      this.detail = null;
      this.cache = {};
      this.curveState.page = 0;
      this.logState.file = null;
      this.openFolders = [];
      this.ensureShell();
      this.shell.title.textContent = name;
      this.shell.date.innerHTML = "";
      this.shell.controls.innerHTML = "";
      this.shell.kpis.innerHTML = "";
      this.shell.tabs.innerHTML = "";
      this.shell.section.innerHTML = skeleton(5);
      this.writeShelf();
    }
    try {
      if (!this.runs.some((run) => run.name === name)) throw new ApiError(404, `unknown run '${name}'`);
      this.detail = await api.get(`/api/runs/${encodeURIComponent(name)}`);
    } catch (error) {
      if (token !== this.token) return;
      this.detail = null;
      this.shell = null;
      this.main.innerHTML = `<div class="view__pad">${failView(error.status === 404 ? `There is no run named "${name}".` : `Run "${name}" could not be loaded.`, error.message, error)}<p class="runs__back"><a class="btn" href="#/runs">Open the newest run</a></p></div>`;
      return;
    }
    if (token !== this.token) return;
    const info = this.runs.find((run) => run.name === name);
    if (info && info.reports.length) this.app.setContext(name, info.reports[0].seed);
    else this.app.setContext(name, null);
    this.renderShell(info);
    await this.renderSection(token);
  }

  renderRail() {
    const shown = this.runs.filter((run) => !this.filter || run.name.toLowerCase().includes(this.filter) || run.kind.includes(this.filter));
    this.list.innerHTML = shown.length ? shown.map((run) => `
      <a class="runs__item" href="${esc(Links.runs(run.name, this.section))}"${run.name === this.run ? ' aria-current="true"' : ""}>
        <span class="runs__line">${RunStatus.dot(run)}<span class="runs__name">${esc(run.name)}</span><span class="runs__age">${esc(fmtAge(run.mtime).replace(/ ago$/, ""))}</span></span>
        <span class="runs__sub">${esc(`${run.kind}, ${RunStatus.rewardLine(run)}`)}</span>
      </a>`).join("") : `<p class="muted runs__none">No run matches "${esc(this.filter)}".</p>`;
  }

  ensureShell() {
    if (this.shell) return;
    this.main.innerHTML = `
      <div class="view__pad">
        ${PageHead.html({ title: "", form: "narrow", after: `<div data-role="tabs" class="runs__tabs"></div>`, figure: HairlineFigure.html("checkpoint") })}
        <div data-role="section" class="runs__section"></div>
      </div>`;
    const find = (selector) => this.main.querySelector(selector);
    this.shell = { title: find("[data-role=title]"), date: find(".view-head__date"), controls: find("[data-role=controls]"), kpis: find("[data-role=kpis]"), tabs: find("[data-role=tabs]"), section: find("[data-role=section]"), figure: find("hairline-figure[name=checkpoint]") };
  }

  headFigure(name, intensity) {
    const figure = this.shell.figure;
    if (figure.getAttribute("name") !== name) {
      delete figure.dataset.stateKey;
      figure.setAttribute("name", name);
    }
    if (intensity === undefined) figure.removeAttribute("intensity");
    else figure.setAttribute("intensity", intensity.toFixed(2));
  }

  writeShelf() {
    const name = SECTION_FIGURES[this.section] || "checkpoint";
    const detail = this.detail;
    const count = name === "cabinet" && detail ? Math.min(1, detail.files.length / 60) : name === "riffle" && detail ? Math.min(1, detail.analysis.length / 8) : undefined;
    this.headFigure(name, count);
    if (name === "training") this.writeTraining();
    if (name !== "checkpoint") return;
    const shelf = FigureStates.checkpoints(this.runs, this.run);
    this.slotNames = shelf.names;
    FigureStates.write(this.shell.figure, shelf.state);
  }

  renderShell(info) {
    const detail = this.detail;
    const seed = detail.reports[0];
    const episodes = info && info.reports[0] ? info.reports[0].episodes : null;
    this.ensureShell();
    this.shell.title.textContent = detail.name;
    this.shell.date.innerHTML = `<span data-role="dateline">${esc(`${detail.kind}${info ? `, ${fmtAge(info.mtime)}` : ""}`)}</span>${RunStatus.chip(detail)}`;
    this.writeHeadKpis(detail.name);
    this.shell.controls.innerHTML = `${seed !== undefined ? `<a class="btn btn--primary" href="${esc(Links.diagnose(detail.name, seed))}">Open in Diagnose</a>` : ""}${detail.loadable ? `<button type="button" class="btn btn--sm btn--quiet" data-action="actions" aria-haspopup="dialog" aria-expanded="false">Actions</button>` : ""}`;
    this.writeShelf();
    new Tabs(this.shell.tabs, SECTIONS.map((entry) => ({ ...entry, count: entry.value === "logs" ? detail.logs.length : entry.value === "files" ? detail.files.length : entry.value === "gallery" ? detail.analysis.length : undefined })), this.section, (value) => this.app.router.go(Links.runs(this.run, value)));
    this.body = this.shell.section;
    this.actionItems = detail.loadable ? [
      { label: "Record episode", href: "#/episode" },
      { label: "Evaluate", href: Links.evaluate(detail.name) },
      { label: "Record traces", href: Links.recordTraces(detail.name, seed === undefined ? null : seed, episodes) },
    ] : [];
  }

  async cached(key, url) {
    if (!this.cache[key]) this.cache[key] = await api.get(url);
    return this.cache[key];
  }

  async writeHeadKpis(name) {
    const detail = this.detail;
    if (!detail.has_events) return;
    let health;
    try {
      health = await this.cached("health", `/api/runs/${encodeURIComponent(name)}/health`);
    } catch (error) {
      return;
    }
    if (!this.shell || this.run !== name || this.detail !== detail) return;
    if (health.kind !== "pretrain") {
      this.shell.kpis.innerHTML = "";
      return;
    }
    const accuracy = health.charts.find((chart) => chart.key === "accuracy");
    const tiles = accuracy ? accuracy.series.filter((series) => series.values.length).map((series) => ({ label: `${series.label} head`, value: fmtPct(series.values[series.values.length - 1], 1), note: `last of ${plural(series.values.length, "epoch")}`, title: `Behaviour cloning accuracy: last logged agreement with the teacher on held-out pairs, ${series.label} head. Tag ${series.tag}.` })) : [];
    this.shell.kpis.innerHTML = tiles.length ? kpiRow(tiles, { columns: Math.max(3, tiles.length) }) : `<p class="muted">pretrain/accuracy is not logged in this run.</p>`;
  }

  async renderSection(token) {
    clearInterval(this.followTimer);
    this.followTimer = null;
    this.charts = [];
    const body = this.body;
    body.innerHTML = skeleton(4);
    try {
      if (this.section === "summary") await this.renderSummary(token);
      else if (this.section === "health") await this.renderHealth(token);
      else if (this.section === "curves") await this.renderCurves(token);
      else if (this.section === "logs") this.renderLogs();
      else if (this.section === "files") this.renderFiles();
      else this.renderGallery();
    } catch (error) {
      if (token !== this.token) return;
      body.innerHTML = errorBlock(`The ${this.section} section of "${this.run}" could not be loaded.`, error.message);
    }
  }

  trainingValues() {
    const health = this.cache.health;
    if (health) {
      const chart = health.charts.find((entry) => entry.key === (health.kind === "ppo" ? "episode_reward" : "accuracy"));
      if (chart && chart.series.length) return chart.series[0].values;
    }
    const curves = this.cache.curves;
    if (curves) {
      const tag = ["episode/total_reward", "pretrain/accuracy/operator"].find((name) => curves.tags[name]);
      if (tag) return curves.tags[tag].values;
    }
    return null;
  }

  writeTraining() {
    const figure = this.shell ? this.shell.figure : null;
    if (!figure || figure.getAttribute("name") !== "training") return;
    const jobs = this.app.jobs || [];
    const live = jobs.some((job) => job.status === "running" && job.overrides && job.overrides["io.run_name"] === this.run);
    FigureStates.write(figure, FigureStates.training(this.trainingValues(), live));
  }

  panel(title, help, bodyHtml, extra, note) {
    return panelHtml(title, bodyHtml, { info: infoText(help), note, attrs: extra });
  }

  async renderSummary(token) {
    const detail = this.detail;
    const base = `/api/runs/${encodeURIComponent(this.run)}`;
    const [findings, ...reports] = await Promise.all([
      this.cached("findings", `${base}/findings`),
      ...detail.reports.map((seed) => this.cached(`evaluation.${seed}`, `${base}/evaluation/${seed}`)),
    ]);
    if (token !== this.token) return;

    const parts = [];
    if (detail.loadable === false) parts.push(errorBlock(`This checkpoint does not load on the current architecture: ${detail.reason}. It cannot be evaluated or recorded.`));

    const traceLink = detail.loadable ? `<a class="btn btn--sm" href="${esc(Links.recordTraces(this.run, detail.reports[0] === undefined ? null : detail.reports[0], null))}">Record traces</a>` : "";
    const scope = detail.reports.length ? `Rules over report seed ${detail.reports[0]}, its traces and the training curves.` : "Rules over the training curves. Report and trace rules need an evaluation.";
    parts.push(`<div class="runs__findings">${FindingsList.strip(findings.findings, { actions: { report: detail.loadable ? `<a class="btn btn--sm" href="${esc(Links.evaluate(this.run))}">Evaluate this run</a>` : "", traces: traceLink, verdicts: traceLink }, tail: Popover.infoButton("Findings", infoText(scope)) })}</div>`);

    reports.forEach((report, index) => {
      parts.push(`<section class="panel" data-region="evaluation">${panelHead("Evaluation", { note: `seed ${report.meta.seed}, ${plural(report.meta.episodes, "episode")}, ${report.meta.greedy ? "greedy" : "sampled"}`, info: AgentTable.info(report), actions: `<span data-role="agent-set" data-report="${index}"></span>` })}<div class="panel__body"><div data-role="agents" data-report="${index}"></div>
        <details class="disclosure runs__settings"><summary>Report settings: reward weights, environment, caps</summary><div class="runs__meta3">
          <div><h4 class="ctl__label">Reward</h4>${kvTable(Object.entries(report.meta.reward).map(([key, value]) => [key, esc(JSON.stringify(value))]))}</div>
          <div><h4 class="ctl__label">Caps and run</h4>${kvTable([...Object.entries(report.meta.caps), ...Object.entries(report.meta).filter(([, value]) => typeof value !== "object")].map(([key, value]) => [key, esc(String(value))]))}</div>
          <div><h4 class="ctl__label">Environment</h4>${kvTable(Object.entries(report.meta.environment).map(([key, value]) => [key, esc(JSON.stringify(value))]))}</div>
        </div></details></div></section>`);
    });
    if (!reports.length) parts.push(this.panel("Evaluation", "", emptyBlock(detail.loadable ? "This run has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds, about 150 s at 5." : "This run has no evaluation report.", detail.loadable ? `<a class="btn btn--primary" href="${esc(Links.evaluate(this.run))}">Evaluate this run</a>` : "")));

    const docs = Object.entries(detail.markdown);
    if (docs.length) parts.push(this.panel("Model documents", "Written by the training script.", docs.map(([file, text]) => `<details class="disclosure" data-doc="${esc(file)}"><summary>${esc(file)}</summary>${MarkdownTable.render(text)}</details>`).join(""), "", `${plural(docs.length, "file")}`));

    this.body.innerHTML = parts.join("");
    this.reports = reports;
    for (const host of this.body.querySelectorAll("[data-role=agent-set]")) AgentTable.switcher(host, () => this.drawAgentTables());
    this.drawAgentTables();
  }

  drawAgentTables() {
    if (this.section !== "summary" || !this.reports) return;
    this.reports.forEach((report, index) => {
      const node = this.body.querySelector(`[data-role=agents][data-report="${index}"]`);
      if (node) AgentTable.render(node, report);
    });
  }

  smoothingControl() {
    return `<label class="ctl runs__slider"><span class="ctl__label">Smoothing</span><input type="range" min="0" max="0.95" step="0.05" value="${this.smoothing}" data-role="smoothing" aria-label="Smoothing"><output>${fmtFixed(this.smoothing, 2)}</output></label>`;
  }

  static dedupe(series) {
    const out = [];
    for (const entry of series) {
      const twin = out.find((other) => other.values.length === entry.values.length && other.values.every((value, index) => value === entry.values[index]));
      if (twin) twin.merged.push(entry.label);
      else out.push({ ...entry, merged: [entry.label] });
    }
    return out.map((entry) => ({ ...entry, label: entry.merged.length > 2 ? `${entry.merged.length} groups, identical` : entry.merged.join(" = ") }));
  }

  mountChart(node, series, options) {
    const chart = new LineChart(node, { height: 200, smoothing: this.smoothing, ...options });
    chart.domain = options.domain || null;
    chart.options.onHover = (step) => {
      for (const other of this.charts) if (other !== chart && other.domain && other.domain === chart.domain) other.cross(step);
    };
    chart.render(series);
    this.charts.push(chart);
    return chart;
  }

  static readsHtml(groups) {
    if (!groups.length) return "";
    return `<div class="runs__reads">${groups.map((group) => `<div class="reads" data-hint="${esc(group.hint)}"><span class="ctl__label">${esc(group.title)}</span>${group.reads.map((read) => `<span class="read"><span class="read__name">${esc(read.label)}</span><b>${esc(read.value)}</b>${read.note ? `<small>${esc(read.note)}</small>` : ""}</span>`).join("")}</div>`).join("")}</div>`;
  }

  controlsInfo() {
    return Popover.infoButton("Charts", infoText("Fixed charts. Charts on the same step axis share the crosshair.", "A tag with a single logged value is listed as a value above the charts."));
  }

  async renderHealth(token) {
    if (!this.detail.has_events) {
      this.body.innerHTML = this.panel("Training health", "", emptyBlock("No TensorBoard events in this run."));
      return;
    }
    const health = await this.cached("health", `/api/runs/${encodeURIComponent(this.run)}/health`);
    if (token !== this.token) return;
    const blocks = [];
    const sparse = [];
    health.charts.forEach((chart, index) => {
      const series = RunsView.dedupe(chart.series);
      if (chart.series.length > 0 && chart.series.every((entry) => entry.values.length <= 1)) {
        const shown = chart.series.filter((entry) => entry.values.length);
        const steps = [...new Set(shown.map((entry) => entry.steps[0]))];
        sparse.push({ title: chart.title, hint: `${chart.title}: one logged value per tag${steps.length === 1 ? `, step ${steps[0]}` : ""}`, reads: shown.map((entry) => ({ label: entry.label, value: `${fmtFixed(entry.values[0], 3)}${chart.unit ? ` ${chart.unit}` : ""}`, note: steps.length === 1 ? "" : `step ${entry.steps[0]}` })).concat(steps.length === 1 ? [{ label: "step", value: String(steps[0]) }] : []) });
        return;
      }
      const split = series.length > 3;
      const frames = split ? series.map((entry, s) => `<div class="runs__multi"><h4 class="ctl__label">${esc(entry.label)}</h4><div data-chart="${index}" data-series="${s}"></div></div>`).join("") : `<div data-chart="${index}"></div>`;
      const points = chart.series.length ? Math.max(...chart.series.map((entry) => entry.steps.length)) : 0;
      const merged = series.filter((entry) => entry.merged.length > 1).map((entry) => `${entry.merged.join(", ")} are identical`).join("; ");
      blocks.push(`<section class="panel" data-health="${esc(chart.key)}">${panelHead(chart.title, { note: points ? `n = ${plural(points, "point")}` : "", info: merged && split ? infoText(`${merged}.`) : "" })}<div class="panel__body">${frames}</div></section>`);
    });
    this.body.innerHTML = `
      <div class="runs__controls"><span class="diag__spacer"></span>${this.smoothingControl()}${this.controlsInfo()}</div>
      ${RunsView.readsHtml(sparse)}
      <div class="runs__charts">${blocks.join("")}</div>`;
    this.writeTraining();

    health.charts.forEach((chart, index) => {
      const series = RunsView.dedupe(chart.series);
      const domain = chart.series.length ? CurveSet.group(chart.series[0].tag) : null;
      const xLabel = domain === "episode" ? "episode" : domain === "sample" ? "minibatch" : domain === "batch" ? "update" : "step";
      if (chart.series.length > 0 && chart.series.every((entry) => entry.values.length <= 1)) return;
      if (series.length > 3) {
        series.forEach((entry, s) => this.mountChart(this.body.querySelector(`[data-chart="${index}"][data-series="${s}"]`), [{ ...entry, style: "solid" }], { height: 130, xLabel, yLabel: chart.unit, domain }));
      } else {
        this.mountChart(this.body.querySelector(`[data-chart="${index}"]`), series.map((entry, s) => ({ ...entry, style: STYLES[s] })), { xLabel, yLabel: chart.unit, domain, caption: series.filter((entry) => entry.merged.length > 1).map((entry) => `${entry.merged.join(", ")} are identical`).join("; ") });
      }
    });
  }

  async renderCurves(token) {
    if (!this.detail.has_events) {
      this.body.innerHTML = this.panel("Curves", "", emptyBlock("No TensorBoard events in this run."));
      return;
    }
    const curves = await this.cached("curves", `/api/runs/${encodeURIComponent(this.run)}/curves`);
    if (token !== this.token) return;
    this.curveEntries = CurveSet.build(curves.tags);
    const others = this.runs.filter((run) => run.has_events && run.name !== this.run);
    this.body.innerHTML = `
      <div class="runs__controls">
        <label class="runs__search">${icon("search")}<input class="input" type="search" placeholder="Search tags" aria-label="Search tags" data-role="curve-search" value="${esc(this.curveState.search)}"></label>
        <span class="runs__count" data-role="page-count"></span>
        <span class="ctl"><span class="ctl__label">Scale</span><span data-role="curve-scale"></span></span>
        ${this.smoothingControl()}
        <label class="ctl"><span class="ctl__label">Overlay</span><select class="select" data-role="curve-overlay" aria-label="Overlay run"><option value="">none</option>${others.map((run) => `<option value="${esc(run.name)}"${run.name === this.curveState.overlay ? " selected" : ""}>${esc(run.name)}</option>`).join("")}</select></label>
        <span class="diag__spacer"></span>
        <span class="runs__pager" data-role="pager"><button type="button" class="btn btn--sm" data-action="page-prev">Previous</button><span data-role="page-label" class="muted"></span><button type="button" class="btn btn--sm" data-action="page-next">Next</button></span>
        ${this.controlsInfo()}
      </div>
      <div data-role="curve-error"></div>
      <div data-role="curve-reads"></div>
      <div data-role="curve-grid"></div>`;
    new Segmented(this.body.querySelector("[data-role=curve-scale]"), [{ value: "linear", label: "linear" }, { value: "log", label: "log" }], this.curveState.log ? "log" : "linear", (value) => {
      this.curveState.log = value === "log";
      for (const chart of this.charts) chart.setOptions({ log: this.curveState.log });
    }, "Scale");
    this.writeTraining();
    await this.drawCurves();
  }

  get pins() {
    return Store.read("runs.pins", []);
  }

  static leaf(key) {
    const cut = key.indexOf("/");
    return cut < 0 ? key : key.slice(cut + 1);
  }

  async drawCurves() {
    const tags = this.cache.curves.tags;
    const state = this.curveState;
    const errorNode = this.body.querySelector("[data-role=curve-error]");
    errorNode.innerHTML = "";
    let overlay = null;
    if (state.overlay) {
      try {
        overlay = (await this.cached(`overlay.${state.overlay}`, `/api/runs/${encodeURIComponent(state.overlay)}/curves`)).tags;
      } catch (error) {
        errorNode.innerHTML = errorBlock(`The curves of "${state.overlay}" could not be loaded.`, error.message);
      }
    }

    const pins = this.pins;
    const needle = state.search.toLowerCase();
    const matched = this.curveEntries.filter((entry) => !needle || entry.series.some((series) => series.tag.toLowerCase().includes(needle)));
    const single = matched.filter((entry) => entry.series.every((series) => tags[series.tag].values.length <= 1));
    const charted = matched.filter((entry) => !single.includes(entry));
    const ordered = [...charted.filter((entry) => pins.includes(entry.key)), ...charted.filter((entry) => !pins.includes(entry.key))];
    const pages = Math.max(1, Math.ceil(ordered.length / PAGE));
    state.page = Math.max(0, Math.min(pages - 1, state.page));
    const shown = ordered.slice(state.page * PAGE, (state.page + 1) * PAGE);

    this.body.querySelector("[data-role=pager]").hidden = pages < 2;
    this.body.querySelector("[data-role=page-label]").textContent = `page ${state.page + 1} of ${pages}`;
    this.body.querySelector("[data-role=page-count]").textContent = `${plural(Object.keys(tags).length, "tag")}`;
    this.body.querySelector("[data-role=page-count]").dataset.hint = `${plural(ordered.length, "chart")} and ${plural(single.length, "single value")} from ${plural(Object.keys(tags).length, "tag")}${pages > 1 ? `, page ${state.page + 1} of ${pages}` : ""}`;
    this.body.querySelector("[data-action=page-prev]").disabled = state.page === 0;
    this.body.querySelector("[data-action=page-next]").disabled = state.page >= pages - 1;

    const groups = new Map();
    for (const entry of single) {
      const group = CurveSet.group(entry.key);
      if (!groups.has(group)) groups.set(group, []);
      for (const series of entry.series) {
        const data = tags[series.tag];
        if (data.values.length) groups.get(group).push({ label: RunsView.leaf(series.tag), value: fmtFixed(data.values[0], 3), note: `step ${data.steps[0]}` });
      }
    }
    this.body.querySelector("[data-role=curve-reads]").innerHTML = RunsView.readsHtml([...groups].filter(([, reads]) => reads.length).map(([group, reads]) => ({ title: group, hint: `Tags under ${group}/ with a single logged value`, reads })));

    const grid = this.body.querySelector("[data-role=curve-grid]");
    this.charts = [];
    if (!shown.length) {
      grid.innerHTML = single.length ? "" : `<p class="muted">No tag contains "${esc(state.search)}".</p>`;
      return;
    }
    let lastGroup = null;
    const html = [];
    shown.forEach((entry, index) => {
      const pinned = pins.includes(entry.key);
      const group = pinned ? "Pinned" : CurveSet.group(entry.key);
      if (group !== lastGroup) {
        if (lastGroup !== null) html.push("</div>");
        html.push(`<div class="sechead runs__group"><h2>${esc(group)}</h2></div><div class="runs__charts">`);
        lastGroup = group;
      }
      const title = RunsView.leaf(entry.title);
      html.push(`<section class="panel" data-curve="${esc(entry.key)}">${panelHead(title, { note: pinned ? CurveSet.group(entry.key) : "", titleClass: "runs__tag", titleAttrs: ` data-hint="${esc(entry.title)}"`, actions: `<button type="button" class="tbtn tbtn--sm" data-action="pin" data-key="${esc(entry.key)}" aria-pressed="${pinned}" aria-label="${pinned ? "Unpin" : "Pin"} ${esc(entry.title)}" data-hint="${pinned ? "Unpin" : "Pin"}">${icon("flag")}</button>` })}<div class="panel__body"><div data-chart="${index}"></div></div></section>`);
    });
    html.push("</div>");
    grid.innerHTML = html.join("");

    shown.forEach((entry, index) => {
      const series = entry.series.map((item, s) => ({ label: item.label, steps: tags[item.tag].steps, values: tags[item.tag].values, style: STYLES[s] }));
      if (overlay) {
        for (const item of entry.series) {
          if (overlay[item.tag]) series.push({ label: `${state.overlay}${entry.series.length > 1 ? ` ${item.label}` : ""}`, steps: overlay[item.tag].steps, values: overlay[item.tag].values, style: "second" });
        }
        if (series.length > entry.series.length && entry.series.length === 1) series[0].label = this.run;
      }
      this.mountChart(grid.querySelector(`[data-chart="${index}"]`), series, { height: 180, log: state.log, xLabel: "step", yLabel: "", domain: CurveSet.group(entry.key) });
    });
  }

  renderLogs() {
    const logs = this.detail.logs;
    if (!logs.length) {
      this.body.innerHTML = this.panel("Logs", "", emptyBlock("This run has no log file."));
      return;
    }
    const state = this.logState;
    if (!logs.some((log) => log.name === state.file)) state.file = logs[0].name;
    this.body.innerHTML = `
      <div class="runs__controls">
        <label class="ctl"><span class="ctl__label">File</span><select class="select" data-role="log-file" aria-label="Log file">${logs.map((log) => `<option${log.name === state.file ? " selected" : ""}>${esc(log.name)}</option>`).join("")}</select></label>
        <label class="runs__search">${icon("search")}<input class="input" type="search" placeholder="Search in log" aria-label="Search in log" data-role="log-search" value="${esc(state.search)}"></label>
        <span class="runs__count" data-role="log-count"></span>
        <span class="diag__spacer"></span>
        <button type="button" class="btn btn--sm" data-action="log-follow" aria-pressed="${state.follow}" data-hint="Reload the file every 3 s and keep the end in view">Follow</button>
      </div>
      <pre class="job__log runs__log" data-role="log" tabindex="0" aria-label="Log text" data-hint="The last 256 KB of the file"></pre>`;
    this.fillLog(true);
    this.syncFollow();
  }

  fillLog(jump) {
    const state = this.logState;
    const node = this.body.querySelector("[data-role=log]");
    const log = this.detail.logs.find((entry) => entry.name === state.file);
    const count = LogText.fill(node, log.text, state.search);
    const lines = plural(log.text.split("\n").length, "line");
    const counter = this.body.querySelector("[data-role=log-count]");
    counter.textContent = state.search ? plural(count, "match") : lines;
    counter.dataset.hint = `${lines}, the last 256 KB of ${log.name}`;
    if (state.follow) node.scrollTop = node.scrollHeight;
    else if (jump && state.search) {
      const mark = node.querySelector("mark");
      if (mark) mark.scrollIntoView({ block: "center" });
    }
  }

  syncFollow() {
    clearInterval(this.followTimer);
    this.followTimer = null;
    if (!this.logState.follow) return;
    this.followTimer = setInterval(async () => {
      const run = this.run;
      let detail;
      try {
        detail = await api.get(`/api/runs/${encodeURIComponent(run)}`);
      } catch (error) {
        return;
      }
      if (run !== this.run || this.section !== "logs") return;
      this.detail = detail;
      this.fillLog(false);
    }, 3000);
  }

  renderFiles() {
    const files = this.detail.files;
    const total = files.reduce((sum, file) => sum + file.size, 0);
    if (!files.length) {
      this.body.innerHTML = panelHtml("Files", emptyBlock("The run directory is empty."), { info: infoText(`Every file under runs/${this.run}.`) });
      return;
    }
    const groups = new Map();
    for (const file of files) {
      const cut = file.rel.lastIndexOf("/");
      const folder = cut < 0 ? "" : file.rel.slice(0, cut + 1);
      if (!groups.has(folder)) groups.set(folder, []);
      groups.get(folder).push({ ...file, name: file.rel.slice(cut + 1) });
    }
    const open = new Set(this.openFolders || []);
    const bodies = [...groups].sort(([a], [b]) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b))).map(([folder, entries]) => {
      const rows = entries.map((file) => `<tr><td class="mono files__name">${esc(file.name)}</td><td class="num">${esc(fmtBytes(file.size))}</td></tr>`).join("");
      if (!folder) return `<tbody>${rows}</tbody>`;
      const size = entries.reduce((sum, file) => sum + file.size, 0);
      const shown = open.has(folder);
      return `<tbody class="files__group${shown ? " is-open" : ""}" data-folder="${esc(folder)}"><tr class="files__head"><td class="mono"><button type="button" class="files__toggle" data-action="folder" aria-expanded="${shown}">${icon("chevron-right")}<span>${esc(folder)}</span></button></td><td class="num">${plural(entries.length, "file")}, ${esc(fmtBytes(size))}</td></tr>${rows}</tbody>`;
    }).join("");
    this.body.innerHTML = panelHtml("Files", `<div class="dtable dtable--dense dtable--static files"><table><thead><tr><th>Path</th><th class="num">Size</th></tr></thead>${bodies}</table></div>`, { note: `${plural(files.length, "file")}, ${fmtBytes(total)}`, info: infoText(`Every file under runs/${this.run}. A folder row opens its files.`), flush: true, className: "files__panel" });
  }

  renderGallery() {
    const images = this.detail.analysis;
    this.body.innerHTML = this.panel("Gallery", "Images written to the analysis folder of the run.", images.length ? `<div class="runs__gallery">${images.map((image) => `<button type="button" class="runs__thumb" data-action="lightbox" data-src="${esc(image.url)}" data-name="${esc(image.name)}"><img src="${esc(image.url)}" alt="${esc(image.name)}" loading="lazy"><span>${esc(image.name)}</span></button>`).join("")}</div>` : emptyBlock("This run has no analysis images."));
  }

  onClick(event) {
    const record = event.target.closest('a[href="#/episode"]');
    if (record) this.app.setContext(this.run, this.detail.reports.length ? this.detail.reports[0] : null);
    const button = event.target.closest("button[data-action]");
    if (!button) return;
    if (button.dataset.action === "actions") {
      const menu = Popover.toggle(button, Popover.menu(this.actionItems), { label: "Run actions" });
      if (menu) menu.addEventListener("click", (pick) => {
        if (pick.target.closest('a[href="#/episode"]')) this.app.setContext(this.run, this.detail.reports.length ? this.detail.reports[0] : null);
        Popover.close();
      });
      return;
    }
    const action = button.dataset.action;
    if (action === "page-prev" || action === "page-next") {
      this.curveState.page += action === "page-next" ? 1 : -1;
      this.drawCurves();
      this.root.scrollTop = 0;
    } else if (action === "pin") {
      const pins = this.pins;
      const key = button.dataset.key;
      Store.write("runs.pins", pins.includes(key) ? pins.filter((pin) => pin !== key) : [...pins, key]);
      this.drawCurves();
    } else if (action === "lightbox") {
      this.app.openLightbox(button.dataset.src, button.dataset.name);
    } else if (action === "folder") {
      const group = button.closest("[data-folder]");
      const shown = group.classList.toggle("is-open");
      button.setAttribute("aria-expanded", String(shown));
      const open = new Set(this.openFolders || []);
      if (shown) open.add(group.dataset.folder);
      else open.delete(group.dataset.folder);
      this.openFolders = [...open];
    } else if (action === "log-follow") {
      this.logState.follow = !this.logState.follow;
      button.setAttribute("aria-pressed", String(this.logState.follow));
      this.fillLog(false);
      this.syncFollow();
    }
  }

  onInput(event) {
    const input = event.target;
    if (!(input instanceof HTMLElement) || !input.dataset.role) return;
    const role = input.dataset.role;
    if (role === "smoothing" && event.type === "input") {
      this.smoothing = Number(input.value);
      Store.write("runs.smoothing", this.smoothing);
      input.parentElement.querySelector("output").textContent = fmtFixed(this.smoothing, 2);
      for (const chart of this.charts) chart.setOptions({ smoothing: this.smoothing });
    } else if (role === "curve-search" && event.type === "input") {
      this.curveState.search = input.value.trim();
      this.curveState.page = 0;
      this.drawCurves();
    } else if (role === "curve-overlay" && event.type === "change") {
      this.curveState.overlay = input.value;
      this.drawCurves();
    } else if (role === "log-file" && event.type === "change") {
      this.logState.file = input.value;
      this.fillLog(true);
    } else if (role === "log-search" && event.type === "input") {
      this.logState.search = input.value.trim();
      this.fillLog(true);
    }
  }
}
