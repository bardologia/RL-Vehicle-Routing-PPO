import { api, ApiError } from "../api.js";
import { AgentTable } from "../diagnose/agent_table.js";
import { FindingsList } from "../diagnose/findings_list.js";
import { Links } from "../diagnose/links.js";
import { el, esc, fmtAge, fmtBytes, fmtFixed, fmtPct, fmtSigned, icon, isNum, plural, Store } from "../format.js";
import { LineChart } from "../ui/charts.js";
import { kpiRow } from "../ui/kpi.js";
import { LogText } from "../ui/logtext.js";
import { MarkdownTable } from "../ui/markdown_table.js";
import { Figure } from "../fx/figure.js";
import { emptyBlock, errorBlock, plainChip, skeleton, statusChip } from "../ui/status.js";
import { kvTable } from "../ui/table.js";
import { Tabs } from "../ui/tabs.js";

const SECTIONS = [
  { value: "summary", label: "Summary" },
  { value: "health", label: "Training health" },
  { value: "curves", label: "Curves" },
  { value: "logs", label: "Logs" },
  { value: "files", label: "Files" },
  { value: "gallery", label: "Gallery" },
];

const STYLES = ["solid", "dashed", "dotted"];
const PAGE = 24;

export class RunStatus {
  static word(run) {
    if (!run.has_checkpoint) return "NO CHECKPOINT";
    return run.loadable ? "LOADABLE" : "INCOMPATIBLE";
  }

  static chip(run) {
    const word = RunStatus.word(run);
    if (word === "NO CHECKPOINT") return plainChip(word);
    return statusChip(word, run.loadable ? null : run.reason);
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

    root.appendChild(el(`
      <div class="runs">
        <aside class="runs__rail" aria-label="Runs">
          <label class="runs__search">${icon("search")}<input class="input" type="search" placeholder="Search runs" aria-label="Search runs" data-role="search"></label>
          <div class="runs__list" data-role="list"></div>
          ${Figure.html("cabinet", "runs__figure")}
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
      this.main.innerHTML = `<div class="view__pad">${errorBlock("The run list could not be loaded.", error.message)}</div>`;
      return;
    }
    if (token !== this.token) return;

    const context = this.app.context;
    const name = parts[0] || (this.runs.some((run) => run.name === context.run) ? context.run : null) || (this.runs[0] ? this.runs[0].name : null);
    const section = SECTIONS.some((entry) => entry.value === parts[1]) ? parts[1] : "summary";
    if (!name) {
      this.renderRail();
      this.main.innerHTML = `<div class="view__pad">${emptyBlock("The runs directory is empty. A pretrain run is the usual first step.", `<a class="btn btn--primary" href="#/launch/pretrain">Launch pretrain</a>`)}</div>`;
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
      this.main.innerHTML = `<div class="view__pad">${skeleton(5)}</div>`;
    }
    try {
      if (!this.runs.some((run) => run.name === name)) throw new ApiError(404, `unknown run '${name}'`);
      this.detail = await api.get(`/api/runs/${encodeURIComponent(name)}`);
    } catch (error) {
      if (token !== this.token) return;
      this.detail = null;
      this.main.innerHTML = `<div class="view__pad"><section class="panel"><header class="panel__head"><h3 class="panel__title">${error.status === 404 ? "404: unknown run" : "Run not loaded"}</h3></header><div class="panel__body">${errorBlock(error.status === 404 ? `There is no run named "${name}".` : `Run "${name}" could not be loaded.`, error.message)}<p><a class="btn" href="#/runs">Open the newest run</a></p></div></section></div>`;
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
        <span class="runs__name">${esc(run.name)}</span>
        <span class="runs__meta"><span>${esc(run.kind)}</span><span>${esc(fmtAge(run.mtime))}</span></span>
        <span class="runs__meta">${RunStatus.chip({ ...run, reason: null })}</span>
        <span class="runs__meta runs__reward">${esc(RunStatus.rewardLine(run))}</span>
      </a>`).join("") : `<p class="muted runs__none">No run matches "${esc(this.filter)}".</p>`;
  }

  renderShell(info) {
    const detail = this.detail;
    const seed = detail.reports[0];
    const episodes = info && info.reports[0] ? info.reports[0].episodes : null;
    this.main.innerHTML = `
      <div class="view__pad">
        <header class="runs__head">
          <div class="runs__title">
            <h2>${esc(detail.name)}</h2>
            <span class="muted">${esc(detail.kind)}${info ? `, ${esc(fmtAge(info.mtime))}` : ""}</span>
            ${RunStatus.chip(detail)}
          </div>
          <div class="runs__actions">
            ${seed !== undefined ? `<a class="btn" href="${esc(Links.diagnose(detail.name, seed))}">Open in Diagnose</a>` : ""}
            ${detail.loadable ? `<a class="btn" href="#/episode">Record episode</a>` : ""}
            ${detail.loadable ? `<a class="btn" href="${esc(Links.evaluate(detail.name))}">Evaluate</a>` : ""}
            ${detail.loadable ? `<a class="btn" href="${esc(Links.recordTraces(detail.name, seed === undefined ? null : seed, episodes))}">Record traces</a>` : ""}
          </div>
        </header>
        <div data-role="tabs" class="runs__tabs"></div>
        <div data-role="section" class="runs__section"></div>
      </div>`;
    new Tabs(this.main.querySelector("[data-role=tabs]"), SECTIONS.map((entry) => ({ ...entry, count: entry.value === "logs" ? detail.logs.length : entry.value === "files" ? detail.files.length : entry.value === "gallery" ? detail.analysis.length : undefined })), this.section, (value) => this.app.router.go(Links.runs(this.run, value)));
    this.body = this.main.querySelector("[data-role=section]");
  }

  async cached(key, url) {
    if (!this.cache[key]) this.cache[key] = await api.get(url);
    return this.cache[key];
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

  panel(title, help, bodyHtml, extra) {
    return `<section class="panel"${extra || ""}><header class="panel__head"><h3 class="panel__title">${esc(title)}</h3><span class="panel__help">${esc(help || "")}</span></header><div class="panel__body">${bodyHtml}</div></section>`;
  }

  async renderSummary(token) {
    const detail = this.detail;
    const base = `/api/runs/${encodeURIComponent(this.run)}`;
    const [findings, health, ...reports] = await Promise.all([
      this.cached("findings", `${base}/findings`),
      detail.has_events ? this.cached("health", `${base}/health`) : Promise.resolve(null),
      ...detail.reports.map((seed) => this.cached(`evaluation.${seed}`, `${base}/evaluation/${seed}`)),
    ]);
    if (token !== this.token) return;

    const parts = [];
    if (detail.loadable === false) parts.push(`<div class="errorblock" role="alert">${statusChip("INCOMPATIBLE")}<span class="errorblock__cause">This checkpoint does not load on the current architecture: ${esc(detail.reason)}. It cannot be evaluated or recorded.</span></div>`);

    const traceLink = detail.loadable ? `<a class="btn btn--sm" href="${esc(Links.recordTraces(this.run, detail.reports[0] === undefined ? null : detail.reports[0], null))}">Record traces</a>` : "";
    parts.push(this.panel("Findings", detail.reports.length ? `Rules over report seed ${detail.reports[0]}, its traces and the training curves.` : "Rules over the training curves. Report and trace rules need an evaluation.", FindingsList.html(findings.findings, { actions: { report: detail.loadable ? `<a class="btn btn--sm" href="${esc(Links.evaluate(this.run))}">Evaluate this run</a>` : "", traces: traceLink, verdicts: traceLink }, columns: true })));

    if (health && health.kind === "pretrain") {
      const accuracy = health.charts.find((chart) => chart.key === "accuracy");
      const tiles = accuracy ? accuracy.series.filter((series) => series.values.length).map((series) => ({ label: `${series.label} head accuracy`, value: fmtPct(series.values[series.values.length - 1], 1), note: `last of ${plural(series.values.length, "epoch")}`, title: series.tag })) : [];
      parts.push(this.panel("Behaviour cloning accuracy", "Last logged agreement with the teacher on held-out pairs, per head.", tiles.length ? kpiRow(tiles) : `<p class="muted">pretrain/accuracy is not logged in this run.</p>`, ' data-role="accuracy"'));
    }

    reports.forEach((report, index) => {
      parts.push(`<section class="panel"><header class="panel__head"><h3 class="panel__title">Evaluation, seed ${report.meta.seed}</h3><span class="panel__help">${plural(report.meta.episodes, "episode")}, ${report.meta.greedy ? "greedy" : "sampled"} policy.</span><span class="panel__actions"><a class="btn btn--sm" href="${esc(Links.diagnose(this.run, report.meta.seed))}">Open in Diagnose</a></span></header><div class="panel__body"><div data-role="agents" data-report="${index}"></div>
        <details class="disclosure"><summary>Report settings: reward weights, environment, caps</summary><div class="runs__meta3">
          <div><h4 class="subhead">Reward</h4>${kvTable(Object.entries(report.meta.reward).map(([key, value]) => [key, esc(JSON.stringify(value))]))}</div>
          <div><h4 class="subhead">Caps and run</h4>${kvTable([...Object.entries(report.meta.caps), ...Object.entries(report.meta).filter(([, value]) => typeof value !== "object")].map(([key, value]) => [key, esc(String(value))]))}</div>
          <div><h4 class="subhead">Environment</h4>${kvTable(Object.entries(report.meta.environment).map(([key, value]) => [key, esc(JSON.stringify(value))]))}</div>
        </div></details></div></section>`);
    });
    if (!reports.length) parts.push(this.panel("Evaluation", "", emptyBlock(detail.loadable ? "This run has no evaluation report. The teacher dominates the time: about 30 s per episode at 3 rounds, about 150 s at 5." : "This run has no evaluation report.", detail.loadable ? `<a class="btn btn--primary" href="${esc(Links.evaluate(this.run))}">Evaluate this run</a>` : "")));

    const docs = Object.entries(detail.markdown);
    if (docs.length) parts.push(this.panel("Model documents", "Written by the training script.", docs.map(([file, text]) => `<details class="disclosure" data-doc="${esc(file)}"><summary>${esc(file)}</summary>${MarkdownTable.render(text)}</details>`).join("")));

    this.body.innerHTML = parts.join("");
    this.reports = reports;
    this.drawAgentTables();
  }

  drawAgentTables() {
    if (this.section !== "summary" || !this.reports) return;
    this.reports.forEach((report, index) => {
      const node = this.body.querySelector(`[data-report="${index}"]`);
      if (node) AgentTable.render(node, report);
    });
  }

  smoothingControl() {
    return `<label class="fieldlabel runs__slider"><span>Smoothing (EMA)</span><input type="range" min="0" max="0.95" step="0.05" value="${this.smoothing}" data-role="smoothing" aria-label="Smoothing"><output>${fmtFixed(this.smoothing, 2)}</output></label>`;
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

  async renderHealth(token) {
    if (!this.detail.has_events) {
      this.body.innerHTML = this.panel("Training health", "", emptyBlock("No TensorBoard events in this run."));
      return;
    }
    const health = await this.cached("health", `/api/runs/${encodeURIComponent(this.run)}/health`);
    if (token !== this.token) return;
    const blocks = [];
    health.charts.forEach((chart, index) => {
      const series = RunsView.dedupe(chart.series);
      const split = series.length > 3;
      const sparse = chart.series.length > 0 && chart.series.every((entry) => entry.values.length <= 1);
      const frames = sparse ? kpiRow(chart.series.filter((entry) => entry.values.length).map((entry) => ({ label: entry.label, value: fmtFixed(entry.values[0], 3), unit: chart.unit, note: `one value, step ${entry.steps[0]}`, title: entry.tag }))) : split ? series.map((entry, s) => `<div class="runs__multi"><h4 class="subhead">${esc(entry.label)}</h4><div data-chart="${index}" data-series="${s}"></div></div>`).join("") : `<div data-chart="${index}"></div>`;
      const points = chart.series.length ? Math.max(...chart.series.map((entry) => entry.steps.length)) : 0;
      const merged = series.filter((entry) => entry.merged.length > 1).map((entry) => `${entry.merged.join(", ")} are identical`).join("; ");
      blocks.push(`<section class="panel" data-health="${esc(chart.key)}"><header class="panel__head"><h3 class="panel__title">${esc(chart.title)}</h3><span class="panel__help">${points ? `n = ${plural(points, "point")}` : ""}${merged && (sparse || split) ? `. ${esc(merged)}` : ""}</span></header><div class="panel__body">${frames}</div></section>`);
    });
    this.body.innerHTML = `
      <div class="runs__controls"><span class="muted">${health.kind === "ppo" ? "PPO training" : "Behaviour cloning"}: fixed charts. Charts on the same step axis share the crosshair.</span><span class="diag__spacer"></span>${this.smoothingControl()}</div>
      <div class="runs__charts">${blocks.join("")}</div>`;

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
        <label class="check"><input type="checkbox" data-role="curve-log"${this.curveState.log ? " checked" : ""}><span>Log scale</span></label>
        ${this.smoothingControl()}
        <label class="fieldlabel"><span>Overlay run</span><select class="select" data-role="curve-overlay"><option value="">none</option>${others.map((run) => `<option value="${esc(run.name)}"${run.name === this.curveState.overlay ? " selected" : ""}>${esc(run.name)}</option>`).join("")}</select></label>
        <span class="diag__spacer"></span>
        <span class="runs__pager"><button type="button" class="btn btn--sm" data-action="page-prev">Previous</button><span data-role="page-label" class="muted"></span><button type="button" class="btn btn--sm" data-action="page-next">Next</button></span>
      </div>
      <div data-role="curve-error"></div>
      <div data-role="curve-grid"></div>`;
    await this.drawCurves();
  }

  get pins() {
    return Store.read("runs.pins", []);
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
    const ordered = [...matched.filter((entry) => pins.includes(entry.key)), ...matched.filter((entry) => !pins.includes(entry.key))];
    const pages = Math.max(1, Math.ceil(ordered.length / PAGE));
    state.page = Math.max(0, Math.min(pages - 1, state.page));
    const shown = ordered.slice(state.page * PAGE, (state.page + 1) * PAGE);

    this.body.querySelector("[data-role=page-label]").textContent = `page ${state.page + 1} of ${pages}, ${plural(ordered.length, "chart")} from ${plural(Object.keys(tags).length, "tag")}`;
    this.body.querySelector("[data-action=page-prev]").disabled = state.page === 0;
    this.body.querySelector("[data-action=page-next]").disabled = state.page >= pages - 1;

    const grid = this.body.querySelector("[data-role=curve-grid]");
    this.charts = [];
    if (!shown.length) {
      grid.innerHTML = `<p class="muted">No tag contains "${esc(state.search)}".</p>`;
      return;
    }
    let lastGroup = null;
    const html = [];
    shown.forEach((entry, index) => {
      const pinned = pins.includes(entry.key);
      const group = pinned ? "Pinned" : CurveSet.group(entry.key);
      if (group !== lastGroup) {
        if (lastGroup !== null) html.push("</div>");
        html.push(`<h3 class="diag__section">${esc(group)}</h3><div class="runs__charts">`);
        lastGroup = group;
      }
      html.push(`<section class="panel" data-curve="${esc(entry.key)}"><header class="panel__head"><h3 class="panel__title runs__tag" title="${esc(entry.title)}">${esc(entry.title)}</h3><span class="panel__help"></span><span class="panel__actions"><button type="button" class="btn btn--sm" data-action="pin" data-key="${esc(entry.key)}" aria-pressed="${pinned}">${pinned ? "Pinned" : "Pin"}</button></span></header><div class="panel__body"><div data-chart="${index}"></div></div></section>`);
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
      this.mountChart(grid.querySelector(`[data-chart="${index}"]`), series, { height: 180, log: state.log, xLabel: "step", yLabel: "logged value", domain: CurveSet.group(entry.key) });
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
        <label class="fieldlabel"><span>File</span><select class="select" data-role="log-file">${logs.map((log) => `<option${log.name === state.file ? " selected" : ""}>${esc(log.name)}</option>`).join("")}</select></label>
        <label class="runs__search">${icon("search")}<input class="input" type="search" placeholder="Search in log" aria-label="Search in log" data-role="log-search" value="${esc(state.search)}"></label>
        <span class="muted" data-role="log-count"></span>
        <span class="diag__spacer"></span>
        <label class="check"><input type="checkbox" data-role="log-follow"${state.follow ? " checked" : ""}><span>Follow</span></label>
        <span class="muted">last 256 KB</span>
      </div>
      <pre class="job__log runs__log" data-role="log" tabindex="0" aria-label="Log text"></pre>`;
    this.fillLog(true);
    this.syncFollow();
  }

  fillLog(jump) {
    const state = this.logState;
    const node = this.body.querySelector("[data-role=log]");
    const log = this.detail.logs.find((entry) => entry.name === state.file);
    const count = LogText.fill(node, log.text, state.search);
    this.body.querySelector("[data-role=log-count]").textContent = state.search ? plural(count, "match") : `${plural(log.text.split("\n").length, "line")}`;
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
    const rows = files.map((file) => `<tr><td><code>${esc(file.rel)}</code></td><td class="num">${esc(fmtBytes(file.size))}</td></tr>`).join("");
    this.body.innerHTML = this.panel("Files", `${plural(files.length, "file")}, ${fmtBytes(total)} in runs/${this.run}`, files.length ? `<div class="dtable dtable--dense dtable--static"><table><thead><tr><th>Path</th><th class="num">Size</th></tr></thead><tbody>${rows}</tbody></table></div>` : emptyBlock("The run directory is empty."));
  }

  renderGallery() {
    const images = this.detail.analysis;
    this.body.innerHTML = this.panel("Gallery", "Images written to the analysis folder of the run.", images.length ? `<div class="runs__gallery">${images.map((image) => `<button type="button" class="runs__thumb" data-action="lightbox" data-src="${esc(image.url)}" data-name="${esc(image.name)}"><img src="${esc(image.url)}" alt="${esc(image.name)}" loading="lazy"><span>${esc(image.name)}</span></button>`).join("")}</div>` : `<div class="emptyblock">${Figure.html("drawer")}<p>This run has no analysis images.</p></div>`);
  }

  onClick(event) {
    const record = event.target.closest('a[href="#/episode"]');
    if (record) this.app.setContext(this.run, this.detail.reports.length ? this.detail.reports[0] : null);
    const button = event.target.closest("button[data-action]");
    if (!button) return;
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
    } else if (role === "curve-log" && event.type === "change") {
      this.curveState.log = input.checked;
      for (const chart of this.charts) chart.setOptions({ log: this.curveState.log });
    } else if (role === "curve-overlay" && event.type === "change") {
      this.curveState.overlay = input.value;
      this.drawCurves();
    } else if (role === "log-file" && event.type === "change") {
      this.logState.file = input.value;
      this.fillLog(true);
    } else if (role === "log-search" && event.type === "input") {
      this.logState.search = input.value.trim();
      this.fillLog(true);
    } else if (role === "log-follow" && event.type === "change") {
      this.logState.follow = input.checked;
      this.fillLog(false);
      this.syncFollow();
    }
  }
}
