import { api } from "../api.js";
import { esc, fmtAge, fmtSigned, isNum, NONE, Store } from "../format.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead } from "../ui/panel.js";
import { errorBlock, failView, skeleton, statusChip } from "../ui/status.js";
import { FigureStates } from "../ui/figure_states.js";
import { AgentPicker } from "./agent_picker.js";
import { Agents, Sources } from "./encoding.js";

export class StartPage {
  static SOURCES = [
    { value: "seed", label: "Random seed", hint: "The environment sampler draws jobs, vehicles and disruptions from this seed." },
    { value: "evaluation", label: "Evaluation report", hint: "Replays one episode of a stored report with its environment, reward weights and caps, so totals match the report." },
  ];

  static LANES = [
    { agent: "model", label: "Model" },
    { agent: "teacher", label: "Teacher" },
    { agent: "insertion_only", label: "Insertion only" },
    { agent: "do_nothing", label: "Do nothing" },
  ];

  constructor(app, container) {
    this.app = app;
    this.container = container;
    this.kind = Store.read("episode.source", "seed");
    this.runs = [];
    this.picker = null;
    this.recent = [];
    const extra = `<div class="start__form" data-role="form"></div>`;
    container.innerHTML = `<div class="view__pad start">${PageHead.html({ title: "Episode", lead: "Record one episode with every agent on the same start.", extra, form: "hero", figure: HairlineFigure.html("recording") })}<div class="start__body" data-role="body"></div></div>`;
    this.form = container.querySelector("[data-role=form]");
    this.body = container.querySelector("[data-role=body]");
    this.figure = container.querySelector("hairline-figure[name=recording]");
    this.body.addEventListener("pointerover", (event) => this.followRow(event.target));
    this.body.addEventListener("focusin", (event) => this.followRow(event.target));
    this.body.addEventListener("pointerleave", () => this.followRow(null));
    this.body.addEventListener("click", (event) => this.onRow(event));
    this.form.addEventListener("click", (event) => this.onClick(event));
    this.form.addEventListener("change", (event) => this.onChange(event));
    this.form.addEventListener("input", (event) => {
      if (event.target.dataset.field) this.syncRecord();
    });
    app.onHealth(() => this.syncRecord());
  }

  followRow(target) {
    const row = target instanceof Element ? target.closest("tr[data-recent]") : null;
    const episode = row ? this.recent[Number(row.dataset.recent)] : this.recent[0];
    FigureStates.write(this.figure, FigureStates.recordingFromEpisode(episode || null));
  }

  async show() {
    this.form.innerHTML = "";
    this.body.innerHTML = skeleton(5);
    let checkpoints;
    let runs;
    let episodes;
    try {
      [checkpoints, runs, episodes] = await Promise.all([api.get("/api/checkpoints"), api.get("/api/runs"), api.get("/api/episodes")]);
    } catch (error) {
      this.body.innerHTML = failView("The episode start page could not be loaded.", error.message, error);
      return;
    }
    this.checkpoints = checkpoints.checkpoints;
    this.runs = runs.runs.filter((run) => run.reports.length);
    this.render(episodes.episodes.slice(0, 10));
  }

  readSaved(useContext) {
    const context = this.app.context;
    const saved = Store.read("episode.form", { seed: 1, rounds: 5, edits: 8, run: null, report: null, index: 0 });
    if (useContext && context.run && this.runs.some((run) => run.name === context.run)) saved.run = context.run;
    if (!this.runs.some((run) => run.name === saved.run)) saved.run = this.runs.length ? this.runs[0].name : null;
    const run = this.runs.find((entry) => entry.name === saved.run);
    const reports = run ? run.reports : [];
    if (!reports.some((report) => report.seed === saved.report)) saved.report = reports.length ? reports[0].seed : null;
    return saved;
  }

  numberField(name, label, value, ch, title) {
    return `<label class="ctl"><span class="ctl__label">${esc(label)}</span><input class="input input--num" data-field="${name}" inputmode="numeric" style="--ch:${ch}" value="${esc(value)}"${title ? ` title="${esc(title)}"` : ""}><span class="field__error" role="alert" data-error="${name}"></span></label>`;
  }

  fieldsHtml() {
    const saved = this.values;
    if (this.kind === "seed") return `${this.numberField("seed", "Seed", saved.seed, 10)}${this.numberField("rounds", "Rounds", saved.rounds, 3)}${this.numberField("edits", "Edits", saved.edits, 3)}`;
    if (!this.runs.length) return `<span class="start__none">No run has an evaluation report yet. <a href="#/launch/evaluate">Evaluate a run</a></span>`;
    const run = this.runs.find((entry) => entry.name === saved.run);
    const reports = run ? run.reports : [];
    const report = reports.find((entry) => entry.seed === saved.report);
    const runSelect = `<select class="select" data-field="run">${this.runs.map((entry) => `<option value="${esc(entry.name)}"${entry.name === saved.run ? " selected" : ""}>${esc(entry.name)}</option>`).join("")}</select>`;
    const reportSelect = `<select class="select" data-field="report">${reports.map((entry) => `<option value="${entry.seed}"${entry.seed === saved.report ? " selected" : ""}>${entry.seed}</option>`).join("")}</select>`;
    return `${PageHead.control("Run", runSelect)}${PageHead.control("Report seed", reportSelect)}${this.numberField("index", "Episode", saved.index, 4, report ? `0 to ${report.episodes - 1}` : "")}`;
  }

  sourceHtml() {
    const items = StartPage.SOURCES.map((source) => `<button type="button" role="radio" data-source="${source.value}" aria-checked="${this.kind === source.value}" data-hint-title="${esc(source.label)}" data-hint="${esc(source.hint)}">${esc(source.label)}</button>`).join("");
    return `<span class="ctl" role="group" aria-label="Source"><span class="ctl__label">Source</span><span class="segmented" role="radiogroup" aria-label="Episode source">${items}</span></span>`;
  }

  renderForm() {
    this.form.innerHTML = `
      <div class="ctls">${this.sourceHtml()}<a class="btn btn--sm" href="#/lab" data-hint-title="Scenario built by hand" data-hint="Place jobs and vehicles on the map or start from a template with a declared expectation, then run the agents on it.">Open the Lab</a></div>
      <div class="ctls" data-role="fields">${this.fieldsHtml()}</div>
      <div class="ctls"><span class="ctl start__agents" role="group" aria-label="Agents"><span class="ctl__label">Agents</span><span data-role="picker"></span></span></div>
      <div class="start__record">
        <button type="button" class="btn btn--primary" data-action="record" data-hint-title="Record" data-hint="Every selected agent gets its own lane on the same episode. Cheap agents take about a second each; the first model lane loads the checkpoint, about 5 s.">Record</button>
        <span data-role="record-note" class="start__note"></span>
      </div>
      <div data-role="record-error"></div>`;
    this.picker = new AgentPicker(this.form.querySelector("[data-role=picker]"), { checkpoints: this.checkpoints, preferredRun: this.app.context.run, storeKey: "episode.agents" });
    this.syncRecord();
  }

  laneCell(episode, agent) {
    const keys = Object.keys(episode.agents).filter((key) => Agents.style(key) === agent);
    if (!keys.length) return `<td class="num faint">${NONE}</td>`;
    const entry = episode.agents[keys[0]];
    const value = entry.status === "finished" && isNum(entry.total_reward) ? fmtSigned(entry.total_reward, 2) : statusChip(entry.status);
    const others = keys.length > 1 ? ` title="${esc(keys.map((key) => `${Agents.label(key)}: ${isNum(episode.agents[key].total_reward) ? fmtSigned(episode.agents[key].total_reward, 2) : episode.agents[key].status}`).join("; "))}"` : "";
    return `<td class="num"${others}>${value}${keys.length > 1 ? `<small class="start__more">+${keys.length - 1}</small>` : ""}</td>`;
  }

  runsOf(episode) {
    const runs = new Set();
    if (episode.source.kind === "evaluation") runs.add(episode.source.run);
    for (const key of Object.keys(episode.agents)) {
      const parsed = Agents.parse(key);
      if (parsed.agent === "model" && parsed.run) runs.add(parsed.run);
    }
    return [...runs].join(", ") || NONE;
  }

  href(episode) {
    const keys = Object.keys(episode.agents).sort((a, b) => Agents.order(a) - Agents.order(b));
    const first = keys.find((key) => episode.agents[key].status === "finished") || keys[0];
    return `#/episode/ep/${episode.episode_id}/${encodeURIComponent(first)}/0`;
  }

  recentHtml(recent) {
    const lanes = StartPage.LANES.map((lane) => `<th class="num"><span class="lanekey lanekey--${lane.agent}"><i></i>${esc(lane.label)}</span></th>`).join("");
    const rows = recent.length ? recent.map((episode, index) => `
      <tr data-recent="${index}" data-href="${esc(this.href(episode))}">
        <td class="nowrap mono"><a href="${esc(this.href(episode))}">${esc(fmtAge(episode.created))}</a></td>
        <td class="mono" title="${esc(Sources.text(episode.source))}">${esc(Sources.short(episode.source))}</td>
        <td>${esc(this.runsOf(episode))}</td>
        ${StartPage.LANES.map((lane) => this.laneCell(episode, lane.agent)).join("")}
      </tr>`).join("") : `<tr class="dtable__empty"><td colspan="7">No recordings yet.</td></tr>`;
    return `
      <section class="panel panel--flush">
        ${panelHead("Recent recordings", { note: "last ten, return in pts", info: infoText("The last ten recordings, stored under logs/webui/episodes. Each lane column holds the return of that agent; a lane that did not finish shows its state.") })}
        <div class="panel__body"><div class="dtable dtable--static start__recent"><table><thead><tr><th>Recorded</th><th>Source</th><th>Run</th>${lanes}</tr></thead><tbody>${rows}</tbody></table></div></div>
      </section>`;
  }

  render(recent) {
    this.recent = recent;
    this.values = this.readSaved(true);
    this.renderForm();
    this.body.innerHTML = this.recentHtml(recent);
    this.followRow(null);
  }

  onRow(event) {
    if (!(event.target instanceof Element) || event.target.closest("a, button")) return;
    const row = event.target.closest("tr[data-href]");
    if (row) this.app.router.go(row.dataset.href);
  }

  syncRecord() {
    const button = this.form.querySelector("[data-action=record]");
    if (!button) return;
    const note = this.form.querySelector("[data-role=record-note]");
    const up = this.app.servicesUp;
    const needsReport = this.kind === "evaluation" && !this.values.report;
    const invalid = this.validate();
    button.disabled = !up || needsReport || invalid.length > 0;
    if (!this.app.health) note.textContent = "checking OSRM and VROOM";
    else if (!up) note.innerHTML = statusChip("DOWN", "routing services");
    else if (needsReport) note.textContent = "needs a run with an evaluation report";
    else if (invalid.length) note.innerHTML = `${statusChip("FAIL")} ${esc(invalid.join(" "))}`;
    else note.textContent = "";
  }

  limits() {
    const run = this.runs.find((entry) => entry.name === this.values.run);
    const report = run ? run.reports.find((entry) => entry.seed === this.values.report) : null;
    return {
      seed: { label: "Seed", min: 0, max: 4294967295, kind: "seed" },
      rounds: { label: "Rounds", min: 1, max: 10, kind: "seed" },
      edits: { label: "Edits per round", min: 1, max: 8, kind: "seed" },
      index: { label: "Episode index", min: 0, max: report ? report.episodes - 1 : 0, kind: "evaluation" },
    };
  }

  parse(name, limit) {
    const node = this.form.querySelector(`[data-field="${name}"]`);
    if (!node) return { node: null, value: this.values[name], problem: null, short: null };
    const text = node.value.trim();
    if (!/^\d+$/.test(text)) return { node, value: null, problem: `${limit.label} must be a whole number from ${limit.min} to ${limit.max}.`, short: "whole number" };
    const value = Number(text);
    if (value < limit.min || value > limit.max) return { node, value: null, problem: `${limit.label} must be from ${limit.min} to ${limit.max}.`, short: `${limit.min} to ${limit.max}` };
    return { node, value, problem: null, short: null };
  }

  validate() {
    const problems = [];
    for (const [name, limit] of Object.entries(this.limits())) {
      const { node, problem, short } = this.parse(name, limit);
      if (!node) continue;
      node.setAttribute("aria-invalid", String(Boolean(problem)));
      this.form.querySelector(`[data-error="${name}"]`).textContent = short || "";
      if (problem && limit.kind === this.kind) problems.push(problem);
    }
    return problems;
  }

  readForm() {
    const select = (name) => {
      const node = this.form.querySelector(`[data-field="${name}"]`);
      return node ? node.value : null;
    };
    const values = { ...this.values, run: select("run") || this.values.run, report: select("report") ? Number(select("report")) : this.values.report };
    this.values = values;
    for (const [name, limit] of Object.entries(this.limits())) {
      const { value } = this.parse(name, limit);
      if (value !== null) values[name] = value;
    }
    Store.write("episode.form", this.values);
  }

  setSource(kind) {
    if (kind === this.kind) return;
    this.readForm();
    this.kind = kind;
    Store.write("episode.source", kind);
    for (const button of this.form.querySelectorAll("button[data-source]")) button.setAttribute("aria-checked", String(button.dataset.source === kind));
    this.form.querySelector("[data-role=fields]").innerHTML = this.fieldsHtml();
    this.syncRecord();
  }

  onChange(event) {
    const target = event.target;
    if (!target.dataset.field) return;
    this.readForm();
    if (target.dataset.field === "run") {
      this.values = this.readSaved();
      this.form.querySelector("[data-role=fields]").innerHTML = this.fieldsHtml();
    }
    this.syncRecord();
  }

  onClick(event) {
    const source = event.target.closest("button[data-source]");
    if (source) {
      this.setSource(source.dataset.source);
      return;
    }
    const button = event.target.closest("button[data-action=record]");
    if (button && !button.disabled) this.record();
  }

  async record() {
    this.readForm();
    const errorNode = this.form.querySelector("[data-role=record-error]");
    errorNode.innerHTML = "";
    const invalid = this.validate();
    if (invalid.length) {
      errorNode.innerHTML = errorBlock("The recording was not started.", invalid.join(" "));
      return;
    }
    const agents = this.picker.agents();
    if (!agents.length) {
      errorNode.innerHTML = errorBlock("Select at least one agent.");
      return;
    }
    const body = this.kind === "seed"
      ? { source: { kind: "seed", seed: this.values.seed }, caps: { rounds: this.values.rounds, edits: this.values.edits }, agents }
      : { source: { kind: "evaluation", run: this.values.run, report_seed: this.values.report, index: this.values.index }, agents };
    let created;
    try {
      created = await api.post("/api/episodes", body);
    } catch (error) {
      errorNode.innerHTML = errorBlock("The recording was not started.", error.message);
      return;
    }
    this.app.router.go(`#/episode/ep/${created.episode_id}/${encodeURIComponent(created.agent_keys[0])}/0`);
  }
}
