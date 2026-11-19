import { api } from "../api.js";
import { esc, fmtAge, fmtSigned, isNum, Store } from "../format.js";
import { Figure } from "../fx/figure.js";
import { PageHead } from "../ui/page_head.js";
import { infoText, panelHead } from "../ui/panel.js";
import { errorBlock, skeleton, statusChip } from "../ui/status.js";
import { AgentPicker } from "./agent_picker.js";
import { Agents } from "./encoding.js";

export class StartPage {
  constructor(app, container) {
    this.app = app;
    this.container = container;
    this.kind = Store.read("episode.source", "seed");
    this.runs = [];
    this.picker = null;
    container.addEventListener("click", (event) => this.onClick(event));
    container.addEventListener("change", (event) => this.onChange(event));
    container.addEventListener("input", (event) => {
      if (event.target.dataset.field) this.syncRecord();
    });
    app.onHealth(() => this.syncRecord());
  }

  static sourceText(source) {
    if (source.kind === "seed") return `Seed ${source.seed}`;
    if (source.kind === "evaluation") return `Evaluation episode ${source.index} of ${source.run}, report seed ${source.report_seed}`;
    if (source.template) return `Scenario from template ${source.template}`;
    if (source.name) return `Scenario ${source.name}`;
    return "Scenario from the Lab";
  }

  async show() {
    this.container.innerHTML = `<div class="view__pad">${skeleton(5)}</div>`;
    let checkpoints;
    let runs;
    let episodes;
    try {
      [checkpoints, runs, episodes] = await Promise.all([api.get("/api/checkpoints"), api.get("/api/runs"), api.get("/api/episodes")]);
    } catch (error) {
      this.container.innerHTML = `<div class="view__pad">${errorBlock("The episode start page could not be loaded.", error.message)}</div>`;
      return;
    }
    this.checkpoints = checkpoints.checkpoints;
    this.runs = runs.runs.filter((run) => run.reports.length);
    this.render(episodes.episodes.slice(0, 10));
  }

  render(recent) {
    const context = this.app.context;
    const saved = Store.read("episode.form", { seed: 1, rounds: 5, edits: 8, run: null, report: null, index: 0 });
    if (context.run && this.runs.some((run) => run.name === context.run)) saved.run = context.run;
    if (!this.runs.some((run) => run.name === saved.run)) saved.run = this.runs.length ? this.runs[0].name : null;
    const run = this.runs.find((entry) => entry.name === saved.run);
    const reports = run ? run.reports : [];
    if (!reports.some((report) => report.seed === saved.report)) saved.report = reports.length ? reports[0].seed : null;
    const report = reports.find((entry) => entry.seed === saved.report);
    this.form = saved;

    const card = (kind, title, text, body) => `
      <label class="source${this.kind === kind ? " is-selected" : ""}" data-source="${kind}">
        <span class="source__head"><input type="radio" name="episode-source" value="${kind}"${this.kind === kind ? " checked" : ""}><b>${title}</b></span>
        <span class="source__text muted">${text}</span>
        <span class="source__body">${body}</span>
      </label>`;

    const seedBody = `
      <span class="fieldrow">
        <span class="fieldlabel"><span>Seed</span><input class="input input--num" data-field="seed" inputmode="numeric" value="${esc(saved.seed)}"><span class="field__error" role="alert" data-error="seed"></span></span>
        <span class="fieldlabel"><span>Rounds</span><input class="input input--num" data-field="rounds" inputmode="numeric" value="${esc(saved.rounds)}"><span class="field__error" role="alert" data-error="rounds"></span></span>
        <span class="fieldlabel"><span>Edits</span><input class="input input--num" data-field="edits" inputmode="numeric" value="${esc(saved.edits)}"><span class="field__error" role="alert" data-error="edits"></span></span>
      </span>`;

    const evaluationBody = this.runs.length ? `
      <span class="fieldrow">
        <span class="fieldlabel"><span>Run</span><select class="select" data-field="run">${this.runs.map((entry) => `<option value="${esc(entry.name)}"${entry.name === saved.run ? " selected" : ""}>${esc(entry.name)}</option>`).join("")}</select></span>
        <span class="fieldlabel"><span>Report seed</span><select class="select" data-field="report">${reports.map((entry) => `<option value="${entry.seed}"${entry.seed === saved.report ? " selected" : ""}>${entry.seed}</option>`).join("")}</select></span>
        <span class="fieldlabel"><span>Episode</span><input class="input input--num" data-field="index"${report ? ` title="0 to ${report.episodes - 1}"` : ""} inputmode="numeric" value="${esc(saved.index)}"><span class="field__error" role="alert" data-error="index"></span></span>
      </span>` : `<span class="muted">No run has an evaluation report yet. <a href="#/launch/evaluate">Evaluate a run</a> first.</span>`;

    const recentRows = recent.length ? recent.map((episode) => {
      const keys = Object.keys(episode.agents).sort((a, b) => Agents.order(a) - Agents.order(b));
      const first = keys.find((key) => episode.agents[key].status === "finished") || keys[0];
      const agents = keys.map((key) => {
        const entry = episode.agents[key];
        const tail = entry.status === "finished" && isNum(entry.total_reward) ? fmtSigned(entry.total_reward, 2) : entry.status === "failed" ? statusChip("FAIL") : esc(entry.status);
        return `<span class="recent__agent"><span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span> ${tail}</span>`;
      }).join("");
      return `<tr><td class="nowrap">${fmtAge(episode.created)}</td><td>${esc(StartPage.sourceText(episode.source))}</td><td>${agents}</td><td><a class="btn btn--sm" href="#/episode/ep/${esc(episode.episode_id)}/${encodeURIComponent(first)}/0">Open</a></td></tr>`;
    }).join("") : `<tr class="dtable__empty"><td colspan="4">No recordings yet. Pick a source above and record one.</td></tr>`;

    this.container.innerHTML = `
      <div class="view__pad start">
        ${PageHead.html({ title: "Episode", lead: "Record one episode with every agent on the same start.", form: "narrow", figure: Figure.html("patch") })}
        <div class="start__top">
          <div>
        <div class="start__sources" role="radiogroup" aria-label="Episode source">
          ${card("seed", "Random episode from a seed", "The environment sampler draws jobs, vehicles and disruptions from this seed.", seedBody)}
          ${card("evaluation", "Episode of an evaluation report", "Replays one episode of a stored report with its environment, reward weights and caps, so totals match the report.", evaluationBody)}
          <div class="source source--link">
            <span class="source__head"><b>Scenario built by hand</b></span>
            <span class="source__text muted">Place jobs and vehicles on the map or start from a template with a declared expectation, then run the agents on it.</span>
            <span class="source__body"><a class="btn" href="#/lab">Open the Lab</a></span>
          </div>
        </div>
        <section class="panel start__agents">
          ${panelHead("Agents", { info: infoText("Each agent gets its own lane on the same episode.", "The episode is replayed decision by decision with the policy distribution, the reward terms and the plan on the map.") })}
          <div class="panel__body">
            <div data-role="picker"></div>
            <div class="start__record">
              <button type="button" class="btn btn--primary" data-action="record">Record</button>
              <span data-role="record-note" class="muted"></span>
            </div>
            <div data-role="record-error"></div>
          </div>
        </section>
          </div>
        </div>
        <section class="panel panel--flush">
          ${panelHead("Recent recordings", { note: "last ten", info: infoText("The last ten recordings, stored under logs/webui/episodes.") })}
          <div class="panel__body"><div class="dtable dtable--static"><table><thead><tr><th>Recorded</th><th>Source</th><th>Agents and return</th><th></th></tr></thead><tbody>${recentRows}</tbody></table></div></div>
        </section>
      </div>`;

    this.picker = new AgentPicker(this.container.querySelector("[data-role=picker]"), { checkpoints: this.checkpoints, preferredRun: context.run, storeKey: "episode.agents" });
    this.syncRecord();
  }

  syncRecord() {
    const button = this.container.querySelector("[data-action=record]");
    if (!button) return;
    const note = this.container.querySelector("[data-role=record-note]");
    const up = this.app.servicesUp;
    const needsReport = this.kind === "evaluation" && !this.form.report;
    const invalid = this.validate();
    button.disabled = !up || needsReport || invalid.length > 0;
    if (!this.app.health) note.textContent = "Checking OSRM and VROOM.";
    else if (!up) note.innerHTML = `${statusChip("DOWN", "routing services")} Start them from the repository root: <code>sg docker -c "docker compose up -d"</code>`;
    else if (needsReport) note.textContent = "This source needs a run with an evaluation report.";
    else if (invalid.length) note.innerHTML = `${statusChip("FAIL")} ${esc(invalid.join(" "))}`;
    else note.textContent = "Cheap agents take about a second each; the first model lane loads the checkpoint (about 5 s).";
  }

  limits() {
    const run = this.runs.find((entry) => entry.name === this.form.run);
    const report = run ? run.reports.find((entry) => entry.seed === this.form.report) : null;
    return {
      seed: { label: "Seed", min: 0, max: 4294967295, kind: "seed" },
      rounds: { label: "Rounds", min: 1, max: 10, kind: "seed" },
      edits: { label: "Edits per round", min: 1, max: 8, kind: "seed" },
      index: { label: "Episode index", min: 0, max: report ? report.episodes - 1 : 0, kind: "evaluation" },
    };
  }

  parse(name, limit) {
    const node = this.container.querySelector(`[data-field="${name}"]`);
    if (!node) return { node: null, value: this.form[name], problem: null, short: null };
    const text = node.value.trim();
    if (!/^\d+$/.test(text)) return { node, value: null, problem: `${limit.label} must be a whole number from ${limit.min} to ${limit.max}.`, short: "not a whole number" };
    const value = Number(text);
    if (value < limit.min || value > limit.max) return { node, value: null, problem: `${limit.label} must be from ${limit.min} to ${limit.max}.`, short: `allowed: ${limit.min} to ${limit.max}` };
    return { node, value, problem: null, short: null };
  }

  validate() {
    const problems = [];
    for (const [name, limit] of Object.entries(this.limits())) {
      const { node, problem, short } = this.parse(name, limit);
      if (!node) continue;
      node.setAttribute("aria-invalid", String(Boolean(problem)));
      this.container.querySelector(`[data-error="${name}"]`).textContent = short || "";
      if (problem && limit.kind === this.kind) problems.push(problem);
    }
    return problems;
  }

  readForm() {
    const select = (name) => {
      const node = this.container.querySelector(`[data-field="${name}"]`);
      return node ? node.value : null;
    };
    const form = { ...this.form, run: select("run") || this.form.run, report: select("report") ? Number(select("report")) : this.form.report };
    this.form = form;
    for (const [name, limit] of Object.entries(this.limits())) {
      const { value } = this.parse(name, limit);
      if (value !== null) form[name] = value;
    }
    Store.write("episode.form", this.form);
  }

  onChange(event) {
    const target = event.target;
    if (target.name === "episode-source") {
      this.kind = target.value;
      Store.write("episode.source", this.kind);
      for (const node of this.container.querySelectorAll(".source[data-source]")) node.classList.toggle("is-selected", node.dataset.source === this.kind);
      this.syncRecord();
      return;
    }
    if (target.dataset.field) {
      this.readForm();
      if (target.dataset.field === "run") this.show();
      else this.syncRecord();
    }
  }

  onClick(event) {
    const button = event.target.closest("button[data-action=record]");
    if (button && !button.disabled) this.record();
  }

  async record() {
    this.readForm();
    const errorNode = this.container.querySelector("[data-role=record-error]");
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
      ? { source: { kind: "seed", seed: this.form.seed }, caps: { rounds: this.form.rounds, edits: this.form.edits }, agents }
      : { source: { kind: "evaluation", run: this.form.run, report_seed: this.form.report, index: this.form.index }, agents };
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
