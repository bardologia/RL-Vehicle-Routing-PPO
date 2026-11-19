import { el, esc, Store } from "../format.js";

export class AgentPicker {
  static AGENTS = [
    { key: "model", label: "model", hint: "The policy of the chosen checkpoint. The first model lane loads the checkpoint, about 5 s." },
    { key: "insertion_only", label: "insertion only", hint: "Inserts while an insertion pays off. About a second." },
    { key: "do_nothing", label: "do nothing", hint: "Commits every round untouched. About a second." },
    { key: "teacher", label: "teacher", hint: "Regret insertion with removals. Slow: about 30 to 150 s." },
  ];

  constructor(container, { checkpoints, preferredRun, storeKey, teacherDefault }) {
    this.container = container;
    this.storeKey = storeKey;
    this.checkpoints = checkpoints;
    const loadable = checkpoints.filter((entry) => entry.loadable);
    const saved = Store.read(storeKey, null) || { model: true, run: null, sampled: false, insertion_only: true, do_nothing: true, teacher: Boolean(teacherDefault) };
    if (preferredRun && loadable.some((entry) => entry.run === preferredRun)) saved.run = preferredRun;
    if (!loadable.some((entry) => entry.run === saved.run)) saved.run = loadable.length ? loadable[0].run : null;
    if (!saved.run) saved.model = false;
    this.state = saved;
    container.addEventListener("click", (event) => this.onClick(event));
    container.addEventListener("change", (event) => this.onChange(event));
    this.render();
  }

  get noModel() {
    return !this.checkpoints.some((entry) => entry.loadable);
  }

  chip(agent) {
    const disabled = agent.key === "model" && this.noModel;
    return `<button type="button" class="chip" data-agent="${agent.key}" aria-pressed="${Boolean(this.state[agent.key])}"${disabled ? " disabled" : ""} data-hint-title="${esc(agent.label)}" data-hint="${esc(agent.hint)}">${esc(agent.label)}</button>`;
  }

  modelControls() {
    const state = this.state;
    const options = this.checkpoints.map((entry) => `<option value="${esc(entry.run)}"${entry.loadable ? "" : " disabled"}${entry.run === state.run ? " selected" : ""}>${esc(entry.run)}${entry.loadable ? "" : " (architecture mismatch)"}</option>`).join("");
    const mode = (value, label) => `<button type="button" role="radio" data-mode="${value}" aria-checked="${(value === "sampled") === Boolean(state.sampled)}">${label}</button>`;
    return `
      <span class="picker__model"${state.model ? "" : " hidden"}>
        <select class="select" data-role="run" aria-label="Checkpoint">${options}</select>
        <span class="segmented" role="radiogroup" aria-label="Action selection">${mode("greedy", "greedy")}${mode("sampled", "sampled")}</span>
      </span>`;
  }

  render() {
    this.container.innerHTML = "";
    this.container.appendChild(el(`
      <div class="picker">
        <span class="chips" role="group" aria-label="Agents">${AgentPicker.AGENTS.map((agent) => this.chip(agent)).join("")}</span>
        ${this.noModel ? `<span class="picker__none">No loadable checkpoint. <a href="#/launch/pretrain">Launch pretrain</a></span>` : this.modelControls()}
      </div>`));
  }

  onClick(event) {
    const chip = event.target.closest("button[data-agent]");
    if (chip && !chip.disabled) {
      this.state[chip.dataset.agent] = !this.state[chip.dataset.agent];
      chip.setAttribute("aria-pressed", String(this.state[chip.dataset.agent]));
      if (chip.dataset.agent === "model") this.container.querySelector(".picker__model").hidden = !this.state.model;
      this.persist();
      return;
    }
    const mode = event.target.closest("button[data-mode]");
    if (mode) {
      this.state.sampled = mode.dataset.mode === "sampled";
      for (const button of this.container.querySelectorAll("button[data-mode]")) button.setAttribute("aria-checked", String(button === mode));
      this.persist();
    }
  }

  onChange(event) {
    if (event.target.dataset.role !== "run") return;
    this.state.run = event.target.value || null;
    this.persist();
  }

  persist() {
    Store.write(this.storeKey, this.state);
    this.container.dispatchEvent(new Event("picker-change", { bubbles: true }));
  }

  agents() {
    const out = [];
    if (this.state.model && this.state.run) out.push({ agent: "model", run_name: this.state.run, greedy: !this.state.sampled });
    for (const name of ["teacher", "insertion_only", "do_nothing"]) {
      if (this.state[name]) out.push({ agent: name, run_name: null, greedy: true });
    }
    return out;
  }
}
