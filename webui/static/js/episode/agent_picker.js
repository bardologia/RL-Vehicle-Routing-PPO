import { el, esc, Store } from "../format.js";

export class AgentPicker {
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
    container.addEventListener("change", () => this.read());
    this.render();
  }

  render() {
    const state = this.state;
    const options = this.checkpoints.map((entry) => `<option value="${esc(entry.run)}"${entry.loadable ? "" : " disabled"}${entry.run === state.run ? " selected" : ""}>${esc(entry.run)}${entry.loadable ? "" : " (architecture mismatch)"}</option>`).join("");
    const noModel = !this.checkpoints.some((entry) => entry.loadable);
    this.container.innerHTML = "";
    this.container.appendChild(el(`
      <div class="picker">
        <div class="picker__row">
          <label class="check${noModel ? " is-disabled" : ""}"><input type="checkbox" data-agent="model"${state.model ? " checked" : ""}${noModel ? " disabled" : ""}><span>model</span></label>
          <select class="select" data-role="run" aria-label="Checkpoint"${noModel ? " disabled" : ""}>${options}</select>
          <select class="select" data-role="mode" aria-label="Action selection"><option value="greedy"${state.sampled ? "" : " selected"}>greedy</option><option value="sampled"${state.sampled ? " selected" : ""}>sampled</option></select>
        </div>
        ${noModel ? `<p class="muted">No loadable checkpoint. <a href="#/launch/pretrain">Launch pretrain</a> to create one.</p>` : ""}
        <div class="picker__row"><label class="check"><input type="checkbox" data-agent="insertion_only"${state.insertion_only ? " checked" : ""}><span>insertion only</span></label><span class="muted">inserts while an insertion pays off</span></div>
        <div class="picker__row"><label class="check"><input type="checkbox" data-agent="do_nothing"${state.do_nothing ? " checked" : ""}><span>do nothing</span></label><span class="muted">commits every round untouched</span></div>
        <div class="picker__row"><label class="check"><input type="checkbox" data-agent="teacher"${state.teacher ? " checked" : ""}><span>teacher</span></label><span class="muted">slow: about 30 to 150 s</span></div>
      </div>`));
  }

  read() {
    const box = (name) => this.container.querySelector(`[data-agent="${name}"]`).checked;
    this.state = {
      model: box("model"),
      run: this.container.querySelector("[data-role=run]").value || null,
      sampled: this.container.querySelector("[data-role=mode]").value === "sampled",
      insertion_only: box("insertion_only"),
      do_nothing: box("do_nothing"),
      teacher: box("teacher"),
    };
    Store.write(this.storeKey, this.state);
  }

  agents() {
    this.read();
    const out = [];
    if (this.state.model && this.state.run) out.push({ agent: "model", run_name: this.state.run, greedy: !this.state.sampled });
    for (const name of ["teacher", "insertion_only", "do_nothing"]) {
      if (this.state[name]) out.push({ agent: name, run_name: null, greedy: true });
    }
    return out;
  }
}
