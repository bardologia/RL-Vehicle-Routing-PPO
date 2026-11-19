import { api } from "../api.js";
import { esc, icon } from "../format.js";
import { FigureStates } from "../ui/figure_states.js";
import { failView, skeleton } from "../ui/status.js";
import { ConfigForm } from "./config_form.js";
import { RunPicker } from "./form_widgets.js";
import { ConfigTemplates, RunConfigCopy } from "./rail.js";

export class LaunchPage extends ConfigForm {
  static FOLLOW = { pretrain: "train", train: "evaluate", evaluate: "diagnose" };

  constructor(view, host, project) {
    super();
    this.view = view;
    this.app = view.app;
    this.host = host;
    this.project = project;
    this.key = null;
    this.detail = null;
    this.detach = true;
    this.pendingImport = null;
    this.pendingQuery = null;
    this.loadSeq = 0;
    this.fired = 0;
    this.launching = false;
    this.copyControl = new RunConfigCopy(this, project.roots.runs);
    this.templateControl = new ConfigTemplates(this);
  }

  async show(key, section, query) {
    const queried = Object.keys(query || {}).length > 0;
    if (queried) this.pendingQuery = { ...query };

    if (key === this.key && this.config) {
      if (section) this.setActiveSection(section);
      this.applyPending();
      return;
    }

    const seq = ++this.loadSeq;
    this.key = key;
    this.detail = null;
    this.resetForm();
    this.detach = true;
    this.activeSection = section || null;
    RunPicker.forget();
    this.renderSkeleton();

    let detail;
    try {
      detail = await api.get(`/api/scripts/${encodeURIComponent(key)}`);
    } catch (error) {
      if (seq !== this.loadSeq) return;
      this.host.innerHTML = failView(`The entry point "${key}" could not be loaded.`, error.message, error);
      return;
    }
    if (seq !== this.loadSeq) return;
    this.detail = detail;
    this.renderShell(detail);

    let cfg;
    try {
      cfg = await api.get(`/api/scripts/${encodeURIComponent(key)}/config`);
    } catch (error) {
      if (seq !== this.loadSeq) return;
      this.renderConfigError(error.message);
      return;
    }
    if (seq !== this.loadSeq) return;

    this.config = cfg;
    this.renderConfig(cfg);
    this.refresh();
    this.applyPending();
  }

  renderSkeleton() {
    this.host.innerHTML = `<div class="launch-page">${this.view.head({ title: "Loading", lead: "", dateline: "", figure: true })}<div class="launch-page__wait">${skeleton(4)}</div></div>`;
    this.headFigure = this.host.querySelector("hairline-figure");
    this.writeDispatch();
  }

  renderShell(detail) {
    this.host.innerHTML = `
      <div class="launch-page">
        ${this.view.head({ title: detail.title, lead: detail.purpose, dateline: this.facts(), figure: true })}
        <div class="launch-work">
          <aside class="launch-rail" aria-label="Launch settings" data-role="rail"></aside>
          <div class="launch-main">
            <div class="tabs launch-main__tabs" role="tablist" aria-label="Entry point">
              <button type="button" role="tab" aria-selected="true" data-pane="config">Configuration</button>
              <button type="button" role="tab" aria-selected="false" data-pane="source">Source</button>
            </div>
            <div class="launch-main__pane" data-role="config">${skeleton(4)}</div>
            <div class="launch-main__pane" data-role="source" hidden><pre class="launch-source" data-role="code"></pre></div>
          </div>
        </div>
      </div>`;

    this.headFigure = this.host.querySelector("hairline-figure");
    this.configHost = this.host.querySelector("[data-role=config]");
    this.factsEl = this.host.querySelector("[data-role=dateline]");
    this.host.querySelectorAll(".launch-main__tabs button").forEach((tab) => tab.addEventListener("click", () => this.setPane(tab.dataset.pane)));
    this.renderSource(detail);
    this.renderRail();
    this.writeDispatch();
  }

  facts() {
    const bits = [this.detail.category, this.detail.file, this.detail.config_class];
    if (this.config) {
      const shown = this.states.length;
      bits.push(`${shown} of ${this.config.leaves.length} fields`, `${this.sections.length} sections`);
    }
    return bits.join(" · ");
  }

  renderSource(detail) {
    const lines = detail.source.replace(/\n$/, "").split("\n");
    this.host.querySelector("[data-role=code]").innerHTML = lines.map((line) => `<span class="launch-source__line">${esc(line) || " "}</span>`).join("");
  }

  setPane(pane) {
    this.host.querySelectorAll(".launch-main__tabs button").forEach((tab) => tab.setAttribute("aria-selected", String(tab.dataset.pane === pane)));
    this.host.querySelector("[data-role=config]").hidden = pane !== "config";
    this.host.querySelector("[data-role=source]").hidden = pane !== "source";
  }

  renderRail() {
    const rail = this.host.querySelector("[data-role=rail]");
    rail.innerHTML = "";

    rail.appendChild(this.interpreterBlock());
    const follow = this.followBlock();
    if (follow) rail.appendChild(follow);
    rail.appendChild(this.detachBlock());
    rail.appendChild(this.copyControl.build());
    rail.appendChild(this.templateControl.build());
    rail.appendChild(this.commandBlock());
    rail.appendChild(this.manifestBlock());
    rail.appendChild(this.actionsBlock());
  }

  interpreterBlock() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `<span class="rail-block__label">Interpreter</span><select class="select rail-input" aria-label="Interpreter" data-role="interpreter">${this.project.interpreters.map((entry) => `<option value="${esc(entry.path)}"${entry.path === this.detail.preferred ? " selected" : ""}>${esc(entry.label)}  ·  ${esc(entry.path)}</option>`).join("")}</select>`;
    this.interpreterEl = block.querySelector("select");
    return block;
  }

  followBlock() {
    const next = LaunchPage.FOLLOW[this.detail.key];
    this.followEl = null;
    if (!next) return null;

    const title = (this.view.scripts.find((script) => script.key === next) || { title: next }).title;
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `<span class="rail-block__label">After run</span><select class="select rail-input" aria-label="Follow-up script" data-role="follow"><option value="">nothing</option><option value="${esc(next)}">${esc(title)}</option></select><span class="rail-note" data-role="follow-note"></span>`;
    this.followEl = block.querySelector("select");
    this.followNote = block.querySelector("[data-role=follow-note]");
    return block;
  }

  detachBlock() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `
      <span class="rail-block__label">Execution</span>
      <div class="rail-detach"><button type="button" class="switch" role="switch" aria-label="Detach from server" data-role="detach"><span class="switch__knob"></span></button><span class="rail-detach__label">Detach from server</span></div>
      <p class="rail-note rail-detach__hint" data-role="detach-hint"></p>`;
    this.detachEl = block.querySelector("[data-role=detach]");
    this.detachHint = block.querySelector("[data-role=detach-hint]");
    this.detachEl.addEventListener("click", () => {
      this.detach = !this.detach;
      this.refresh();
    });
    return block;
  }

  commandBlock() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `<div class="rail-row rail-row--head"><span class="rail-block__label">Command</span><button type="button" class="btn btn--sm" data-role="copy">Copy</button></div><code class="rail-command" data-role="command"></code>`;
    this.cmdEl = block.querySelector("[data-role=command]");
    block.querySelector("[data-role=copy]").addEventListener("click", () => {
      navigator.clipboard.writeText(this.commandText()).then(() => this.app.toast.show("Command copied", "ok"), () => this.app.toast.show("The clipboard refused the command", "error"));
    });
    return block;
  }

  manifestBlock() {
    const block = document.createElement("div");
    block.className = "rail-block";
    block.innerHTML = `<span class="rail-block__label">Overrides</span><div class="rail-manifest" data-role="manifest"></div>`;
    this.manifestEl = block.querySelector("[data-role=manifest]");
    return block;
  }

  actionsBlock() {
    const block = document.createElement("div");
    block.className = "rail-block rail-block--actions";
    block.innerHTML = `
      <button type="button" class="btn btn--primary rail-action" data-role="launch">${icon("play")}<span>Launch run</span><small data-role="count"></small></button>
      <button type="button" class="btn rail-action" data-role="schedule" title="Queue this exact configuration to launch when the running job and any earlier queued runs end">${icon("then")}<span>Schedule after current</span><small data-role="count"></small></button>
      <button type="button" class="btn rail-action" data-role="save" title="Store this exact configuration under Saved, to launch or schedule any time later">${icon("save")}<span>Save for later</span><small data-role="count"></small></button>
      <p class="rail-note rail-note--warn" data-role="required-note" hidden></p>`;
    this.launchBtn = block.querySelector("[data-role=launch]");
    this.scheduleBtn = block.querySelector("[data-role=schedule]");
    this.saveBtn = block.querySelector("[data-role=save]");
    this.requiredNote = block.querySelector("[data-role=required-note]");
    this.launchBtn.addEventListener("click", () => this.launch(false));
    this.scheduleBtn.addEventListener("click", () => this.launch(true));
    this.saveBtn.addEventListener("click", () => this.save());
    return block;
  }

  renderConfigError(message) {
    this.configHost.innerHTML = `<div class="launch-error">${failView(`The configuration of ${this.detail.title} could not be resolved.`, message)}<button type="button" class="btn btn--sm" data-role="retry">Retry resolution</button></div>`;
    this.configHost.querySelector("[data-role=retry]").addEventListener("click", () => {
      this.key = null;
      this.show(this.detail.key, null, {});
    });
  }

  renderConfig(cfg) {
    this.configHost.innerHTML = "";
    this.configHost.appendChild(this.buildToolbar(cfg));
    this.renderLayout(this.configHost, cfg);
    this.factsEl.textContent = this.facts();
  }

  navigate(key) {
    history.replaceState(null, "", `#/launch/${this.key}/${key}`);
    this.app.router.current = location.hash;
    this.setActiveSection(key);
  }

  applyPending() {
    if (!this.config) return;
    const query = this.pendingQuery;
    this.pendingQuery = null;
    if (query) {
      const outcome = this.applyRunConfig(query);
      const parts = [`Prefilled ${outcome.applied} field${outcome.applied === 1 ? "" : "s"}`, ...ConfigForm.outcomeParts(outcome)];
      this.app.toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
    }

    const entry = this.pendingImport;
    this.pendingImport = null;
    if (!entry || entry.script !== this.key) return;

    const outcome = this.applyRunConfig(entry.overrides);
    const parts = [`${entry.launch ? "Retrying" : "Imported"} "${entry.name}": ${outcome.applied} override${outcome.applied === 1 ? "" : "s"} set`, ...ConfigForm.outcomeParts(outcome)];
    if (entry.launch) parts.push(...this.restoreLaunch(entry.launch));
    this.app.toast.show(parts.join(" · "), ConfigForm.outcomeKind(outcome));
  }

  restoreLaunch(launch) {
    const notes = [];
    this.detach = Boolean(launch.detach);
    if ([...this.interpreterEl.options].some((option) => option.value === launch.interpreter)) this.interpreterEl.value = launch.interpreter;
    else notes.push(`interpreter ${launch.interpreter} is no longer listed, using the preferred one`);
    if (this.followEl) this.followEl.value = launch.follow_up && [...this.followEl.options].some((option) => option.value === launch.follow_up) ? launch.follow_up : "";
    this.refresh();
    return notes;
  }

  commandText() {
    if (!this.detail) return "";
    return this.commandLine(`python -u ${this.detail.file}`, " ", this.detach);
  }

  countText(n) {
    return n ? `${n} override${n > 1 ? "s" : ""}` : "all defaults";
  }

  refresh() {
    super.refresh();
    if (!this.detail || !this.cmdEl) return;

    const n = Object.keys(this.dirty).length;
    const count = this.countText(n);

    this.cmdEl.textContent = this.commandText();
    if (this.countEl) this.countEl.textContent = count;
    [this.launchBtn, this.scheduleBtn, this.saveBtn].forEach((button) => {
      button.querySelector("[data-role=count]").textContent = count;
      button.classList.toggle("is-armed", n > 0);
    });

    this.detachEl.classList.toggle("is-on", this.detach);
    this.detachEl.setAttribute("aria-checked", String(this.detach));
    this.detachHint.textContent = this.detach
      ? `The run survives a lost connection or a console restart. Output goes to logs/${this.detail.key}_<stamp>.out.`
      : "Output streams to this console. The run dies if the console server goes down.";
    if (this.followEl) {
      this.followEl.disabled = this.detach;
      this.followNote.textContent = this.detach ? "a detached run cannot chain a follow-up" : "";
    }

    this.renderGate();
    this.renderManifest();
    this.writeDispatch();
  }

  missingRequired() {
    const widgets = this.config && this.config.layout ? this.config.layout.widgets : {};
    return Object.entries(widgets).filter(([path, widget]) => widget.required && this.byPath.has(path)).map(([path]) => path).filter((path) => ["", "none", "None"].includes(String(this.effective(this.byPath.get(path))).trim()));
  }

  renderGate() {
    if (!this.launchBtn) return;
    const missing = this.missingRequired();
    const blocked = missing.length > 0;
    this.launchBtn.disabled = this.launching || blocked;
    this.scheduleBtn.disabled = this.launching || blocked;
    this.requiredNote.hidden = !blocked;
    this.requiredNote.textContent = blocked ? `Set ${missing.join(", ")} to launch: ${this.detail.title} loads the checkpoint of that run.` : "";
  }

  renderManifest() {
    const entries = Object.entries(this.dirty);
    if (!entries.length) {
      this.manifestEl.innerHTML = `<p class="rail-note">No overrides. Every field launches at its default.</p>`;
      return;
    }

    this.manifestEl.innerHTML = entries.map(([path, value]) => {
      const leaf = this.byPath.get(path);
      const from = leaf ? leaf.value || "none" : "";
      return `<button type="button" class="rail-override" data-path="${esc(path)}" title="Reset ${esc(path)} to its default"><span class="rail-override__path">${esc(path)}</span><span class="rail-override__change">${esc(from)} <span aria-hidden="true">to</span> <b>${esc(value || "none")}</b></span>${icon("close", "rail-override__x")}</button>`;
    }).join("");
    this.manifestEl.querySelectorAll(".rail-override").forEach((item) => item.addEventListener("click", () => this.resetField(item.dataset.path)));
  }

  writeDispatch() {
    if (!this.headFigure) return;
    const jobs = this.app.jobs || [];
    FigureStates.write(this.headFigure, { armed: Boolean(this.config), loads: Math.min(6, Object.keys(this.dirty).length), queued: jobs.some((job) => ["queued", "pending", "scheduled"].includes(job.status)), fired: this.fired });
  }

  launchBody(queue) {
    return {
      script_key: this.detail.key,
      interpreter: this.interpreterEl.value,
      overrides: { ...this.dirty },
      follow_up: this.detach || !this.followEl ? null : this.followEl.value || null,
      detach: this.detach,
      queue,
    };
  }

  async launch(queue) {
    if (!this.detail || this.launching || this.missingRequired().length) return;
    this.launching = true;
    this.launchBtn.disabled = true;
    this.scheduleBtn.disabled = true;
    try {
      const res = await api.post("/api/run", this.launchBody(queue));
      this.fired += 1;
      this.writeDispatch();
      this.app.toast.show(res.queued ? `Scheduled ${this.detail.title} to run after the current job` : `Launched ${this.detail.title}`, "ok");
      this.app.pollJobs();
      this.app.router.go(`#/console/${res.job_id}`);
    } catch (error) {
      this.app.toast.show(error.message || (queue ? "Schedule failed" : "Launch failed"), "error");
    } finally {
      this.launching = false;
      this.renderGate();
    }
  }

  async save() {
    if (!this.detail) return;
    const name = window.prompt("Name this saved run (optional):", "");
    if (name === null) return;

    const body = this.launchBody(false);
    this.saveBtn.disabled = true;
    try {
      const res = await api.post("/api/saved-runs", { script_key: body.script_key, title: this.detail.title, name: name.trim(), interpreter: body.interpreter, overrides: body.overrides, follow_up: body.follow_up, detach: body.detach });
      this.app.toast.show(`Saved ${res.entry.name || this.detail.title} under Saved`, "ok");
    } catch (error) {
      this.app.toast.show(error.message || "Save failed", "error");
    } finally {
      this.saveBtn.disabled = false;
    }
  }
}
