import { api } from "../api.js";
import { esc, icon } from "../format.js";
import { emptyBlock, failView, skeleton } from "../ui/status.js";

export class StoreCards {
  static overrides(entry) {
    const rows = Object.entries(entry.overrides || {});
    if (!rows.length) return `<p class="saved-card__none">All defaults, no overrides.</p>`;
    return `<dl class="saved-card__overrides">${rows.map(([path, value]) => `<dt>${esc(path)}</dt><dd>${esc(value)}</dd>`).join("")}</dl>`;
  }

  static facts(facts) {
    return `<div class="saved-card__facts">${facts.map((fact) => `<span>${esc(fact)}</span>`).join("")}</div>`;
  }
}

export class SavedRunsPage {
  constructor(view, host) {
    this.view = view;
    this.app = view.app;
    this.host = host;
    this.entries = [];
  }

  async render() {
    this.host.innerHTML = `
      <div class="stores">
        ${this.view.head({ title: "Saved runs", lead: "Launch definitions kept for later, each with its entry point, interpreter, overrides and run mode.", figure: true, figureName: "drawer" })}
        <div class="saved-grid" data-role="grid">${skeleton(4)}</div>
      </div>`;
    this.grid = this.host.querySelector("[data-role=grid]");
    await this.load();
  }

  async load() {
    try {
      this.entries = (await api.get("/api/saved-runs")).saved;
    } catch (error) {
      this.grid.innerHTML = failView("The saved runs could not be loaded.", error.message, error);
      return;
    }
    this.paint();
  }

  paint() {
    if (!this.entries.length) {
      this.grid.innerHTML = emptyBlock("No saved runs yet. Configure a script on its launch page and press Save for later.", `<a class="btn btn--sm" href="#/launch">Open the entry points</a>`);
      return;
    }

    this.grid.innerHTML = this.entries.map((entry) => {
      const facts = [entry.detach ? "detached" : "attached"];
      if (entry.follow_up) facts.push(`then ${entry.follow_up}`);
      facts.push(entry.saved_at.replace("T", " "));
      return `
        <article class="saved-card" data-id="${esc(entry.saved_id)}">
          <div class="saved-card__top"><span class="saved-card__script">${esc(entry.title)}</span><button type="button" class="tbtn tbtn--sm" data-action="delete" title="Delete this saved run" aria-label="Delete this saved run">${icon("trash")}</button></div>
          <h3 class="saved-card__name">${esc(entry.name || entry.title)}</h3>
          ${StoreCards.overrides(entry)}
          ${StoreCards.facts(facts)}
          <div class="saved-card__actions">
            <button type="button" class="btn btn--sm btn--primary" data-action="launch">${icon("play")}<span>Launch now</span></button>
            <button type="button" class="btn btn--sm" data-action="schedule" title="Queue this saved run to launch when the running job and any earlier queued runs end">${icon("then")}<span>Schedule</span></button>
          </div>
        </article>`;
    }).join("");

    this.grid.querySelectorAll(".saved-card").forEach((card) => {
      const entry = this.entries.find((item) => item.saved_id === card.dataset.id);
      card.querySelector("[data-action=launch]").addEventListener("click", () => this.run(entry, false));
      card.querySelector("[data-action=schedule]").addEventListener("click", () => this.run(entry, true));
      card.querySelector("[data-action=delete]").addEventListener("click", () => this.remove(entry));
    });
  }

  async run(entry, queue) {
    const label = entry.name || entry.title;
    try {
      const res = await api.post(`/api/saved-runs/${entry.saved_id}/run`, { queue: Boolean(queue) });
      this.app.toast.show(res.queued ? `Scheduled ${label} to run after the current job` : `Launched ${label}`, "ok");
      this.app.pollJobs();
      this.app.router.go(`#/console/${res.job_id}`);
    } catch (error) {
      this.app.toast.show(error.message || (queue ? "Schedule failed" : "Launch failed"), "error");
    }
  }

  async remove(entry) {
    const label = entry.name || entry.title;
    if (!window.confirm(`Delete saved run "${label}"?`)) return;
    try {
      await api.post(`/api/saved-runs/${entry.saved_id}/delete`, {});
      this.app.toast.show(`Deleted ${label}`, "ok");
      await this.load();
    } catch (error) {
      this.app.toast.show(error.message || "Delete failed", "error");
    }
  }
}

export class TemplatesPage {
  constructor(view, host) {
    this.view = view;
    this.app = view.app;
    this.host = host;
    this.entries = [];
  }

  async render() {
    this.host.innerHTML = `
      <div class="stores">
        ${this.view.head({ title: "Config templates", lead: "Named override sets that seed a launch form; the interpreter and run mode stay with the page that imports them.", figure: true, figureName: "riffle" })}
        <div class="saved-grid" data-role="grid">${skeleton(4)}</div>
      </div>`;
    this.grid = this.host.querySelector("[data-role=grid]");
    await this.load();
  }

  async load() {
    try {
      this.entries = (await api.get("/api/config-templates")).templates;
    } catch (error) {
      this.grid.innerHTML = failView("The config templates could not be loaded.", error.message, error);
      return;
    }
    this.paint();
  }

  paint() {
    if (!this.entries.length) {
      this.grid.innerHTML = emptyBlock("No config templates yet. Set the fields you reuse on a launch page and press Save as template in the rail.", `<a class="btn btn--sm" href="#/launch">Open the entry points</a>`);
      return;
    }

    this.grid.innerHTML = this.entries.map((entry) => {
      const count = Object.keys(entry.overrides).length;
      return `
        <article class="saved-card" data-id="${esc(entry.template_id)}">
          <div class="saved-card__top"><span class="saved-card__script">${esc(entry.title)}</span><button type="button" class="tbtn tbtn--sm" data-action="delete" title="Delete this template" aria-label="Delete this template">${icon("trash")}</button></div>
          <h3 class="saved-card__name">${esc(entry.name)}</h3>
          ${StoreCards.overrides(entry)}
          ${StoreCards.facts([`${count} override${count === 1 ? "" : "s"}`, entry.saved_at.replace("T", " ")])}
          <div class="saved-card__actions">
            <button type="button" class="btn btn--sm btn--primary" data-action="open" title="Open the launch page of this entry point with the template applied to the form">${icon("external")}<span>Open and import</span></button>
            <button type="button" class="btn btn--sm" data-action="rename">${icon("edit")}<span>Rename</span></button>
          </div>
        </article>`;
    }).join("");

    this.grid.querySelectorAll(".saved-card").forEach((card) => {
      const entry = this.entries.find((item) => item.template_id === card.dataset.id);
      card.querySelector("[data-action=open]").addEventListener("click", () => this.open(entry));
      card.querySelector("[data-action=rename]").addEventListener("click", () => this.rename(entry));
      card.querySelector("[data-action=delete]").addEventListener("click", () => this.remove(entry));
    });
  }

  open(entry) {
    this.view.page.pendingImport = { name: entry.name, script: entry.script, overrides: entry.overrides, launch: null };
    this.app.router.go(`#/launch/${entry.script}`);
  }

  async rename(entry) {
    const raw = window.prompt(`Rename template "${entry.name}":`, entry.name);
    if (raw === null) return;
    try {
      await api.post(`/api/config-templates/${entry.template_id}/rename`, { name: raw });
      this.app.toast.show(`Renamed to "${raw.trim()}"`, "ok");
      await this.load();
    } catch (error) {
      this.app.toast.show(error.message || "Rename failed", "error");
    }
  }

  async remove(entry) {
    if (!window.confirm(`Delete template "${entry.name}"?`)) return;
    try {
      await api.post(`/api/config-templates/${entry.template_id}/delete`, {});
      this.app.toast.show(`Deleted template "${entry.name}"`, "ok");
      await this.load();
    } catch (error) {
      this.app.toast.show(error.message || "Delete failed", "error");
    }
  }
}
