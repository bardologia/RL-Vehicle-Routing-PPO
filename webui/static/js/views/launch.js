import { api } from "../api.js";
import { el, esc } from "../format.js";
import { PageHead } from "../ui/page_head.js";
import { failView, skeleton } from "../ui/status.js";
import { LaunchPage } from "../launch/launch_page.js";
import { ScriptCatalogView } from "../launch/script_catalog_view.js";
import { SequencesPage } from "../launch/sequences.js";
import { SavedRunsPage, TemplatesPage } from "../launch/stores.js";

export class LaunchView {
  static PAGES = [
    { key: "", label: "Scripts" },
    { key: "saved", label: "Saved" },
    { key: "templates", label: "Templates" },
    { key: "sequences", label: "Sequences" },
  ];

  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.scripts = null;
    this.project = null;
    this.page = null;
    this.sequences = null;
    this.current = "";
    this.pendingImport = null;

    root.appendChild(el(`<div class="view__pad launch-view"><div data-role="host">${skeleton(4)}</div></div>`));
    this.host = root.querySelector("[data-role=host]");
    app.onJobs(() => {
      if (this.page && this.current === "script") this.page.writeDispatch();
    });
  }

  head({ title, lead = "", dateline = "", figure = false, figureName = "dispatch" }) {
    const section = this.current === "script" ? "" : this.current;
    const nav = `<nav class="segmented launch-nav" aria-label="Launch pages">${LaunchView.PAGES.map((page) => `<a href="#/launch${page.key ? `/${page.key}` : ""}"${page.key === section ? ' aria-current="page"' : ""}>${esc(page.label)}</a>`).join("")}</nav>`;
    return PageHead.html({ title, lead, controls: nav, dateline, figure: figure ? HairlineFigure.html(figureName) : "", form: "narrow", className: "launch-head" });
  }

  set pendingImport(entry) {
    this.pending = entry;
    if (this.page) this.page.pendingImport = entry;
  }

  get pendingImport() {
    return this.pending;
  }

  focusSearch() {
    const input = this.host.querySelector(".catalog__search, .cfg-toolbar__search");
    if (input) input.focus();
  }

  async ensureCatalog() {
    if (this.scripts && this.project) return true;
    try {
      [this.scripts, this.project] = await Promise.all([api.get("/api/scripts").then((data) => data.scripts), api.get("/api/project")]);
    } catch (error) {
      this.host.innerHTML = failView("The entry points could not be loaded.", error.message, error);
      return false;
    }
    this.sequences = new SequencesPage(this, this.host, this.project);
    return true;
  }

  async enter(parts, query) {
    if (!(await this.ensureCatalog())) return;
    const [first, second] = parts;
    if (this.current === "sequences" && first !== "sequences") this.sequences.leave();

    if (!first) {
      this.current = "";
      this.page = null;
      new ScriptCatalogView(this, this.host).render(this.scripts);
      return;
    }

    if (first === "saved") {
      this.current = "saved";
      await new SavedRunsPage(this, this.host).render();
      return;
    }

    if (first === "templates") {
      this.current = "templates";
      await new TemplatesPage(this, this.host).render();
      return;
    }

    if (first === "sequences") {
      this.current = "sequences";
      await this.sequences.render();
      return;
    }

    const known = this.scripts.some((script) => script.key === first);
    if (!known) {
      this.current = "";
      this.host.innerHTML = failView(`There is no entry point called "${first}".`, "", { status: 404 });
      return;
    }

    const reuse = this.current === "script" && this.page && this.page.key === first && this.host.contains(this.page.configHost || null);
    this.current = "script";
    if (!reuse) this.page = new LaunchPage(this, this.host, this.project);
    if (this.pending) {
      this.page.pendingImport = this.pending;
      this.pending = null;
    }
    await this.page.show(first, second || null, query);
  }

  leave() {
    if (this.current === "sequences") this.sequences.leave();
  }
}
