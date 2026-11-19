import { api } from "./api.js";
import { el, esc, plural, Store } from "./format.js";
import { BackdropOrb } from "./fx/orb.js";
import { keys } from "./keys.js";
import { KeySheet } from "./ui/keysheet.js";
import { Popover } from "./ui/popover.js";
import { Router } from "./router.js";
import { FigureState, statusChip, Toast } from "./ui/status.js";
import { tooltip, Tooltip } from "./ui/tooltip.js";
import { ConsoleView } from "./views/console.js";
import { DiagnoseView } from "./views/diagnose.js";
import { EpisodeView } from "./views/episode.js";
import { LabView } from "./views/lab.js";
import { LaunchView } from "./views/launch.js";
import { OverviewView } from "./views/overview.js";
import { RunsView } from "./views/runs.js";

const TABS = [
  { key: "overview", label: "Overview", view: OverviewView },
  { key: "diagnose", label: "Diagnose", view: DiagnoseView },
  { key: "episode", label: "Episode", view: EpisodeView, fixed: true },
  { key: "lab", label: "Lab", view: LabView, fixed: true },
  { key: "runs", label: "Runs", view: RunsView },
  { key: "launch", label: "Launch", view: LaunchView },
  { key: "console", label: "Console", view: ConsoleView },
];

const QUIET_VIEWS = ["episode", "lab"];
const DOCKER = 'sg docker -c "docker compose up -d"';

class App {
  constructor() {
    this.api = api;
    this.keys = keys;
    this.views = {};
    this.sections = {};
    this.active = null;
    this.health = null;
    this.healthListeners = [];
    this.jobs = null;
    this.jobListeners = [];
    this.toast = new Toast(document.getElementById("toast"));
    this.backdrop = new BackdropOrb(document.getElementById("bg-orb"));
  }

  get context() {
    return Store.read("context", { run: null, seed: null });
  }

  setContext(run, seed) {
    Store.write("context", { run, seed });
  }

  bindHints() {
    let hinting = false;
    document.body.addEventListener("pointermove", (event) => {
      const target = event.target instanceof Element ? event.target.closest("[data-hint]") : null;
      if (target) {
        tooltip.show(Tooltip.text(target.dataset.hintTitle ? esc(target.dataset.hintTitle) : "", esc(target.dataset.hint)), event.clientX, event.clientY);
        hinting = true;
      } else if (hinting) {
        tooltip.hide();
        hinting = false;
      }
    });
    document.body.addEventListener("pointerleave", () => {
      if (hinting) tooltip.hide();
      hinting = false;
    });
  }

  measureScrollbar() {
    const probe = el(`<div style="position:absolute;top:-200px;width:100px;height:100px;overflow-y:scroll"></div>`);
    document.body.appendChild(probe);
    document.documentElement.style.setProperty("--sbw", `${probe.offsetWidth - probe.clientWidth}px`);
    probe.remove();
  }

  mount() {
    this.measureScrollbar();
    FigureState.watch(document.body);
    const tabs = document.getElementById("tabs");
    const holder = document.getElementById("views");
    TABS.forEach((tab, index) => {
      tabs.appendChild(el(`<a href="#/${tab.key}" data-tab="${tab.key}" title="Key ${index + 1}">${tab.label}</a>`));
      const section = el(`<section class="view${tab.fixed ? " view--fixed" : ""}" data-view="${tab.key}" aria-label="${tab.label}"></section>`);
      holder.appendChild(section);
      this.sections[tab.key] = section;
      this.views[tab.key] = new tab.view(this, section);
      keys.bindGlobal(String(index + 1), `Open ${tab.label}`, () => this.router.go(`#/${tab.key}`));
    });
    this.tabBar = el(`<span class="topbar__bar is-instant" aria-hidden="true"></span>`);
    tabs.appendChild(this.tabBar);
    window.addEventListener("resize", () => this.placeTabBar(true));
    document.fonts.ready.then(() => this.placeTabBar(true));

    const lightbox = document.getElementById("lightbox");
    lightbox.addEventListener("click", () => lightbox.classList.remove("is-open"));

    this.keySheet = new KeySheet(document.getElementById("keysheet"), keys, () => TABS.find((tab) => tab.key === this.active).label);
    document.getElementById("keys-toggle").addEventListener("click", () => this.keySheet.toggle());
    this.serviceNode = document.getElementById("svc");
    this.serviceNode.addEventListener("click", () => this.toggleServices());
    this.bindHints();
    keys.bindGlobal("?", "Show this key list", () => this.keySheet.toggle());
    keys.bindGlobal("/", "Focus the search of the current view", () => {
      const view = this.views[this.active];
      if (!view.focusSearch) return false;
      view.focusSearch();
      return true;
    });
    keys.bindGlobal("Escape", "Leave a search box, close the key list", () => {
      if (!(document.activeElement instanceof HTMLElement) || !document.activeElement.matches("input[type=search]")) return false;
      document.activeElement.blur();
      return true;
    });
  }

  placeTabBar(instant) {
    const link = document.querySelector('#tabs a[aria-current="page"]');
    if (!link) return;
    this.tabBar.classList.toggle("is-instant", Boolean(instant));
    this.tabBar.style.width = `${link.offsetWidth}px`;
    this.tabBar.style.transform = `translateX(${link.offsetLeft}px)`;
  }

  finishBoot() {
    const boot = document.getElementById("boot");
    if (!boot) return;
    boot.classList.add("is-done");
    setTimeout(() => boot.remove(), 300);
  }

  openLightbox(source, caption) {
    const lightbox = document.getElementById("lightbox");
    lightbox.querySelector("img").src = source;
    lightbox.querySelector(".lightbox__cap").textContent = caption || "";
    lightbox.classList.add("is-open");
  }

  onHealth(listener) {
    this.healthListeners.push(listener);
    if (this.health) listener(this.health);
  }

  get servicesUp() {
    return Boolean(this.health && this.health.osrm && this.health.vroom);
  }

  async pollHealth() {
    let health;
    try {
      health = await api.get("/api/health");
    } catch (error) {
      health = { osrm: false, vroom: false, osrm_url: "", vroom_url: "", unreachable: true };
    }
    this.health = health;
    this.renderServices();
    for (const listener of this.healthListeners) listener(health);
  }

  serviceWord() {
    const health = this.health;
    if (health.unreachable) return "Backend unreachable";
    if (!health.osrm && !health.vroom) return "OSRM and VROOM down";
    if (!health.osrm) return "OSRM down";
    if (!health.vroom) return "VROOM down";
    return "services";
  }

  renderServices() {
    const down = !this.servicesUp;
    this.serviceNode.classList.remove("is-pending");
    this.serviceNode.classList.toggle("is-down", down);
    this.serviceNode.querySelector("span").textContent = this.serviceWord();
    if (!Popover.isOpenFor(this.serviceNode)) return;
    const node = Popover.current.node;
    node.querySelector(".svcpop__body").innerHTML = this.servicesBody();
    const figure = node.querySelector("hairline-figure");
    const state = this.serviceState();
    if (JSON.stringify(figure.state) !== JSON.stringify(state)) figure.state = state;
  }

  serviceState() {
    const health = this.health || { osrm: false, vroom: false, unreachable: false };
    return { osrm: Boolean(health.osrm), vroom: Boolean(health.vroom), reachable: !health.unreachable };
  }

  servicesBody() {
    const health = this.health || { osrm: false, vroom: false, osrm_url: "", vroom_url: "" };
    const down = !health.osrm || !health.vroom;
    const row = (name, key) => `<b>${name}</b><span>${esc(health[`${key}_url`] || "")}</span>${statusChip(health[key] ? "UP" : "DOWN")}`;
    return `
      <div class="svcpop__rows">${row("OSRM", "osrm")}${row("VROOM", "vroom")}</div>
      ${health.unreachable ? `<p class="svcpop__text">The console backend did not answer.</p>` : ""}
      <p class="svcpop__text">${down ? "Recording, solving and teacher verdicts are disabled. Start the services from the repository root:" : "Recording, solving and teacher verdicts need both routing services. Started from the repository root with:"}</p>
      <code class="svcpop__cmd">${esc(DOCKER)}</code>`;
  }

  servicesHtml() {
    return `<div class="svcpop__figure">${FigureState.html("services", this.serviceState(), 'aria-hidden="true"')}</div><div class="svcpop__body">${this.servicesBody()}</div>`;
  }

  toggleServices() {
    Popover.toggle(this.serviceNode, this.servicesHtml(), { label: "Routing services", className: "svcpop" });
  }

  onJobs(listener) {
    this.jobListeners.push(listener);
    if (this.jobs) listener(this.jobs);
  }

  async pollJobs() {
    const link = document.getElementById("jobs-link");
    try {
      const { jobs } = await api.get("/api/jobs");
      this.jobs = jobs;
      for (const listener of this.jobListeners) listener(jobs);
      const running = jobs.filter((job) => ["running", "queued", "pending"].includes(job.status)).length;
      link.textContent = running ? `${plural(running, "job")} running` : "No jobs running";
      link.classList.toggle("is-live", running > 0);
    } catch (error) {
      link.textContent = "Backend unreachable";
      link.classList.remove("is-live");
    }
  }

  route({ tab, parts, query }) {
    if (this.active && this.active !== tab) {
      const previous = this.views[this.active];
      if (previous.leave) previous.leave();
    }
    const first = this.active === null;
    this.active = tab;
    document.documentElement.dataset.view = tab;
    this.backdrop.setStopped(QUIET_VIEWS.includes(tab));
    keys.setScope(tab);
    for (const [key, section] of Object.entries(this.sections)) section.classList.toggle("is-active", key === tab);
    for (const link of document.querySelectorAll("#tabs a")) {
      if (link.dataset.tab === tab) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
    this.placeTabBar(first);
    if (first) this.finishBoot();
    this.views[tab].enter(parts, query);
  }

  start() {
    this.mount();
    this.router = new Router(TABS.map((tab) => tab.key), (route) => this.route(route));
    this.pollHealth();
    this.pollJobs();
    setInterval(() => this.pollHealth(), 15000);
    setInterval(() => this.pollJobs(), 15000);
    this.router.sync();
  }
}

window.app = new App();
window.app.start();
