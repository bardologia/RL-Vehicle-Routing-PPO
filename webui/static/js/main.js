import { api } from "./api.js";
import { el, plural, Store } from "./format.js";
import { Figure } from "./fx/figure.js";
import { BackdropOrb } from "./fx/orb.js";
import { keys } from "./keys.js";
import { KeySheet } from "./ui/keysheet.js";
import { Router } from "./router.js";
import { statusChip, Toast } from "./ui/status.js";
import { ConsoleView } from "./views/console.js";
import { DiagnoseView } from "./views/diagnose.js";
import { EpisodeView } from "./views/episode.js";
import { LabView } from "./views/lab.js";
import { LaunchView } from "./views/launch.js";
import { OverviewView } from "./views/overview.js";
import { RunsView } from "./views/runs.js";

const TABS = [
  { key: "overview", label: "Overview", view: OverviewView, figure: "router" },
  { key: "diagnose", label: "Diagnose", view: DiagnoseView, figure: "loupe" },
  { key: "episode", label: "Episode", view: EpisodeView, fixed: true },
  { key: "lab", label: "Lab", view: LabView, fixed: true },
  { key: "runs", label: "Runs", view: RunsView, fixed: true },
  { key: "launch", label: "Launch", view: LaunchView },
  { key: "console", label: "Console", view: ConsoleView, figure: "terminal" },
];

const QUIET_VIEWS = ["episode", "lab"];

class App {
  constructor() {
    this.api = api;
    this.keys = keys;
    this.views = {};
    this.sections = {};
    this.active = null;
    this.health = null;
    this.healthListeners = [];
    this.toast = new Toast(document.getElementById("toast"));
    this.backdrop = new BackdropOrb(document.getElementById("bg-orb"));
  }

  get context() {
    return Store.read("context", { run: null, seed: null });
  }

  setContext(run, seed) {
    Store.write("context", { run, seed });
  }

  mount() {
    const tabs = document.getElementById("tabs");
    const holder = document.getElementById("views");
    TABS.forEach((tab, index) => {
      tabs.appendChild(el(`<a href="#/${tab.key}" data-tab="${tab.key}" title="Key ${index + 1}">${tab.label}</a>`));
      const section = el(`<section class="view${tab.fixed ? " view--fixed" : ""}" data-view="${tab.key}" aria-label="${tab.label}"></section>`);
      holder.appendChild(section);
      this.sections[tab.key] = section;
      this.views[tab.key] = new tab.view(this, section);
      const head = tab.figure ? section.querySelector("header.diag__bar, header.pagehead") : null;
      if (head) Figure.pagehead(head, tab.figure);
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
    for (const service of ["osrm", "vroom"]) {
      const node = document.getElementById(`svc-${service}`);
      node.innerHTML = statusChip(health[service] ? "UP" : "DOWN", null, service.toUpperCase());
      node.title = health[`${service}_url`];
    }
    for (const listener of this.healthListeners) listener(health);
  }

  async pollJobs() {
    const link = document.getElementById("jobs-link");
    try {
      const { jobs } = await api.get("/api/jobs");
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
