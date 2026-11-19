import { api } from "../api.js";
import { esc, icon, plural } from "../format.js";
import { Orb } from "../fx/orb.js";
import { FigureStates } from "../ui/figure_states.js";
import { infoText, panelHead } from "../ui/panel.js";
import { statusChip } from "../ui/status.js";
import { DialGauge, INK, LinearMeter, SegMeter } from "./gauges.js";
import { ActivityTimeline, HeatmapCalendar, SERIES, Spark, TimeStrip } from "./time_strip.js";

const DOCKER = 'sg docker -c "docker compose up -d"';

export class ByteFormat {
  static gb(bytes) {
    return (bytes / 1073741824).toFixed(1);
  }

  static mb(bytes) {
    const gb = bytes / 1073741824;
    if (gb >= 1) return `${gb.toFixed(1)} G`;
    return `${Math.round(bytes / 1048576)} M`;
  }

  static size(bytes) {
    const tb = bytes / 1099511627776;
    if (tb >= 1) return `${tb.toFixed(2)} TB`;
    return bytes >= 1073741824 ? `${ByteFormat.gb(bytes)} GB` : `${Math.round(bytes / 1048576)} MB`;
  }

  static uptime(seconds) {
    const d = Math.floor(seconds / 86400);
    const h = Math.floor((seconds % 86400) / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (d > 0) return `${d}d ${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
    return `${h}:${String(m).padStart(2, "0")}`;
  }

  static energy(wh) {
    if (!wh) return "";
    return wh < 10 ? `${wh.toFixed(2)} Wh` : `${Math.round(wh)} Wh`;
  }
}

export class BoardMarkup {
  static figure(name, attrs, className) {
    return `<div class="sb-fig${className ? ` ${className}` : ""}"><div class="fig-fill">${HairlineFigure.html(name, `${attrs || ""} aria-hidden="true"`)}</div></div>`;
  }

  static bar(role, tone) {
    return `<div class="sb-bar"><i class="sb-bar__fill${tone ? ` sb-bar__fill--${tone}` : ""}" data-role="${role}"></i></div>`;
  }

  static metric(label, role, title) {
    return `<div class="sb-metric"${title ? ` title="${esc(title)}"` : ""}><span data-role="${role}-label">${esc(label)}</span><span data-role="${role}">--</span></div>`;
  }

  static ranges(role, spans, current) {
    return `<span class="segmented" role="radiogroup" aria-label="Window" data-role="${role}">${spans.map((span) => `<button type="button" role="radio" data-range="${span}" aria-checked="${span === current}">${span}</button>`).join("")}</span>`;
  }

  static watchdog(limits) {
    const tiles = [
      [`${Math.round(limits.cpu_alert)}%`, "cpu alert"],
      [`${limits.load_ratio.toFixed(1)} / core`, "load limit"],
      [`${Math.round(limits.ram_warn)}%`, "ram warn"],
      [`${Math.round(limits.ram_kill)}%`, "ram kill"],
      [`${Math.round(limits.mine_share * 100)}%`, "own ram share"],
      [`< ${Math.round(limits.disk_warn)}% free`, "disk warn"],
      [`${Math.round(limits.interval)} s`, "interval"],
      [`${Math.round(limits.cooldown)} s`, "kill cooldown"],
    ].map(([value, key]) => `<div class="sb-tile"><dt>${esc(key)}</dt><dd>${esc(value)}</dd></div>`).join("");
    const info = infoText("The watchdog samples CPU, load, RAM and disk every interval and raises an alert when a reading crosses its limit.", "When RAM passes the kill limit and your processes hold more than the own RAM share, it stops every running console job, then waits the cooldown before it may stop them again.", "Launch politeness sets the priority every new launch starts under. Auto-throttle lowers the priority of running console jobs once an alert blames them.", `Auto-nuke, when armed, runs Nuke jobs once a danger alert you cause has stood for ${Math.round(limits.nuke_after)} s, then waits ${Math.round(limits.nuke_cooldown)} s before it may fire again. Only the RAM alert at the kill limit with your share above the own RAM share counts as yours.`);
    return `
      <section class="panel span-4 sb-card sb-card--watch" aria-label="Watchdog">
        ${panelHead("Watchdog", { info, actions: `<span class="sb-light" data-role="wd-light" aria-hidden="true"></span><span class="sb-mode" data-role="wd-mode">--</span>` })}
        <div class="sb-watch">
          <dl class="sb-tiles">${tiles}</dl>
          ${BoardMarkup.figure("watchdog", 'data-role="wd-figure" autopilot="off"', "sb-watch__fig")}
        </div>
        <div class="sb-alerts" data-role="wd-alerts"></div>
        <div class="sb-actions">
          <button type="button" class="btn btn--sm" data-role="detach" title="Detach the console server from this terminal and login session into its own systemd scope, so it survives logout and a closed terminal">Keep alive: off</button>
          <label class="ctl"><span class="ctl__label">Launch</span><select class="select" data-role="politeness" aria-label="Launch politeness" title="Priority every new launch starts under. Running jobs keep theirs."><option value="off">off</option><option value="polite">polite</option><option value="background">background</option></select></label>
          <button type="button" class="btn btn--sm" data-role="throttle" aria-pressed="false" title="When on, the first alert that blames console jobs drops them to nice 19 and idle I/O. Priorities only go down, so a throttled job stays throttled until it ends.">Auto-throttle</button>
          <button type="button" class="btn btn--sm" data-role="auto-nuke" aria-pressed="false" title="When armed, nukes every console job and project script of yours once a danger alert you cause persists">Auto-nuke</button>
        </div>
      </section>`;
  }

  static kill() {
    const info = infoText("Nuke jobs sends SIGTERM, then SIGKILL after a grace period, to every console job and every project entry script running under your user, and clears the launch queue. Nothing else on the machine is touched and the console is spared.", "Kill front end stops every console server of this repository. Running and detached jobs keep going and are adopted again when the console restarts with webui/run.sh.");
    return `
      <section class="panel span-4 sb-card sb-card--kill" aria-label="Kill switches">
        ${panelHead("Kill switches", { info, actions: `<span class="sb-light" data-role="nuke-light" aria-hidden="true"></span><span class="sb-mode" data-role="nuke-mode">ready</span>` })}
        <div class="sb-kill">
          <div class="sb-kill__cell">
            ${BoardMarkup.figure("nuke", 'data-role="nuke-figure"', "sb-kill__fig")}
            <p class="sb-hint">Stops every console job and project script of yours here; the console is spared.</p>
            <button type="button" class="btn btn--danger" data-role="nuke">${icon("warning")}<span>Nuke jobs</span></button>
          </div>
          <div class="sb-kill__cell">
            ${BoardMarkup.figure("guillotine", 'data-role="kill-figure"', "sb-kill__fig")}
            <p class="sb-hint">Stops every console server of this repository; running jobs survive and are adopted on restart.</p>
            <button type="button" class="btn" data-role="kill">${icon("stop")}<span>Kill front end</span></button>
          </div>
        </div>
      </section>`;
  }

  static services() {
    const info = infoText("Recording, solving and teacher verdicts call OSRM for travel times and VROOM for the teacher plan. Both run as docker services next to the console.");
    return `
      <section class="panel span-4 sb-card sb-card--svc" aria-label="Routing services">
        ${panelHead("Routing services", { info, actions: `<button type="button" class="btn btn--sm" data-role="svc-check">${icon("refresh")}<span>Check now</span></button>` })}
        <div class="sb-svc">
          ${BoardMarkup.figure("services", 'data-role="svc-figure"', "sb-svc__fig")}
          <div class="sb-svc__rows" data-role="svc-rows"></div>
          <p class="sb-hint" data-role="svc-text"></p>
          <code class="sb-cmd">${esc(DOCKER)}</code>
        </div>
      </section>`;
  }

  static gpus(sys) {
    const cards = sys.gpus.map((gpu, i) => `
      <article class="gcard" data-gpu="${i}">
        <header class="gcard__head"><span class="gcard__idx">gpu ${gpu.index}</span><span class="gcard__name">${esc(gpu.name)}</span><span class="gcard__who" data-role="who"></span></header>
        <div class="gcard__cluster">
          <canvas class="gcard__dial" data-role="dial"></canvas>
          <div class="gcard__side">
            <canvas class="gcard__graph" data-role="graph" title="Last 72 s: utilisation and VRAM in percent"></canvas>
            <div class="gcard__legend"><span><i class="sb-key sb-key--util"></i>util</span><span><i class="sb-key sb-key--vram"></i>vram</span></div>
          </div>
          <div class="gcard__live"><div class="fig-fill">${HairlineFigure.html("gpuwire", 'data-role="wire" aria-hidden="true"')}</div></div>
        </div>
        <div class="gcard__meters"><canvas class="sb-meter" data-role="temp"></canvas><canvas class="sb-meter" data-role="power"></canvas></div>
        <canvas class="gcard__health" data-role="health" title="Last hour: temperature in degrees Celsius and power in percent of the limit"></canvas>
        <div class="gcard__legend"><span><i class="sb-key sb-key--temp"></i>temp, last hour</span><span><i class="sb-key sb-key--power"></i>power, last hour</span></div>
        <div class="gcard__vram"><span class="gcard__vlabel">VRAM</span><canvas class="gcard__seg" data-role="vseg"></canvas><span class="gcard__vtext" data-role="vram">--</span></div>
      </article>`).join("");
    const empty = `<div class="sb-empty">${esc(sys.gpus_known ? "No CUDA devices are visible to the backend." : "nvidia-smi is not answering, GPU occupancy is unknown.")}</div>`;
    const shelf = sys.gpus.length % 2 === 1 ? `<div class="gcard gcard--shelf">${BoardMarkup.figure("gpus", 'data-role="gpu-shelf"', "gcard__shelf")}</div>` : "";
    const info = infoText("Read from nvidia-smi every second. The dial shows utilisation, the strip the last 72 s of utilisation and VRAM, the meters temperature and power draw, the second strip the last hour from the telemetry log.", "The holder names the users with processes on the device; console jobs count as yours.");
    return `
      <section class="panel span-8 sb-card sb-card--gpus" aria-label="CUDA devices">
        ${panelHead("CUDA devices", { note: plural(sys.gpus.length, "device"), info })}
        <div class="sb-gpugrid">${sys.gpus.length ? cards + shelf : empty}</div>
      </section>`;
  }

  static cpu(sys) {
    const info = infoText("Busy is the share of all cores not idle over the last sample. Load is the run-queue length averaged over 1, 5 and 15 minutes. A core counts as dispatched above 50% busy.");
    return `
      <section class="panel span-4 sb-card sb-card--cpu" aria-label="Processor">
        ${panelHead("Processor", { note: `${sys.cpu.count} cores`, info })}
        <div class="sb-cpu">
          <canvas class="sb-cpu__dial" data-role="cpu-dial"></canvas>
          <div class="sb-cpu__side">
            <dl class="sb-load"><div><dt data-role="load1">--</dt><dd>load 1m</dd></div><div><dt data-role="load5">--</dt><dd>5m</dd></div><div><dt data-role="load15">--</dt><dd>15m</dd></div></dl>
            ${BoardMarkup.metric("avg usage", "cpu-avg")}
            ${BoardMarkup.bar("cpu-bar")}
            ${BoardMarkup.metric("cores above 50%", "cpu-active")}
            ${BoardMarkup.bar("cores-bar", "live")}
          </div>
        </div>
        <canvas class="sb-spark" data-role="cpu-graph" title="Last 72 s of total CPU busy, in percent"></canvas>
        ${BoardMarkup.figure("cores", 'data-role="cores-figure"', "sb-cpu__fig")}
      </section>`;
  }

  static resources(sys) {
    const info = infoText("Disk shows the file system that holds the repository, then the space the repository and its runs folder take on it, measured with du every few minutes.", "RAM counts total minus available; the strip shows the last 72 s.");
    return `
      <section class="panel span-4 sb-card sb-card--res" aria-label="Storage and memory">
        ${panelHead("Storage", { note: "", info })}
        <div class="sb-res">
          <div class="sb-res__info">
            ${BoardMarkup.metric("", "disk")}
            ${BoardMarkup.bar("disk-bar")}
            ${BoardMarkup.metric("free, all users", "disk-free")}
            ${BoardMarkup.metric("repository", "disk-repo")}
            ${BoardMarkup.bar("disk-repo-bar", "live")}
            ${BoardMarkup.metric("runs", "disk-runs")}
            ${BoardMarkup.bar("disk-runs-bar", "accent")}
          </div>
          ${BoardMarkup.figure("cabinet", "", "sb-res__fig")}
        </div>
        <div class="sb-rule"><h3 class="panel__title">Memory</h3><span class="panel__note" data-role="mem-total"></span></div>
        <div class="sb-res">
          <div class="sb-res__info">
            ${BoardMarkup.metric("ram", "ram")}
            ${BoardMarkup.bar("ram-bar", "live")}
            ${BoardMarkup.metric("swap", "swap")}
            ${BoardMarkup.bar("swap-bar", "warn")}
            <canvas class="sb-spark sb-spark--mem" data-role="mem-graph" title="Last 72 s of RAM in use, in percent"></canvas>
          </div>
          ${BoardMarkup.figure("drawer", "", "sb-res__fig")}
        </div>
      </section>`;
  }

  static activity(range) {
    const info = infoText("Every GPU sample is logged to logs/webui/telemetry and bucketed here: a block is coloured by who held the device, its strength by the share of the block the device was busy.");
    return `
      <section class="panel span-4 sb-card sb-card--act" aria-label="GPU activity">
        ${panelHead("GPU activity", { note: "", info, actions: BoardMarkup.ranges("act-ranges", ["day", "week", "month"], range) })}
        ${BoardMarkup.figure("patch", "", "sb-act__fig")}
        <div class="sb-act__plot"><canvas class="sb-act__canvas" data-role="act-graph"></canvas></div>
        <div class="sb-foot">
          <span class="sb-legend"><span><i class="sb-key sb-key--mine"></i>you</span><span><i class="sb-key sb-key--others"></i>others</span><span><i class="sb-key sb-key--idle"></i>idle</span></span>
          <span class="sb-read" data-role="act-read">hover a block</span>
        </div>
      </section>`;
  }

  static jobs() {
    const info = infoText("The newest console jobs with their energy draw while they ran. A row opens the job in the Console.");
    return `
      <section class="panel span-4 sb-card sb-card--jobs" aria-label="Jobs">
        ${panelHead("Jobs", { note: "", info, actions: `<a class="btn btn--sm" href="#/console">${icon("external")}<span>Open Console</span></a>` })}
        <ul class="sb-jobs" data-role="jobs"><li class="sb-empty">No jobs yet.</li></ul>
        ${BoardMarkup.figure("scheduler", 'data-role="jobs-figure"', "sb-jobs__fig")}
      </section>`;
  }

  static host(range) {
    const info = infoText("CPU, RAM and network throughput logged every 30 s to logs/webui/telemetry and kept across restarts. Hover a plot for its readings.");
    return `
      <section class="panel span-12 sb-card sb-card--host" aria-label="Host history">
        ${panelHead("Host history", { note: "", info, actions: BoardMarkup.ranges("hh-ranges", ["hour", "day", "week", "month"], range) })}
        <div class="sb-host">
          ${BoardMarkup.figure("phosphor", 'autopilot="off"', "sb-host__fig")}
          <div class="sb-host__cell"><span class="sb-legend"><span><i class="sb-key sb-key--util"></i>cpu %</span><span><i class="sb-key sb-key--vram"></i>ram %</span></span><canvas class="sb-host__canvas" data-role="hh-cpu"></canvas></div>
          <div class="sb-host__cell"><span class="sb-legend"><span><i class="sb-key sb-key--rx"></i>net in MB/s</span><span><i class="sb-key sb-key--tx"></i>net out MB/s</span></span><canvas class="sb-host__canvas" data-role="hh-net"></canvas></div>
          ${BoardMarkup.figure("router", "", "sb-host__fig")}
        </div>
      </section>`;
  }

  static internals() {
    const info = infoText("The newest lines the console server wrote to its own log, logs/webui/webui_server.log.");
    return `
      <section class="panel span-3 sb-card sb-card--figure" aria-hidden="true">${BoardMarkup.figure("terminal", "", "sb-term__fig")}</section>
      <section class="panel span-9 sb-card sb-card--log" aria-label="Server internals">
        ${panelHead("Server internals", { note: "", info })}
        <div class="sb-log" data-role="log">${Orb.inline("Reading the server log", "listening")}</div>
      </section>`;
  }

  static heatmap() {
    const info = infoText("Thirty days of GPU activity, one column per day and one row per four-hour slot. Colour names the holder, strength the busy share.");
    return `
      <section class="panel span-12 sb-card sb-card--heat" aria-label="GPU heatmap">
        ${panelHead("GPU heatmap, 30 days", { note: "", info })}
        <div class="sb-heat">
          <div class="sb-heat__main">
            <canvas class="sb-heat__canvas" data-role="heat-graph"></canvas>
            <div class="sb-foot">
              <span class="sb-legend"><span><i class="sb-key sb-key--mine"></i>you</span><span><i class="sb-key sb-key--others"></i>others</span><span><i class="sb-key sb-key--idle"></i>idle</span></span>
              <span class="sb-read" data-role="heat-read">hover a cell; rows are 4 h slots, columns are days</span>
            </div>
          </div>
          ${BoardMarkup.figure("calendar", 'data-role="heat-figure" autopilot="off"', "sb-heat__fig")}
        </div>
      </section>`;
  }

  static processes(user) {
    const info = infoText("Your processes on this machine, busiest first. Memory is the proportional set size, so shared pages are split across the processes that map them.");
    return `
      <section class="panel panel--flush span-12 sb-card sb-card--procs" aria-label="Processes">
        ${panelHead(`Processes of ${user}`, { note: "", info })}
        <div class="panel__body sb-procs">
          <div class="dtable dtable--dense dtable--static">
            <table>
              <thead><tr><th class="num">PID</th><th class="num">CPU%</th><th class="num">Mem</th><th class="num">GPU</th><th class="num">Thr</th><th>S</th><th>Command</th></tr></thead>
              <tbody data-role="procs"></tbody>
            </table>
          </div>
        </div>
      </section>`;
  }

  static board(sys, ranges) {
    return `
      <div class="grid-12 sb-grid">
        ${BoardMarkup.watchdog(sys.alerts.limits)}
        ${BoardMarkup.kill()}
        ${BoardMarkup.services()}
        ${BoardMarkup.gpus(sys)}
        ${BoardMarkup.cpu(sys)}
        ${BoardMarkup.resources(sys)}
        ${BoardMarkup.activity(ranges.activity)}
        ${BoardMarkup.jobs()}
        ${BoardMarkup.host(ranges.host)}
        ${BoardMarkup.internals()}
        ${BoardMarkup.heatmap()}
        ${BoardMarkup.processes(sys.user)}
      </div>`;
  }
}

export class StatusBoard {
  static POLL_MS = 1000;
  static CADENCE = { jobs: 5000, log: 5000, hostHour: 30000, activity: 60000, host: 60000, heatmap: 300000 };

  constructor(app, host, head) {
    this.app = app;
    this.host = host;
    this.head = head;
    this.nodes = {};
    this.built = false;
    this.gpuCount = -1;
    this.gpuEls = [];
    this.cpuDial = null;
    this.timers = [];
    this.polling = false;
    this.ranges = { activity: "day", host: "day" };
    this.states = { activity: null, host: null, heat: null, hostHour: {} };
    this.server = null;
    this.politeness = null;
    this.down = false;
    this.healthListener = (health) => this.renderServices(health);
    this.resizer = new ResizeObserver(() => this.redrawCharts());
  }

  node(role) {
    return this.host.querySelector(`[data-role="${role}"]`);
  }

  text(role, value) {
    const target = this.node(role);
    if (target) target.textContent = value;
  }

  bar(role, pct) {
    const target = this.node(role);
    if (!target) return;
    target.style.transform = `scaleX(${Math.max(0, Math.min(100, pct)) / 100})`;
    target.classList.toggle("is-hot", pct >= 90);
  }

  async poll() {
    if (this.down || this.polling) return;
    this.polling = true;
    let sys;
    try {
      sys = await api.get("/api/system");
    } catch (error) {
      this.polling = false;
      this.head.dateline.textContent = `backend unreachable: ${error.message}`;
      return;
    }
    this.polling = false;
    if (!this.built || sys.gpus.length !== this.gpuCount) this.build(sys);
    this.update(sys);
  }

  disposeGauges() {
    for (const card of this.gpuEls) [card.dial, card.temp, card.power, card.vseg].forEach((gauge) => gauge.dispose());
    if (this.cpuDial) this.cpuDial.dispose();
    this.gpuEls = [];
    this.cpuDial = null;
  }

  build(sys) {
    this.disposeGauges();
    this.built = true;
    this.gpuCount = sys.gpus.length;
    this.histMax = sys.history.max_samples;
    this.host.innerHTML = BoardMarkup.board(sys, this.ranges);
    this.resizer.disconnect();
    this.resizer.observe(this.host);

    this.gpuEls = [...this.host.querySelectorAll(".gcard[data-gpu]")].map((card, i) => {
      const part = (role) => card.querySelector(`[data-role="${role}"]`);
      return {
        who: part("who"),
        graph: part("graph"),
        health: part("health"),
        vram: part("vram"),
        wire: card.querySelector("hairline-figure[name=gpuwire]"),
        dial: new DialGauge(part("dial"), { label: "util %", color: INK.accent }),
        temp: new LinearMeter(part("temp"), { min: 20, max: 100, label: "temp °C", color: INK.live, zones: [{ from: 70, to: 85, color: INK.warn }, { from: 85, to: 100, color: INK.fail }] }),
        power: new LinearMeter(part("power"), { max: sys.gpus[i].power_limit || 75, label: "power W", color: SERIES.power, digits: 1 }),
        vseg: new SegMeter(part("vseg"), { color: INK.live }),
      };
    });
    this.cpuDial = new DialGauge(this.node("cpu-dial"), { label: "busy %", color: INK.accent, zones: [{ from: 75, to: 90, color: INK.warn }, { from: 90, to: 100, color: INK.fail }] });
    this.activity = new ActivityTimeline(this.node("act-graph"), this.node("act-read"));
    this.heatmap = new HeatmapCalendar(this.node("heat-graph"), this.node("heat-read"));

    this.wireRanges("act-ranges", "activity", () => this.pollActivity());
    this.wireRanges("hh-ranges", "host", () => this.pollHost());
    this.wireControls();
    this.renderActivity();
    this.renderHost();
    this.renderHeatmap();
    if (this.app.health) this.renderServices(this.app.health);
    if (this.jobs) this.renderJobs(this.jobs);
    this.pollLog();
  }

  wireRanges(role, key, reload) {
    const group = this.node(role);
    const buttons = [...group.querySelectorAll("button[data-range]")];
    for (const button of buttons) {
      button.addEventListener("click", () => {
        if (this.ranges[key] === button.dataset.range) return;
        this.ranges[key] = button.dataset.range;
        for (const other of buttons) other.setAttribute("aria-checked", String(other === button));
        reload();
      });
    }
  }

  wireControls() {
    this.node("nuke").addEventListener("click", () => this.nuke());
    this.node("kill").addEventListener("click", () => this.killFrontEnd());
    this.node("detach").addEventListener("click", () => this.detach());
    this.node("politeness").addEventListener("change", (event) => this.sendPoliteness({ level: event.target.value }, event.target));
    this.node("throttle").addEventListener("click", (event) => this.sendPoliteness({ auto_throttle: !this.politeness.auto_throttle }, event.currentTarget));
    this.node("auto-nuke").addEventListener("click", (event) => this.armNuke(event.currentTarget));
    this.node("svc-check").addEventListener("click", () => this.app.pollHealth());
  }

  update(sys) {
    this.renderMast(sys);
    this.renderWatchdog(sys);
    this.renderServer(sys.server);
    this.renderPoliteness(sys.politeness);
    this.renderGpus(sys);
    this.renderCpu(sys);
    this.renderResources(sys);
    this.renderProcesses(sys.procs);
  }

  renderMast(sys) {
    this.head.title.textContent = sys.host;
    const bits = [`up ${ByteFormat.uptime(sys.uptime)}`, `${sys.cpu.count} cores`, sys.gpus_known ? plural(sys.gpus.length, "CUDA device") : "CUDA devices unknown", `${ByteFormat.size(sys.mem.total)} RAM`];
    this.head.dateline.textContent = bits.join(" · ");
    this.renderMastFigures(sys);
  }

  renderMastFigures(sys) {
    const jobs = this.jobs || [];
    const running = jobs.filter((job) => job.status === "running").slice(0, 4);
    const waiting = jobs.filter((job) => job.status === "queued" || job.status === "scheduled").length;
    const busy = Math.round(Math.max(0, Math.min(100, sys.cpu.total))) / 100;
    FigureStates.write(this.head.fleet, { routes: running.map(() => ({ stops: Math.max(1, Math.round(busy * 6)), drive: busy })), unassigned: Math.min(6, waiting), idle: 4 - running.length, solving: false });
    const recent = sys.history.cpu.slice(-48);
    FigureStates.write(this.head.load, recent.length ? FigureStates.training(recent.map((value) => value / 100), running.length > 0) : { values: Array(24).fill(null), best: null, live: false });
  }

  renderWatchdog(sys) {
    const alerts = sys.alerts;
    const active = alerts.active;
    this.node("wd-light").className = `sb-light${alerts.armed ? (active.length ? " is-alert" : " is-armed") : ""}`;
    this.text("wd-mode", alerts.armed ? (active.length ? plural(active.length, "active alert") : "armed, all clear") : "offline");

    const recent = alerts.events.slice(-3).reverse();
    const rows = active.map((alert) => ({ level: alert.level, message: alert.message, time: alert.since })).concat(recent.map((event) => ({ level: event.level, message: event.message, time: event.time })));
    this.node("wd-alerts").innerHTML = rows.map((row) => `<div class="sb-alert sb-alert--${esc(row.level)}"><span class="sb-alert__time">${esc(row.time.slice(11, 19))}</span><span class="sb-alert__msg">${esc(row.message)}</span></div>`).join("");

    this.renderAutoNuke(alerts.auto_nuke);

    const kinds = new Set(active.map((alert) => alert.kind));
    const lim = alerts.limits;
    const ratio = sys.cpu.count ? sys.cpu.load[0] / sys.cpu.count : 0;
    const levels = [sys.cpu.total / 100, (0.8 * ratio) / lim.load_ratio, 1 - sys.mem.available / sys.mem.total, 1 - sys.disk.free / sys.disk.total];
    const limits = [lim.cpu_alert / 100, 0.8, lim.ram_warn / 100, 1 - lim.disk_warn / 100];
    const flags = [kinds.has("cpu"), kinds.has("load"), kinds.has("ram") || kinds.has("ram_kill"), kinds.has("disk")];
    FigureStates.write(this.node("wd-figure"), { armed: alerts.armed, gauges: levels.map((level, k) => ({ level: Math.round(level * 100) / 100, limit: limits[k], alert: flags[k] })) });
  }

  renderAutoNuke(armed) {
    this.autoNuke = armed;
    const button = this.node("auto-nuke");
    button.setAttribute("aria-pressed", String(armed));
    button.classList.toggle("is-warn", armed);
    button.textContent = armed ? "Auto-nuke: armed" : "Auto-nuke: off";
  }

  renderServer(server) {
    this.server = server;
    const button = this.node("detach");
    const partial = server.detached && server.shield_error;
    button.innerHTML = `<span>${server.detached ? (partial ? "Keep alive: partial" : "Keep alive: on") : "Keep alive: off"}</span>`;
    button.classList.toggle("is-on", Boolean(server.detached) && !partial);
    button.classList.toggle("is-warn", Boolean(partial));
    button.disabled = Boolean(server.detached);
    if (server.detached) button.title = partial ? `Detached as pid ${server.pid}, but the session shield is incomplete: ${server.shield_error}. Output continues in ${server.log_path}.` : `Shielded in ${server.scope} as pid ${server.pid}; survives logout and a closed terminal. Output continues in ${server.log_path}.`;
  }

  renderPoliteness(politeness) {
    this.politeness = politeness;
    const select = this.node("politeness");
    if (document.activeElement !== select) select.value = politeness.level;
    select.title = `New launches start under ${politeness.label}. Running jobs keep theirs.`;
    const toggle = this.node("throttle");
    toggle.setAttribute("aria-pressed", String(politeness.auto_throttle));
    toggle.textContent = politeness.auto_throttle ? "Auto-throttle: on" : "Auto-throttle: off";
  }

  renderGpus(sys) {
    const history = sys.history.gpus;
    sys.gpus.forEach((gpu, i) => {
      const card = this.gpuEls[i];
      const util = gpu.util || 0;
      const memPct = gpu.mem_total ? (100 * gpu.mem_used) / gpu.mem_total : 0;
      card.dial.set(util);
      card.temp.set(gpu.temp);
      if (gpu.power_limit) card.power.range(gpu.power_limit);
      card.power.set(gpu.power);
      card.vseg.set(memPct / 100);
      FigureStates.write(card.wire, { util: Math.round(util), power: gpu.power_limit && gpu.power != null ? Math.round(Math.max(0, Math.min(1, gpu.power / gpu.power_limit)) * 100) / 100 : undefined });
      card.vram.textContent = `${ByteFormat.gb(gpu.mem_used * 1048576)} / ${ByteFormat.gb(gpu.mem_total * 1048576)} GB`;

      const holders = gpu.holders.join(", ");
      card.who.textContent = holders || (gpu.stale ? "stale memory" : "idle");
      card.who.className = `gcard__who${holders ? (gpu.others ? " is-others" : " is-mine") : gpu.stale ? " is-stale" : ""}`;

      const recent = history[String(gpu.index)] || { util: [], mem: [] };
      Spark.draw(card.graph, [{ data: recent.mem, color: SERIES.ram, fill: 0.08 }, { data: recent.util, color: SERIES.cpu, fill: 0.14 }], this.histMax);
      const hour = this.states.hostHour[String(gpu.index)] || [];
      if (hour.length > 1) Spark.draw(card.health, [{ data: hour.map((b) => b.temp || 0), color: SERIES.temp, fill: 0.05 }, { data: hour.map((b) => (gpu.power_limit ? (100 * (b.power || 0)) / gpu.power_limit : b.power || 0)), color: SERIES.power, fill: 0.05 }], 120);
    });
    const devices = sys.gpus.map((gpu) => ({ holder: gpu.mine ? "mine" : gpu.others ? "others" : gpu.stale ? "stale" : "idle", util: Math.round(gpu.util || 0) }));
    if (devices.length) FigureStates.write(this.node("gpu-shelf"), { devices });
  }

  renderCpu(sys) {
    const cpu = sys.cpu;
    this.cpuDial.set(cpu.total);
    ["load1", "load5", "load15"].forEach((role, i) => this.text(role, cpu.load[i].toFixed(1)));
    FigureStates.write(this.node("cores-figure"), { cores: cpu.cores.map((u) => Math.round(u)) });
    const avg = cpu.cores.reduce((sum, u) => sum + u, 0) / cpu.cores.length;
    const active = cpu.cores.filter((u) => u >= 50).length;
    this.bar("cpu-bar", avg);
    this.bar("cores-bar", (100 * active) / cpu.cores.length);
    this.text("cpu-avg", `${avg.toFixed(1)} %`);
    this.text("cpu-active", `${active} / ${cpu.cores.length}`);
    Spark.draw(this.node("cpu-graph"), [{ data: sys.history.cpu, color: SERIES.cpu, fill: 0.14 }], this.histMax);
  }

  renderResources(sys) {
    const mem = sys.mem;
    const disk = sys.disk;
    const used = mem.total - mem.available;
    this.bar("ram-bar", (100 * used) / mem.total);
    this.text("ram", `${ByteFormat.gb(used)} / ${ByteFormat.gb(mem.total)} GB`);
    this.text("mem-total", ByteFormat.size(mem.total));
    const swapUsed = mem.swap_total - mem.swap_free;
    this.bar("swap-bar", mem.swap_total ? (100 * swapUsed) / mem.swap_total : 0);
    this.text("swap", mem.swap_total ? `${ByteFormat.gb(swapUsed)} / ${ByteFormat.gb(mem.swap_total)} GB` : "none");
    Spark.draw(this.node("mem-graph"), [{ data: sys.history.ram, color: SERIES.ram, fill: 0.1 }], this.histMax);

    this.bar("disk-bar", (100 * disk.used) / disk.total);
    this.text("disk-label", disk.path);
    this.node("disk-label").title = disk.path;
    this.text("disk", `${ByteFormat.size(disk.used)} / ${ByteFormat.size(disk.total)}`);
    this.text("disk-free", ByteFormat.size(disk.free));
    this.host.querySelector(".sb-card--res .panel__note").textContent = ByteFormat.size(disk.total);
    this.text("disk-repo", disk.repo_used != null ? ByteFormat.size(disk.repo_used) : "measuring");
    this.bar("disk-repo-bar", disk.repo_used ? (100 * disk.repo_used) / disk.total : 0);
    this.text("disk-runs", disk.runs_used != null ? ByteFormat.size(disk.runs_used) : "measuring");
    this.bar("disk-runs-bar", disk.runs_used ? (100 * disk.runs_used) / disk.total : 0);
  }

  renderProcesses(procs) {
    const body = this.node("procs");
    this.host.querySelector(".sb-card--procs .panel__note").textContent = `${procs.length} process${procs.length === 1 ? "" : "es"}`;
    if (!procs.length) {
      body.innerHTML = `<tr><td colspan="7" class="sb-empty">No processes.</td></tr>`;
      return;
    }
    body.innerHTML = procs.map((proc) => {
      const tone = proc.cpu >= 100 ? " is-hot" : proc.cpu >= 25 ? " is-mid" : "";
      return `<tr${proc.state === "R" ? ' class="is-run"' : ""}><td class="num">${proc.pid}</td><td class="num sb-cpuval${tone}">${proc.cpu.toFixed(1)}</td><td class="num">${ByteFormat.mb(proc.pss != null ? proc.pss : proc.rss)}</td><td class="num">${proc.gpu ? ByteFormat.mb(proc.gpu * 1048576) : "--"}</td><td class="num">${proc.threads}</td><td class="mono">${esc(proc.state)}</td><td class="mono sb-cmd-cell" title="${esc(proc.cmd)}">${esc(proc.cmd)}</td></tr>`;
    }).join("");
  }

  renderServices(health) {
    if (!this.built) return;
    const row = (name, key) => `<b>${name}</b><span>${esc(health[`${key}_url`] || "")}</span>${statusChip(health[key] ? "UP" : "DOWN")}`;
    this.node("svc-rows").innerHTML = row("OSRM", "osrm") + row("VROOM", "vroom");
    const down = !health.osrm || !health.vroom;
    this.text("svc-text", health.unreachable ? "The console backend did not answer." : down ? "Recording, solving and teacher verdicts are disabled. Start the services from the repository root with:" : "Both services answer. They were started from the repository root with:");
    FigureStates.write(this.node("svc-figure"), { osrm: Boolean(health.osrm), vroom: Boolean(health.vroom), reachable: !health.unreachable });
  }

  async pollJobs() {
    try {
      this.jobs = (await api.get("/api/jobs")).jobs;
    } catch (error) {
      return;
    }
    if (this.built) this.renderJobs(this.jobs);
  }

  renderJobs(jobs) {
    const running = jobs.filter((job) => job.status === "running").length;
    this.host.querySelector(".sb-card--jobs .panel__note").textContent = running ? `${running} running` : plural(jobs.length, "job");
    const waiting = jobs.filter((job) => ["queued", "pending", "scheduled"].includes(job.status));
    FigureStates.write(this.node("jobs-figure"), { jobs: jobs.filter((job) => job.status === "running").map((job) => ({ id: job.job_id, running: true, position: null, of: null })).concat(waiting.map((job, i) => ({ id: job.job_id, running: false, position: i + 1, of: waiting.length }))) });

    const list = this.node("jobs");
    if (!jobs.length) {
      list.innerHTML = `<li class="sb-empty">No jobs yet. Launch a script to see it here.</li>`;
      return;
    }
    const followers = new Map(jobs.filter((job) => job.follow_of).map((job) => [job.follow_of, job]));
    list.innerHTML = jobs.filter((job) => !job.follow_of).slice(0, 8).map((job) => {
      const next = followers.get(job.job_id);
      return this.jobRow(job, false) + (next ? this.jobRow(next, true) : "");
    }).join("");
  }

  jobRow(job, follow) {
    const word = job.status === "failed" && job.stopped ? "stopped" : job.status;
    const orb = job.status === "running" ? Orb.html("working", 20, "running") : "";
    const meta = [job.started.replace("T", " ").slice(5, 16), ByteFormat.energy(job.energy_wh), job.description].filter(Boolean).join(" · ");
    return `<li class="sb-job${follow ? " sb-job--follow" : ""}"><a href="#/console/${esc(job.job_id)}" title="${esc(job.command)}">${follow ? icon("then", "sb-job__then") : ""}<span class="sb-job__text"><span class="sb-job__name">${esc(job.title || job.script)}</span><span class="sb-job__meta">${esc(meta)}</span></span><span class="sb-job__state">${orb}${statusChip(word)}</span></a></li>`;
  }

  async pollLog() {
    let lines;
    try {
      lines = (await api.get("/api/server-log?limit=150")).lines;
    } catch (error) {
      return;
    }
    const box = this.node("log");
    if (!box) return;
    this.host.querySelector(".sb-card--log .panel__note").textContent = plural(lines.length, "recent line");
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    box.innerHTML = lines.length ? lines.map((line) => `<div class="sb-logline is-${esc(line.level.toLowerCase())}"><span class="sb-logline__time">${esc(line.time)}</span><span class="sb-logline__lvl">${esc(line.level)}</span><span class="sb-logline__msg">${esc(line.message)}</span></div>`).join("") : `<div class="sb-empty">The server has logged nothing yet.</div>`;
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  async pollActivity() {
    const range = this.ranges.activity;
    let data;
    try {
      data = await api.get(`/api/gpu-activity?range=${range}`);
    } catch (error) {
      return;
    }
    if (range !== this.ranges.activity) return;
    this.states.activity = data;
    this.renderActivity();
  }

  renderActivity() {
    const state = this.states.activity;
    if (!this.built || !state) return;
    const blocks = state.bucket_s >= 3600 ? `${Math.round(state.bucket_s / 3600)} h blocks` : `${Math.round(state.bucket_s / 60)} min blocks`;
    const note = this.host.querySelector(".sb-card--act .panel__note");
    note.textContent = blocks;
    note.title = `${plural(state.gpus.length, "device")}, ${blocks}`;
    this.activity.render(state);
  }

  async pollHost() {
    const range = this.ranges.host;
    let data;
    try {
      data = await api.get(`/api/host-history?range=${range}`);
    } catch (error) {
      return;
    }
    if (range !== this.ranges.host) return;
    this.states.host = data;
    if (range === "hour") this.keepHour(data);
    this.renderHost();
  }

  async pollHostHour() {
    try {
      this.keepHour(await api.get("/api/host-history?range=hour"));
    } catch (error) {
      return;
    }
  }

  keepHour(data) {
    this.states.hostHour = {};
    for (const gpu of data.gpus) this.states.hostHour[String(gpu.index)] = gpu.buckets;
  }

  renderHost() {
    const state = this.states.host;
    if (!this.built || !state) return;
    this.host.querySelector(".sb-card--host .panel__note").textContent = plural(state.host.length, "point");
    TimeStrip.draw(this.node("hh-cpu"), state, [
      { points: TimeStrip.points(state.host, "cpu"), color: SERIES.cpu, label: "cpu" },
      { points: TimeStrip.points(state.host, "ram"), color: SERIES.ram, label: "ram" },
    ], { yMax: 100, unit: "%" });
    const netMax = TimeStrip.niceMax(Math.max(0.5, ...state.host.map((s) => Math.max(s.rx || 0, s.tx || 0))));
    TimeStrip.draw(this.node("hh-net"), state, [
      { points: TimeStrip.points(state.host, "rx"), color: SERIES.rx, label: "in" },
      { points: TimeStrip.points(state.host, "tx"), color: SERIES.tx, label: "out" },
    ], { yMax: netMax, unit: " MB/s" });
  }

  async pollHeatmap() {
    try {
      this.states.heat = await api.get("/api/gpu-activity?range=month");
    } catch (error) {
      return;
    }
    this.renderHeatmap();
  }

  renderHeatmap() {
    const state = this.states.heat;
    if (!this.built || !state) return;
    this.host.querySelector(".sb-card--heat .panel__note").textContent = `${plural(state.gpus.length, "device")} · 4 h cells`;
    this.heatmap.render(state);
    if (state.gpus.length) FigureStates.write(this.node("heat-figure"), { days: this.heatmap.figureDays() });
  }

  redrawCharts() {
    if (!this.built) return;
    this.activity.draw();
    this.heatmap.draw();
    this.renderHost();
  }

  async nuke() {
    if (!window.confirm("Nuke jobs: this stops every console job and every project entry script running under your user on this machine, and clears the launch queue. The console itself is spared. Continue?")) return;
    const button = this.node("nuke");
    button.disabled = true;
    this.fireNuke(true);
    try {
      const res = await api.post("/api/system/nuke");
      this.app.toast.show(`Nuke: cleared ${res.cancelled} queued launch${res.cancelled === 1 ? "" : "es"}, terminated ${res.signalled}, force-killed ${res.killed}`, "ok");
      this.app.pollJobs();
      this.pollJobs();
    } catch (error) {
      this.app.toast.show(`Nuke failed: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      setTimeout(() => this.fireNuke(false), 2400);
    }
  }

  fireNuke(on) {
    FigureStates.write(this.node("nuke-figure"), { firing: on });
    this.node("nuke-light").className = `sb-light${on ? " is-alert" : ""}`;
    this.text("nuke-mode", on ? "firing" : "ready");
  }

  async killFrontEnd() {
    if (!window.confirm("Kill front end: this stops every console server of this repository, this one included. Running and detached jobs keep going and are adopted again on the next start; queued launches are lost. Continue?")) return;
    const button = this.node("kill");
    button.disabled = true;
    try {
      const res = await api.post("/api/system/shutdown");
      this.down = true;
      FigureStates.write(this.node("kill-figure"), { down: true });
      button.innerHTML = `${icon("stop")}<span>Front end down</span>`;
      const others = res.siblings.length ? `, plus ${plural(res.siblings.length, "other instance")}: pids ${res.siblings.join(", ")}` : "";
      this.app.toast.show(`Front end stopped (pid ${res.pid}${others}). Running jobs keep going; restart it with webui/run.sh.`, "ok");
    } catch (error) {
      this.app.toast.show(`Shutdown failed: ${error.message}`, "error");
      button.disabled = false;
    }
  }

  async detach() {
    const button = this.node("detach");
    button.disabled = true;
    try {
      const res = await api.post("/api/system/detach");
      if (res.shield_error) this.app.toast.show(`Console detached as pid ${res.pid}, but the session shield is incomplete: ${res.shield_error}`, "warn");
      else this.app.toast.show(`Console detached into ${res.scope} as pid ${res.pid}; output continues in ${res.log_path}`, "ok");
      this.renderServer(res);
    } catch (error) {
      this.app.toast.show(`Detach failed: ${error.message}`, "error");
      button.disabled = false;
    }
  }

  async armNuke(button) {
    const arming = !this.autoNuke;
    if (arming && !window.confirm("Arm auto-nuke: once a danger alert you cause has stood for about 30 s (RAM at the kill limit with most of it yours), this stops every console job and every project entry script running under your user, and clears the launch queue. The console itself is spared. Arm it?")) return;
    button.disabled = true;
    try {
      const res = await api.post("/api/impact/arm", { armed: arming });
      this.renderAutoNuke(res.auto_nuke);
      this.app.toast.show(`Auto-nuke ${res.auto_nuke ? "armed" : "disarmed"}`, res.auto_nuke ? "warn" : "ok");
    } catch (error) {
      this.app.toast.show(`Auto-nuke toggle failed: ${error.message}`, "error");
    } finally {
      button.disabled = false;
    }
  }

  async sendPoliteness(payload, control) {
    control.disabled = true;
    try {
      const res = await api.post("/api/impact/politeness", payload);
      this.renderPoliteness(res);
      this.app.toast.show(`Launch politeness ${res.level} (${res.label}), auto-throttle ${res.auto_throttle ? "on" : "off"}`, "ok");
    } catch (error) {
      this.app.toast.show(`Politeness update failed: ${error.message}`, "error");
    } finally {
      control.disabled = false;
    }
  }

  every(ms, task) {
    this.timers.push(setInterval(() => {
      if (document.visibilityState === "visible") task();
    }, ms));
  }

  stop() {
    for (const timer of this.timers) clearInterval(timer);
    this.timers = [];
    this.app.healthListeners = this.app.healthListeners.filter((listener) => listener !== this.healthListener);
  }

  start() {
    this.stop();
    this.poll();
    this.pollJobs();
    this.pollLog();
    this.pollHostHour();
    this.pollHost();
    this.pollActivity();
    this.pollHeatmap();
    this.app.onHealth(this.healthListener);
    this.every(StatusBoard.POLL_MS, () => this.poll());
    this.every(StatusBoard.CADENCE.jobs, () => this.pollJobs());
    this.every(StatusBoard.CADENCE.log, () => this.pollLog());
    this.every(StatusBoard.CADENCE.hostHour, () => this.pollHostHour());
    this.every(StatusBoard.CADENCE.host, () => this.pollHost());
    this.every(StatusBoard.CADENCE.activity, () => this.pollActivity());
    this.every(StatusBoard.CADENCE.heatmap, () => this.pollHeatmap());
  }
}
