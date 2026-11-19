import { api } from "../api.js";
import { esc } from "../format.js";
import { Orb } from "../fx/orb.js";
import { LogText } from "../ui/logtext.js";
import { statusChip } from "../ui/status.js";
import { StreamHub } from "./stream_hub.js";

export class TileLogPanel {
  static ERROR_RE = /Traceback \(most recent call last\)|^\s*[\w.]*(?:Error|Exception|Interrupt)(?::|$)|\bERROR\b|\bCRITICAL\b|\bFAILED\b|CUDA out of memory|Segmentation fault|core dumped/;

  constructor(tile) {
    this.tile = tile;
    this.hits = [];
    this.index = 0;
    this.lines = [];
    this.marks = [];
    this.markIndex = -1;

    this.el = document.createElement("div");
    this.el.className = "console-tile__log";
    this.el.hidden = true;
    this.el.innerHTML = `
      <div class="console-tile__loghead">
        <span class="console-tile__lognote" data-role="note"></span>
        <label class="console-tile__search"><input class="input" type="search" placeholder="Search" spellcheck="false" aria-label="Search the transcript of ${esc(tile.job.title)}" data-role="search"><span class="console-tile__matches" data-role="matches"></span></label>
        <button type="button" class="btn btn--sm" data-role="jump">Next error</button>
        <button type="button" class="btn btn--sm" data-role="reload">Reload</button>
      </div>
      <pre class="console-tile__logbody" data-role="body" tabindex="0"></pre>`;

    this.note = this.el.querySelector("[data-role=note]");
    this.search = this.el.querySelector("[data-role=search]");
    this.matches = this.el.querySelector("[data-role=matches]");
    this.jump = this.el.querySelector("[data-role=jump]");
    this.body = this.el.querySelector("[data-role=body]");

    this.jump.addEventListener("click", () => this.jumpError());
    this.el.querySelector("[data-role=reload]").addEventListener("click", () => this.load());
    this.search.addEventListener("input", () => this.highlight(true));
    this.search.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      this.nextMatch();
    });
  }

  get showing() {
    return !this.el.hidden;
  }

  toggle() {
    const showing = this.el.hidden;

    this.el.hidden = !showing;
    this.tile.logBtn.setAttribute("aria-pressed", String(showing));
    this.tile.showPanels();

    if (showing) this.load();
    else this.tile.fit();
  }

  async load() {
    this.note.innerHTML = `${Orb.html("listening", 20)}<span>Reading</span>`;

    let data;
    try {
      data = await api.get(`/api/jobs/${this.tile.job.job_id}/log`);
    } catch (error) {
      this.fail(error.status === 0 ? "backend unreachable" : error.message);
      return;
    }

    if (!data.available) {
      this.fail(data.reason);
      return;
    }

    this.paint(data);
  }

  fail(note) {
    this.body.textContent = "";
    this.lines = [];
    this.note.textContent = note;
    this.jump.disabled = true;
    this.matches.textContent = "";
  }

  paint(data) {
    this.lines = data.text.split("\n");
    this.render();

    const bits = [data.path, `${Math.round(data.size / 1024)} KB`];
    if (data.truncated) bits.push(`last ${Math.round(data.shown / 1024)} KB shown`);
    bits.push(this.hits.length ? `${this.hits.length} error lines` : "no error lines");

    this.note.textContent = bits.join(" · ");
    this.note.title = data.path;
    this.jump.disabled = !this.hits.length;
    this.index = this.hits.length - 1;

    if (this.marks.length) this.marks[0].scrollIntoView({ block: "center" });
    else if (this.hits.length) this.hits[this.index].scrollIntoView({ block: "center" });
    else this.body.scrollTop = this.body.scrollHeight;
  }

  render() {
    const query = this.search.value.trim();
    const parts = [];
    let count = 0;

    this.lines.forEach((line, row) => {
      const error = TileLogPanel.ERROR_RE.test(line);
      const marked = query ? LogText.html(line, query) : { html: esc(line), count: 0 };
      count += marked.count;
      parts.push(`<span class="console-tile__logline${error ? " is-error" : ""}" data-row="${row}">${marked.html || " "}</span>`);
    });

    this.body.innerHTML = parts.join("");
    this.hits = [...this.body.querySelectorAll(".is-error")];
    this.marks = [...this.body.querySelectorAll("mark")];
    this.markIndex = this.marks.length ? 0 : -1;
    this.matches.textContent = query ? `${count} ${count === 1 ? "match" : "matches"}` : "";
    this.marks.forEach((mark, index) => mark.classList.toggle("is-current", index === 0));
  }

  highlight(scroll) {
    this.render();
    if (scroll && this.marks.length) this.marks[0].scrollIntoView({ block: "center" });
  }

  nextMatch() {
    if (!this.marks.length) return;
    this.marks[this.markIndex].classList.remove("is-current");
    this.markIndex = (this.markIndex + 1) % this.marks.length;
    this.marks[this.markIndex].classList.add("is-current");
    this.marks[this.markIndex].scrollIntoView({ block: "center" });
  }

  jumpError() {
    if (!this.hits.length) return;
    this.index = (this.index + 1) % this.hits.length;
    this.hits[this.index].scrollIntoView({ block: "center" });
  }
}

export class TerminalGauge {
  static PASSES = 3;

  constructor(options) {
    this.host = document.createElement("div");
    this.host.className = "console-tile__out console-gauge";
    this.host.setAttribute("aria-hidden", "true");
    this.host.style.cssText = "position:absolute; left:-10000px; top:0; visibility:hidden; pointer-events:none;";
    document.body.appendChild(this.host);

    this.term = new Terminal({ ...options, cols: 2, rows: 2 });
    this.fit = new FitAddon.FitAddon();
    this.term.loadAddon(this.fit);
    this.term.open(this.host);
  }

  propose(font, width, height) {
    this.host.style.width = `${width}px`;
    this.host.style.height = `${height}px`;
    if (this.term.options.fontSize !== font) this.term.options.fontSize = font;

    let dims = this.fit.proposeDimensions();
    for (let pass = 0; pass < TerminalGauge.PASSES && dims && dims.cols && dims.rows; pass++) {
      const cols = Math.min(ConsoleTile.MAX_COLS, dims.cols);
      const rows = Math.max(2, dims.rows);
      if (this.term.cols === cols && this.term.rows === rows) break;
      this.term.resize(cols, rows);
      dims = this.fit.proposeDimensions();
    }
    return dims;
  }
}

export class ConsoleTile {
  static LIVE_STATUSES = new Set(["pending", "running", "queued", "scheduled"]);
  static MAX_COLS = 120;
  static FONT_STEP = 0.25;

  static terminalOptions() {
    const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return {
      cols: 120,
      rows: 24,
      convertEol: true,
      disableStdin: true,
      cursorBlink: false,
      cursorInactiveStyle: "none",
      scrollback: 10000,
      allowTransparency: true,
      fontFamily: '"JetBrains Mono", "JetBrains Mono Log", ui-monospace, monospace',
      fontSize: 12,
      fontWeight: 600,
      fontWeightBold: 800,
      lineHeight: 1.25,
      minimumContrastRatio: 7,
      theme: {
        background: "rgba(0, 0, 0, 0)",
        foreground: token("--ink-dim"),
        cursor: "rgba(0, 0, 0, 0)",
        selectionBackground: "rgba(163, 192, 255, 0.28)",
        black: token("--ink-hint"),
        red: token("--fail"),
        green: token("--pass"),
        yellow: token("--warn"),
        blue: token("--accent"),
        magenta: token("--lane-teacher"),
        cyan: token("--live"),
        white: token("--ink-dim"),
        brightBlack: token("--ink-faint"),
        brightRed: token("--fail-ink"),
        brightGreen: "#bbf7d0",
        brightYellow: "#fef08a",
        brightBlue: token("--accent-deep"),
        brightMagenta: "#f5d0fe",
        brightCyan: "#a5f3fc",
        brightWhite: token("--ink"),
      },
    };
  }

  static quantize(font) {
    return Math.round(font / ConsoleTile.FONT_STEP) * ConsoleTile.FONT_STEP;
  }

  constructor(job, manager, host) {
    this.job = job;
    this.manager = manager;
    this.sub = null;
    this.opened = false;
    this.desc = job.description || "";
    this.streamLost = false;

    this.root = document.createElement("article");
    this.root.className = "console-tile";
    this.root.dataset.job = job.job_id;
    this.root.innerHTML = `
      <header class="console-tile__bar">
        <span class="console-tile__name" title="${esc(job.command)}">${esc(job.script)}</span>
        <span class="console-tile__meta" data-role="meta"></span>
        <span data-role="state"></span>
        <a class="btn btn--sm" data-role="run" hidden>Open run</a>
        <button type="button" class="btn btn--sm" data-role="log" aria-pressed="false" title="Read the log file as plain text, live frames flattened, jumped to the last error">.out</button>
        <button type="button" class="btn btn--sm btn--danger" data-role="stop">Stop</button>
        <button type="button" class="btn btn--sm" data-role="retry" title="Reopen the launch page with this job's overrides, interpreter and run mode filled in">Retry</button>
        <button type="button" class="btn btn--sm" data-role="close">Close</button>
      </header>`;

    this.metaEl = this.root.querySelector("[data-role=meta]");
    this.stateEl = this.root.querySelector("[data-role=state]");
    this.runBtn = this.root.querySelector("[data-role=run]");
    this.logBtn = this.root.querySelector("[data-role=log]");
    this.stopBtn = this.root.querySelector("[data-role=stop]");
    this.retryBtn = this.root.querySelector("[data-role=retry]");

    this.metaEl.textContent = this.meta(this.baseMeta(job));
    if (this.desc) this.metaEl.title = this.desc;

    this.logBtn.addEventListener("click", () => this.log.toggle());
    this.stopBtn.addEventListener("click", () => this.stop());
    this.retryBtn.addEventListener("click", () => this.manager.retry(this.job));
    this.root.querySelector("[data-role=close]").addEventListener("click", () => this.manager.close(this.job.job_id));

    this.log = new TileLogPanel(this);
    this.outEl = document.createElement("div");
    this.outEl.className = "console-tile__out";
    this.root.append(this.log.el, this.outEl);
    host.appendChild(this.root);

    this.term = new Terminal(ConsoleTile.terminalOptions());
    this.fitAddon = new FitAddon.FitAddon();
    this.term.loadAddon(this.fitAddon);

    this.resizer = new ResizeObserver(() => this.manager.queueFit());
    this.resizer.observe(this.outEl);

    this.logBtn.hidden = !job.log_path;
    this.setStatus(job.status);
    this.connect();
  }

  connect() {
    this.sub = StreamHub.instance().subscribe(this.job.job_id, null, (data) => this.onEvent(data));
  }

  get streamOpen() {
    return this.sub !== null && !this.sub.closed && StreamHub.instance().connected;
  }

  disconnect() {
    if (!this.sub) return;
    this.sub.close();
    this.sub = null;
  }

  linkEvent(data) {
    if (data.type === "link") {
      if (data.state === "lost" && !this.streamLost) {
        this.streamLost = true;
        this.note("live stream interrupted, reconnecting; the job list keeps the status current meanwhile", "33");
      }
      return true;
    }
    if (data.type === "truncated") {
      this.note(`stream truncated: ${data.dropped} event(s) dropped, this page could not keep up; the on-disk log is complete`, "31");
      return true;
    }
    return false;
  }

  onEvent(data) {
    if (this.linkEvent(data)) return;
    this.streamLost = false;

    if (data.type === "unavailable") {
      this.note(`live stream unavailable: ${data.error}`, "33");
      this.disconnect();
    } else if (data.type === "chunk") {
      this.term.write(data.data);
    } else if (data.type === "line") {
      this.term.writeln(data.text);
    } else if (data.type === "status") {
      this.onStatus(data);
    } else if (data.type === "end") {
      this.disconnect();
    }
  }

  onStatus(data) {
    if (data.status === "running") {
      this.note(this.runNote(data), data.adopted && !data.log ? "33" : "36");
      this.metaEl.textContent = this.meta(data.adopted ? `pid ${data.pid} · adopted` : data.detached ? `pid ${data.pid} · detached` : `pid ${data.pid}`);
      if (data.log) {
        this.job.log_path = data.log;
        this.logBtn.hidden = false;
      }
      this.setStatus("running");
    } else if (data.status === "scheduled") {
      this.note(`scheduled to run after ${data.after || "the current job"}`, "33");
    } else if (data.status === "queued") {
      this.note(`queued at position ${data.position} · ${this.queueBlocker(data.blocker)}`, "33");
      this.setStatus("queued");
    } else if (data.status === "cancelled") {
      this.note("cancelled before start", "33");
      this.setStatus("cancelled");
      this.manager.refresh();
    } else {
      this.note(`process ${data.status} (exit ${data.code})`, data.code === 0 ? "36" : "31");
      this.setStatus(data.status);
      this.manager.refresh();
    }
  }

  queueBlocker(blocker) {
    if (!blocker) return "starts when the previous job ends";
    const what = blocker.adopted ? "adopted process" : "job";
    const pid = blocker.pid ? ` (pid ${blocker.pid})` : "";
    return `held by ${what} ${blocker.script}${pid} · starts when it ends`;
  }

  runNote(data) {
    const kind = data.adopted ? "adopted process already running under your user" : data.detached ? "running detached" : "process started";
    if (!data.adopted && !data.detached) return `${kind} (pid ${data.pid})`;
    if (data.log) return `${kind} (pid ${data.pid}) · following ${data.log}, stop still kills it`;
    return `${kind} (pid ${data.pid}) · output is not captured, stop still kills it`;
  }

  meta(base) {
    return this.desc ? `${base} · ${this.desc}` : base;
  }

  baseMeta(job) {
    const bits = [job.pid ? `pid ${job.pid}` : "not started"];
    if (job.adopted) bits.push("adopted");
    else if (job.detach) bits.push("detached");
    return bits.join(" · ");
  }

  stop() {
    if (this.job.status === "running") {
      const seconds = Math.max(0, (Date.now() - Date.parse(this.job.started)) / 1000);
      const elapsed = seconds >= 3600 ? `${Math.floor(seconds / 3600)} h ${Math.round((seconds % 3600) / 60)} min` : `${Math.round(seconds / 60)} min`;
      if (!window.confirm(`Stop ${this.job.script}?\n\nIt has been running for ${elapsed}. The process group is signalled at once and everything since the last checkpoint is lost.`)) return;
    }
    this.manager.stop(this.job.job_id);
  }

  note(text, color) {
    this.term.write(`\r\n\x1b[2;${color}m${text}\x1b[0m\r\n`);
  }

  setStatus(status) {
    this.job.status = status;
    const word = status === "failed" && this.job.stopped ? "stopped" : status;
    this.stateEl.innerHTML = statusChip(word);
    this.root.dataset.status = status;
    this.stopBtn.textContent = status === "scheduled" || status === "queued" ? "Cancel" : "Stop";
    this.stopBtn.disabled = !["running", "scheduled", "queued", "pending"].includes(status);
    this.retryBtn.hidden = ConsoleTile.LIVE_STATUSES.has(status) || status === "untracked";
    this.showRun();
  }

  showRun() {
    const run = this.job.overrides ? this.job.overrides["io.run_name"] : null;
    const shown = this.job.status === "finished" && Boolean(run) && this.manager.runNames.has(run);
    this.runBtn.hidden = !shown;
    if (shown) this.runBtn.href = `#/runs/${encodeURIComponent(run)}/summary`;
  }

  syncFromList(job) {
    if (job.pid && job.pid !== this.job.pid) {
      this.job.pid = job.pid;
      this.metaEl.textContent = this.meta(this.baseMeta(job));
    }
    if (job.description && job.description !== this.desc) {
      this.desc = job.description;
      this.metaEl.textContent = this.meta(this.baseMeta(job));
      this.metaEl.title = this.desc;
    }
    if (job.log_path && !this.job.log_path) {
      this.job.log_path = job.log_path;
      this.logBtn.hidden = false;
    }
    this.job.stopped = job.stopped;
    this.showRun();
    if (job.status === this.job.status || this.streamOpen) return;

    this.note(`process ${job.status}, reported by the job list; the live stream missed it`, job.status === "finished" ? "36" : "31");
    this.setStatus(job.status);
  }

  markUntracked() {
    if (this.job.status === "untracked") return;

    this.disconnect();
    this.note("the console no longer tracks this job (it restarted or pruned its history); a process still running reappears in the job list as an adopted job", "33");
    this.setStatus("untracked");
  }

  showPanels() {
    this.outEl.hidden = this.log.showing;
  }

  fit() {
    if (this.log.showing) return;
    this.fitInto(this.term, this.fitAddon, this.outEl);
  }

  fitInto(term, fitAddon, host) {
    const w = host.clientWidth;
    const h = host.clientHeight;
    if (!w || !h) return;

    const gauge = this.manager.gauge;
    let font = ConsoleTile.quantize(Math.max(8, Math.min(16, (w - 16) / 72)));
    let dims = gauge.propose(font, w, h);
    if (dims && dims.cols && dims.cols < ConsoleTile.MAX_COLS) {
      font = ConsoleTile.quantize(Math.max(7, (font * dims.cols) / ConsoleTile.MAX_COLS));
      dims = gauge.propose(font, w, h);
    }
    if (!dims || !dims.cols || !dims.rows) return;

    if (term.options.fontSize !== font) term.options.fontSize = font;
    term.resize(Math.min(ConsoleTile.MAX_COLS, dims.cols), Math.max(2, dims.rows));

    if (!this.opened) {
      term.open(host);
      this.opened = true;
    }

    const check = fitAddon.proposeDimensions();
    if (check && check.cols && check.rows) term.resize(Math.min(ConsoleTile.MAX_COLS, check.cols), Math.max(2, check.rows));
    term.scrollToBottom();
  }

  dispose() {
    this.resizer.disconnect();
    this.disconnect();
    this.term.dispose();
    this.root.remove();
  }
}
