import { api } from "../api.js";
import { esc, icon, Store } from "../format.js";
import { infoText } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { skeleton } from "../ui/status.js";
import { ConfigForm } from "./config_form.js";
import { PythonLiteral } from "./python_literal.js";
import { ConfigTemplates, RunConfigCopy } from "./rail.js";

export class SequenceBoard {
  static STATUSES = ["idle", "pending", "held", "launching", "running", "done", "failed", "cancelled", "skipped", "bypassed"];
  static ACTIVE = ["launching", "running"];
  static ORB_STATES = { launching: "connecting", running: "working" };
  static SETTLED = ["done", "skipped", "bypassed"];
  static BROKEN = ["failed", "cancelled"];
  static SVG_NS = "http://www.w3.org/2000/svg";
  static MARGIN = 48;
  static GRID = 24;
  static PORT_Y = 34;

  constructor(host, callbacks) {
    this.host = host;
    this.callbacks = callbacks;
    this.surface = null;
    this.svg = null;
    this.entry = null;
    this.live = null;
    this.selected = "";
    this.linking = "";
    this.drag = null;
    this.dragUsed = false;
  }

  render(entry, live, selected, linking) {
    this.entry = entry;
    this.live = live;
    this.selected = selected;
    this.linking = linking;
    this.host.innerHTML = "";

    const surface = document.createElement("div");
    surface.className = "seq-surface";
    this.surface = surface;

    const svg = document.createElementNS(SequenceBoard.SVG_NS, "svg");
    svg.classList.add("seq-edges");
    this.svg = svg;

    surface.appendChild(svg);
    surface.appendChild(this.startBlock());
    entry.blocks.forEach((block) => surface.appendChild(this.block(block)));

    surface.addEventListener("click", (event) => {
      if (this.dragUsed) {
        this.dragUsed = false;
        return;
      }
      if (event.target === surface && this.callbacks.onBackground) this.callbacks.onBackground();
    });

    this.host.appendChild(surface);
    this.redraw();
  }

  redraw() {
    if (!this.surface || !this.surface.isConnected) return;
    this.resize();
    this.drawEdges();
  }

  resize() {
    const blocks = [...this.surface.querySelectorAll(".seq-block")];
    const right = Math.max(...blocks.map((el) => el.offsetLeft + el.offsetWidth), 0);
    const bottom = Math.max(...blocks.map((el) => el.offsetTop + el.offsetHeight), 0);
    this.surface.style.width = `${Math.max(right + SequenceBoard.MARGIN, this.host.clientWidth)}px`;
    this.surface.style.height = `${Math.max(bottom + SequenceBoard.MARGIN, 360)}px`;
  }

  paintStates(live) {
    this.live = live;
    if (!this.surface) return;

    this.surface.querySelectorAll(".seq-block[data-block]").forEach((el) => {
      const state = this.state(el.dataset.block);
      SequenceBoard.STATUSES.forEach((status) => el.classList.toggle(`is-${status}`, state.status === status));
      el.querySelector(".seq-block__state").innerHTML = this.stateHtml(state.status);
      el.querySelector("[data-action=release]").hidden = state.status !== "held";
      el.querySelector("[data-action=job]").hidden = !state.job_id;
      const error = el.querySelector(".seq-block__error");
      error.hidden = !state.error;
      error.textContent = state.error || "";
      error.title = state.error || "";
    });
    this.paintEdgeStates();
  }

  stateHtml(status) {
    const tone = { done: "pass", running: "live", launching: "live", failed: "fail", cancelled: "warn", skipped: "warn", held: "warn" }[status] || "none";
    return `<span class="state state--${tone}"><i></i>${esc(status)}</span>`;
  }

  state(blockId) {
    const state = this.live && this.live.blocks ? this.live.blocks[blockId] : null;
    return state || { status: "idle", job_id: null, error: "" };
  }

  startBlock() {
    const el = document.createElement("div");
    el.className = "seq-block seq-block--start";
    el.style.left = "24px";
    el.style.top = "24px";
    el.innerHTML = `<div class="seq-block__head"><span class="seq-block__kind">start</span></div><h4 class="seq-block__title">Start</h4><p class="seq-block__hint">blocks run with their own settings; scenarios are sampled inline</p>`;
    return el;
  }

  static axes(count) {
    return `${count} ${count === 1 ? "axis" : "axes"}`;
  }

  block(block) {
    const state = this.state(block.block_id);
    const overrides = Object.keys(block.overrides).length;
    const open = block.decisions.filter((decision) => !decision.choice).length;
    const el = document.createElement("div");
    el.className = `seq-block is-${state.status}${block.skip ? " is-skip" : ""}${block.checkpoint ? " is-checkpoint" : ""}${this.selected === block.block_id ? " is-selected" : ""}${this.linking === block.block_id ? " is-linking" : ""}`;
    el.dataset.block = block.block_id;
    el.style.left = `${block.x}px`;
    el.style.top = `${block.y}px`;
    el.tabIndex = 0;

    el.innerHTML = `
      <div class="seq-block__head"><span class="seq-block__kind">run</span><span class="seq-block__state">${this.stateHtml(state.status)}</span></div>
      <h4 class="seq-block__title" title="${esc(block.script)}">${esc(block.name || block.title)}</h4>
      <p class="seq-block__meta">${esc(block.script)} · ${overrides} override${overrides === 1 ? "" : "s"}</p>
      ${block.decisions.length ? `<p class="seq-block__flag seq-block__flag--decide" title="${esc(block.decisions.map((decision) => `${decision.path} = ${decision.choice || "undecided"}`).join("\n"))}">decides ${SequenceBoard.axes(block.decisions.length)}${open ? ` · ${open} open` : ""}</p>` : ""}
      ${block.checkpoint ? `<p class="seq-block__flag">checkpoint: holds until released</p>` : ""}
      ${block.skip ? `<p class="seq-block__flag seq-block__flag--skip">skip: bypassed on run</p>` : ""}
      <p class="seq-block__error" title="${esc(state.error || "")}"${state.error ? "" : " hidden"}>${esc(state.error || "")}</p>
      <div class="seq-block__actions">
        <button type="button" class="btn btn--sm" data-action="link" title="Click here, then click the block that should run after this one">Link</button>
        <button type="button" class="btn btn--sm" data-action="skip" aria-pressed="${block.skip}" title="A skipped block is bypassed once its own predecessors finish: it does not launch and then counts as satisfied for the blocks after it">${block.skip ? "Unskip" : "Skip"}</button>
        <button type="button" class="btn btn--sm" data-action="release"${state.status === "held" ? "" : " hidden"} title="Release this checkpoint so the sequence continues">Release</button>
        <button type="button" class="btn btn--sm" data-action="job"${state.job_id ? "" : " hidden"} title="Open this block's job in the console">Job</button>
      </div>`;

    el.addEventListener("click", (event) => {
      event.stopPropagation();
      if (this.dragUsed) {
        this.dragUsed = false;
        return;
      }
      this.callbacks.onSelect(block.block_id);
    });
    el.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.target === el) this.callbacks.onSelect(block.block_id);
    });
    const action = (name, handler) => el.querySelector(`[data-action=${name}]`).addEventListener("click", (event) => {
      event.stopPropagation();
      handler();
    });
    action("link", () => this.callbacks.onLink(block.block_id));
    action("skip", () => this.callbacks.onToggleSkip(block.block_id));
    action("release", () => this.callbacks.onRelease(block.block_id));
    action("job", () => this.callbacks.onOpenJob(this.state(block.block_id).job_id));

    el.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      this.dragStart(event, block, el);
    });
    return el;
  }

  dragStart(event, block, el) {
    event.preventDefault();
    this.dragUsed = false;
    this.drag = { dx: event.clientX - block.x, dy: event.clientY - block.y, moved: false };

    const move = (moveEvent) => {
      const grid = SequenceBoard.GRID;
      const rawX = Math.max(0, moveEvent.clientX - this.drag.dx);
      const rawY = Math.max(0, moveEvent.clientY - this.drag.dy);
      if (!this.drag.moved && Math.abs(rawX - block.x) + Math.abs(rawY - block.y) < 4) return;
      block.x = Math.round(rawX / grid) * grid;
      block.y = Math.round(rawY / grid) * grid;
      this.drag.moved = true;
      el.style.left = `${block.x}px`;
      el.style.top = `${block.y}px`;
      this.drawEdges();
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      const moved = this.drag && this.drag.moved;
      this.drag = null;
      this.dragUsed = Boolean(moved);
      if (moved) {
        this.redraw();
        this.callbacks.onMoved();
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  anchor(el) {
    return { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight };
  }

  drawEdges() {
    if (!this.svg) return;
    this.svg.innerHTML = "";

    const byId = new Map([...this.surface.querySelectorAll(".seq-block[data-block]")].map((el) => [el.dataset.block, el]));
    const start = this.surface.querySelector(".seq-block--start");
    const paths = [];

    this.entry.blocks.forEach((block) => {
      const target = byId.get(block.block_id);
      if (!target) return;
      if (!block.predecessors.length && start) paths.push([this.anchor(start), this.anchor(target), true, "__start__", block.block_id]);
      block.predecessors.forEach((pred) => {
        const source = byId.get(pred);
        if (source) paths.push([this.anchor(source), this.anchor(target), false, pred, block.block_id]);
      });
    });

    paths.forEach(([from, to, dashed, sourceId, targetId], index) => {
      const group = document.createElementNS(SequenceBoard.SVG_NS, "g");
      group.classList.add("seq-edgegroup");
      group.dataset.source = sourceId;
      group.dataset.target = targetId;

      const path = document.createElementNS(SequenceBoard.SVG_NS, "path");
      path.setAttribute("d", this.edgePath(from, to));
      path.setAttribute("id", `seq-edge-${index}`);
      path.classList.add("seq-edge");
      if (dashed) path.classList.add("seq-edge--start");
      group.appendChild(path);

      const dot = document.createElementNS(SequenceBoard.SVG_NS, "circle");
      dot.setAttribute("r", "3");
      dot.classList.add("seq-edge__dot");
      const motion = document.createElementNS(SequenceBoard.SVG_NS, "animateMotion");
      motion.setAttribute("dur", "1.6s");
      motion.setAttribute("repeatCount", "indefinite");
      const mpath = document.createElementNS(SequenceBoard.SVG_NS, "mpath");
      mpath.setAttribute("href", `#seq-edge-${index}`);
      motion.appendChild(mpath);
      dot.appendChild(motion);
      group.appendChild(dot);

      this.svg.appendChild(group);
    });

    this.paintEdgeStates();
  }

  paintEdgeStates() {
    if (!this.svg) return;
    const flowing = this.flowingEdges();
    this.svg.querySelectorAll(".seq-edgegroup").forEach((group) => {
      const status = this.state(group.dataset.target).status;
      const flows = flowing.has(`${group.dataset.source}->${group.dataset.target}`);
      group.classList.toggle("is-flowing", flows);
      group.classList.toggle("is-done", !flows && SequenceBoard.SETTLED.includes(status));
      group.classList.toggle("is-failed", SequenceBoard.BROKEN.includes(status));
    });
  }

  flowingEdges() {
    const byId = new Map(this.entry.blocks.map((block) => [block.block_id, block]));
    const flowing = new Set();
    const feed = (blockId) => {
      const block = byId.get(blockId);
      const preds = block && block.predecessors.length ? block.predecessors : ["__start__"];
      preds.forEach((pred) => {
        const key = `${pred}->${blockId}`;
        if (flowing.has(key)) return;
        flowing.add(key);
        if (pred !== "__start__" && this.state(pred).status === "bypassed") feed(pred);
      });
    };
    this.entry.blocks.forEach((block) => {
      if (SequenceBoard.ACTIVE.includes(this.state(block.block_id).status)) feed(block.block_id);
    });
    return flowing;
  }

  edgePath(from, to) {
    const fromCx = from.x + from.w / 2;
    const fromCy = from.y + from.h / 2;
    const toCx = to.x + to.w / 2;
    const toCy = to.y + to.h / 2;

    if (Math.abs(toCx - fromCx) >= Math.abs(toCy - fromCy)) {
      const fromPort = from.y + Math.min(SequenceBoard.PORT_Y, from.h / 2);
      const toPort = to.y + Math.min(SequenceBoard.PORT_Y, to.h / 2);
      const rightward = toCx >= fromCx;
      const x1 = rightward ? from.x + from.w : from.x;
      const x2 = rightward ? to.x : to.x + to.w;
      const dx = Math.max(40, Math.abs(x2 - x1) / 2) * (rightward ? 1 : -1);
      return `M ${x1} ${fromPort} C ${x1 + dx} ${fromPort}, ${x2 - dx} ${toPort}, ${x2} ${toPort}`;
    }

    const downward = toCy >= fromCy;
    const y1 = downward ? from.y + from.h : from.y;
    const y2 = downward ? to.y : to.y + to.h;
    const dy = Math.max(40, Math.abs(y2 - y1) / 2) * (downward ? 1 : -1);
    return `M ${fromCx} ${y1} C ${fromCx} ${y1 + dy}, ${toCx} ${y2 - dy}, ${toCx} ${y2}`;
  }
}

export class SequencesPage extends ConfigForm {
  constructor(view, host, project) {
    super();
    this.view = view;
    this.app = view.app;
    this.host = host;
    this.project = project;
    this.copyControl = new RunConfigCopy(this, project.roots.runs);
    this.templateControl = new ConfigTemplates(this);
    this.sequences = [];
    this.current = null;
    this.live = null;
    this.selected = "";
    this.linking = "";
    this.editorKey = "";
    this.pollTimer = null;
    this.board = null;
    this.detail = null;
    this.configCache = new Map();
    this.busy = false;
    this.escHandler = (event) => {
      if (event.key === "Escape" && this.selected && !event.target.closest("input, select, textarea")) this.closeDrawer();
    };
  }

  async quickAdd(script, title) {
    let entry = null;
    const active = Store.read("sequence.active", "");
    try {
      if (active) entry = (await api.get(`/api/sequences/${active}`)).entry;
    } catch (error) {
      entry = null;
    }

    try {
      if (!entry) {
        entry = (await api.post("/api/sequences", { name: "" })).entry;
        Store.write("sequence.active", entry.sequence_id);
      }
      const n = entry.blocks.length;
      entry.blocks.push({ script, title, overrides: {}, decisions: [], predecessors: n ? [entry.blocks[n - 1].block_id] : [], checkpoint: false, skip: false, x: 288 + n * 264, y: 24 });
      await api.post(`/api/sequences/${entry.sequence_id}/save`, entry);
      this.app.toast.show(`Added ${title} to sequence "${entry.name}"`, "ok");
    } catch (error) {
      this.app.toast.show(error.message || "Could not add the block", "error");
    }
  }

  async render() {
    this.host.innerHTML = `
      <div class="stores stores--sequences">
        ${this.view.head({ title: "Sequences", lead: "Chains of launches that run block by block, each after its predecessors finish, in a workspace of their own.", figure: true, figureName: "elevator" })}
        <div class="seq-layout" data-role="layout">${skeleton(4)}</div>
      </div>`;
    this.layoutHost = this.host.querySelector("[data-role=layout]");
    this.headFigure = this.host.querySelector("hairline-figure");

    await this.loadList();
    const active = this.current ? this.current.sequence_id : Store.read("sequence.active", "");
    const pick = this.sequences.find((entry) => entry.sequence_id === active) || this.sequences[0];
    if (pick) await this.open(pick.sequence_id);
    this.draw();
    this.startPoll();
    document.removeEventListener("keydown", this.escHandler);
    document.addEventListener("keydown", this.escHandler);
  }

  leave() {
    this.stopPoll();
    document.removeEventListener("keydown", this.escHandler);
  }

  async loadList() {
    try {
      this.sequences = (await api.get("/api/sequences")).sequences;
    } catch (error) {
      this.sequences = [];
      this.app.toast.show(error.message, "error");
    }
  }

  async open(sequenceId) {
    try {
      const res = await api.get(`/api/sequences/${sequenceId}`);
      this.current = res.entry;
      this.live = res.state;
      this.selected = "";
      this.linking = "";
      Store.write("sequence.active", sequenceId);
    } catch (error) {
      this.app.toast.show(error.message || "Could not open the sequence", "error");
    }
  }

  async create() {
    const name = window.prompt("Name of the new sequence:", "");
    if (name === null) return;
    try {
      const created = await api.post("/api/sequences", { name });
      await this.loadList();
      await this.open(created.entry.sequence_id);
      this.draw();
    } catch (error) {
      this.app.toast.show(error.message || "Create failed", "error");
    }
  }

  async deleteSequence(entry) {
    if (!window.confirm(`Delete sequence "${entry.name}"?`)) return;
    try {
      await api.post(`/api/sequences/${entry.sequence_id}/delete`, {});
      if (this.current && this.current.sequence_id === entry.sequence_id) this.current = null;
      await this.loadList();
      this.draw();
    } catch (error) {
      this.app.toast.show(error.message || "Delete failed", "error");
    }
  }

  async persist() {
    if (!this.current) return false;
    try {
      const saved = await api.post(`/api/sequences/${this.current.sequence_id}/save`, this.current);
      const byId = new Map(this.current.blocks.filter((block) => block.block_id).map((block) => [block.block_id, block]));
      saved.entry.blocks.forEach((clean, index) => Object.assign(byId.get(clean.block_id) || this.current.blocks[index], clean));
      this.current.updated_at = saved.entry.updated_at;
      const listed = this.sequences.find((entry) => entry.sequence_id === this.current.sequence_id);
      if (listed) Object.assign(listed, { name: saved.entry.name, blocks: saved.entry.blocks, updated_at: saved.entry.updated_at });
      return true;
    } catch (error) {
      this.app.toast.show(error.message || "Save failed", "error");
      return false;
    }
  }

  async act(path, body, success, kind = "ok") {
    if (this.busy) return null;
    this.busy = true;
    try {
      const res = await api.post(`/api/sequences/${this.current.sequence_id}/${path}`, body || {});
      if (res.state !== undefined) this.live = res.state;
      if (success) this.app.toast.show(success, kind);
      if (this.board) this.board.paintStates(this.live);
      this.paintStatus();
      return res;
    } catch (error) {
      this.app.toast.show(error.message || "The sequence did not answer", "error");
      return null;
    } finally {
      this.busy = false;
    }
  }

  async release(blockId) {
    const pending = this.upstreamUndecided(blockId);
    if (pending.length && !window.confirm(`These earlier blocks have no verdict recorded yet:\n\n${pending.map((item) => item.label).join("\n")}\n\nReleasing now runs this block on the pre-verdict values. Continue?`)) {
      this.selectBlock(pending[0].block_id);
      return;
    }
    await this.act("release", { block_id: blockId }, "Checkpoint released");
  }

  openDecisions(blockId) {
    const block = this.current.blocks.find((item) => item.block_id === blockId);
    return block ? block.decisions.filter((decision) => !decision.choice) : [];
  }

  upstreamUndecided(blockId) {
    const byId = new Map(this.current.blocks.map((block) => [block.block_id, block]));
    const seen = new Set();
    const pending = [];
    const stack = [...(byId.has(blockId) ? byId.get(blockId).predecessors : [])];

    while (stack.length) {
      const id = stack.pop();
      if (seen.has(id) || !byId.has(id)) continue;
      seen.add(id);
      this.openDecisions(id).forEach((decision) => pending.push({ block_id: id, label: `${byId.get(id).name || byId.get(id).title}: ${decision.path}` }));
      stack.push(...byId.get(id).predecessors);
    }

    return pending;
  }

  async start(resumeDir = "", bypass = []) {
    if (!(await this.persist())) return;
    await this.act("start", resumeDir ? { resume_dir: resumeDir, bypass } : {}, resumeDir ? "Sequence resumed in the previous workspace" : "Sequence started");
  }

  async resume() {
    const previous = this.live && this.live.run_dir ? this.live.run_dir : "";
    const input = window.prompt("Workspace directory of the execution to resume (runs/sequences/...). Blocks it records as finished are bypassed on this start only; the rest runs inside this workspace:", previous);
    if (input === null || !input.trim()) return;
    const dir = input.trim();

    let scan;
    try {
      scan = await api.post(`/api/sequences/${this.current.sequence_id}/resume-scan`, { resume_dir: dir });
    } catch (error) {
      this.app.toast.show(error.message || "Resume scan failed", "error");
      return;
    }

    if (!scan.recorded) {
      if (!window.confirm("This workspace holds no recorded execution state, so finished blocks cannot be detected. Start inside it using the current skip marks?")) return;
      await this.start(dir);
      return;
    }

    const byId = new Map(this.current.blocks.map((block) => [block.block_id, block]));
    const names = scan.done.map((id) => byId.get(id).name || byId.get(id).title);
    const ask = names.length ? `Recorded as finished in this workspace, bypassed on this start only:\n\n${names.join("\n")}\n\nThe remaining blocks run inside the workspace. Continue?` : "No block is recorded as finished in this workspace. Start inside it from the beginning?";
    if (!window.confirm(ask)) return;
    await this.start(dir, scan.done);
  }

  async refreshBoard() {
    let cleared = false;
    const active = this.live && ["running", "held"].includes(this.live.status);
    if (this.current && this.live && !active) {
      const res = await this.act("clear", {}, "");
      if (!res) return;
      cleared = res.cleared;
    }
    await this.loadList();
    if (this.current) await this.open(this.current.sequence_id);
    this.draw();
    this.app.toast.show(cleared ? "Execution state cleared, board reset to idle" : "Sequence view refreshed from the server", "ok");
  }

  startPoll() {
    this.stopPoll();
    this.pollTimer = setInterval(() => this.poll(), 3000);
  }

  stopPoll() {
    clearInterval(this.pollTimer);
    this.pollTimer = null;
  }

  async poll() {
    if (!this.current) return;
    let res;
    try {
      res = await api.get(`/api/sequences/${this.current.sequence_id}`);
    } catch (error) {
      return;
    }

    if (!this.selected && res.entry.updated_at !== this.current.updated_at) {
      this.current = res.entry;
      this.live = res.state;
      this.draw();
      return;
    }

    const changed = JSON.stringify(res.state) !== JSON.stringify(this.live);
    this.live = res.state;
    if (!changed) return;
    if (this.board) this.board.paintStates(this.live);
    this.paintStatus();
  }

  draw() {
    this.layoutHost.innerHTML = "";
    this.layoutHost.appendChild(this.buildRail());
    this.layoutHost.appendChild(this.buildMain());
    if (this.board) this.board.redraw();
    this.paintStatus();
  }

  buildRail() {
    const rail = document.createElement("aside");
    rail.className = "seq-rail";
    rail.innerHTML = `<button type="button" class="btn btn--primary seq-rail__new" data-action="new">${icon("plus")}<span>New sequence</span></button>`;
    rail.querySelector("[data-action=new]").addEventListener("click", () => this.create());

    if (!this.sequences.length) rail.insertAdjacentHTML("beforeend", `<p class="seq-rail__empty">No sequences yet. Create one here, or add an entry point from its card in the catalog.</p>`);

    this.sequences.forEach((entry) => {
      const active = this.current && entry.sequence_id === this.current.sequence_id;
      const status = entry.state ? entry.state.status : "idle";
      const card = document.createElement("div");
      card.className = `seq-rail__card${active ? " is-active" : ""}`;
      card.tabIndex = 0;
      card.innerHTML = `
        <div class="seq-rail__row"><span class="seq-rail__name">${esc(entry.name)}</span><button type="button" class="tbtn tbtn--sm" data-action="delete" title="Delete this sequence" aria-label="Delete this sequence">${icon("trash")}</button></div>
        <div class="seq-rail__row"><span class="seq-rail__facts">${entry.blocks.length} block${entry.blocks.length === 1 ? "" : "s"}</span><span class="seq-status">${esc(status)}</span></div>`;
      const openIt = async () => {
        await this.open(entry.sequence_id);
        this.draw();
      };
      card.addEventListener("click", openIt);
      card.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && event.target === card) openIt();
      });
      card.querySelector("[data-action=delete]").addEventListener("click", async (event) => {
        event.stopPropagation();
        await this.deleteSequence(entry);
      });
      rail.appendChild(card);
    });
    return rail;
  }

  buildMain() {
    const main = document.createElement("div");
    main.className = "seq-main";
    if (!this.current) {
      main.innerHTML = `<p class="seq-empty">Select a sequence above or create a new one.</p>`;
      this.board = null;
      return main;
    }

    main.appendChild(this.buildBar());
    main.insertAdjacentHTML("beforeend", `<p class="seq-bar__dir" data-role="dir" hidden></p>`);
    this.dirEl = main.querySelector("[data-role=dir]");

    const boardHost = document.createElement("div");
    boardHost.className = "seq-board";
    main.appendChild(boardHost);

    this.drawerEl = document.createElement("div");
    this.drawerEl.className = "seq-drawer";
    this.drawerEl.hidden = true;
    main.appendChild(this.drawerEl);

    this.board = new SequenceBoard(boardHost, {
      onSelect: (blockId) => this.selectBlock(blockId),
      onLink: (blockId) => this.toggleLinking(blockId),
      onToggleSkip: (blockId) => this.toggleSkip(blockId),
      onMoved: () => this.persist(),
      onRelease: (blockId) => this.release(blockId),
      onOpenJob: (jobId) => this.app.router.go(`#/console/${jobId}`),
      onBackground: () => this.closeDrawer(),
    });
    this.board.render(this.current, this.live, this.selected, this.linking);
    if (this.selected) this.renderBlockEditor();
    return main;
  }

  buildBar() {
    const bar = document.createElement("div");
    bar.className = "seq-bar";
    bar.innerHTML = `
      <input class="input seq-bar__name" type="text" spellcheck="false" aria-label="Sequence name" data-role="name">
      <select class="select seq-bar__add" aria-label="Add a block" data-role="add"><option value="">Add a block</option>${this.view.scripts.map((script) => `<option value="${esc(script.key)}">${esc(script.title)}</option>`).join("")}</select>
      <div class="rail-detach"><button type="button" class="switch" role="switch" data-role="detach" title="Detach each block's run from the console server"><span class="switch__knob"></span></button><span class="rail-detach__label">Detach runs</span></div>
      <div class="rail-detach"><button type="button" class="switch" role="switch" data-role="isolate" title="Each start creates a fresh runs/sequences/<name>_<stamp> directory and redirects every block's io.runs_dir into it, so repeated executions never collide"><span class="switch__knob"></span></button><span class="rail-detach__label">Isolated dir</span></div>
      <span class="seq-bar__status" data-role="status"></span>
      <span class="seq-bar__spacer"></span>
      <button type="button" class="btn btn--sm" data-role="refresh" title="Reload this sequence from the server and redraw the board; a finished, failed or stopped execution state is cleared so the board returns to idle">${icon("refresh")}<span>Refresh</span></button>
      <button type="button" class="btn btn--sm" data-role="resume" title="Start inside the workspace of a previous execution; blocks it records as finished are bypassed on that start only">Resume</button>
      <button type="button" class="btn btn--sm btn--primary" data-role="run"></button>`;

    const name = bar.querySelector("[data-role=name]");
    name.value = this.current.name;
    name.addEventListener("change", async () => {
      this.current.name = name.value;
      if (await this.persist()) await this.loadList();
      this.draw();
    });

    const add = bar.querySelector("[data-role=add]");
    add.addEventListener("change", async () => {
      if (!add.value) return;
      const script = this.view.scripts.find((item) => item.key === add.value);
      const n = this.current.blocks.length;
      this.current.blocks.push({ script: script.key, title: script.title, overrides: {}, decisions: [], predecessors: [], checkpoint: false, skip: false, x: 288 + n * 264, y: 24 });
      add.value = "";
      if (await this.persist()) this.draw();
    });

    const toggle = (role, key) => {
      const button = bar.querySelector(`[data-role=${role}]`);
      button.classList.toggle("is-on", this.current[key]);
      button.setAttribute("aria-checked", String(this.current[key]));
      button.addEventListener("click", async () => {
        this.current[key] = !this.current[key];
        if (await this.persist()) this.draw();
      });
    };
    toggle("detach", "detach");
    toggle("isolate", "isolate");

    this.statusEl = bar.querySelector("[data-role=status]");
    this.runBtn = bar.querySelector("[data-role=run]");
    this.resumeBtn = bar.querySelector("[data-role=resume]");
    bar.querySelector("[data-role=refresh]").addEventListener("click", () => this.refreshBoard());
    this.resumeBtn.addEventListener("click", () => this.resume());
    this.runBtn.addEventListener("click", () => {
      const active = this.live && ["running", "held"].includes(this.live.status);
      if (active) this.act("stop", {}, "Sequence stopped", "warn");
      else this.start();
    });
    return bar;
  }

  paintStatus() {
    if (this.headFigure) {
      const states = this.live ? Object.values(this.live.blocks) : [];
      const fired = states.filter((state) => ["running", "done", "failed"].includes(state.status)).length;
      const total = this.current ? this.current.blocks.length : 0;
      const running = Boolean(this.live && ["running", "held"].includes(this.live.status));
      const intensity = total ? (running ? 0.6 + 0.4 * Math.min(1, fired / total) : 0.35) : 0.15;
      const value = String(Math.round(intensity * 100) / 100);
      if (this.headFigure.getAttribute("intensity") !== value) this.headFigure.setAttribute("intensity", value);
    }
    if (!this.statusEl || !this.current) return;

    const status = this.live ? this.live.status : "idle";
    const active = ["running", "held"].includes(status);
    this.statusEl.textContent = status;
    this.statusEl.dataset.status = status;
    this.runBtn.textContent = active ? "Stop sequence" : "Start sequence";
    this.runBtn.classList.toggle("btn--primary", !active);
    this.runBtn.classList.toggle("btn--danger", active);
    this.resumeBtn.hidden = active || !this.current.isolate;

    const runDir = this.live ? this.live.run_dir : "";
    this.dirEl.hidden = !runDir;
    this.dirEl.textContent = runDir ? `workspace ${runDir}` : "";
  }

  selectBlock(blockId) {
    if (this.linking && this.linking !== blockId) {
      this.toggleEdge(this.linking, blockId);
      return;
    }
    this.linking = "";
    this.selected = blockId;
    this.draw();
  }

  async toggleSkip(blockId) {
    const block = this.current.blocks.find((item) => item.block_id === blockId);
    block.skip = !block.skip;
    if (!(await this.persist())) block.skip = !block.skip;
    this.draw();
  }

  toggleLinking(blockId) {
    this.linking = this.linking === blockId ? "" : blockId;
    if (this.linking) this.app.toast.show("Now click the block that should run after this one; press Link again to cancel", "ok");
    this.board.render(this.current, this.live, this.selected, this.linking);
  }

  async toggleEdge(sourceId, targetId) {
    const target = this.current.blocks.find((item) => item.block_id === targetId);
    const before = target.predecessors.slice();
    const index = target.predecessors.indexOf(sourceId);
    if (index >= 0) target.predecessors.splice(index, 1);
    else target.predecessors.push(sourceId);
    this.linking = "";
    if (!(await this.persist())) target.predecessors = before;
    this.draw();
  }

  closeDrawer() {
    if (!this.selected) return;
    this.selected = "";
    this.linking = "";
    this.draw();
  }

  drawerShell(title, hint) {
    this.drawerEl.hidden = false;
    this.drawerEl.setAttribute("role", "dialog");
    this.drawerEl.setAttribute("aria-label", title);
    this.drawerEl.innerHTML = `<div class="seq-drawer__head"><h3 class="seq-drawer__title">${esc(title)}</h3><span class="seq-drawer__hint">${esc(hint)}</span><button type="button" class="tbtn tbtn--sm" data-action="close" title="Close" aria-label="Close">${icon("close")}</button></div><div class="seq-drawer__body"></div>`;
    this.drawerEl.querySelector("[data-action=close]").addEventListener("click", () => this.closeDrawer());
    return this.drawerEl.querySelector(".seq-drawer__body");
  }

  async renderBlockEditor() {
    const block = this.current.blocks.find((item) => item.block_id === this.selected);
    if (!block) {
      this.selected = "";
      return;
    }

    this.resetForm();
    this.detail = { key: block.script, title: block.title };
    const body = this.drawerShell(block.name || block.title, `${block.script} · edit and apply; waiting blocks pick edits up at launch`);
    const byId = new Map(this.current.blocks.map((item) => [item.block_id, item]));

    const controls = document.createElement("div");
    controls.className = "seq-drawer__controls";
    controls.innerHTML = `
      <input class="input" type="text" spellcheck="false" placeholder="Block name" aria-label="Block name" data-role="name">
      <button type="button" class="btn btn--sm" data-role="checkpoint" aria-pressed="${block.checkpoint}" title="A checkpoint holds the line here: the block waits until released by hand, so it can be edited against the previous results first">Checkpoint</button>
      <button type="button" class="btn btn--sm" data-role="skip" aria-pressed="${block.skip}" title="A skipped block is bypassed: it does not launch and counts as satisfied for the blocks after it">Skip</button>
      <div class="seq-drawer__preds"><span class="seq-drawer__predlabel">After</span>${block.predecessors.length ? block.predecessors.map((pred) => `<button type="button" class="chip" data-pred="${esc(pred)}" title="Remove this link">${esc((byId.get(pred) || {}).title || pred)}${icon("close")}</button>`).join("") : `<span class="seq-drawer__prednone">the start block</span>`}</div>`;

    const name = controls.querySelector("[data-role=name]");
    name.value = block.name;
    name.addEventListener("change", async () => {
      block.name = name.value;
      if (await this.persist()) this.board.render(this.current, this.live, this.selected, this.linking);
    });
    controls.querySelector("[data-role=checkpoint]").addEventListener("click", async () => {
      block.checkpoint = !block.checkpoint;
      if (await this.persist()) this.draw();
    });
    controls.querySelector("[data-role=skip]").addEventListener("click", () => this.toggleSkip(block.block_id));
    controls.querySelectorAll("[data-pred]").forEach((chip) => chip.addEventListener("click", async () => {
      block.predecessors = block.predecessors.filter((pred) => pred !== chip.dataset.pred);
      if (await this.persist()) this.draw();
    }));
    body.appendChild(controls);

    const decideHost = document.createElement("div");
    decideHost.className = "seq-decide";
    body.appendChild(decideHost);

    const configHost = document.createElement("div");
    configHost.className = "seq-drawer__config";
    configHost.innerHTML = skeleton(4);
    body.appendChild(configHost);

    const actions = document.createElement("div");
    actions.className = "seq-drawer__actions";
    actions.innerHTML = `<button type="button" class="btn btn--sm btn--danger" data-role="remove">Remove block</button><button type="button" class="btn btn--sm btn--primary" data-role="apply" disabled>Apply to block</button>`;
    body.appendChild(actions);

    actions.querySelector("[data-role=remove]").addEventListener("click", async () => {
      if (!window.confirm(`Remove block "${block.name || block.title}" from the sequence?`)) return;
      this.current.blocks = this.current.blocks.filter((item) => item.block_id !== block.block_id);
      this.current.blocks.forEach((item) => {
        item.predecessors = item.predecessors.filter((pred) => pred !== block.block_id);
      });
      this.selected = "";
      if (await this.persist()) this.draw();
    });

    const editorKey = `${block.block_id}:${block.script}`;
    this.editorKey = editorKey;

    let cfg = this.configCache.get(block.script);
    if (!cfg) {
      try {
        cfg = await api.get(`/api/scripts/${block.script}/config`);
        this.configCache.set(block.script, cfg);
      } catch (error) {
        if (this.editorKey === editorKey) configHost.innerHTML = `<p class="cfg-note">${esc(error.message)}</p>`;
        return;
      }
    }
    if (this.editorKey !== editorKey) return;

    this.config = cfg;
    this.buildDecisions(decideHost, block, cfg);
    configHost.innerHTML = "";
    const tools = document.createElement("div");
    tools.className = "seq-copy";
    tools.append(this.copyControl.build(), this.templateControl.build());
    configHost.appendChild(tools);
    configHost.appendChild(this.buildToolbar(cfg));
    this.renderLayout(configHost, cfg);
    this.applyRunConfig(block.overrides);
    this.paintProvenance(block);

    const apply = actions.querySelector("[data-role=apply]");
    apply.disabled = false;
    apply.addEventListener("click", async () => {
      const steered = this.steerDecisions(block);
      const applied = { ...this.dirty };
      block.overrides = applied;
      if (!(await this.persist())) return;

      const reverted = this.revertedAxes(block, applied, steered);
      this.applyRunConfig(block.overrides);
      this.buildDecisions(decideHost, block, cfg);
      this.paintProvenance(block);
      this.board.render(this.current, this.live, this.selected, this.linking);
      const count = Object.keys(block.overrides).length;
      const parts = [`Block updated: ${count} override${count === 1 ? "" : "s"}`];
      if (steered.length) parts.push(`pre-verdict value updated for ${steered.join(", ")}`);
      reverted.forEach(({ path, decider }) => parts.push(`${path} reverted: decided by '${decider}'`));
      this.app.toast.show(parts.join(" · "), reverted.length ? "warn" : "ok");
    });
  }

  steerDecisions(block) {
    const steered = [];
    block.decisions.forEach((decision) => {
      const leaf = this.byPath.get(decision.path);
      if (!leaf) return;
      const edited = this.dirty[decision.path] !== undefined ? this.dirty[decision.path] : String(leaf.value);
      if (edited === decision.waiting) return;
      decision.waiting = edited;
      steered.push(decision.path);
    });
    return steered;
  }

  revertedAxes(block, applied, steered) {
    const notes = block.field_notes;
    const paths = new Set([...Object.keys(applied), ...Object.keys(block.overrides)]);
    const reverted = [];
    paths.forEach((path) => {
      if (!notes[path] || steered.includes(path)) return;
      if (applied[path] === block.overrides[path]) return;
      const decider = this.current.blocks.find((other) => other.block_id !== block.block_id && other.decisions.some((decision) => decision.path === path));
      reverted.push({ path, decider: decider ? decider.name || decider.title : "an earlier block" });
    });
    return reverted;
  }

  paintProvenance(block) {
    const titles = {
      frozen: "frozen: this value is the verdict of an earlier block in the sequence",
      waiting: "waiting: a later block decides this field; until then it holds the pre-experiment value",
    };
    this.states.forEach(({ leaf, row }) => {
      row.classList.remove("cfg-row--frozen", "cfg-row--waiting");
      const note = titles[block.field_notes[leaf.path]] ? block.field_notes[leaf.path] : null;
      const name = row.querySelector(".cfg-row__name");
      if (name.dataset.base === undefined) name.dataset.base = name.dataset.hint;
      name.dataset.hint = note ? `${titles[note]}. ${name.dataset.base}` : name.dataset.base;
      if (note) row.classList.add(`cfg-row--${note}`);
    });
  }

  buildDecisions(host, block, cfg) {
    const info = Popover.infoButton("Decided here", infoText("Config axes this block puts under test. Blocks after this one run the verdict, blocks before it run the pre-verdict value; both are written on save, so a verdict is recorded once instead of edited into every later block.", "Editing a decided axis in the form below rewrites its pre-verdict value on apply. A candidate with commas of its own, a tuple or a list, is typed in brackets."));
    host.innerHTML = `
      <div class="seq-decide__head"><span class="seq-decide__title">Decided here</span><span class="seq-decide__count">${block.decisions.length ? SequenceBoard.axes(block.decisions.length) : "none"}</span>${info}</div>
      <div class="seq-decide__list"></div>
      <div class="seq-decide__add">
        <input class="input seq-decide__picker" type="text" spellcheck="false" placeholder="config path under test" aria-label="Config path under test" list="seq-decide-paths">
        <datalist id="seq-decide-paths">${cfg.leaves.filter((leaf) => leaf.editable).map((leaf) => `<option value="${esc(leaf.path)}"></option>`).join("")}</datalist>
        <button type="button" class="btn btn--sm" data-role="add-axis">Add axis</button>
      </div>`;
    const list = host.querySelector(".seq-decide__list");
    block.decisions.forEach((decision) => list.appendChild(this.decisionRow(block, decision, cfg)));
    const picker = host.querySelector(".seq-decide__picker");
    host.querySelector("[data-role=add-axis]").addEventListener("click", () => this.addDecision(block, picker.value.trim(), cfg));
    picker.addEventListener("keydown", (event) => {
      if (event.key === "Enter") this.addDecision(block, picker.value.trim(), cfg);
    });
  }

  decisionRow(block, decision, cfg) {
    const leaf = cfg.leaves.find((item) => item.path === decision.path) || null;
    const values = decision.options.slice();
    if (decision.choice && !values.includes(decision.choice)) values.unshift(decision.choice);
    const row = document.createElement("div");
    row.className = `seq-decide__item${decision.choice ? " is-decided" : ""}`;
    row.innerHTML = `
      <code class="seq-decide__path${leaf ? "" : " is-foreign"}" title="${esc(leaf ? `${leaf.type} · default ${leaf.value}` : "this block's entry point does not expose the path; the axis is still frozen into the blocks that do")}">${esc(decision.path)}</code>
      <label class="ctl seq-decide__field"><span class="ctl__label">Before</span><input class="input" type="text" spellcheck="false" data-role="waiting" title="Value every block that is not yet after this one runs with while the question is open"></label>
      <label class="ctl seq-decide__field seq-decide__field--wide"><span class="ctl__label">Candidates</span><input class="input" type="text" spellcheck="false" data-role="options" title="Comma-separated values the experiment compares; they become the verdict choices. A comma inside brackets, braces or quotes belongs to the value"></label>
      <label class="ctl seq-decide__field"><span class="ctl__label">Verdict</span><select class="select" data-role="verdict" title="Winner of this experiment; picking it freezes the value into every block after this one"><option value="">undecided</option>${values.map((value) => `<option value="${esc(value)}"${value === decision.choice ? " selected" : ""}>${esc(value)}</option>`).join("")}</select></label>
      <button type="button" class="tbtn tbtn--sm seq-decide__remove" data-role="remove" title="Stop deciding this axis here; the blocks it stamped keep their current values as plain overrides" aria-label="Stop deciding ${esc(decision.path)}">${icon("close")}</button>`;
    const waiting = row.querySelector("[data-role=waiting]");
    waiting.value = decision.waiting;
    waiting.addEventListener("change", async () => {
      decision.waiting = waiting.value;
      if (await this.persist()) this.draw();
    });
    const options = row.querySelector("[data-role=options]");
    options.value = decision.options.join(", ");
    options.addEventListener("change", async () => {
      decision.options = PythonLiteral.split(options.value);
      if (await this.persist()) this.draw();
    });
    row.querySelector("[data-role=verdict]").addEventListener("change", (event) => this.recordVerdict(block, decision.path, event.target.value));
    row.querySelector("[data-role=remove]").addEventListener("click", async () => {
      block.decisions = block.decisions.filter((item) => item.path !== decision.path);
      if (await this.persist()) this.draw();
    });
    return row;
  }

  async addDecision(block, path, cfg) {
    if (!path) return;
    if (block.decisions.some((decision) => decision.path === path)) {
      this.app.toast.show(`This block already decides ${path}`, "warn");
      return;
    }
    const leaf = cfg.leaves.find((item) => item.path === path) || null;
    block.decisions.push({
      path,
      options: leaf && Array.isArray(leaf.choices) && leaf.choices.length ? leaf.choices.slice() : leaf && leaf.type === "bool" ? ["False", "True"] : [],
      waiting: block.overrides[path] !== undefined ? block.overrides[path] : leaf ? String(leaf.value) : "",
      choice: "",
    });
    if (await this.persist()) this.draw();
  }

  async recordVerdict(block, path, choice) {
    let res;
    try {
      res = await api.post(`/api/sequences/${this.current.sequence_id}/decide`, { block_id: block.block_id, path, choice });
    } catch (error) {
      this.app.toast.show(error.message || "Could not record the verdict", "error");
      this.draw();
      return;
    }
    this.current = res.entry;
    this.app.toast.show(choice ? `${path} frozen to ${choice} in every block after this one` : `${path} reopened; the blocks after this one went back to the pre-verdict value`, "ok");
    this.draw();
  }

  refresh() {
    super.refresh();
    if (this.countEl) {
      const n = Object.keys(this.dirty).length;
      this.countEl.textContent = n ? `${n} override${n > 1 ? "s" : ""}` : "all defaults";
    }
  }
}

