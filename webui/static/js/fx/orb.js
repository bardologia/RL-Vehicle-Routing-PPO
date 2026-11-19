import { esc } from "../format.js";

export class Motion {
  static media = window.matchMedia("(prefers-reduced-motion: reduce)");
  static listeners = new Set();

  static get reduced() {
    return Motion.media.matches;
  }

  static subscribe(listener) {
    Motion.listeners.add(listener);
    return () => Motion.listeners.delete(listener);
  }

  static now() {
    return performance.now() / 1000;
  }

  static pixelRatio() {
    return Math.min(2, window.devicePixelRatio || 1);
  }
}

Motion.media.addEventListener("change", () => {
  for (const listener of [...Motion.listeners]) listener(Motion.reduced);
});

class ThinkingOrbElement extends HTMLElement {
  static STATES = ["working", "searching", "solving", "listening", "connecting", "weaving", "composing", "breathing", "shaping"];
  static SIZES = [20, 64];

  static get observedAttributes() {
    return ["state", "size", "speed", "paused"];
  }

  constructor() {
    super();
    this.canvas = document.createElement("canvas");
    this.raf = null;
    this.visible = true;
    this.unmotion = null;
    this.observer = null;
    this.loop = this.loop.bind(this);
    this.sync = this.sync.bind(this);
  }

  get state() {
    const state = this.getAttribute("state") || "working";
    if (!ThinkingOrbElement.STATES.includes(state)) throw new Error(`thinking-orb: unknown state "${state}"`);
    return state;
  }

  get size() {
    const size = Number(this.getAttribute("size") || 20);
    if (!ThinkingOrbElement.SIZES.includes(size)) throw new Error(`thinking-orb: size must be 20 or 64, got "${this.getAttribute("size")}"`);
    return size;
  }

  connectedCallback() {
    if (!this.canvas.isConnected) this.appendChild(this.canvas);
    if (this.hasAttribute("aria-label")) this.setAttribute("role", "img");
    else this.setAttribute("aria-hidden", "true");
    this.configure();
    this.observer = new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.sync();
    });
    this.observer.observe(this);
    document.addEventListener("visibilitychange", this.sync);
    this.unmotion = Motion.subscribe(() => this.configure());
  }

  disconnectedCallback() {
    this.stop();
    if (this.observer) this.observer.disconnect();
    if (this.unmotion) this.unmotion();
    this.observer = null;
    this.unmotion = null;
    document.removeEventListener("visibilitychange", this.sync);
  }

  attributeChangedCallback() {
    if (this.isConnected) this.configure();
  }

  configure() {
    const size = this.size;
    const ratio = Motion.pixelRatio();
    const preset = window.ThinkingOrbsEngine.resolvePreset(this.state, size);
    this.ratio = ratio;
    this.px = size;
    this.draw = window.ThinkingOrbsEngine.MODE_DRAWS[preset.mode];
    this.opts = preset.opts;
    this.speed = preset.speed * Number(this.getAttribute("speed") || 1);
    this.dark = getComputedStyle(this).getPropertyValue("--orb-theme").trim() === "dark";

    this.canvas.width = Math.round(size * ratio);
    this.canvas.height = Math.round(size * ratio);
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;

    this.ctx = this.canvas.getContext("2d");
    this.frame(Motion.reduced ? 0.6 : Motion.now() * this.speed);
    this.sync();
  }

  frame(time) {
    this.ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
    this.ctx.clearRect(0, 0, this.px, this.px);
    this.draw(this.ctx, this.px, time, this.dark, this.opts);
  }

  loop() {
    this.frame(Motion.now() * this.speed);
    this.raf = requestAnimationFrame(this.loop);
  }

  stop() {
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  sync() {
    const run = this.isConnected && this.visible && document.visibilityState !== "hidden" && !this.hasAttribute("paused") && !Motion.reduced;
    if (run && this.raf === null) this.raf = requestAnimationFrame(this.loop);
    if (!run) this.stop();
  }
}

export class BackdropOrb {
  static HOLD_S = 14;
  static FADE_S = 2.4;
  static SPEED = 0.45;
  static SKIP = ["working", "shaping"];
  static DENSITY = { composing: 2 };
  static DOT = 0.6;
  static FPS = 30;

  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.states = ThinkingOrbElement.STATES.filter((state) => !BackdropOrb.SKIP.includes(state));
    this.presets = Object.fromEntries(this.states.map((state) => [state, this.preset(state)]));
    this.index = Math.floor(Math.random() * this.states.length);
    this.since = Motion.now();
    this.last = 0;
    this.stopped = false;
    this.raf = 0;
    this.loop = this.loop.bind(this);
    this.sync = this.sync.bind(this);

    new ResizeObserver(() => this.measure()).observe(canvas);
    document.addEventListener("visibilitychange", this.sync);
    Motion.subscribe(this.sync);
    this.measure();
    this.sync();
  }

  preset(state) {
    const engine = window.ThinkingOrbsEngine;
    const base = engine.resolvePreset(state, 64);
    const opts = engine.scaleRadii(engine.scaleCounts(base.opts, BackdropOrb.DENSITY[state] || 3), BackdropOrb.DOT);
    return { draw: engine.MODE_DRAWS[base.mode], speed: base.speed * BackdropOrb.SPEED, opts };
  }

  measure() {
    this.ratio = Motion.pixelRatio();
    this.px = this.canvas.clientWidth;
    if (!this.px) return;
    this.canvas.width = Math.round(this.px * this.ratio);
    this.canvas.height = Math.round(this.px * this.ratio);
    this.draw(Motion.now());
  }

  running() {
    return !this.stopped && !Motion.reduced && document.visibilityState !== "hidden";
  }

  sync() {
    const run = this.running();
    if (run && !this.raf) this.raf = requestAnimationFrame(this.loop);
    if (run) return;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (!this.stopped) this.draw(Motion.now());
  }

  setStopped(stopped) {
    this.stopped = stopped;
    if (!stopped) this.measure();
    this.sync();
  }

  advance(now) {
    if (now - this.since < BackdropOrb.HOLD_S + BackdropOrb.FADE_S) return;
    this.index = (this.index + 1) % this.states.length;
    this.since = now;
  }

  paint(state, now, alpha) {
    const preset = this.presets[state];
    this.ctx.globalAlpha = alpha;
    preset.draw(this.ctx, this.px, Motion.reduced ? 0.6 : now * preset.speed, true, preset.opts);
  }

  draw(now) {
    if (!this.px) return;
    const share = Math.min(1, Math.max(0, (now - this.since - BackdropOrb.HOLD_S) / BackdropOrb.FADE_S));
    const mix = Motion.reduced ? 0 : share * share * (3 - 2 * share);
    const next = this.states[(this.index + 1) % this.states.length];

    this.ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0);
    this.ctx.clearRect(0, 0, this.px, this.px);
    if (mix < 1) this.paint(this.states[this.index], now, 1 - mix);
    if (mix > 0) this.paint(next, now, mix);
    this.ctx.globalAlpha = 1;
  }

  loop() {
    this.raf = 0;
    if (!this.running()) return;
    const now = Motion.now();
    if (now - this.last >= 1 / BackdropOrb.FPS - 0.004) {
      this.last = now;
      this.advance(now);
      this.draw(now);
    }
    this.raf = requestAnimationFrame(this.loop);
  }
}

export class Orb {
  static html(state, size = 20, label = "") {
    return `<thinking-orb state="${state}" size="${size}"${label ? ` aria-label="${esc(label)}"` : ""}></thinking-orb>`;
  }

  static inline(text, state = "connecting") {
    return `<span class="orb-busy">${Orb.html(state, 20)}<span>${esc(text)}</span></span>`;
  }

  static panel(text, state = "searching") {
    return `<div class="orb-panel">${Orb.html(state, 64)}<span class="orb-panel__text">${esc(text)}</span></div>`;
  }
}

export class Busy {
  static FADE_MS = 240;

  static cover(host, text, state = "searching") {
    const node = document.createElement("div");
    node.className = "fx-busy";
    node.innerHTML = Orb.panel(text, state);
    host.classList.add("fx-busy-host");
    host.appendChild(node);
    return Busy.handle(node, () => {
      if (!host.querySelector(":scope > .fx-busy")) host.classList.remove("fx-busy-host");
    });
  }

  static inline(host, text, state = "connecting") {
    const node = document.createElement("span");
    node.className = "fx-inline";
    node.innerHTML = Orb.inline(text, state);
    host.appendChild(node);
    return Busy.handle(node, () => {});
  }

  static handle(node, after) {
    let finished = false;
    const remove = () => {
      node.remove();
      after();
    };
    return {
      done() {
        if (finished) return;
        finished = true;
        if (Motion.reduced) {
          remove();
          return;
        }
        node.classList.add("is-fading");
        setTimeout(remove, Busy.FADE_MS);
      },
    };
  }
}

customElements.define("thinking-orb", ThinkingOrbElement);
