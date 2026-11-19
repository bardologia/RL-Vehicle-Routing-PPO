"use strict";

class HairlineCatalog {
  static FIGURES = { ...window.HairlineStock, ...RoutingFigures.CATALOG };

  static spec(name) {
    const spec = HairlineCatalog.FIGURES[name];
    if (!spec) throw new Error(`hairline-figure: unknown figure "${name}"`);
    return spec;
  }

  static stock(name) {
    return Object.prototype.hasOwnProperty.call(window.HairlineStock, name) && !Object.prototype.hasOwnProperty.call(RoutingFigures.CATALOG, name);
  }

  static value(spec, intensity) {
    const raw = intensity == null || intensity === "" ? 0.5 : Number(intensity);
    if (!Number.isFinite(raw)) throw new Error(`hairline-figure: intensity must be a number, got "${intensity}"`);
    const [lo, mid, hi] = spec.range;
    const i             = Math.min(1, Math.max(0, raw));
    const value         = i <= 0.5 ? lo + (i / 0.5) * (mid - lo) : mid + ((i - 0.5) / 0.5) * (hi - mid);
    return Math.round(value * 1000) / 1000;
  }
}

class HairlineAutopilot {
  static PERIOD_X  = 14;
  static PERIOD_Y  = 10;
  static REACH_X   = 0.28;
  static REACH_Y   = 0.25;
  static CYCLE     = 3.2;
  static HOVER     = 0.5;
  static DRIFT     = 0.06;
  static GLIDE     = HairlineAutopilot.DRIFT + (2 * (1 - HairlineAutopilot.DRIFT)) / (1 - HairlineAutopilot.HOVER);
  static STEP_MS   = 60;
  static HOP_MS    = 800;
  static RESUME_MS = 2200;
  static JITTER    = 0.04;
  static HOPS      = [[0.3, 0.2], [0.7, 0.6], [0.4, 0.3], [0.75, 0.75], [0.25, 0.4], [0.6, 0.8], [0.35, 0.5], [0.7, 0.2]];

  constructor(stage, hop = false) {
    this.stage    = stage;
    this.hop      = hop;
    this.phase    = Math.random() * Math.PI * 2;
    this.clock    = Math.random() * HairlineAutopilot.CYCLE;
    this.travel   = 0;
    this.hops     = Math.floor(Math.random() * HairlineAutopilot.HOPS.length);
    this.visible  = false;
    this.held     = false;
    this.timer    = 0;
    this.last     = 0;
    this.resume   = 0;
    this.frame    = this.frame.bind(this);
    this.hold     = (e) => { if (e.isTrusted) this._hold(); };
    this.release  = (e) => { if (e.isTrusted) this._release(); };
    this.sync     = () => this._sync();
    this.io       = new IntersectionObserver(([entry]) => { this.visible = entry.isIntersecting; this._sync(); }, { rootMargin: "80px" });
    this.unmotion = FxMotion.subscribe(this.sync);
    stage.addEventListener("pointermove", this.hold);
    stage.addEventListener("pointerdown", this.hold);
    stage.addEventListener("pointerleave", this.release);
    document.addEventListener("visibilitychange", this.sync);
    this.io.observe(stage);
  }

  get step() {
    return this.hop ? HairlineAutopilot.HOP_MS : HairlineAutopilot.STEP_MS;
  }

  _target() {
    if (this.hop) {
      const [u, v] = HairlineAutopilot.HOPS[this.hops % HairlineAutopilot.HOPS.length];
      const jitter = () => (Math.random() * 2 - 1) * HairlineAutopilot.JITTER;
      this.hops   += 1;
      return [u + jitter(), v + jitter()];
    }
    const u = 0.5 + HairlineAutopilot.REACH_X * this._sweep(this.travel / HairlineAutopilot.PERIOD_X + this.phase);
    const v = 0.5 + HairlineAutopilot.REACH_Y * this._sweep(this.travel / HairlineAutopilot.PERIOD_Y + 1.7 * this.phase);
    return [u, v];
  }

  frame() {
    const now    = performance.now();
    this.timer   = setTimeout(this.frame, this.step);
    const dt     = Math.min((2 * this.step) / 1000, (now - this.last) / 1000);
    this.last    = now;
    this.clock  += dt;
    this.travel += this._speed(this.clock) * dt;
    const box    = this.stage.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const [u, v] = this._target();
    this.stage.dispatchEvent(new PointerEvent("pointermove", { clientX: box.left + u * box.width, clientY: box.top + v * box.height, pointerType: "mouse" }));
  }

  _speed(clock) {
    const share = (clock / HairlineAutopilot.CYCLE) % 1;
    if (share < HairlineAutopilot.HOVER) return HairlineAutopilot.DRIFT;
    const swell = Math.sin((Math.PI * (share - HairlineAutopilot.HOVER)) / (1 - HairlineAutopilot.HOVER)) ** 2;
    return HairlineAutopilot.DRIFT + (HairlineAutopilot.GLIDE - HairlineAutopilot.DRIFT) * swell;
  }

  _sweep(cycles) {
    return (2 / Math.PI) * Math.asin(Math.sin(2 * Math.PI * cycles));
  }

  _hold() {
    clearTimeout(this.resume);
    this.held = true;
    this._sync();
  }

  _release() {
    clearTimeout(this.resume);
    this.resume = setTimeout(() => {
      this.held = false;
      this._sync();
    }, HairlineAutopilot.RESUME_MS);
  }

  _sync() {
    const run = this.visible && !this.held && document.visibilityState !== "hidden" && !window.REDUCED_MOTION && !HairlineFigureBase.held;
    if (run && !this.timer) this.timer = setTimeout(this.frame, this.step);
    if (!run && this.timer) {
      clearTimeout(this.timer);
      this.timer = 0;
    }
  }

  destroy() {
    clearTimeout(this.timer);
    clearTimeout(this.resume);
    this.io.disconnect();
    this.unmotion();
    this.stage.removeEventListener("pointermove", this.hold);
    this.stage.removeEventListener("pointerdown", this.hold);
    this.stage.removeEventListener("pointerleave", this.release);
    document.removeEventListener("visibilitychange", this.sync);
  }
}

class HairlineFigureElement extends HTMLElement {
  static NS = "http://www.w3.org/2000/svg";

  static get observedAttributes() {
    return ["name", "intensity"];
  }

  constructor() {
    super();
    this.figure    = null;
    this.svg       = null;
    this.autopilot = null;
    this.value     = null;
    this.ownLabel  = false;
    this.data      = undefined;
  }

  get state() {
    return this.data;
  }

  set state(value) {
    this.data = value;
    if (this.figure) this._applyState();
  }

  connectedCallback() {
    if (!this.figure) this._mount();
  }

  attributeChangedCallback(attribute, before, after) {
    if (!this.figure || before === after) return;
    if (attribute === "name") {
      this.data = undefined;
      this._unmount();
      this._mount();
      return;
    }
    const value = HairlineCatalog.value(HairlineCatalog.spec(this.getAttribute("name")), after);
    if (value === this.value) return;
    this.value = value;
    this.figure.set(value);
  }

  disconnectedCallback() {
    if (this.figure) this._unmount();
  }

  _mount() {
    const name = this.getAttribute("name");
    const spec = HairlineCatalog.spec(name);
    HL.inject(this.getRootNode());
    this.setAttribute("data-hairline", name);
    this.setAttribute("data-nofig", "");
    if (!this.hasAttribute("aria-hidden")) {
      this.setAttribute("role", "img");
      this.ownLabel = !this.hasAttribute("aria-label");
      if (this.ownLabel) this._label();
    }

    this.svg = document.createElementNS(HairlineFigureElement.NS, "svg");
    this.svg.setAttribute("viewBox", "0 0 400 320");
    this.svg.setAttribute("aria-hidden", "true");
    this.appendChild(this.svg);

    this.value  = HairlineCatalog.value(spec, this.getAttribute("intensity"));
    this.figure = spec.mount({ stage: this, svg: this.svg, read: { textContent: null } }, this.value);
    this._applyState();
    if (this.getAttribute("autopilot") !== "off") this.autopilot = new HairlineAutopilot(this, HairlineCatalog.stock(name));
  }

  _label() {
    this.setAttribute("aria-label", RoutingFigures.LABELS[this.getAttribute("name")]);
  }

  _applyState() {
    if (this.data === undefined) return;
    if (typeof this.figure.setState !== "function") throw new Error(`hairline-figure: figure "${this.getAttribute("name")}" takes no state`);
    this.figure.setState(this.data);
  }

  _unmount() {
    if (this.autopilot) this.autopilot.destroy();
    this.figure.destroy();
    this.svg.remove();
    if (this.ownLabel) this.removeAttribute("aria-label");
    this.figure    = null;
    this.svg       = null;
    this.autopilot = null;
    this.ownLabel  = false;
  }
}

class HairlineFigure {
  static html(name, attrs = "") {
    return `<hairline-figure name="${name}"${attrs ? " " + attrs : ""}></hairline-figure>`;
  }

  static empty(name, className, messageHtml) {
    return `<div class="${className} figure-empty">${HairlineFigure.html(name, 'aria-hidden="true"')}<div class="figure-empty__text">${messageHtml}</div></div>`;
  }
}

class MastFigures {
  constructor(mast) {
    this.mast      = mast;
    this.figures   = [...mast.querySelectorAll("hairline-figure")];
    this.autopilot = new HairlineAutopilot(mast);
    this.relay     = (e) => this._relay(e);
    this.leave     = (e) => { if (e.isTrusted) this._rest(); };
    mast.addEventListener("pointermove", this.relay);
    mast.addEventListener("pointerleave", this.leave);
  }

  _relay(e) {
    const box = this.mast.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const u = (e.clientX - box.left) / box.width;
    const v = (e.clientY - box.top) / box.height;
    for (const figure of this.figures) {
      const r = figure.getBoundingClientRect();
      if (!r.width) continue;
      figure.dispatchEvent(new PointerEvent("pointermove", { clientX: r.left + u * r.width, clientY: r.top + v * r.height, pointerType: "mouse" }));
    }
  }

  _rest() {
    for (const figure of this.figures) figure.dispatchEvent(new PointerEvent("pointerleave", { pointerType: "mouse" }));
  }

  destroy() {
    this.autopilot.destroy();
    this.mast.removeEventListener("pointermove", this.relay);
    this.mast.removeEventListener("pointerleave", this.leave);
  }
}

customElements.define("hairline-figure", HairlineFigureElement);
window.HairlineFigure = HairlineFigure;
window.MastFigures    = MastFigures;
