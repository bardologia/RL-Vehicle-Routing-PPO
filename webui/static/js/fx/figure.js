import { Motion } from "./orb.js";

class FigureAutopilot {
  static PERIOD_X = 14;
  static PERIOD_Y = 10;
  static REACH_X = 0.28;
  static REACH_Y = 0.25;
  static CYCLE = 3.2;
  static HOVER = 0.5;
  static DRIFT = 0.06;
  static GLIDE = FigureAutopilot.DRIFT + (2 * (1 - FigureAutopilot.DRIFT)) / (1 - FigureAutopilot.HOVER);
  static STEP_MS = 60;
  static RESUME_MS = 2200;

  constructor(stage) {
    this.stage = stage;
    this.phase = Math.random() * Math.PI * 2;
    this.clock = Math.random() * FigureAutopilot.CYCLE;
    this.travel = 0;
    this.visible = false;
    this.held = false;
    this.timer = 0;
    this.last = 0;
    this.resume = 0;
    this.frame = this.frame.bind(this);
    this.sync = this.sync.bind(this);
    this.hold = (event) => {
      if (event.isTrusted) this.setHeld(true);
    };
    this.release = (event) => {
      if (event.isTrusted) this.setHeld(false);
    };
    this.observer = new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
      this.sync();
    }, { rootMargin: "80px" });
    this.unmotion = Motion.subscribe(this.sync);
    stage.addEventListener("pointermove", this.hold);
    stage.addEventListener("pointerdown", this.hold);
    stage.addEventListener("pointerleave", this.release);
    document.addEventListener("visibilitychange", this.sync);
    this.observer.observe(stage);
  }

  static sweep(cycles) {
    return (2 / Math.PI) * Math.asin(Math.sin(2 * Math.PI * cycles));
  }

  static speed(clock) {
    const share = (clock / FigureAutopilot.CYCLE) % 1;
    if (share < FigureAutopilot.HOVER) return FigureAutopilot.DRIFT;
    const swell = Math.sin((Math.PI * (share - FigureAutopilot.HOVER)) / (1 - FigureAutopilot.HOVER)) ** 2;
    return FigureAutopilot.DRIFT + (FigureAutopilot.GLIDE - FigureAutopilot.DRIFT) * swell;
  }

  frame() {
    const now = performance.now();
    this.timer = setTimeout(this.frame, FigureAutopilot.STEP_MS);
    const dt = Math.min((2 * FigureAutopilot.STEP_MS) / 1000, (now - this.last) / 1000);
    this.last = now;
    this.clock += dt;
    this.travel += FigureAutopilot.speed(this.clock) * dt;
    const box = this.stage.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const u = 0.5 + FigureAutopilot.REACH_X * FigureAutopilot.sweep(this.travel / FigureAutopilot.PERIOD_X + this.phase);
    const v = 0.5 + FigureAutopilot.REACH_Y * FigureAutopilot.sweep(this.travel / FigureAutopilot.PERIOD_Y + 1.7 * this.phase);
    this.stage.dispatchEvent(new PointerEvent("pointermove", { clientX: box.left + u * box.width, clientY: box.top + v * box.height, pointerType: "mouse" }));
  }

  setHeld(held) {
    clearTimeout(this.resume);
    if (held) {
      this.held = true;
      this.sync();
      return;
    }
    this.resume = setTimeout(() => {
      this.held = false;
      this.sync();
    }, FigureAutopilot.RESUME_MS);
  }

  sync() {
    const run = this.visible && !this.held && document.visibilityState !== "hidden" && !Motion.reduced;
    if (run && !this.timer) this.timer = setTimeout(this.frame, FigureAutopilot.STEP_MS);
    if (!run && this.timer) {
      clearTimeout(this.timer);
      this.timer = 0;
    }
  }

  destroy() {
    clearTimeout(this.timer);
    clearTimeout(this.resume);
    this.observer.disconnect();
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
    return ["name"];
  }

  constructor() {
    super();
    this.figure = null;
    this.svg = null;
    this.autopilot = null;
  }

  connectedCallback() {
    if (!this.figure) this.mount();
  }

  attributeChangedCallback(attribute, before, after) {
    if (!this.figure || before === after) return;
    this.unmount();
    this.mount();
  }

  disconnectedCallback() {
    if (this.figure) this.unmount();
  }

  mount() {
    const name = this.getAttribute("name");
    const spec = window.HairlineStock[name];
    if (!spec) throw new Error(`hairline-figure: unknown figure "${name}"`);
    window.HL.inject(this.getRootNode());
    this.setAttribute("data-hairline", name);
    this.setAttribute("data-nofig", "");
    if (this.hasAttribute("aria-label")) this.setAttribute("role", "img");
    else this.setAttribute("aria-hidden", "true");

    this.svg = document.createElementNS(HairlineFigureElement.NS, "svg");
    this.svg.setAttribute("viewBox", "0 0 400 320");
    this.svg.setAttribute("aria-hidden", "true");
    this.appendChild(this.svg);

    this.figure = spec.mount({ stage: this, svg: this.svg, read: { textContent: null } }, spec.range[1]);
    if (this.getAttribute("autopilot") !== "off" && !Motion.reduced) this.autopilot = new FigureAutopilot(this);
  }

  unmount() {
    if (this.autopilot) this.autopilot.destroy();
    this.figure.destroy();
    this.svg.remove();
    this.figure = null;
    this.svg = null;
    this.autopilot = null;
  }
}

export class Figure {
  static NAMES = ["drawer", "terminal", "plug", "riffle", "loupe", "sieve", "elevator", "slow", "cabinet", "dish", "router", "patch", "phosphor"];

  static html(name, className = "") {
    if (!Figure.NAMES.includes(name)) throw new Error(`Figure: unknown figure "${name}"`);
    return `<hairline-figure name="${name}"${className ? ` class="${className}"` : ""}></hairline-figure>`;
  }

  static pagehead(head, name) {
    const holder = document.createElement("template");
    holder.innerHTML = Figure.html(name, "pagefig");
    head.classList.add("has-pagefig");
    head.appendChild(holder.content.firstElementChild);
  }
}

customElements.define("hairline-figure", HairlineFigureElement);
