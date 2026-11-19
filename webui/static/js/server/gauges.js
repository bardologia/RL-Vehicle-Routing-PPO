export const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const INK = {
  ink: "245, 248, 250",
  dim: "214, 223, 228",
  faint: "180, 193, 201",
  hint: "138, 152, 163",
  accent: "163, 192, 255",
  live: "94, 234, 212",
  warn: "252, 211, 77",
  fail: "248, 113, 113",
  pass: "74, 222, 128",
};

export class CanvasBase {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = 0;
    this.h = 0;
    this.raf = null;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(this.canvas);
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return;
    this.w = Math.max(1, rect.width);
    this.h = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.onResize();
  }

  dispose() {
    this.observer.disconnect();
    if (this.raf != null) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  onResize() {}

  font(weight, size) {
    return `${weight} ${size}px "JetBrains Mono", ui-monospace, monospace`;
  }
}

export class AnimatedGauge extends CanvasBase {
  constructor(canvas, opts, tuning) {
    super(canvas);
    this.min = opts.min != null ? opts.min : tuning.min;
    this.max = opts.max != null ? opts.max : tuning.max;
    this.k = tuning.k || 0;
    this.c = tuning.c || 0;
    this.ease = tuning.ease;
    this.settle = tuning.settle;
    this.v = this.min;
    this.t = this.min;
    this.vel = 0;
    this.peak = this.min;
    this.peakAt = 0;
    this.last = 0;
    this.tick = this.tick.bind(this);
    document.fonts.ready.then(() => {
      if (this.ready) this.draw();
    });
  }

  onResize() {
    if (this.ready) this.draw();
  }

  range(max) {
    if (max != null && max > this.min && max !== this.max) this.max = max;
  }

  set(value) {
    const now = performance.now();
    const v = Math.max(this.min, Math.min(this.max, value == null ? this.min : value));
    if (v >= this.peak || now - this.peakAt > 60000) {
      this.peak = v;
      this.peakAt = now;
    }
    this.t = v;
    if (REDUCED_MOTION) {
      this.v = v;
      this.vel = 0;
      this.draw();
      return;
    }
    this.wake();
  }

  wake() {
    if (this.raf != null) return;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.tick);
  }

  tick(now) {
    const dt = Math.min(0.05, Math.max(0.001, (now - this.last) / 1000));
    this.last = now;
    if (this.k) {
      const n = Math.max(1, Math.ceil(dt * 240));
      const h = dt / n;
      for (let i = 0; i < n; i++) {
        this.vel += (-this.k * (this.v - this.t) - this.c * this.vel) * h;
        this.v += this.vel * h;
      }
    } else {
      this.v += (this.t - this.v) * this.ease;
    }
    const span = this.max - this.min;
    const landed = Math.abs(this.t - this.v) < span * this.settle && Math.abs(this.vel) < span * this.settle * 12;
    if (landed) {
      this.v = this.t;
      this.vel = 0;
    }
    this.draw();
    if (landed) {
      this.raf = null;
      return;
    }
    this.raf = requestAnimationFrame(this.tick);
  }
}

export class DialGauge extends AnimatedGauge {
  static TRAIL = 24;

  constructor(canvas, opts) {
    super(canvas, opts, { min: 0, max: 100, k: 90, c: 13, settle: 0.0005 });
    this.label = opts.label || "";
    this.unit = opts.unit || "";
    this.color = opts.color || INK.accent;
    this.zones = opts.zones || [];
    this.majors = opts.majors != null ? opts.majors : 5;
    this.minors = opts.minors != null ? opts.minors : 4;
    this.digits = opts.digits != null ? opts.digits : 0;
    this.a0 = -Math.PI * 1.2;
    this.a1 = Math.PI * 0.2;
    this.trail = [];
    this.ready = true;
    this.draw();
  }

  set(value) {
    if (value != null) {
      this.trail.push(Math.max(this.min, Math.min(this.max, value)));
      if (this.trail.length > DialGauge.TRAIL) this.trail.shift();
    }
    super.set(value);
  }

  alarm() {
    return this.zones.length > 0 && this.v >= this.zones[this.zones.length - 1].from;
  }

  angle(v) {
    return this.a0 + ((v - this.min) / (this.max - this.min)) * (this.a1 - this.a0);
  }

  arc(cx, cy, r, from, to, color, lw) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.arc(cx, cy, r, from, to);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.stroke();
  }

  ray(cx, cy, a, r0, r1, color, lw) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
    ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.stroke();
  }

  draw() {
    if (this.w < 40 || this.h < 40) return;
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    ctx.clearRect(0, 0, w, h);

    const r = Math.max(14, Math.min((w - 80) / 2, (h - 48) / 1.492));
    const cx = w / 2;
    const top = 0.904 * r + 27;
    const cy = top + Math.max(0, h - (top + 0.588 * (r + 22) + 8)) / 2;
    const alarm = this.alarm();

    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.97, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(9, 13, 17, 0.8)";
    ctx.fill();
    ctx.strokeStyle = `rgba(${INK.ink}, 0.14)`;
    ctx.lineWidth = 1;
    ctx.stroke();

    const lw = Math.max(1.6, r * 0.04);
    this.arc(cx, cy, r, this.a0, this.a1, `rgba(${INK.ink}, 0.14)`, 1);
    if (this.v > this.min) this.arc(cx, cy, r, this.a0, this.angle(this.v), `rgba(${this.color}, 0.92)`, lw);

    this.zones.forEach((zone, index) => {
      const hot = alarm && index === this.zones.length - 1;
      this.arc(cx, cy, r + 4, this.angle(zone.from), this.angle(zone.to), `rgba(${zone.color}, ${hot ? 1 : 0.7})`, hot ? 2.2 : 1.4);
    });

    const steps = this.majors * this.minors;
    for (let i = 0; i <= steps; i++) {
      const major = i % this.minors === 0;
      const a = this.a0 + (i / steps) * (this.a1 - this.a0);
      this.ray(cx, cy, a, r + 7, r + (major ? 12 : 9), `rgba(${INK.ink}, ${major ? 0.5 : 0.2})`, 1);
    }

    ctx.font = this.font(600, 11);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = `rgb(${INK.hint})`;
    for (let m = 0; m <= this.majors; m++) {
      const value = this.min + (m / this.majors) * (this.max - this.min);
      const a = this.angle(value);
      ctx.fillText(String(Math.round(value)), cx + Math.cos(a) * (r + 22), cy + Math.sin(a) * (r + 22));
    }

    if (this.peak > this.v + (this.max - this.min) * 0.01) this.ray(cx, cy, this.angle(this.peak), r - 7, r - 2, `rgb(${INK.warn})`, 1.4);

    const n = this.trail.length;
    this.trail.forEach((value, index) => {
      const age = (n - 1 - index) / Math.max(1, DialGauge.TRAIL - 1);
      this.ray(cx, cy, this.angle(value), r * 0.34, r * 0.76, `rgba(${INK.ink}, ${(0.1 * (1 - age)).toFixed(3)})`, 1);
    });

    this.ray(cx, cy, this.angle(this.v), -r * 0.12, r * 0.78, `rgb(${INK.ink})`, 1.6);
    ctx.beginPath();
    ctx.arc(cx, cy, Math.max(2.6, r * 0.07), 0, Math.PI * 2);
    ctx.fillStyle = "rgb(9, 13, 17)";
    ctx.strokeStyle = alarm ? `rgb(${this.zones[this.zones.length - 1].color})` : `rgba(${INK.ink}, 0.8)`;
    ctx.lineWidth = 1.2;
    ctx.fill();
    ctx.stroke();

    const vy = cy + r * 0.42;
    ctx.font = this.font(700, r > 50 ? 18 : 14);
    ctx.fillStyle = `rgb(${INK.ink})`;
    ctx.fillText(`${this.v.toFixed(this.digits)}${this.unit}`, cx, vy);
    ctx.font = this.font(600, 11);
    ctx.fillStyle = `rgb(${INK.hint})`;
    ctx.fillText(this.label, cx, vy + (r > 50 ? 17 : 14));
  }
}

export class LinearMeter extends AnimatedGauge {
  constructor(canvas, opts) {
    super(canvas, opts, { min: 0, max: 100, ease: 0.14, settle: 0.0005 });
    this.label = opts.label || "";
    this.unit = opts.unit || "";
    this.color = opts.color || INK.accent;
    this.zones = opts.zones || [];
    this.digits = opts.digits != null ? opts.digits : 0;
    this.ready = true;
    this.draw();
  }

  x(v, x0, x1) {
    return x0 + ((v - this.min) / (this.max - this.min)) * (x1 - x0);
  }

  draw() {
    if (this.w < 60 || this.h < 30) return;
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    ctx.clearRect(0, 0, w, h);

    const x0 = 2;
    const x1 = w - 2;
    const ty = h - 11;

    ctx.textBaseline = "middle";
    ctx.font = this.font(700, 11);
    ctx.textAlign = "left";
    ctx.fillStyle = `rgb(${INK.hint})`;
    ctx.fillText(this.label, x0, 8);
    ctx.font = this.font(700, 12);
    ctx.textAlign = "right";
    ctx.fillStyle = `rgb(${INK.ink})`;
    ctx.fillText(`${this.v.toFixed(this.digits)}${this.unit}`, x1, 8);

    this.zones.forEach((zone) => {
      ctx.beginPath();
      ctx.moveTo(this.x(zone.from, x0, x1), ty - 7);
      ctx.lineTo(this.x(zone.to, x0, x1), ty - 7);
      ctx.strokeStyle = `rgba(${zone.color}, 0.75)`;
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    ctx.beginPath();
    ctx.moveTo(x0, ty);
    ctx.lineTo(x1, ty);
    ctx.strokeStyle = `rgba(${INK.ink}, 0.14)`;
    ctx.lineWidth = 2;
    ctx.stroke();

    if (this.v > this.min) {
      ctx.beginPath();
      ctx.moveTo(x0, ty);
      ctx.lineTo(this.x(this.v, x0, x1), ty);
      ctx.strokeStyle = `rgba(${this.color}, 0.75)`;
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    for (let i = 0; i <= 10; i++) {
      const tx = x0 + (i / 10) * (x1 - x0);
      const major = i % 5 === 0;
      ctx.beginPath();
      ctx.moveTo(tx, ty + 3);
      ctx.lineTo(tx, ty + (major ? 9 : 6));
      ctx.strokeStyle = `rgba(${INK.ink}, ${major ? 0.42 : 0.2})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (this.peak > this.v + (this.max - this.min) * 0.01) {
      const px = this.x(this.peak, x0, x1);
      ctx.beginPath();
      ctx.moveTo(px, ty - 9);
      ctx.lineTo(px, ty - 3);
      ctx.strokeStyle = `rgb(${INK.warn})`;
      ctx.lineWidth = 1.6;
      ctx.stroke();
    }

    const nx = this.x(this.v, x0, x1);
    ctx.beginPath();
    ctx.moveTo(nx, ty - 9);
    ctx.lineTo(nx, ty + 4);
    ctx.strokeStyle = `rgb(${INK.ink})`;
    ctx.lineWidth = 1.8;
    ctx.stroke();
  }
}

export class SegMeter extends AnimatedGauge {
  constructor(canvas, opts) {
    super(canvas, opts, { min: 0, max: 1, ease: 0.16, settle: 0.0006 });
    this.color = opts.color || INK.live;
    this.redFrom = opts.redFrom != null ? opts.redFrom : 0.92;
    this.ready = true;
    this.draw();
  }

  draw() {
    if (this.w < 30 || this.h < 6) return;
    const ctx = this.ctx;
    const w = this.w;
    const h = this.h;
    ctx.clearRect(0, 0, w, h);

    const gap = 2;
    const segs = Math.max(16, Math.min(56, Math.floor(w / 14)));
    const sw = (w - (segs - 1) * gap) / segs;
    const lit = this.v * segs;

    for (let i = 0; i < segs; i++) {
      const on = i + 1 <= lit ? 1 : Math.max(0, lit - i);
      const red = (i + 1) / segs > this.redFrom;
      const tint = red ? INK.fail : this.color;
      const alpha = on > 0 ? 0.25 + 0.65 * on : 0.1;
      ctx.fillStyle = on > 0 ? `rgba(${tint}, ${alpha.toFixed(3)})` : `rgba(${INK.ink}, ${alpha.toFixed(3)})`;
      ctx.fillRect(i * (sw + gap), 1, sw, h - 2);
    }
  }
}
