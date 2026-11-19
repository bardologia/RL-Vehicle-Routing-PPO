"use strict";

class HairlineGeometry {
  static samples(points) {
    const kept = points.filter((p, i) => {
      const prev = points[(i + points.length - 1) % points.length];
      return Math.hypot(p[0] - prev[0], p[1] - prev[1]) > 1e-6;
    });
    const cu = kept.reduce((sum, p) => sum + p[0], 0) / kept.length;
    const cv = kept.reduce((sum, p) => sum + p[1], 0) / kept.length;
    return kept.map((p, i) => {
      const prev = kept[(i + kept.length - 1) % kept.length];
      const next = kept[(i + 1) % kept.length];
      const tu   = next[0] - prev[0];
      const tv   = next[1] - prev[1];
      const len  = Math.hypot(tu, tv);
      const out  = tv * (p[0] - cu) - tu * (p[1] - cv) >= 0 ? 1 : -1;
      return { u: p[0], v: p[1], nu: (out * tv) / len, nv: (-out * tu) / len };
    });
  }

  static inside(hull, p) {
    let sign = 0;
    for (let i = 0; i < hull.length; i++) {
      const a     = hull[i];
      const b     = hull[(i + 1) % hull.length];
      const cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
      if (Math.abs(cross) < 1e-9) continue;
      if (sign === 0) sign = Math.sign(cross);
      else if (Math.sign(cross) !== sign) return false;
    }
    return true;
  }
}

class HairlineFigureBase {
  static FRAME_MS = { calm: 1000 / 60 - 2, busy: 1000 / 20 - 2 };
  static MAX_DT   = 0.05;
  static SHOWN    = 0.35;
  static frameMs  = HairlineFigureBase.FRAME_MS.calm;
  static held     = false;
  static live     = new Set();

  static busy(on) {
    HairlineFigureBase.frameMs = on ? HairlineFigureBase.FRAME_MS.busy : HairlineFigureBase.FRAME_MS.calm;
  }

  static hold(on) {
    if (on === HairlineFigureBase.held) return;
    HairlineFigureBase.held = on;
    HL.setReducedMotion(on || window.REDUCED_MOTION);
    for (const fn of [...FxMotion.listeners]) fn(window.REDUCED_MOTION);
    if (on) return;
    for (const figure of HairlineFigureBase.live) figure.loop.wake();
  }

  constructor(svg, amount, scale) {
    this.amount = amount;
    this.chosen = null;
    this.C      = HL.Cam(45, 0.5, scale);
    this.g      = HL.mk("g", {}, svg);
    this.paced  = -Infinity;
    this.owed   = 0;
    this.parked = false;
  }

  static nearest(values, target) {
    let best = 0;
    for (let i = 1; i < values.length; i++) if (Math.abs(values[i] - target) < Math.abs(values[best] - target)) best = i;
    return best;
  }

  _camera(points, cy) {
    HL.fit(this.C, points, 200, cy);
    this.P     = HL.proj(this.C);
    this.front = HL.facing(this.C);
  }

  _shifted(dx, dy, dz) {
    return (x, y, z) => this.P(x + dx, y + dy, z + dz);
  }

  _onScreen(r, d) {
    return [(r + d) * Math.SQRT1_2, (d - r) * Math.SQRT1_2];
  }

  _screenBox(r0, d0, r1, d1, radius) {
    const corners = [[r1, d0], [r1, d1], [r0, d1], [r0, d0]].map(([r, d]) => this._onScreen(r, d));
    return HairlineGeometry.samples(HL.fillet(corners, [radius, radius, radius, radius], 4));
  }

  _plinth(x0, y0, x1, y1, r, depth) {
    const [ring, inner] = HL.rings(x0, y0, x1, y1, r, 2.2);
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ring, inner, -depth, 0));
  }

  _pace(dt, now) {
    if (this.parked || HairlineFigureBase.held) return false;
    this.owed += dt;
    if (now - this.paced < HairlineFigureBase.frameMs) return true;
    const owed = Math.min(HairlineFigureBase.MAX_DT, this.owed);
    this.paced = now;
    this.owed  = 0;
    return this._tick(owed, now);
  }

  _watch(stage) {
    const io     = new IntersectionObserver(([entry]) => {
      const parked = !entry.isIntersecting || entry.intersectionRatio < HairlineFigureBase.SHOWN;
      if (parked === this.parked) return;
      this.parked = parked;
      if (!parked) this.loop.wake();
    }, { rootMargin: "0px", threshold: [0, HairlineFigureBase.SHOWN, 1] });
    io.observe(stage);
    return () => io.disconnect();
  }

  _start(stage, svg) {
    this.bag  = HL.disposer();
    this.loop = HL.register(stage, (dt, now) => this._pace(dt, now));
    this.bag.add(this.loop.unregister);
    this.bag.add(this._watch(stage));
    this.bag.add(HL.pointer(stage, { move: (p) => this._move(p), leave: () => this._leave() }));
    this.bag.add(() => svg.replaceChildren());
    HairlineFigureBase.live.add(this);
    this.bag.add(() => HairlineFigureBase.live.delete(this));
  }

  _advance(items, now) {
    let moving = false;
    for (const item of items) {
      const value = HL.tval(item.tw, now);
      if (!HL.tdone(item.tw, now)) moving = true;
      if (value === item.drawn) continue;
      item.drawn = value;
      this._draw(item, value);
    }
    return moving;
  }

  _leave() {
    this._choose(null);
  }

  _choose(key) {
    if (key === this.chosen) return;
    this.chosen = key;
    this._retarget(performance.now());
  }

  set(value) {
    this.amount = value;
    this._retarget(performance.now());
  }

  destroy() {
    this.bag.dispose();
  }
}

