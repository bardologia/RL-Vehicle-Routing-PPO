"use strict";

class FxCheck {
  static object(name, path, value, keys) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${name}: ${path} must be an object with ${keys.join(", ")}, got ${value}`);
    const prefix = path === "state" ? "" : `${path}.`;
    for (const key of Object.keys(value)) if (!keys.includes(key)) throw new Error(`${name}: ${prefix}${key} is not a known key (expected ${keys.join(", ")})`);
    for (const key of keys) if (!(key in value)) throw new Error(`${name}: ${prefix}${key} is missing`);
    return value;
  }

  static number(name, key, value, lo = -Infinity, hi = Infinity) {
    if (typeof value !== "number" || !Number.isFinite(value)) throw new Error(`${name}: ${key} must be a finite number, got ${value}`);
    if (value < lo || value > hi) throw new Error(`${name}: ${key} must be in [${lo}, ${hi}], got ${value}`);
    return value;
  }

  static positive(name, key, value) {
    FxCheck.number(name, key, value);
    if (value <= 0) throw new Error(`${name}: ${key} must be above 0, got ${value}`);
    return value;
  }

  static integer(name, key, value, lo, hi) {
    FxCheck.number(name, key, value, lo, hi);
    if (!Number.isInteger(value)) throw new Error(`${name}: ${key} must be an integer, got ${value}`);
    return value;
  }

  static boolean(name, key, value) {
    if (typeof value !== "boolean") throw new Error(`${name}: ${key} must be a boolean, got ${value}`);
    return value;
  }

  static list(name, key, value, length) {
    const listed = Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));
    if (!listed) throw new Error(`${name}: ${key} must be an array of length ${length}, got ${value}`);
    if (value.length !== length) throw new Error(`${name}: ${key} must have length ${length}, got ${value.length}`);
    return value;
  }

  static shares(name, key, value, length) {
    FxCheck.list(name, key, value, length);
    let sum = 0;
    for (let i = 0; i < value.length; i++) sum += FxCheck.number(name, `${key}[${i}]`, value[i], 0, 1);
    if (sum !== 0 && Math.abs(sum - 1) > 1e-6) throw new Error(`${name}: ${key} must sum to 1 within 1e-6 or be all zeros, got a sum of ${sum}`);
    return Array.from(value);
  }
}

class FxLight {
  constructor() {
    this.el = null;
    this.lo = false;
  }

  set(el) {
    if (el === this.el) return;
    if (this.el) {
      this.el.classList.remove("hi");
      if (this.lo) this.el.classList.add("lo");
    }
    this.el = el;
    this.lo = false;
    if (!el) return;
    this.lo = el.classList.contains("lo");
    el.classList.remove("lo");
    el.classList.add("hi");
  }
}

class FxPrism {
  static EMPTY = { sil: "", crease: "" };

  constructor(P, front, ring, inner) {
    const o              = P(0, 0, 0);
    const ex             = P(1, 0, 0);
    const ey             = P(0, 1, 0);
    const ez             = P(0, 0, 1);
    this.ax              = ex[0] - o[0];
    this.ay              = ex[1] - o[1];
    this.bx              = ey[0] - o[0];
    this.by              = ey[1] - o[1];
    this.cy              = ez[1] - o[1];
    const [lower, upper] = FxPrism.chains(ring.map((q) => P(q.u, q.v, 0)));
    this.lower           = lower;
    this.upper           = upper;
    this.crease          = inner ? HL.run(inner, front).map((q) => P(q.u, q.v, 0)) : [];
  }

  static chains(flat) {
    const n   = flat.length;
    let left  = 0;
    let right = 0;
    for (let i = 1; i < n; i++) {
      const [x, y] = flat[i];
      if (x < flat[left][0] - 1e-6 || (Math.abs(x - flat[left][0]) <= 1e-6 && y > flat[left][1])) left = i;
      if (x > flat[right][0] + 1e-6 || (Math.abs(x - flat[right][0]) <= 1e-6 && y > flat[right][1])) right = i;
    }
    const a = FxPrism.walk(flat, left, right);
    const b = FxPrism.walk(flat, right, left);
    return FxPrism.meanY(a) >= FxPrism.meanY(b) ? [a, b] : [b.reverse(), a.reverse()];
  }

  static walk(flat, from, to) {
    const out = [];
    for (let i = from; ; i = (i + 1) % flat.length) {
      out.push(flat[i]);
      if (i === to) return out;
    }
  }

  static meanY(points) {
    return points.reduce((sum, p) => sum + p[1], 0) / points.length;
  }

  static r(v) {
    return Math.round(v * 100) / 100;
  }

  static closed(lower, ox, lo, upper, hi) {
    let d = "M";
    for (const p of lower) d += `${FxPrism.r(p[0] + ox)} ${FxPrism.r(p[1] + lo)}L`;
    for (const p of upper) d += `${FxPrism.r(p[0] + ox)} ${FxPrism.r(p[1] + hi)}L`;
    return d.slice(0, -1) + "Z";
  }

  static line(points, ox, oy) {
    if (points.length < 2) return "";
    let d = "M";
    for (const p of points) d += `${FxPrism.r(p[0] + ox)} ${FxPrism.r(p[1] + oy)}L`;
    return d.slice(0, -1);
  }

  path(x, y, z0, z1) {
    const ox = x * this.ax + y * this.bx;
    const oy = x * this.ay + y * this.by;
    const lo = oy + z0 * this.cy;
    const hi = oy + z1 * this.cy;
    return { sil: FxPrism.closed(this.lower, ox, lo, this.upper, hi), crease: FxPrism.line(this.crease, ox, hi) };
  }

  put(solid, x, y, z0, z1) {
    HL.put(solid, this.path(x, y, z0, z1));
  }
}

class FxRim {
  static STEPS = 16;

  constructor(P, r, plane) {
    const o     = P(0, 0, 0);
    this.points = Array.from({ length: FxRim.STEPS }, (_, k) => {
      const a = (k / FxRim.STEPS) * Math.PI * 2;
      const q = plane === "x" ? P(0, r * Math.cos(a), r * Math.sin(a)) : P(r * Math.cos(a), 0, r * Math.sin(a));
      return [q[0] - o[0], q[1] - o[1]];
    });
    this.P      = P;
  }

  path(x, y, z) {
    const [ox, oy] = this.P(x, y, z);
    let d          = "M";
    for (const p of this.points) d += `${FxPrism.r(p[0] + ox)} ${FxPrism.r(p[1] + oy)}L`;
    return d.slice(0, -1) + "Z";
  }
}

class FxShift {
  constructor(P) {
    const o = P(0, 0, 0);
    const x = P(1, 0, 0);
    const y = P(0, 1, 0);
    const z = P(0, 0, 1);
    this.ax = x[0] - o[0];
    this.ay = x[1] - o[1];
    this.bx = y[0] - o[0];
    this.by = y[1] - o[1];
    this.cy = z[1] - o[1];
  }

  of(x, y, z) {
    return `translate(${FxPrism.r(x * this.ax + y * this.bx)} ${FxPrism.r(x * this.ay + y * this.by + z * this.cy)})`;
  }
}

class FxDepth {
  constructor(parent) {
    this.layer = HL.mk("g", {}, parent);
    this.items = [];
    this.order = [];
    this.dirty = true;
  }

  add(depth) {
    const item = { group: HL.mk("g", {}, this.layer), depth };
    this.items.push(item);
    this.dirty = true;
    return item;
  }

  move(item, depth) {
    if (item.depth === depth) return;
    item.depth = depth;
    this.dirty = true;
  }

  remove(item) {
    item.group.remove();
    this.items = this.items.filter((other) => other !== item);
    this.order = this.order.filter((other) => other !== item);
    this.dirty = true;
  }

  sort() {
    if (!this.dirty) return;
    this.dirty   = false;
    const sorted = this.items.slice().sort((a, b) => a.depth - b.depth);
    let same     = sorted.length === this.order.length;
    for (let i = 0; same && i < sorted.length; i++) same = sorted[i] === this.order[i];
    if (same) return;
    for (const item of sorted) this.layer.appendChild(item.group);
    this.order = sorted;
  }
}

class FxPolyline {
  constructor(points) {
    this.points  = points;
    this.lengths = [0];
    for (let k = 1; k < points.length; k++) this.lengths.push(this.lengths[k - 1] + Math.hypot(points[k][0] - points[k - 1][0], points[k][1] - points[k - 1][1]));
    this.total   = this.lengths[this.lengths.length - 1];
  }

  at(t) {
    const target = HL.clamp(t, 0, 1) * this.total;
    for (let k = 1; k < this.points.length; k++) {
      if (target > this.lengths[k] && k < this.points.length - 1) continue;
      const span = this.lengths[k] - this.lengths[k - 1];
      const s    = span > 0 ? HL.clamp((target - this.lengths[k - 1]) / span, 0, 1) : 1;
      return [HL.lerp(this.points[k - 1][0], this.points[k][0], s), HL.lerp(this.points[k - 1][1], this.points[k][1], s)];
    }
    return this.points[0].slice();
  }
}

class FxRank {
  static terciles(values) {
    const live = values.map((v, i) => [v, i]).filter(([v]) => v > 0).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    const out  = values.map(() => "dot off");
    const n    = live.length;
    let rank   = 0;
    live.forEach(([v, i], k) => {
      if (k > 0 && v < live[k - 1][0]) rank = k;
      out[i] = rank < n / 3 ? "dot" : rank < (2 * n) / 3 ? "dot m" : "dot off";
    });
    return out;
  }

  static modal(shares) {
    let best = 0;
    for (let i = 1; i < shares.length; i++) if (shares[i] > 0 && (best === 0 || shares[i] > shares[best])) best = i;
    return best;
  }

  static runnerUp(shares, modal) {
    let best = -1;
    for (let i = 1; i < shares.length; i++) if (i !== modal && shares[i] > 0 && (best < 0 || shares[i] > shares[best])) best = i;
    return best;
  }

  static smooth(s) {
    const t = HL.clamp(s, 0, 1);
    return t * t * (3 - 2 * t);
  }
}

class FxBalanceFigure extends HairlineFigureBase {
  static NAME    = "balance";
  static RANGE   = [0.4, 0.6, 0.9];
  static SCALE   = 3.45;
  static ARM     = 34;
  static POST    = 40;
  static STRING  = 18;
  static PAN     = 9;
  static ALONG   = [Math.SQRT1_2, -Math.SQRT1_2];
  static ACROSS  = [Math.SQRT1_2, Math.SQRT1_2];
  static COIN    = [3, 1.3, 1.65];
  static COINS   = 8;
  static TILT    = 9;
  static SWAY_S  = 4.2;
  static DROP_S  = 2.2;
  static FALL    = 0.5;
  static HOLD    = 0.9;
  static FADE    = 0.4;
  static DROP_H  = 12;
  static DEFAULT = { mine: 1200000, others: 1150000, alarm: false };

  constructor({ stage, svg }, value) {
    super(svg, value, FxBalanceFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    this.drop  = { pan: null, born: -1e9, drawn: "" };
    this.next  = 0.8;
    this._camera(this._extent(), 150);
    this.shift = new FxShift(this.P);
    this.coin  = new FxPrism(this.P, this.front, HL.circ(FxBalanceFigure.COIN[0], 28), HL.circ(FxBalanceFigure.COIN[0] - 0.5, 28));
    this._buildStand();
    this._buildPans();
    this._buildBeam();
    this._apply(FxBalanceFigure.DEFAULT);
    this.tilt  = HL.spring(this.theta, { k: 60, c: 9 });
    this.drawn = NaN;
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const reach    = FxBalanceFigure.ARM + FxBalanceFigure.PAN;
    const [ax, ay] = FxBalanceFigure.ALONG;
    const swing    = HL.rad(FxBalanceFigure.TILT + FxBalanceFigure.RANGE[2]);
    const low      = FxBalanceFigure.POST - FxBalanceFigure.STRING - FxBalanceFigure.ARM * Math.sin(swing) - 2;
    return [[-14, -14, -4], [14, 14, -4], [ax * reach, ay * reach, low], [-ax * reach, -ay * reach, low], [ax * reach, ay * reach, FxBalanceFigure.POST + 10], [-ax * reach, -ay * reach, FxBalanceFigure.POST + 10]];
  }

  _buildStand() {
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, HL.circ(14, 64), HL.circ(12.6, 64), -4, 0));
    const [ring, inner] = HL.rings(-1.8, -1.8, 1.8, 1.8, 1.2, 0.5);
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ring, inner, 0, FxBalanceFigure.POST));
  }

  _buildPans() {
    const [dx, dy] = FxBalanceFigure.ALONG;
    const [ex, ey] = FxBalanceFigure.ACROSS;
    const pivot    = FxBalanceFigure.POST + 2;
    this.pans      = [-1, 1].map((side, k) => {
      const group   = HL.mk("g", {}, this.g);
      const x       = side * FxBalanceFigure.ARM * dx;
      const y       = side * FxBalanceFigure.ARM * dy;
      const z       = pivot - FxBalanceFigure.STRING;
      const rim     = (s) => this.P(x + s * FxBalanceFigure.PAN * 0.85 * dx, y + s * FxBalanceFigure.PAN * 0.85 * dy, z);
      const top     = this.P(x, y, pivot - 1.4);
      HL.mk("path", { class: "nf", d: HL.seg(top, rim(-1)) + HL.seg(top, rim(1)) }, group);
      const el      = HL.solid(group);
      HL.put(el, HL.prism(this._shifted(x, y, 0), this.front, HL.circ(FxBalanceFigure.PAN, 48), HL.circ(FxBalanceFigure.PAN - 1, 48), z - 1.4, z));
      const dots    = Array.from({ length: k + 1 }, (_, j) => {
        const along = (j - k / 2) * 2.4;
        const out   = 0.5 * FxBalanceFigure.PAN;
        const dot   = HL.flatDot(group, this.C, 0.9, "dot m");
        HL.place(dot, this.P(x + out * ex + along * dx, y + out * ey + along * dy, z));
        return dot;
      });
      const stackAt = [x - 0.42 * FxBalanceFigure.PAN * ex, y - 0.42 * FxBalanceFigure.PAN * ey];
      const coins   = Array.from({ length: FxBalanceFigure.COINS + 1 }, () => HL.solid(group));
      return { side, k, group, el, dots, coins, stackAt, z, count: -1, shift: "" };
    });
  }

  _buildBeam() {
    this.beam      = HL.solid(this.g);
    this.beamRing  = HL.rrect(-FxBalanceFigure.ARM - 3, -2.2, FxBalanceFigure.ARM + 3, 2.2, 1.6, 4);
    this.beamInner = HL.rrect(-FxBalanceFigure.ARM - 2.2, -1.4, FxBalanceFigure.ARM + 2.2, 1.4, 1, 4);
  }

  setState(state) {
    const name = FxBalanceFigure.NAME;
    FxCheck.object(name, "state", state, ["mine", "others", "alarm"]);
    const next = { mine: FxCheck.number(name, "mine", state.mine), others: FxCheck.number(name, "others", state.others), alarm: FxCheck.boolean(name, "alarm", state.alarm) };
    const now  = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    const big    = Math.max(Math.abs(state.mine), Math.abs(state.others), 1);
    const unit   = big / 64;
    const coins  = (v) => Math.min(FxBalanceFigure.COINS, Math.ceil(Math.log2(1 + Math.max(0, v) / unit)));
    this.data    = state;
    this.counts  = [coins(state.mine), coins(state.others)];
    this.theta   = HL.clamp((state.mine - state.others) / (0.02 * big), -1, 1) * HL.rad(FxBalanceFigure.TILT);
    this.heavier = this.theta > 0 ? 0 : this.theta < 0 ? 1 : this.counts[1] > this.counts[0] ? 1 : 0;
    for (const pan of this.pans) {
      for (const dot of pan.dots) dot.setAttribute("class", pan.k === this.heavier ? "dot" : "dot m");
      if (pan.count === this.counts[pan.k]) continue;
      pan.count = this.counts[pan.k];
      this._stack(pan);
    }
  }

  _stack(pan) {
    const [, thick, pitch] = FxBalanceFigure.COIN;
    pan.coins.forEach((coin, i) => {
      if (i === FxBalanceFigure.COINS) return;
      const z0 = pan.z + i * pitch;
      HL.put(coin, i < pan.count ? this.coin.path(pan.stackAt[0], pan.stackAt[1], z0, z0 + thick) : FxPrism.EMPTY);
    });
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const sway  = still ? 0 : HL.rad(this.amount) * Math.sin((2 * Math.PI * this.clock) / FxBalanceFigure.SWAY_S);

    this.tilt.t = this.theta + sway;

    const moving = HL.stepS(this.tilt, dt);
    if (this.tilt.x !== this.drawn) {
      this.drawn = this.tilt.x;
      this._drawScale(this.tilt.x);
    }
    if (!still) this._drop();
    return still ? moving : true;
  }

  _drawScale(theta) {
    const [dx, dy] = FxBalanceFigure.ALONG;
    const [ex, ey] = FxBalanceFigure.ACROSS;
    const c        = Math.cos(theta);
    const s        = Math.sin(theta);
    const pivot    = FxBalanceFigure.POST + 2;
    const at       = (u, v, w) => this.P(u * c * dx + v * ex - w * s * dx, u * c * dy + v * ey - w * s * dy, pivot + u * s + w * c);
    const keep     = (q) => this.front({ nu: q.nu * dx + q.nv * ex, nv: q.nu * dy + q.nv * ey });
    const top      = this.beamRing.map((q) => at(q.u, q.v, 1.4));
    const bottom   = this.beamRing.map((q) => at(q.u, q.v, -1.4));
    HL.put(this.beam, { sil: HL.poly(HL.hull(top.concat(bottom))), crease: HL.open(HL.run(this.beamInner, keep).map((q) => at(q.u, q.v, 1.4))) });
    for (const pan of this.pans) {
      const u     = pan.side * FxBalanceFigure.ARM;
      const shift = this.shift.of((u * c - u) * dx, (u * c - u) * dy, u * s);
      if (shift === pan.shift) continue;
      pan.shift   = shift;
      pan.group.setAttribute("transform", shift);
    }
  }

  _drop() {
    if (this.clock >= this.next) {
      this.next      = this.clock + FxBalanceFigure.DROP_S;
      this.drop.pan  = this.pans[this.heavier];
      this.drop.born = this.clock;
      for (const pan of this.pans) HL.put(pan.coins[FxBalanceFigure.COINS], FxPrism.EMPTY);
    }
    const pan = this.drop.pan;
    if (!pan) return;
    const age                    = this.clock - this.drop.born;
    const [, thick, pitch]       = FxBalanceFigure.COIN;
    const rest                   = pan.z + pan.count * pitch;
    const fall                   = HL.clamp(age / FxBalanceFigure.FALL, 0, 1);
    const fade                   = HL.clamp((age - FxBalanceFigure.FALL - FxBalanceFigure.HOLD) / FxBalanceFigure.FADE, 0, 1);
    const z                      = rest + FxBalanceFigure.DROP_H * (1 - fall * fall);
    const h                      = thick * (1 - fade);
    const key                    = `${z}|${h}`;
    if (key === this.drop.drawn) return;
    this.drop.drawn              = key;
    HL.put(pan.coins[FxBalanceFigure.COINS], h < 0.1 ? FxPrism.EMPTY : this.coin.path(pan.stackAt[0], pan.stackAt[1], z, z + h));
  }

  _move(p) {
    const centre = this.P(0, 0, FxBalanceFigure.POST)[0];
    this._choose(p[0] < centre ? 0 : 1);
  }

  _retarget() {
    const lit  = this.chosen === null ? this.heavier : this.chosen;
    const pan  = this.pans[lit];
    const coin = this.data.alarm && lit === 1 && pan.count > 0 ? pan.coins[pan.count - 1].sil : null;
    this.light.set(coin || pan.el.sil);
    this.loop.wake();
  }
}


class FxTermsFigure extends HairlineFigureBase {
  static NAME    = "terms";
  static RANGE   = [3, 6, 10];
  static SCALE   = 3.6;
  static COUNT   = 6;
  static HEIGHTS = [30, 25.2, 20.8, 16.8, 13.2, 10];
  static FLAT    = 1.2;
  static WIDTH   = 11.6;
  static GAP     = 1.4;
  static DEPTH   = 16;
  static REACH   = 1.6;
  static STAGGER = 45;
  static CAP     = 1.4;
  static PUCK    = [2.5, 1.9];
  static STEP_S  = 2.5;
  static LEAD_S  = 4;
  static HOP_S   = 0.7;
  static ARC     = 6;
  static REST    = 0.35;
  static DEFAULT = { values: [-3.1, 1.2, -0.4, 2.6, -1.8, -4.7] };

  constructor({ stage, svg }, value) {
    super(svg, value, FxTermsFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    this.tour  = { at: 0, since: 0, hop: null };
    this.span  = FxTermsFigure.COUNT * (FxTermsFigure.WIDTH + FxTermsFigure.GAP) - FxTermsFigure.GAP;
    this._camera(this._extent(), 170);
    this._buildBase();
    this._buildSteps();
    this._buildPuck();
    this._apply(FxTermsFigure.DEFAULT, performance.now());
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const half = this.span / 2 + 6;
    const top  = FxTermsFigure.HEIGHTS[0] + FxTermsFigure.RANGE[2] + FxTermsFigure.PUCK[1];
    return [[...this._onScreen(-half, -6), -4], [...this._onScreen(half, -6), -4], [...this._onScreen(-half, FxTermsFigure.DEPTH + 6), -4], [...this._onScreen(half, FxTermsFigure.DEPTH + 6), -4], [...this._onScreen(0, 0), top]];
  }

  _buildBase() {
    const half  = this.span / 2 + 6;
    const ring  = this._screenBox(-half, -6, half, FxTermsFigure.DEPTH + 6, 6);
    const inner = this._screenBox(-half + 2.2, -3.8, half - 2.2, FxTermsFigure.DEPTH + 3.8, 4);
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ring, inner, -4, 0));
  }

  _buildSteps() {
    const pitch = FxTermsFigure.WIDTH + FxTermsFigure.GAP;
    const d     = FxTermsFigure.DEPTH;
    this.steps  = Array.from({ length: FxTermsFigure.COUNT }, (_, k) => {
      const r0    = -this.span / 2 + k * pitch;
      const r1    = r0 + FxTermsFigure.WIDTH;
      const group = HL.mk("g", {}, this.g);
      const el    = HL.solid(group);
      const cap   = HL.solid(group);
      cap.sil.classList.add("dash");
      const dot   = HL.flatDot(group, this.C, 0.9, "dot m");
      const step  = { k, el, cap, dot, centre: this._onScreen((r0 + r1) / 2, d / 2), front: this._onScreen((r0 + r1) / 2, d - 3), shape: new FxPrism(this.P, this.front, this._screenBox(r0, 0, r1, d, 2), this._screenBox(r0 + 1, 1, r1 - 1, d - 1, 1.2)), capShape: new FxPrism(this.P, this.front, this._screenBox(r0 + 0.9, 0.9, r1 - 0.9, d - 0.9, 1.6), null), hw: HL.tween(FxTermsFigure.FLAT), tw: HL.tween(0), value: null, drawn: "", restX: this.P(...this._onScreen((r0 + r1) / 2, d / 2), 0)[0] };
      return step;
    });
  }

  _buildPuck() {
    const [r] = FxTermsFigure.PUCK;
    this.puck = { el: HL.solid(this.g), shape: new FxPrism(this.P, this.front, HL.circ(r, 24), HL.circ(r - 0.5, 24)), drawn: "" };
  }

  setState(state) {
    const name   = FxTermsFigure.NAME;
    FxCheck.object(name, "state", state, ["values"]);
    FxCheck.list(name, "values", state.values, FxTermsFigure.COUNT);
    const values = Array.from(state.values, (v, k) => (v === null ? null : FxCheck.number(name, `values[${k}]`, v)));
    const now    = performance.now();
    this._apply({ values }, now);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state, now) {
    const ranked = state.values.map((v, k) => [v, k]).filter(([v]) => v !== null).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
    const rank   = new Map();
    ranked.forEach(([v, k], n) => rank.set(k, n > 0 && v === ranked[n - 1][0] ? rank.get(ranked[n - 1][1]) : n));
    this.order   = ranked.map(([, k]) => k);
    this.leader  = this.order.length ? this.order[0] : 0;
    for (const step of this.steps) {
      const v    = state.values[step.k];
      step.value = v;
      HL.tset(step.hw, v === null ? FxTermsFigure.FLAT : FxTermsFigure.HEIGHTS[rank.get(step.k)], now, step.k * 30);
      step.el.sil.classList.toggle("dash", v === null);
      step.dot.setAttribute("class", v === null ? "dot off" : v > 0 ? "dot" : v < 0 ? "dot off" : "dot m");
      step.dot.setAttribute("visibility", v === null ? "hidden" : "visible");
      step.drawn = "";
    }
    if (this.order.length && !this.order.includes(this.tour.at)) this.tour = { at: this.leader, since: this.clock, hop: null };
  }

  _top(k, now) {
    const step = this.steps[k];
    return HL.tval(step.hw, now) + HL.tval(step.tw, now) + (step.value !== null && step.value < 0 ? FxTermsFigure.CAP : 0);
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    let moving  = false;
    for (const step of this.steps) moving = this._drawStep(step, now) || moving;
    if (!still) this._runTour();
    this._drawPuck(now);
    return still ? moving : true;
  }

  _drawStep(step, now) {
    const h    = HL.tval(step.hw, now);
    const lift = HL.tval(step.tw, now);
    const busy = !HL.tdone(step.hw, now) || !HL.tdone(step.tw, now);
    const key  = `${h}|${lift}`;
    if (key === step.drawn) return busy;
    step.drawn = key;
    const top  = h + lift;
    step.shape.put(step.el, 0, 0, 0, top);
    const cap  = step.value !== null && step.value < 0;
    HL.put(step.cap, cap ? step.capShape.path(0, 0, top, top + FxTermsFigure.CAP) : FxPrism.EMPTY);
    HL.place(step.dot, this.P(step.front[0], step.front[1], top + (cap ? FxTermsFigure.CAP : 0)));
    return busy;
  }

  _runTour() {
    const n    = this.order.length;
    const tour = this.tour;
    if (n < 2) return;
    if (tour.hop) {
      if (this.clock - tour.hop.t0 < FxTermsFigure.HOP_S) return;
      tour.at    = tour.hop.to;
      tour.since = tour.hop.t0;
      tour.hop   = null;
      return;
    }
    const stay = tour.at === this.leader ? FxTermsFigure.LEAD_S : FxTermsFigure.STEP_S;
    if (this.clock - tour.since < stay) return;
    tour.hop   = { from: tour.at, to: this.order[(this.order.indexOf(tour.at) + 1) % n], t0: this.clock };
  }

  _drawPuck(now) {
    const tour = this.tour;
    if (!this.order.length) return HL.put(this.puck.el, FxPrism.EMPTY);
    const pose = tour.hop ? this._hopPose(tour.hop, now) : [...this.steps[tour.at].centre, this._top(tour.at, now)];
    const key  = pose.join("|");
    if (key === this.puck.drawn) return;
    this.puck.drawn = key;
    this.puck.shape.put(this.puck.el, pose[0], pose[1], pose[2], pose[2] + FxTermsFigure.PUCK[1]);
  }

  _hopPose(hop, now) {
    const s    = FxRank.smooth((this.clock - hop.t0) / FxTermsFigure.HOP_S);
    const a    = this.steps[hop.from].centre;
    const b    = this.steps[hop.to].centre;
    const za   = this._top(hop.from, now);
    const zb   = this._top(hop.to, now);
    let peak   = 0;
    for (let k = Math.min(hop.from, hop.to); k <= Math.max(hop.from, hop.to); k++) peak = Math.max(peak, this._top(k, now));
    const lift = Math.max(FxTermsFigure.ARC, peak + 3 - (za + zb) / 2);
    return [HL.lerp(a[0], b[0], s), HL.lerp(a[1], b[1], s), HL.lerp(za, zb, s) + lift * Math.sin(Math.PI * s)];
  }

  _move(p) {
    this._choose(HairlineFigureBase.nearest(this.steps.map((step) => step.restX), p[0]));
  }

  _retarget(now) {
    const anchor = this.chosen === null ? this.leader : this.chosen;
    for (const step of this.steps) {
      const distance = Math.abs(step.k - anchor);
      const reach    = this.chosen === null ? (step.k === anchor ? FxTermsFigure.REST : 0) : Math.max(0, 1 - distance / FxTermsFigure.REACH) ** 2;
      HL.tset(step.tw, this.amount * reach, now, distance * FxTermsFigure.STAGGER);
    }
    this.light.set(this.steps[anchor].el.sil);
    this.loop.wake();
  }
}


class FxParts {
  static led(parent, P, x, y, z, cls) {
    const el = HL.mk("circle", { r: 1.2, class: cls || "dot off" }, parent);
    HL.place(el, P(x, y, z));
    return el;
  }

  static box(P, front, parent, x0, y0, x1, y1, z0, z1, r, b) {
    const solid = HL.solid(parent);
    HL.put(solid, HL.prism(P, front, ...HL.rings(x0, y0, x1, y1, r === undefined ? 0.8 : r, b === undefined ? 0.5 : b), z0, z1));
    return solid;
  }

  static shape(P, front, x0, y0, x1, y1, r, b) {
    return new FxPrism(P, front, ...HL.rings(x0, y0, x1, y1, r, b));
  }

  static depot(P, front, parent, x0, y0, x1, y1, h) {
    const body = FxParts.box(P, front, parent, x0, y0, x1, y1, 0, h, 1.4, 0.9);
    FxParts.box(P, front, parent, x0 + 2, y0 + 2, x1 - 2, y1 - 2, h, h + 1.6, 1, 0.6);
    return body;
  }

  static flat(P, points, z) {
    return HL.poly(points.map(([x, y]) => P(x, y, z)));
  }

  static line(P, points, z) {
    return HL.open(points.map(([x, y]) => P(x, y, z)));
  }

  static set(el, cls) {
    if (el.getAttribute("class") !== cls) el.setAttribute("class", cls);
  }

  static show(el, on) {
    const value = on ? "inline" : "none";
    if (el.getAttribute("display") !== value) el.setAttribute("display", value);
  }
}

class FxVan {
  static BODY  = [-3.2, 1.2, 1.5, 3.2];
  static CAB   = [1.5, 3.2, 1.4, 2.5];
  static RIDE  = 0.8;
  static CAB_Z = 0.7;
  static AXLES = [-2.2, 2.3];
  static RIM   = 0.7;

  constructor(P, front, scale) {
    this.P     = P;
    this.front = front;
    this.k     = scale;
    this.shift = new FxShift(P);
    this.kinds = new Map();
  }

  _kind(heading) {
    if (this.kinds.has(heading)) return this.kinds.get(heading);
    const along = heading[0];
    const sign  = heading[1] === "+" ? 1 : -1;
    const k     = this.k;
    const span  = (a0, a1) => (sign > 0 ? [a0 * k, a1 * k] : [-a1 * k, -a0 * k]);
    const shape = ([a0, a1, half]) => {
      const [lo, hi] = span(a0, a1);
      const box      = along === "x" ? [lo, -half * k, hi, half * k] : [-half * k, lo, half * k, hi];
      return FxParts.shape(this.P, this.front, box[0], box[1], box[2], box[3], 0.6 * k, 0.4 * k);
    };
    const r     = FxVan.RIM * k;
    const rim   = new FxRim(this.P, r, along === "x" ? "y" : "x");
    const side  = FxVan.BODY[2] * k;
    const kind  = { sign, body: shape(FxVan.BODY), cab: shape(FxVan.CAB), wheels: FxVan.AXLES.map((a) => (along === "x" ? rim.path(sign * a * k, side, r) : rim.path(side, sign * a * k, r))).join("") };
    this.kinds.set(heading, kind);
    return kind;
  }

  make(parent) {
    const g = HL.mk("g", {}, parent);
    return { g, body: HL.solid(g), cab: HL.solid(g), wheels: HL.mk("path", {}, g), heading: "", grow: -1, at: "", dashed: false };
  }

  pose(van, heading, x, y, z, grow) {
    const size = grow === undefined ? 1 : grow;
    if (heading !== van.heading || size !== van.grow) {
      van.heading = heading;
      van.grow    = size;
      this._shape(van, heading, size);
    }
    const at = this.shift.of(x, y, z);
    if (at === van.at) return;
    van.at = at;
    van.g.setAttribute("transform", at);
  }

  _shape(van, heading, size) {
    if (!heading || size < 0.05) {
      HL.put(van.body, FxPrism.EMPTY);
      HL.put(van.cab, FxPrism.EMPTY);
      van.wheels.setAttribute("d", "");
      return;
    }
    const kind = this._kind(heading);
    const k    = this.k;
    const ride = FxVan.RIDE * k;
    kind.body.put(van.body, 0, 0, ride, ride + FxVan.BODY[3] * k * size);
    kind.cab.put(van.cab, 0, 0, FxVan.CAB_Z * k, FxVan.CAB_Z * k + FxVan.CAB[3] * k * size);
    van.wheels.setAttribute("d", size < 0.6 ? "" : kind.wheels);
    if (kind.sign > 0) van.g.insertBefore(van.body.g, van.cab.g);
    else van.g.insertBefore(van.cab.g, van.body.g);
  }

  dash(van, on) {
    if (on === van.dashed) return;
    van.dashed = on;
    van.body.sil.classList.toggle("dash", on);
    van.cab.sil.classList.toggle("dash", on);
    van.wheels.classList.toggle("dash", on);
  }
}

class FxDoor {
  constructor(P, front, parent, axis, at, centre, half, z0, height) {
    this.P     = P;
    this.axis  = axis;
    this.at    = at;
    this.c     = centre;
    this.half  = half;
    this.z0    = z0;
    this.h     = height;
    const ring = axis === "x" ? HL.rrect(at + 0.1, centre - half, at + 0.9, centre + half, 0.3) : HL.rrect(centre - half, at + 0.1, centre + half, at + 0.9, 0.3);
    this.shape = new FxPrism(P, front, ring, null);
    this.frame = HL.mk("path", { class: "nf", d: HL.open([this._pt(-half - 0.7, z0), this._pt(-half - 0.7, z0 + height + 1.2), this._pt(half + 0.7, z0 + height + 1.2), this._pt(half + 0.7, z0)]) }, parent);
    this.panel = HL.solid(parent);
    this.drawn = NaN;
  }

  _pt(s, z) {
    return this.axis === "x" ? this.P(this.at + 0.9, this.c + s, z) : this.P(this.c + s, this.at + 0.9, z);
  }

  draw(open) {
    if (Math.abs(open - this.drawn) < 0.002) return;
    this.drawn  = open;
    const top   = this.z0 + this.h;
    const z0    = this.z0 + HL.clamp(open, 0, 1) * this.h;
    if (top - z0 < 0.25) return HL.put(this.panel, FxPrism.EMPTY);
    const shape = this.shape.path(0, 0, z0, top);
    let slats   = "";
    for (let z = top - 1.4; z > z0 + 0.4; z -= 1.4) slats += HL.seg(this._pt(-this.half + 0.5, z), this._pt(this.half - 0.5, z));
    HL.put(this.panel, { sil: shape.sil, crease: slats });
  }
}

class FxFleetFigure extends HairlineFigureBase {
  static NAME    = "fleet";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 4.1;
  static PLINTH  = [-33, -25, 33, 25];
  static DEPOT   = [-6.5, -5, 6.5, 5, 7.5];
  static BLOCK   = [12.2, 25.8, 9.6, 18.8, 0.9];
  static LANE    = 2.4;
  static QUADS   = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
  static VAN     = 1.3;
  static LAP_S   = 9;
  static HOLD_S  = 0.5;
  static SCAN_S  = 1.6;
  static POSTS   = 6;
  static CRATES  = 6;
  static PARKED  = [[15.4, -2.6], [15.4, 2.6], [24.6, -2.6], [24.6, 2.6]];
  static PAD     = [[-4.5, 12.5], [0, 12.5], [4.5, 12.5], [-4.5, 17], [0, 17], [4.5, 17]];
  static REST    = 0.5;
  static DEFAULT = { routes: [{ stops: 5, drive: 0.4 }, { stops: 3, drive: 0.3 }, { stops: 4, drive: 0.3 }], unassigned: 2, idle: 1, solving: false };

  constructor({ stage, svg }, value) {
    super(svg, value, FxFleetFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxFleetFigure.PLINTH;
    this._camera(this._extent(), 164);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.vanKit = new FxVan(this.P, this.front, FxFleetFigure.VAN);
    this.crate  = FxParts.shape(this.P, this.front, -1.6, -1.6, 1.6, 1.6, 0.5, 0.4);
    this.post   = FxParts.shape(this.P, this.front, -0.8, -0.8, 0.8, 0.8, 0.5, 0.3);
    this._buildLoops();
    this.scan   = HL.mk("path", { class: "nf dash", d: "" }, this.g);
    this.depth  = new FxDepth(this.g);
    this._buildBlocks();
    this._buildDepot();
    this._buildPosts();
    this._buildPad();
    this._buildVans();
    this._apply(FxFleetFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxFleetFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [0, 0, FxFleetFigure.DEPOT[4] + 1.6], [x0 + 8, y0 + 8, 8], [x1 - 5, y1 - 5, 4]];
  }

  _rect(sx, sy, grow) {
    const [a, b, c, d] = FxFleetFigure.BLOCK;
    const xs           = [sx * (a - grow), sx * (b + grow)].sort((p, q) => p - q);
    const ys           = [sy * (c - grow), sy * (d + grow)].sort((p, q) => p - q);
    return [xs[0], ys[0], xs[1], ys[1]];
  }

  _buildLoops() {
    const layer = HL.mk("g", {}, this.g);
    const L     = FxFleetFigure.LANE;
    this.loops  = FxFleetFigure.QUADS.map(([sx, sy], q) => {
      const [x0, y0, x1, y1] = this._rect(sx, sy, L);
      const near   = [sx > 0 ? x0 : x1, sy > 0 ? y0 : y1];
      const far    = [sx > 0 ? x1 : x0, sy > 0 ? y1 : y0];
      const points = [near, [far[0], near[1]], far, [near[0], far[1]], near];
      const spur   = [[sx * FxFleetFigure.DEPOT[2], sy * 2.4], [near[0], sy * 2.4], near];
      const path   = HL.mk("path", { class: "nf lo", d: FxParts.flat(this.P, points.slice(0, 4), 0) + FxParts.line(this.P, spur, 0) }, layer);
      const centre = [(x0 + x1) / 2, (y0 + y1) / 2];
      return { q, sx, sy, points, path, line: new FxPolyline(points), centre, at: this.P(centre[0], centre[1], 0), stops: 0, drive: 0, active: false, idle: false, posts: [], arcs: [], lap: FxFleetFigure.LAP_S, van: null, lift: HL.tween(0), phase: q * 0.25 };
    });
  }

  _buildBlocks() {
    const h = FxFleetFigure.BLOCK[4];
    for (const loop of this.loops) {
      const [x0, y0, x1, y1] = this._rect(loop.sx, loop.sy, 0);
      loop.blockItem = this.depth.add(loop.centre[0] + loop.centre[1]);
      loop.block     = FxParts.box(this.P, this.front, loop.blockItem.group, x0, y0, x1, y1, 0, h, 1.2, 0.8);
    }
  }

  _buildDepot() {
    const [x0, y0, x1, y1, h] = FxFleetFigure.DEPOT;
    const item                = this.depth.add(0);
    this.depot                = FxParts.depot(this.P, this.front, item.group, x0, y0, x1, y1, h);
    this.door                 = new FxDoor(this.P, this.front, item.group, "x", x1, 0, 2.6, 0, 5);
    this.door.draw(0.15);
  }

  _buildPosts() {
    for (const loop of this.loops) {
      loop.posts = Array.from({ length: FxFleetFigure.POSTS }, () => {
        const item  = this.depth.add(0);
        const solid = HL.solid(item.group);
        const led   = HL.mk("circle", { r: 1.2, class: "dot m", display: "none" }, item.group);
        return { item, solid, led, x: 0, y: 0, shown: false, lit: "" };
      });
    }
  }

  _buildPad() {
    this.pad = FxFleetFigure.PAD.map(([x, y]) => {
      const item  = this.depth.add(x + y);
      const solid = HL.solid(item.group);
      solid.sil.classList.add("dash");
      return { item, solid, x, y, shown: null };
    });
  }

  _buildVans() {
    for (const loop of this.loops) {
      loop.item = this.depth.add(0);
      loop.van  = this.vanKit.make(loop.item.group);
    }
    this.idleVans = FxFleetFigure.PARKED.map(([x, y]) => {
      const item = this.depth.add(x + y);
      const van  = this.vanKit.make(item.group);
      this.vanKit.dash(van, true);
      return { item, van, x, y, shown: null };
    });
  }

  setState(state) {
    const name   = FxFleetFigure.NAME;
    FxCheck.object(name, "state", state, ["routes", "unassigned", "idle", "solving"]);
    if (!Array.isArray(state.routes) || state.routes.length > 4) throw new Error(`${name}: routes must be an array of at most 4 routes, got ${state.routes}`);
    const routes = state.routes.map((route, k) => {
      FxCheck.object(name, `routes[${k}]`, route, ["stops", "drive"]);
      return { stops: FxCheck.integer(name, `routes[${k}].stops`, route.stops, 0, Infinity), drive: FxCheck.number(name, `routes[${k}].drive`, route.drive, 0, 1) };
    });
    const next   = { routes, unassigned: FxCheck.integer(name, "unassigned", state.unassigned, 0, Infinity), idle: FxCheck.integer(name, "idle", state.idle, 0, 4), solving: FxCheck.boolean(name, "solving", state.solving) };
    this._apply(next);
    const now    = performance.now();
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    this.loops.forEach((loop, q) => {
      const route = state.routes[q] || null;
      loop.active = Boolean(route && route.stops > 0);
      loop.idle   = Boolean(route && route.stops === 0);
      loop.stops  = route ? route.stops : 0;
      loop.drive  = route ? route.drive : 0;
      const empty = !state.routes.length;
      FxParts.set(loop.path, loop.active ? "nf" : loop.idle || empty ? "nf dash" : "nf lo");
      FxParts.set(loop.block.sil, loop.active ? "sil" : loop.idle || empty ? "sil dash" : "lo");
      this._placePosts(loop);
    });
    this.pad.forEach((crate, i) => {
      const shown = i < Math.min(FxFleetFigure.CRATES, state.unassigned);
      if (shown === crate.shown) return;
      crate.shown = shown;
      HL.put(crate.solid, shown ? this.crate.path(crate.x, crate.y, 0, 2.8) : FxPrism.EMPTY);
    });
    this.idleVans.forEach((spot, i) => {
      const shown = i < state.idle;
      if (shown === spot.shown) return;
      spot.shown = shown;
      this.vanKit.pose(spot.van, shown ? "x+" : "", spot.x, spot.y, 0);
    });
    let best    = -1;
    this.loops.forEach((loop) => {
      if (loop.active && (best < 0 || loop.stops > this.loops[best].stops)) best = loop.q;
    });
    this.rest   = best < 0 ? 0 : best;
  }

  _placePosts(loop) {
    const n     = loop.active ? Math.min(FxFleetFigure.POSTS, loop.stops) : 0;
    const total = loop.line.total;
    const h     = FxFleetFigure.BLOCK[4];
    const inset = FxFleetFigure.LANE + 1.1;
    loop.arcs   = Array.from({ length: n }, (_, k) => ((k + 0.5) / n) * total);
    loop.lap    = FxFleetFigure.LAP_S / (0.6 + 0.8 * loop.drive) + n * FxFleetFigure.HOLD_S;
    loop.posts.forEach((post, k) => {
      if (k >= n) {
        if (post.shown) HL.put(post.solid, FxPrism.EMPTY);
        FxParts.show(post.led, false);
        post.shown = false;
        return;
      }
      const [px, py] = loop.line.at(loop.arcs[k] / total);
      const [cx, cy] = loop.centre;
      const [x0, y0, x1, y1] = this._rect(loop.sx, loop.sy, 0);
      post.x         = HL.clamp(px + Math.sign(cx - px) * inset, x0 + 1.1, x1 - 1.1);
      post.y         = HL.clamp(py + Math.sign(cy - py) * inset, y0 + 1.1, y1 - 1.1);
      this.post.put(post.solid, post.x, post.y, h, h + 3.4);
      HL.place(post.led, this.P(post.x, post.y, h + 3.4));
      FxParts.show(post.led, true);
      this.depth.move(post.item, Math.max(post.x + post.y, loop.centre[0] + loop.centre[1] + 0.1));
      post.shown     = true;
    });
  }

  _lapArc(loop, t) {
    const total = loop.line.total;
    const n     = loop.arcs.length;
    const moves = loop.lap - n * FxFleetFigure.HOLD_S;
    let left    = t;
    let s       = 0;
    for (let k = 0; k <= n; k++) {
      const target = k < n ? loop.arcs[k] : total;
      const travel = ((target - s) / total) * moves;
      if (left < travel) return { s: s + (left / travel) * (target - s), at: -1 };
      left        -= travel;
      s            = target;
      if (k === n) break;
      if (left < FxFleetFigure.HOLD_S) return { s, at: k };
      left        -= FxFleetFigure.HOLD_S;
    }
    return { s: total, at: -1 };
  }

  _heading(loop, s) {
    const lengths = loop.line.lengths;
    let k         = 1;
    while (k < lengths.length - 1 && s > lengths[k]) k++;
    const [ax, ay] = loop.points[k - 1];
    const [bx, by] = loop.points[k];
    return Math.abs(bx - ax) > Math.abs(by - ay) ? (bx > ax ? "x+" : "x-") : by > ay ? "y+" : "y-";
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still && !this.data.solving) this.clock += dt;
    let moving  = false;
    for (const loop of this.loops) moving = this._drawLoop(loop, now, still) || moving;
    this._drawScan(still);
    this.light.set(this._hiTarget());
    this.depth.sort();
    return still ? moving : true;
  }

  _drawLoop(loop, now, still) {
    const lift = HL.tval(loop.lift, now);
    const busy = !HL.tdone(loop.lift, now);
    if (!loop.active) {
      this.vanKit.pose(loop.van, "", 0, 0, 0);
      return busy;
    }
    const t         = still ? 0 : (this.clock + loop.phase * loop.lap) % loop.lap;
    const { s, at } = this._lapArc(loop, t);
    const [x, y]    = loop.line.at(s / loop.line.total);
    this.vanKit.pose(loop.van, this._heading(loop, Math.min(s, loop.line.total - 0.01)), x, y, lift);
    this.depth.move(loop.item, x + y);
    loop.posts.forEach((post, k) => {
      if (!post.shown) return;
      const lit = k === at ? "dot" : "dot m";
      if (lit === post.lit) return;
      post.lit = lit;
      post.led.setAttribute("class", lit);
    });
    return busy;
  }

  _drawScan(still) {
    if (!this.data.solving) {
      if (this.scan.getAttribute("d")) this.scan.setAttribute("d", "");
      return;
    }
    const [x0, y0, x1, y1] = FxFleetFigure.PLINTH;
    const share            = still ? 0.5 : (performance.now() / 1000 / FxFleetFigure.SCAN_S) % 1;
    const x                = HL.lerp(x0 + 4, x1 - 4, share);
    this.scan.setAttribute("d", HL.seg(this.P(x, y0 + 3, 0.2), this.P(x, y1 - 3, 0.2)));
  }

  _hiTarget() {
    const loop = this.loops[this.chosen === null ? this.rest : this.chosen];
    return loop.active ? loop.van.body.sil : loop.block.sil;
  }

  _move(p) {
    let best = 0;
    let gap  = Infinity;
    for (const loop of this.loops) {
      const d = Math.hypot(loop.at[0] - p[0], loop.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = loop.q;
    }
    this._choose(best);
  }

  _retarget(now) {
    const lit = this.chosen === null ? this.rest : this.chosen;
    for (const loop of this.loops) HL.tset(loop.lift, loop.q !== lit ? 0 : this.chosen === null ? this.amount * FxFleetFigure.REST : this.amount, now, 0);
    this.light.set(this._hiTarget());
    this.loop.wake();
  }
}

class FxRecordingFigure extends HairlineFigureBase {
  static NAME    = "recording";
  static RANGE   = [1, 2, 3.5];
  static SCALE   = 4.05;
  static PLINTH  = [-38, -20, 38, 20];
  static RAILS   = [-12.5, -4.2, 4.2, 12.5];
  static TILES   = 12;
  static X0      = -31;
  static PITCH   = 5.4;
  static TILE    = [2.1, 3.1, 1];
  static GANTRY  = 10.5;
  static BLINK_S = 0.9;
  static SWEEP_S = 6;
  static REST_S  = 2;
  static LIFT    = 0.6;
  static STATUS  = ["queued", "running", "finished", "failed", "cancelled"];
  static DEFAULT = { lanes: [{ frames: 12, total: 12, status: "finished" }, { frames: 7, total: 12, status: "running" }, { frames: 12, total: 12, status: "finished" }, { frames: 0, total: null, status: "queued" }], rounds: 3 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxRecordingFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxRecordingFigure.PLINTH;
    this._camera(this._extent(), 168);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.shift = new FxShift(this.P);
    const [a, b] = FxRecordingFigure.TILE;
    this.tile  = FxParts.shape(this.P, this.front, -a, -b, a, b, 0.6, 0.4);
    this.cuts  = HL.mk("path", { class: "nf dash", d: "" }, this.g);
    this._buildLanes();
    this._buildGantry();
    this._apply(FxRecordingFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxRecordingFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [x0 + 6, y0 + 2, FxRecordingFigure.GANTRY + 3], [x1 - 6, y0 + 2, FxRecordingFigure.GANTRY + 3]];
  }

  _x(i) {
    return FxRecordingFigure.X0 + i * FxRecordingFigure.PITCH;
  }

  _buildLanes() {
    const end  = this._x(FxRecordingFigure.TILES - 1) + 4.5;
    this.lanes = FxRecordingFigure.RAILS.map((y, l) => {
      const rail  = HL.mk("path", { class: "nf lo", d: HL.seg(this.P(FxRecordingFigure.X0 - 4, y, 0), this.P(end, y, 0)) }, this.g);
      const tiles = Array.from({ length: FxRecordingFigure.TILES }, (_, i) => {
        const g     = HL.mk("g", {}, this.g);
        const solid = HL.solid(g);
        this.tile.put(solid, this._x(i), y, 0, FxRecordingFigure.TILE[2]);
        return { i, g, solid, shown: null, cls: "", at: "" };
      });
      const cap   = FxParts.led(this.g, this.P, end, y, 0, "dot off");
      return { l, y, rail, tiles, cap, count: 0, status: "queued", lift: HL.tween(0) };
    });
  }

  _buildGantry() {
    const top    = FxRecordingFigure.GANTRY;
    const y0     = FxRecordingFigure.RAILS[0] - 4.5;
    const y1     = FxRecordingFigure.RAILS[3] + 4.5;
    this.gantry  = HL.mk("g", {}, this.g);
    FxParts.box(this.P, this.front, this.gantry, -0.9, y0 - 0.9, 0.9, y0 + 0.9, 0, top, 0.5, 0.3);
    this.beam    = FxParts.box(this.P, this.front, this.gantry, -0.9, y0 - 0.9, 0.9, y1 + 0.9, top, top + 1.6, 0.5, 0.3);
    FxParts.box(this.P, this.front, this.gantry, -0.9, y1 - 0.9, 0.9, y1 + 0.9, 0, top, 0.5, 0.3);
    this.lamp    = FxParts.led(this.gantry, this.P, 0, y0 + 2, top + 1.6, "dot off");
    this.gantryX = HL.spring(FxRecordingFigure.X0, { k: 40, c: 11, eps: 0.01 });
    this.gantryAt = "";
  }

  setState(state) {
    const name  = FxRecordingFigure.NAME;
    FxCheck.object(name, "state", state, ["lanes", "rounds"]);
    if (!Array.isArray(state.lanes) || state.lanes.length > 4) throw new Error(`${name}: lanes must be an array of at most 4 lanes, got ${state.lanes}`);
    const lanes = state.lanes.map((lane, k) => {
      FxCheck.object(name, `lanes[${k}]`, lane, ["frames", "total", "status"]);
      if (!FxRecordingFigure.STATUS.includes(lane.status)) throw new Error(`${name}: lanes[${k}].status must be one of ${FxRecordingFigure.STATUS.join(", ")}, got ${lane.status}`);
      return { frames: FxCheck.integer(name, `lanes[${k}].frames`, lane.frames, 0, Infinity), total: lane.total === null ? null : FxCheck.integer(name, `lanes[${k}].total`, lane.total, 1, Infinity), status: lane.status };
    });
    const next  = { lanes, rounds: state.rounds === null ? null : FxCheck.integer(name, "rounds", state.rounds, 1, 10) };
    this._apply(next);
    const now   = performance.now();
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data    = state;
    const n      = FxRecordingFigure.TILES;
    this.running = -1;
    this.lanes.forEach((lane, l) => {
      const src   = state.lanes[l] || null;
      lane.status = src ? src.status : "none";
      lane.count  = !src ? 0 : src.total ? Math.min(n, Math.ceil((n * src.frames) / src.total)) : Math.min(n, src.frames);
      if (src && src.status === "finished" && src.total === null && src.frames > 0) lane.count = n;
      FxParts.set(lane.rail, lane.status === "queued" ? "nf dash" : lane.status === "none" ? "nf lo dash" : "nf lo");
      FxParts.set(lane.cap, lane.status === "finished" ? "dot m" : "dot off");
      FxParts.show(lane.cap, lane.status !== "none");
      if (lane.status === "running" && (this.running < 0 || lane.count > this.lanes[this.running].count)) this.running = l;
      lane.tiles.forEach((tile) => {
        const last = tile.i === lane.count - 1;
        const cls  = (lane.status === "failed" || lane.status === "cancelled") && last ? "sil dash" : "sil";
        if (cls !== tile.cls) {
          tile.cls = cls;
          tile.solid.sil.setAttribute("class", cls);
        }
      });
    });
    let cuts     = "";
    const rounds = state.rounds || 1;
    for (let j = 1; j < rounds; j++) {
      const x = FxRecordingFigure.X0 - FxRecordingFigure.PITCH / 2 + (j / rounds) * n * FxRecordingFigure.PITCH;
      cuts   += HL.seg(this.P(x, FxRecordingFigure.RAILS[0] - 5, 0), this.P(x, FxRecordingFigure.RAILS[3] + 5, 0));
    }
    this.cuts.setAttribute("d", cuts);
    FxParts.set(this.lamp, this.running >= 0 ? "dot" : "dot off");
    let rest     = 0;
    this.lanes.forEach((lane, l) => {
      if (lane.count > 0 && this.lanes[rest].count === 0) rest = l;
    });
    this.rest    = this.running >= 0 ? this.running : rest;
  }

  _tick(dt, now) {
    const still  = HL.reducedMotion();
    if (!still) this.clock += dt;
    const blink  = still || (this.clock / FxRecordingFigure.BLINK_S) % 1 < 0.55;
    const period = FxRecordingFigure.SWEEP_S + FxRecordingFigure.REST_S;
    const sweep  = this.running < 0 && !still ? HL.clamp((this.clock % period) / FxRecordingFigure.SWEEP_S, 0, 1) : -1;
    const head   = this.running >= 0 ? this._x(Math.min(FxRecordingFigure.TILES - 1, this.lanes[this.running].count)) - FxRecordingFigure.PITCH / 2 : sweep >= 0 ? HL.lerp(this._x(0) - 4, this._x(FxRecordingFigure.TILES - 1) + 4, FxRank.smooth(sweep)) : this._x(0) - 4;
    this.gantryX.t = head;
    let moving   = HL.stepS(this.gantryX, dt);
    const gx     = this.gantryX.x;
    const at     = this.shift.of(gx, 0, 0);
    if (at !== this.gantryAt) {
      this.gantryAt = at;
      this.gantry.setAttribute("transform", at);
    }
    for (const lane of this.lanes) moving = this._drawLane(lane, now, blink, gx, sweep >= 0) || moving;
    return still ? moving : true;
  }

  _drawLane(lane, now, blink, gx, sweeping) {
    const lift    = HL.tval(lane.lift, now);
    const running = lane.status === "running";
    for (const tile of lane.tiles) {
      const shown = tile.i < lane.count || (running && tile.i === lane.count && blink && lane.count < FxRecordingFigure.TILES);
      if (shown !== tile.shown) {
        tile.shown = shown;
        FxParts.show(tile.g, shown);
      }
      if (!shown) continue;
      const near  = sweeping ? Math.max(0, 1 - Math.abs(this._x(tile.i) - gx) / FxRecordingFigure.PITCH) : 0;
      const z     = FxRecordingFigure.LIFT * near + (tile.i === lane.count - 1 ? lift : 0);
      const at    = this.shift.of(0, 0, z);
      if (at === tile.at) continue;
      tile.at     = at;
      tile.g.setAttribute("transform", at);
    }
    return !HL.tdone(lane.lift, now);
  }

  _hiTarget(lane) {
    if (lane.count > 0) return lane.tiles[lane.count - 1].solid.sil;
    return lane.rail;
  }

  _move(p) {
    const [, y] = HL.unproj(this.C, p[0], p[1], 0);
    this._choose(HairlineFigureBase.nearest(FxRecordingFigure.RAILS, y));
  }

  _retarget(now) {
    const lit = this.chosen === null ? this.rest : this.chosen;
    for (const lane of this.lanes) HL.tset(lane.lift, lane.l !== lit ? 0 : this.chosen === null ? this.amount * 0.5 : this.amount, now, 0);
    this.light.set(this._hiTarget(this.lanes[lit]));
    this.loop.wake();
  }
}

class FxHeadsFigure extends HairlineFigureBase {
  static NAME     = "heads";
  static RANGE    = [1.5, 3, 5];
  static SCALE    = 4.4;
  static PLINTH   = [-30, -22, 30, 22];
  static ROWS     = [{ y: 12, top: 1, xs: [-14, 0, 14] }, { y: 0, top: 3, xs: [-18, -6, 6, 18] }, { y: -12, top: 5, xs: [-21, -15, -9, -3, 3, 9, 15, 21] }];
  static HALF     = 1.8;
  static H        = 15;
  static FLOOR    = 0.8;
  static PUCK     = [1.7, 1.2];
  static STEP_S   = 0.9;
  static REST_S   = 1.5;
  static BREATH   = 0.03;
  static BREATH_S = 4;
  static REST     = 0.5;
  static DEFAULT  = { operator: [0.62, 0.08, 0.3], reference: [0.45, 0.25, 0.3], masked: [false, false, false], vehicle: [0.55, 0.25, 0.15, 0.05], job: [0.4, 0.2, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02], chosen: [0, 0, 0] };

  constructor({ stage, svg }, value) {
    super(svg, value, FxHeadsFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxHeadsFigure.PLINTH;
    this._camera(this._extent(), 172);
    this._plinth(x0, y0, x1, y1, 6, 3);
    const h    = FxHeadsFigure.HALF;
    this.col   = FxParts.shape(this.P, this.front, -h, -h, h, h, 0.6, 0.4);
    this.puckShape = new FxPrism(this.P, this.front, HL.circ(FxHeadsFigure.PUCK[0], 24), HL.circ(FxHeadsFigure.PUCK[0] - 0.5, 24));
    this._buildRows();
    this.puck  = { el: HL.solid(this.g), drawn: "" };
    this._apply(FxHeadsFigure.DEFAULT, performance.now());
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxHeadsFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [x0 + 6, y0 + 6, FxHeadsFigure.ROWS[2].top + FxHeadsFigure.H + 1 + FxHeadsFigure.RANGE[1]], [x1 - 6, y0 + 6, FxHeadsFigure.ROWS[2].top + FxHeadsFigure.H + 1]];
  }

  _buildRows() {
    const rows = FxHeadsFigure.ROWS.map((row, r) => ({ ...row, r })).reverse();
    this.cols  = [];
    this.rows  = [];
    for (const row of rows) {
      FxParts.box(this.P, this.front, this.g, -26, row.y - 5.5, 26, row.y + 5.5, 0, row.top, 1.2, 0.8);
      const group = HL.mk("g", {}, this.g);
      const items = row.xs.map((x, k) => {
        const socket = HL.mk("path", { class: "nf lo", d: FxParts.flat(this.P, [[x - 2.4, row.y - 2.4], [x + 2.4, row.y - 2.4], [x + 2.4, row.y + 2.4], [x - 2.4, row.y + 2.4]], row.top) }, group);
        const led    = FxParts.led(group, this.P, x, row.y + 4, row.top, "dot off");
        const g      = HL.mk("g", {}, group);
        const solid  = HL.solid(g);
        const ref    = row.r === 0 ? HL.mk("path", { class: "nf lo dash", d: "" }, g) : null;
        const item   = { r: row.r, k, x, y: row.y, base: row.top, socket, led, solid, ref, share: 0, masked: false, used: false, hw: HL.tween(FxHeadsFigure.FLOOR), lw: HL.tween(0), phase: (row.r * 8 + k) * 0.7, drawn: NaN, restX: this.P(x, row.y, row.top)[0], restY: this.P(x, row.y, row.top)[1] };
        this.cols.push(item);
        return item;
      });
      this.rows[row.r] = items;
    }
  }

  setState(state) {
    const name   = FxHeadsFigure.NAME;
    FxCheck.object(name, "state", state, ["operator", "reference", "masked", "vehicle", "job", "chosen"]);
    const shares = (key, value, lo, hi) => {
      if (!Array.isArray(value) || value.length < lo || value.length > hi) throw new Error(`${name}: ${key} must be an array of ${lo} to ${hi} shares, got ${value}`);
      return value.map((v, k) => FxCheck.number(name, `${key}[${k}]`, v, 0, 1));
    };
    FxCheck.list(name, "masked", state.masked, 3);
    FxCheck.list(name, "chosen", state.chosen, 3);
    const next   = {
      operator:  shares("operator", state.operator, 3, 3),
      reference: state.reference === null ? null : shares("reference", state.reference, 3, 3),
      masked:    state.masked.map((v, k) => FxCheck.boolean(name, `masked[${k}]`, v)),
      vehicle:   state.vehicle === null ? null : shares("vehicle", state.vehicle, 1, 4),
      job:       state.job === null ? null : shares("job", state.job, 1, 8),
      chosen:    [FxCheck.integer(name, "chosen[0]", state.chosen[0], 0, 2), state.chosen[1] === null ? null : FxCheck.integer(name, "chosen[1]", state.chosen[1], 0, 3), state.chosen[2] === null ? null : FxCheck.integer(name, "chosen[2]", state.chosen[2], 0, 7)],
    };
    const now    = performance.now();
    this._apply(next, now);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state, now) {
    this.data   = state;
    const lists = [state.operator, state.vehicle || [], state.job || []];
    const quiet = state.chosen[0] === 2;
    this.rows.forEach((items, r) => {
      const values = lists[r];
      const ranks  = FxRank.terciles(items.map((item, k) => (k < values.length && !(r === 0 && state.masked[k]) ? values[k] : 0)));
      items.forEach((item) => {
        item.used   = item.k < values.length;
        item.masked = r === 0 && state.masked[item.k];
        item.share  = item.used && !item.masked ? values[item.k] : 0;
        const dim   = r > 0 && quiet;
        FxParts.set(item.socket, item.masked ? "nf dash" : dim || !item.used ? "nf lo dash" : "nf lo");
        FxParts.set(item.led, item.used && !dim ? ranks[item.k] : "dot off");
        FxParts.show(item.solid.g, item.used && !item.masked);
        item.solid.sil.classList.toggle("lo", dim);
        HL.tset(item.hw, FxHeadsFigure.FLOOR + FxHeadsFigure.H * item.share, now, item.k * 30);
        item.drawn  = NaN;
        if (item.ref) {
          const ref = state.reference ? state.reference[item.k] : null;
          const z   = item.base + FxHeadsFigure.FLOOR + FxHeadsFigure.H * (ref || 0);
          const h   = FxHeadsFigure.HALF + 1.2;
          item.ref.setAttribute("d", ref === null || item.masked ? "" : FxParts.flat(this.P, [[item.x - h, item.y - h], [item.x + h, item.y - h], [item.x + h, item.y + h], [item.x - h, item.y + h]], z));
        }
      });
    });
    const [o, v, j] = state.chosen;
    this.path   = [this.rows[0][o]];
    if (!quiet && v !== null && v < this.rows[1].length && this.rows[1][v].used) this.path.push(this.rows[1][v]);
    if (!quiet && j !== null && j < this.rows[2].length && this.rows[2][j].used) this.path.push(this.rows[2][j]);
    this.rest   = this.rows[0][o];
  }

  _top(item, now) {
    const breath = HL.reducedMotion() ? 1 : 1 + FxHeadsFigure.BREATH * Math.sin((2 * Math.PI * this.clock) / FxHeadsFigure.BREATH_S + item.phase);
    return item.base + (item.used && !item.masked ? HL.tval(item.hw, now) * breath : 0) + HL.tval(item.lw, now);
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    let moving  = false;
    for (const item of this.cols) moving = this._drawCol(item, now) || moving;
    this._drawPuck(now, still);
    return still ? moving : true;
  }

  _drawCol(item, now) {
    const busy = !HL.tdone(item.hw, now) || !HL.tdone(item.lw, now);
    if (!item.used || item.masked) return busy;
    const top  = this._top(item, now);
    if (Math.abs(top - item.drawn) < 0.004) return busy;
    item.drawn = top;
    this.col.put(item.solid, item.x, item.y, item.base, top);
    return busy;
  }

  _drawPuck(now, still) {
    const steps  = this.path.length;
    const period = FxHeadsFigure.STEP_S * steps + FxHeadsFigure.REST_S;
    const t      = still ? period - 0.01 : this.clock % period;
    const hop    = Math.min(steps - 1, Math.floor(t / FxHeadsFigure.STEP_S));
    const s      = still || t >= FxHeadsFigure.STEP_S * steps ? 1 : FxRank.smooth((t - hop * FxHeadsFigure.STEP_S) / (FxHeadsFigure.STEP_S * 0.55));
    const from   = this.path[Math.max(0, hop - 1)];
    const to     = this.path[hop];
    const a      = hop === 0 ? to : from;
    const za     = this._top(a, now);
    const zb     = this._top(to, now);
    const arc    = hop === 0 ? 0 : 4 * Math.sin(Math.PI * s);
    const fade   = !still && t > period - 0.35 ? (period - t) / 0.35 : !still && t < 0.25 ? t / 0.25 : 1;
    const pose   = [HL.lerp(a.x, to.x, s), HL.lerp(a.y, to.y, s), HL.lerp(za, zb, s) + arc, fade];
    const key    = pose.map((v) => Math.round(v * 100)).join("|");
    if (key === this.puck.drawn) return;
    this.puck.drawn = key;
    HL.put(this.puck.el, fade < 0.05 ? FxPrism.EMPTY : this.puckShape.path(pose[0], pose[1], pose[2], pose[2] + FxHeadsFigure.PUCK[1] * fade));
  }

  _move(p) {
    let best = this.rest;
    let gap  = Infinity;
    for (const item of this.cols) {
      if (!item.used || item.masked) continue;
      const d = Math.hypot(item.restX - p[0], item.restY - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item;
    }
    this._choose(best);
  }

  _retarget(now) {
    const lit = this.chosen === null ? this.rest : this.chosen;
    for (const item of this.cols) HL.tset(item.lw, item !== lit ? 0 : this.chosen === null ? this.amount * FxHeadsFigure.REST : this.amount, now, 0);
    this.light.set(lit.used && !lit.masked ? lit.solid.sil : lit.socket);
    this.loop.wake();
  }
}

class FxInsertFigure extends HairlineFigureBase {
  static NAME    = "insert";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 5.05;
  static PLINTH  = [-30, -14, 30, 14];
  static ROAD    = [-23, 25, -4.5];
  static POST_Y  = 2.4;
  static TRAY    = [-16, 14, 9.6];
  static PERIOD  = 5;
  static RISE    = 8;
  static PASS_S  = 7;
  static GATE_X  = 23;
  static VAN     = 1.4;
  static DEFAULT = { operator: 0, stops: 4, unassigned: 3, wasted: false, commit: false };

  constructor({ stage, svg }, value) {
    super(svg, value, FxInsertFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxInsertFigure.PLINTH;
    this._camera(this._extent(), 166);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.crate  = FxParts.shape(this.P, this.front, -2, -2, 2, 2, 0.6, 0.45);
    this.postShape = FxParts.shape(this.P, this.front, -1.2, -1.2, 1.2, 1.2, 0.6, 0.4);
    this.vanKit = new FxVan(this.P, this.front, FxInsertFigure.VAN);
    this._buildGround();
    this.depth  = new FxDepth(this.g);
    this._buildDepot();
    this._buildPosts();
    this._buildTray();
    this._buildGate();
    this._buildMover();
    const item  = this.depth.add(0);
    this.van    = this.vanKit.make(item.group);
    this.vanItem = item;
    this.drop   = HL.spring(0, { k: 18, c: 3.2, eps: 0.002 });
    this.phaseOf = -1;
    this._apply(FxInsertFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxInsertFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [0, 0, FxInsertFigure.RISE + 4], [-28, -10, 8], [FxInsertFigure.GATE_X, -10, 10]];
  }

  _buildGround() {
    const [a, b, y] = FxInsertFigure.ROAD;
    FxParts.box(this.P, this.front, this.g, a, y - 3.4, b, y + 3.4, 0, 0.5, 1.2, 0.7);
    const [t0, t1, ty] = FxInsertFigure.TRAY;
    FxParts.box(this.P, this.front, this.g, t0, ty - 3, t1, ty + 3, 0, 0.8, 1, 0.6);
    this.route = HL.mk("path", { class: "nf", d: "" }, this.g);
  }

  _buildDepot() {
    const item = this.depth.add(-40);
    this.depot = FxParts.depot(this.P, this.front, item.group, -29, -11, -23.5, 1.5, 8);
  }

  _buildPosts() {
    this.posts = Array.from({ length: 6 }, () => {
      const item = this.depth.add(0);
      return { item, solid: HL.solid(item.group), led: HL.mk("circle", { r: 1.2, class: "dot m", display: "none" }, item.group), x: 0, shown: false };
    });
  }

  _buildTray() {
    const [t0, t1, ty] = FxInsertFigure.TRAY;
    this.slots = Array.from({ length: 6 }, (_, k) => {
      const x    = t0 + 3.4 + k * 5.4;
      const item = this.depth.add(x + ty);
      const solid = HL.solid(item.group);
      solid.sil.classList.add("dash");
      return { k, x, y: ty, item, solid, drawn: NaN };
    });
  }

  _buildGate() {
    const x   = FxInsertFigure.GATE_X;
    const y   = FxInsertFigure.ROAD[2];
    const far = this.depth.add(x + y - 4);
    FxParts.box(this.P, this.front, far.group, x - 0.8, y - 4.6, x + 0.8, y - 3, 0, 8, 0.5, 0.3);
    const near = this.depth.add(x + y + 4);
    FxParts.box(this.P, this.front, near.group, x - 0.8, y + 3, x + 0.8, y + 4.6, 0, 8, 0.5, 0.3);
    this.bar  = HL.solid(near.group);
    this.barShape = FxParts.shape(this.P, this.front, x - 0.6, y - 3.8, x + 0.6, y + 3.8, 0.4, 0.25);
    this.barDrawn = NaN;
  }

  _buildMover() {
    this.moverItem = this.depth.add(0);
    this.mover     = HL.solid(this.moverItem.group);
    this.moverLed  = HL.mk("circle", { r: 1.2, class: "dot", display: "none" }, this.moverItem.group);
    this.moverKey  = "";
  }

  setState(state) {
    const name = FxInsertFigure.NAME;
    FxCheck.object(name, "state", state, ["operator", "stops", "unassigned", "wasted", "commit"]);
    const next = { operator: FxCheck.integer(name, "operator", state.operator, 0, 2), stops: FxCheck.integer(name, "stops", state.stops, 0, 40), unassigned: FxCheck.integer(name, "unassigned", state.unassigned, 0, Infinity), wasted: FxCheck.boolean(name, "wasted", state.wasted), commit: FxCheck.boolean(name, "commit", state.commit) };
    const now  = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data    = state;
    const n      = Math.min(6, state.stops);
    const xs     = Array.from({ length: n }, (_, k) => HL.lerp(-16, 17, n === 1 ? 0.5 : k / (n - 1)));
    const g      = Math.floor(n / 2);
    this.gap     = n === 0 ? 0 : n === 1 ? xs[0] + 6 : (xs[g - 1] + xs[g]) / 2;
    const shift  = 3.4;
    this.posts.forEach((post, k) => {
      post.shown = k < n;
      post.x     = k < n ? xs[k] + (n > 1 ? (k < g ? -shift : shift) * (2 / n) : 0) : 0;
      if (!post.shown) {
        HL.put(post.solid, FxPrism.EMPTY);
        FxParts.show(post.led, false);
        return;
      }
      this.postShape.put(post.solid, post.x, FxInsertFigure.POST_Y, 0, 4.6);
      HL.place(post.led, this.P(post.x, FxInsertFigure.POST_Y, 4.6));
      FxParts.show(post.led, true);
      this.depth.move(post.item, post.x + FxInsertFigure.POST_Y);
    });
    this.crates  = Math.min(5, state.unassigned);
    this.mover.sil.classList.toggle("dash", state.wasted);
    this.phaseOf = -1;
  }

  _phase(t) {
    const op = this.data.operator;
    const P  = FxInsertFigure.PERIOD;
    if (op === 2) return { kind: "wait", t };
    if (t < 0.6) return { kind: "rise", s: t / 0.6 };
    if (t < 1.4) return { kind: "cross", s: (t - 0.6) / 0.8 };
    if (t < 4.5) return { kind: "land", s: (t - 1.4) / 3.1, fresh: true };
    return { kind: "back", s: (t - 4.5) / (P - 4.5) };
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const t     = still ? 3 : this.clock % FxInsertFigure.PERIOD;
    const phase = this._phase(t);
    this._drawWork(phase, dt, still);
    this._drawVan(still);
    this.light.set(this._hiTarget());
    this.depth.sort();
    return still ? false : true;
  }

  _endpoints() {
    const op   = this.data.operator;
    const slot = op === 0 ? 0 : Math.min(5, this.crates);
    const tray = [this.slots[slot].x, FxInsertFigure.TRAY[2], 0.8];
    const post = [this.gap, FxInsertFigure.POST_Y, 0];
    return op === 0 ? { from: tray, to: post } : { from: post, to: tray };
  }

  _drawWork(phase, dt, still) {
    const op      = this.data.operator;
    let hidden    = -1;
    let jog       = false;
    let pose      = null;
    let asPost    = false;
    let gate      = 0;
    if (op === 2) {
      const t     = phase.t;
      this.drop.t = t > 0.3 && t < 2.6 && this.crates > 0 ? 2 : 0;
      HL.stepS(this.drop, dt);
      if (this.crates > 0) {
        hidden    = 0;
        pose      = [this.slots[0].x, FxInsertFigure.TRAY[2], 0.8 + this.drop.x];
      }
      gate        = this.data.commit ? (still ? 1 : HL.clamp(Math.min((t - 0.5) / 1, (4.8 - t) / 0.6), 0, 1)) : 0;
    } else {
      const { from, to } = this._endpoints();
      const R     = FxInsertFigure.RISE;
      if (op === 0) hidden = 0;
      if (phase.kind === "rise") pose = [from[0], from[1], from[2] + R * FxRank.smooth(phase.s)];
      if (phase.kind === "cross") pose = [HL.lerp(from[0], to[0], FxRank.smooth(phase.s)), HL.lerp(from[1], to[1], FxRank.smooth(phase.s)), from[2] + R + 2 * Math.sin(Math.PI * phase.s)];
      if (phase.kind === "land") {
        if (this.phaseOf !== 2) {
          this.drop.x = to[2] + R;
          this.drop.v = 0;
          this.drop.t = to[2];
        }
        HL.stepS(this.drop, dt);
        pose        = [to[0], to[1], still ? to[2] : this.drop.x];
        asPost      = op === 0;
        jog         = op === 0;
      }
      if (phase.kind === "back") {
        pose        = [to[0], to[1], to[2]];
        asPost      = op === 0;
        jog         = op === 0 && phase.s < 0.5;
        pose.push(1 - phase.s);
        if (op === 0) hidden = phase.s > 0.5 ? -1 : 0;
      }
      if (op === 1 && phase.kind === "back") jog = phase.s > 0.5;
      this.phaseOf = phase.kind === "land" ? 2 : phase.kind === "back" ? 3 : 1;
    }
    this._drawTray(hidden);
    this._drawMover(pose, asPost);
    this._drawRoute(jog || (op === 1 && !pose));
    this._drawBar(gate);
  }

  _drawTray(hidden) {
    this.slots.forEach((slot) => {
      const shown = slot.k < this.crates && slot.k !== hidden;
      const want  = shown ? 1 : 0;
      if (want === slot.drawn) return;
      slot.drawn  = want;
      HL.put(slot.solid, shown ? this.crate.path(slot.x, slot.y, 0.8, 4.2) : FxPrism.EMPTY);
    });
  }

  _drawMover(pose, asPost) {
    const key = pose ? `${pose.map((v) => Math.round(v * 100)).join("|")}|${asPost}` : "";
    if (key === this.moverKey) return;
    this.moverKey = key;
    if (!pose) {
      HL.put(this.mover, FxPrism.EMPTY);
      FxParts.show(this.moverLed, false);
      return;
    }
    const [x, y, z] = pose;
    const grow      = pose.length > 3 ? pose[3] : 1;
    if (grow < 0.05) {
      HL.put(this.mover, FxPrism.EMPTY);
      FxParts.show(this.moverLed, false);
      return;
    }
    const h         = (asPost ? 4.6 : 3.4) * grow;
    (asPost ? this.postShape : this.crate).put(this.mover, x, y, z, z + h);
    FxParts.show(this.moverLed, asPost);
    if (asPost) HL.place(this.moverLed, this.P(x, y, z + h));
    this.depth.move(this.moverItem, x + y + 0.2);
  }

  _drawRoute(jog) {
    const [a, b, y] = FxInsertFigure.ROAD;
    const g         = this.gap;
    const points    = jog ? [[a + 1, y], [g - 2.4, y], [g - 2.4, FxInsertFigure.POST_Y - 1.8], [g + 2.4, FxInsertFigure.POST_Y - 1.8], [g + 2.4, y], [b - 1, y]] : [[a + 1, y], [b - 1, y]];
    const d         = FxParts.line(this.P, points, 0.5);
    if (d !== this.route.getAttribute("d")) this.route.setAttribute("d", d);
  }

  _drawBar(gate) {
    if (Math.abs(gate - this.barDrawn) < 0.002) return;
    this.barDrawn = gate;
    const z       = HL.lerp(9.5, 3.4, gate);
    this.barShape.put(this.bar, 0, 0, z, z + 1.2);
  }

  _drawVan(still) {
    const [a, b, y] = FxInsertFigure.ROAD;
    const t         = still ? 0.2 : (this.clock % (2 * FxInsertFigure.PASS_S)) / FxInsertFigure.PASS_S;
    const out       = t < 1;
    const s         = FxRank.smooth(out ? t : 2 - t);
    const x         = HL.lerp(a + 6, b - 9, s);
    this.vanKit.pose(this.van, out ? "x+" : "x-", x, y, 0.5);
    this.depth.move(this.vanItem, x + y);
  }

  _items() {
    const items = this.posts.filter((post) => post.shown).map((post) => ({ el: post.solid.sil, at: this.P(post.x, FxInsertFigure.POST_Y, 2) }));
    for (const slot of this.slots) if (slot.k < this.crates) items.push({ el: slot.solid.sil, at: this.P(slot.x, slot.y, 2) });
    return items;
  }

  _hiTarget() {
    if (this.chosen !== null) return this.chosen;
    if (this.moverKey) return this.mover.sil;
    return this.van.body.sil;
  }

  _move(p) {
    let best = null;
    let gap  = Infinity;
    for (const item of this._items()) {
      const d = Math.hypot(item.at[0] - p[0], item.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item.el;
    }
    this._choose(gap < 60 ? best : null);
  }

  _retarget() {
    this.light.set(this._hiTarget());
    this.loop.wake();
  }
}

class FxDisruptionFigure extends HairlineFigureBase {
  static NAME    = "disruption";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 4.55;
  static PLINTH  = [-31, -17, 31, 17];
  static DEPOT   = [-28, -13, -15, 2, 9];
  static ROAD    = [-15, 28, -5.5];
  static POSTS   = [-2, 8, 18];
  static POST_Y  = 0.8;
  static CLOCK   = [6, -12.5];
  static DROPS   = [[-3, 7.2], [3, 7.2], [9, 7.2], [15, 7.2], [21, 7.2], [26, 1]];
  static TALLY   = [[-22, 9], [-15.5, 12], [-9, 15]];
  static TYPES   = ["new_job", "remove_job", "new_vehicle", "remove_vehicle"];
  static PERIOD  = 5;
  static GAP_MAX = 5400;
  static VAN     = 1.4;
  static DEFAULT = { type: "new_job", count: 2, gap: 1800, served: 3, failed: 1, dropped: 0 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxDisruptionFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    this.hand  = 0;
    const [x0, y0, x1, y1] = FxDisruptionFigure.PLINTH;
    this._camera(this._extent(), 166);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.crate  = FxParts.shape(this.P, this.front, -2, -2, 2, 2, 0.6, 0.45);
    this.postShape = FxParts.shape(this.P, this.front, -1.2, -1.2, 1.2, 1.2, 0.6, 0.4);
    this.coin   = new FxPrism(this.P, this.front, HL.circ(2.6, 28), HL.circ(2.1, 28));
    this.vanKit = new FxVan(this.P, this.front, FxDisruptionFigure.VAN);
    const [a, b, y] = FxDisruptionFigure.ROAD;
    FxParts.box(this.P, this.front, this.g, a, y - 3.6, b, y + 3.6, 0, 0.5, 1.2, 0.7);
    this.lines  = HL.mk("path", { class: "nf dash", d: "" }, this.g);
    this.depth  = new FxDepth(this.g);
    this._buildDepot();
    this._buildClock();
    this._buildPosts();
    this._buildDrops();
    this._buildTally();
    const item  = this.depth.add(0);
    this.vanItem = item;
    this.van    = this.vanKit.make(item.group);
    this.landing = Array.from({ length: 6 }, () => HL.spring(0, { k: 18, c: 3.2, eps: 0.002 }));
    this._apply(FxDisruptionFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxDisruptionFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [-21, -5, 11], [FxDisruptionFigure.CLOCK[0], FxDisruptionFigure.CLOCK[1], 14], [9, 7.2, 16]];
  }

  _buildDepot() {
    const [x0, y0, x1, y1, h] = FxDisruptionFigure.DEPOT;
    const item = this.depth.add(-40);
    this.depot = FxParts.depot(this.P, this.front, item.group, x0, y0, x1, y1, h);
    this.door  = new FxDoor(this.P, this.front, item.group, "x", x1, FxDisruptionFigure.ROAD[2], 2.8, 0, 5.6);
  }

  _buildClock() {
    const [x, y] = FxDisruptionFigure.CLOCK;
    const item   = this.depth.add(x + y);
    FxParts.box(this.P, this.front, item.group, x - 0.7, y - 0.7, x + 0.7, y + 0.7, 0, 10, 0.4, 0.25);
    this.dial    = HL.solid(item.group);
    HL.put(this.dial, HL.prism(this._shifted(x, y, 0), this.front, HL.circ(4, 36), HL.circ(3.4, 36), 10, 10.7));
    this.handEl  = HL.mk("path", { class: "nf", d: "" }, item.group);
    this.handDrawn = NaN;
  }

  _buildPosts() {
    this.posts = FxDisruptionFigure.POSTS.map((x) => {
      const item = this.depth.add(x + FxDisruptionFigure.POST_Y);
      return { x, item, solid: HL.solid(item.group), led: HL.mk("circle", { r: 1.2, class: "dot m" }, item.group), drawn: NaN, dash: false };
    });
  }

  _buildDrops() {
    this.drops = FxDisruptionFigure.DROPS.map(([x, y]) => {
      const item = this.depth.add(x + y);
      return { x, y, item, solid: HL.solid(item.group), led: HL.mk("circle", { r: 1.2, class: "dot", display: "none" }, item.group), drawn: "" };
    });
  }

  _buildTally() {
    this.tally = FxDisruptionFigure.TALLY.map(([x, y], k) => {
      const item  = this.depth.add(x + y);
      const coins = Array.from({ length: 5 }, () => HL.solid(item.group));
      const cls   = ["sil", "sil dash", "lo"][k];
      for (const coin of coins) coin.sil.setAttribute("class", cls);
      return { k, x, y, coins, count: -1 };
    });
  }

  setState(state) {
    const name = FxDisruptionFigure.NAME;
    FxCheck.object(name, "state", state, ["type", "count", "gap", "served", "failed", "dropped"]);
    if (state.type !== null && !FxDisruptionFigure.TYPES.includes(state.type)) throw new Error(`${name}: type must be one of ${FxDisruptionFigure.TYPES.join(", ")} or null, got ${state.type}`);
    const next = { type: state.type, count: FxCheck.integer(name, "count", state.count, 0, 6), gap: FxCheck.number(name, "gap", state.gap, 0), served: FxCheck.integer(name, "served", state.served, 0, Infinity), failed: FxCheck.integer(name, "failed", state.failed, 0, Infinity), dropped: FxCheck.integer(name, "dropped", state.dropped, 0, Infinity) };
    const now  = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    const counts = [state.served, state.failed, state.dropped];
    for (const stack of this.tally) {
      const count = Math.min(5, counts[stack.k]);
      if (count === stack.count) continue;
      stack.count = count;
      stack.coins.forEach((coin, i) => HL.put(coin, i < count ? this.coin.path(stack.x, stack.y, i * 1.3, i * 1.3 + 1) : FxPrism.EMPTY));
    }
    this.clock = 0;
  }

  _tick(dt, now) {
    const still  = HL.reducedMotion();
    if (!still) this.clock += dt;
    const t      = still ? 2.5 : this.clock % FxDisruptionFigure.PERIOD;
    const type   = this.data.type;
    const back   = t >= 4 ? (t - 4) / 1 : 0;
    this._drawDrops(type === "new_job" ? this.data.count : 0, t, back, dt, still);
    this._drawPosts(type === "remove_job" ? Math.min(3, this.data.count) : 0, t, back);
    this._drawVehicle(type, t, back);
    this._drawClock(dt, still);
    this.light.set(this._hiTarget());
    this.depth.sort();
    return still ? false : true;
  }

  _drawDrops(count, t, back, dt, still) {
    let lines = "";
    this.drops.forEach((drop, k) => {
      const spring = this.landing[k];
      const start  = 0.15 * k;
      if (k >= count) {
        if (drop.drawn) {
          drop.drawn = "";
          HL.put(drop.solid, FxPrism.EMPTY);
          FxParts.show(drop.led, false);
        }
        return;
      }
      let z        = 0;
      if (still) z = 0;
      else if (t < start) z = 16;
      else if (t < start + 0.7) z = 16 * (1 - FxRank.smooth((t - start) / 0.7)) + 1.5;
      else {
        if (spring.t !== 0 || spring.x === 0) {
          spring.x = 1.5;
          spring.v = -14;
          spring.t = 0;
        }
        HL.stepS(spring, dt);
        z = spring.x;
      }
      if (t < start + 0.05) {
        spring.x = 0;
        spring.t = 1;
      }
      const grow   = 1 - back;
      if (!still && t < start + 0.7) lines += HL.seg(this.P(drop.x, drop.y, 16), this.P(drop.x, drop.y, z + 3.4));
      const key    = `${Math.round(z * 100)}|${Math.round(grow * 100)}`;
      if (key === drop.drawn) return;
      drop.drawn   = key;
      HL.put(drop.solid, grow < 0.05 ? FxPrism.EMPTY : this.crate.path(drop.x, drop.y, z, z + 3.4 * grow));
      FxParts.show(drop.led, grow > 0.5 && (still || t >= start + 0.7));
      HL.place(drop.led, this.P(drop.x, drop.y, z + 3.4 * grow));
    });
    if (lines !== this.lines.getAttribute("d")) this.lines.setAttribute("d", lines);
  }

  _drawPosts(count, t, back) {
    this.posts.forEach((post, k) => {
      const leaving = k < count;
      const sink    = leaving ? HL.clamp(t / 1.5, 0, 1) * (1 - back) : 0;
      const h       = HL.lerp(4.6, 0.6, FxRank.smooth(sink));
      if (leaving !== post.dash) {
        post.dash = leaving;
        post.solid.sil.classList.toggle("dash", leaving);
      }
      if (Math.abs(h - post.drawn) < 0.002) return;
      post.drawn    = h;
      this.postShape.put(post.solid, post.x, FxDisruptionFigure.POST_Y, 0, h);
      HL.place(post.led, this.P(post.x, FxDisruptionFigure.POST_Y, h));
      FxParts.set(post.led, leaving && sink > 0.5 ? "dot off" : "dot m");
    });
  }

  _drawVehicle(type, t, back) {
    const y      = FxDisruptionFigure.ROAD[2];
    const parked = 19;
    let x        = parked;
    let heading  = "x+";
    let grow     = 1;
    let dash     = false;
    let open     = 0.1;
    if (type === "new_vehicle") {
      open       = HL.clamp(t / 0.6, 0, 1) * (1 - back) * 0.9 + 0.1;
      const s    = FxRank.smooth(HL.clamp((t - 0.6) / 1.4, 0, 1));
      x          = HL.lerp(-19, -5, s);
      heading    = "x-";
      grow       = t < 0.6 ? 0 : 1 - back;
    } else if (type === "remove_vehicle") {
      dash       = true;
      grow       = 1 - FxRank.smooth(HL.clamp((t - 0.5) / 1, 0, 1)) + back;
    } else if (type === null) {
      open       = 0.9;
    }
    this.door.draw(open);
    this.vanKit.dash(this.van, dash);
    this.vanKit.pose(this.van, heading, x, y, 0.5, Math.min(1, grow));
    this.depth.move(this.vanItem, x + y);
  }

  _drawClock(dt, still) {
    const gap  = this.data.gap;
    const turn = 2 + 6 * Math.min(1, gap / FxDisruptionFigure.GAP_MAX);
    if (!still && gap > 0) this.hand = (this.hand + dt / turn) % 1;
    if (Math.abs(this.hand - this.handDrawn) < 0.0005) return;
    this.handDrawn = this.hand;
    const [x, y]   = FxDisruptionFigure.CLOCK;
    const a        = 2 * Math.PI * this.hand;
    this.handEl.setAttribute("d", HL.seg(this.P(x, y, 10.75), this.P(x + 3 * Math.sin(a), y - 3 * Math.cos(a), 10.75)));
  }

  _hiTarget() {
    if (this.chosen !== null) return this.chosen;
    const type = this.data.type;
    if (type === "new_job" && this.data.count > 0) return this.drops[0].solid.sil;
    if (type === "remove_job" && this.data.count > 0) return this.posts[0].solid.sil;
    if (type === "new_vehicle" || type === "remove_vehicle") return this.van.body.sil;
    return this.depot.sil;
  }

  _move(p) {
    const items = [{ el: this.depot.sil, at: this.P(-21, -5, 5) }, { el: this.dial.sil, at: this.P(...FxDisruptionFigure.CLOCK, 10) }, { el: this.van.body.sil, at: this.P(19, FxDisruptionFigure.ROAD[2], 2) }];
    for (const post of this.posts) items.push({ el: post.solid.sil, at: this.P(post.x, FxDisruptionFigure.POST_Y, 2) });
    for (const stack of this.tally) if (stack.count > 0) items.push({ el: stack.coins[stack.count - 1].sil, at: this.P(stack.x, stack.y, 2) });
    let best = null;
    let gap  = Infinity;
    for (const item of items) {
      const d = Math.hypot(item.at[0] - p[0], item.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item.el;
    }
    this._choose(gap < 70 ? best : null);
  }

  _retarget() {
    this.light.set(this._hiTarget());
    this.loop.wake();
  }
}

class FxDivergeFigure extends HairlineFigureBase {
  static NAME    = "diverge";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 4.1;
  static PLINTH  = [-40, -17, 40, 17];
  static X0      = -35;
  static X1      = 35;
  static HALF    = 3.2;
  static LANE    = 2.2;
  static BEND    = 7;
  static DRIVE_S = 6;
  static REST_S  = 1.5;
  static FADE_S  = 0.5;
  static LEDS    = 8;
  static VAN     = 1.15;
  static DEFAULT = { rounds: 3, split: 1, gap: 0.45, agree: 0.75, teacher: true };

  constructor({ stage, svg }, value) {
    super(svg, value, FxDivergeFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxDivergeFigure.PLINTH;
    this._camera(this._extent(), 170);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.vanKit = new FxVan(this.P, this.front, FxDivergeFigure.VAN);
    FxParts.depot(this.P, this.front, this.g, -39, -6, -35.5, 6, 6);
    this.road   = HL.mk("g", {}, this.g);
    this.cuts   = HL.mk("path", { class: "nf dash", d: "" }, this.g);
    this.leds   = Array.from({ length: FxDivergeFigure.LEDS }, (_, k) => FxParts.led(this.g, this.P, HL.lerp(-30, 30, k / (FxDivergeFigure.LEDS - 1)), 14.5, 0, "dot off"));
    this.depth  = new FxDepth(this.g);
    this.lanes  = ["teacher", "model"].map((key, k) => {
      const postItem = this.depth.add(0);
      const vanItem  = this.depth.add(0);
      const lane     = { key, side: k === 0 ? -1 : 1, postItem, vanItem, post: HL.solid(postItem.group), led: HL.mk("circle", { r: 1.3, class: "dot off", display: "none" }, postItem.group), van: this.vanKit.make(vanItem.group), lift: HL.tween(0), path: null };
      if (key === "teacher") {
        lane.post.sil.classList.add("dash");
        this.vanKit.dash(lane.van, true);
      }
      return lane;
    });
    this.postShape = FxParts.shape(this.P, this.front, -1.4, -1.4, 1.4, 1.4, 0.6, 0.4);
    this._apply(FxDivergeFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxDivergeFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [FxDivergeFigure.X1 - 2, -12, 13], [FxDivergeFigure.X1 - 2, 12, 13]];
  }

  _slab(points, cls) {
    const solid = HL.solid(this.road);
    HL.put(solid, HL.prism(this.P, this.front, HairlineGeometry.samples(points), null, 0, 0.5));
    if (cls) solid.sil.classList.add(cls);
    return solid;
  }

  setState(state) {
    const name = FxDivergeFigure.NAME;
    FxCheck.object(name, "state", state, ["rounds", "split", "gap", "agree", "teacher"]);
    const rounds = FxCheck.integer(name, "rounds", state.rounds, 1, 10);
    const next   = { rounds, split: state.split === null ? null : FxCheck.integer(name, "split", state.split, 0, rounds - 1), gap: FxCheck.number(name, "gap", state.gap, -1, 1), agree: state.agree === null ? null : FxCheck.number(name, "agree", state.agree, 0, 1), teacher: FxCheck.boolean(name, "teacher", state.teacher) };
    const now    = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data  = state;
    const X0   = FxDivergeFigure.X0;
    const X1   = FxDivergeFigure.X1;
    const seg  = (X1 - X0) / state.rounds;
    const xs   = state.split === null ? X0 : X0 + state.split * seg;
    const w    = state.split === null ? 3 : 4 + 8 * Math.abs(state.gap);
    const bend = Math.min(FxDivergeFigure.BEND, (X1 - xs) * 0.4);
    const L    = FxDivergeFigure.LANE;
    this.road.replaceChildren();
    if (xs > X0 + 0.5) {
      const [ring, inner] = HL.rings(X0, -FxDivergeFigure.HALF, xs + 0.5, FxDivergeFigure.HALF, 1.2, 0.7);
      HL.put(HL.solid(this.road), HL.prism(this.P, this.front, ring, inner, 0, 0.5));
    }
    const shown = state.teacher ? this.lanes : this.lanes.filter((lane) => lane.key === "model");
    for (const lane of this.lanes) {
      const on  = shown.includes(lane);
      const y   = state.teacher ? lane.side * w : 0;
      const y0  = xs > X0 + 0.5 ? 0 : y;
      lane.on   = on;
      lane.y    = y;
      lane.path = new FxPolyline([[X0 + 2, y0], [xs, y0], [xs + bend, y], [X1 - 5, y]]);
      if (!on) continue;
      const cls = lane.key === "teacher" ? "dash" : "";
      if (bend > 0.5) this._slab([[xs, y0 - L], [xs + bend, y - L], [xs + bend, y + L], [xs, y0 + L]], cls);
      const [ring, inner] = HL.rings(xs + bend - 0.4, y - L, X1, y + L, 0.8, 0.5);
      const solid = HL.solid(this.road);
      HL.put(solid, HL.prism(this.P, this.front, ring, inner, 0, 0.5));
      if (cls) solid.sil.classList.add(cls);
      if (xs <= X0 + 0.5) {
        const [r2, i2] = HL.rings(X0, y - L, xs + bend, y + L, 0.8, 0.5);
        const s2       = HL.solid(this.road);
        HL.put(s2, HL.prism(this.P, this.front, r2, i2, 0, 0.5));
        if (cls) s2.sil.classList.add(cls);
      }
    }
    let cuts   = "";
    for (let j = 1; j < state.rounds; j++) {
      const x = X0 + j * seg;
      cuts   += HL.seg(this.P(x, -15, 0.55), this.P(x, 15, 0.55));
    }
    this.cuts.setAttribute("d", cuts);
    const better = state.gap >= 0 ? "model" : "teacher";
    for (const lane of this.lanes) {
      const share = lane.key === "model" ? 0.5 + 0.5 * state.gap : 0.5 - 0.5 * state.gap;
      lane.h      = 2 + 10 * share;
      FxParts.show(lane.post.g, lane.on);
      FxParts.show(lane.led, lane.on);
      FxParts.show(lane.van.g, lane.on);
      if (!lane.on) continue;
      this.postShape.put(lane.post, X1 - 2, lane.y, 0.5, 0.5 + lane.h);
      HL.place(lane.led, this.P(X1 - 2, lane.y, 0.5 + lane.h));
      FxParts.set(lane.led, lane.key === better || !state.teacher ? "dot" : "dot m");
      this.depth.move(lane.postItem, X1 - 2 + lane.y);
    }
    const lit  = state.agree === null ? 0 : Math.round(FxDivergeFigure.LEDS * state.agree);
    this.leds.forEach((led, k) => FxParts.set(led, k < lit ? "dot" : "dot off"));
  }

  _tick(dt, now) {
    const still  = HL.reducedMotion();
    if (!still) this.clock += dt;
    const period = FxDivergeFigure.DRIVE_S + FxDivergeFigure.REST_S + FxDivergeFigure.FADE_S;
    const t      = still ? FxDivergeFigure.DRIVE_S : this.clock % period;
    const s      = FxRank.smooth(HL.clamp(t / FxDivergeFigure.DRIVE_S, 0, 1));
    const fadeIn = still ? 1 : HL.clamp(t / 0.3, 0, 1);
    const fade   = still ? 1 : t > FxDivergeFigure.DRIVE_S + FxDivergeFigure.REST_S ? 1 - (t - FxDivergeFigure.DRIVE_S - FxDivergeFigure.REST_S) / FxDivergeFigure.FADE_S : fadeIn;
    let moving   = false;
    for (const lane of this.lanes) {
      if (!lane.on) continue;
      const [x, y] = lane.path.at(s);
      const lift   = HL.tval(lane.lift, now);
      if (!HL.tdone(lane.lift, now)) moving = true;
      this.vanKit.pose(lane.van, "x+", x, y, 0.5 + lift, Math.max(0, Math.min(1, fade)));
      this.depth.move(lane.vanItem, x + y + 0.3);
    }
    this.depth.sort();
    return still ? moving : true;
  }

  _move(p) {
    const [, y] = HL.unproj(this.C, p[0], p[1], 0);
    const on    = this.lanes.filter((lane) => lane.on);
    if (!on.length) return;
    this._choose(on.reduce((best, lane) => (Math.abs(lane.y - y) < Math.abs(best.y - y) ? lane : best)).key);
  }

  _retarget(now) {
    const key = this.chosen === null ? "model" : this.chosen;
    for (const lane of this.lanes) HL.tset(lane.lift, lane.key !== key ? 0 : this.chosen === null ? this.amount * 0.5 : this.amount, now, 0);
    const lane = this.lanes.find((entry) => entry.key === key);
    this.light.set(lane.van.body.sil);
    this.loop.wake();
  }
}

class FxConfusionFigure extends HairlineFigureBase {
  static NAME     = "confusion";
  static RANGE    = [3, 6, 10];
  static SCALE    = 4.1;
  static N        = 3;
  static CELL     = 15;
  static FOOT     = 12.4;
  static H_MIN    = 1.2;
  static H_MAX    = 20;
  static BREATH   = 0.03;
  static BREATH_S = 4;
  static STAGGER  = 40;
  static REST     = 0.4;
  static DEFAULT  = { counts: [40, 6, 12, 3, 20, 4, 10, 2, 30] };

  constructor({ stage, svg }, value) {
    super(svg, value, FxConfusionFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    this.width = FxConfusionFigure.N * FxConfusionFigure.CELL;
    this._camera(this._extent(), 170);
    this._plinth(-3, -3, this.width + 3, this.width + 3, 6, 4);
    this._buildTiles();
    this._apply(FxConfusionFigure.DEFAULT, performance.now());
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const w   = this.width;
    const top = FxConfusionFigure.H_MAX * (1 + FxConfusionFigure.BREATH) + FxConfusionFigure.RANGE[2];
    return [[-3, -3, -4], [w + 3, w + 3, -4], [w + 3, -3, -4], [-3, w + 3, -4], [0, 0, top], [w, 0, top], [0, w, top]];
  }

  _buildTiles() {
    const N     = FxConfusionFigure.N;
    const C     = FxConfusionFigure.CELL;
    const F     = FxConfusionFigure.FOOT;
    const inset = (C - F) / 2;
    const cells = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) cells.push([i, j]);
    cells.sort((a, b) => a[0] + a[1] - (b[0] + b[1]));
    this.tiles = cells.map(([i, j]) => {
      const x0   = i * C + inset;
      const y0   = j * C + inset;
      const ring = HL.rrect(x0, y0, x0 + F, y0 + F, 2, 2);
      const el   = HL.solid(this.g);
      return { i, j, idx: i * N + j, ring, el, shape: new FxPrism(this.P, this.front, ring, HL.rrect(x0 + 0.9, y0 + 0.9, x0 + F - 0.9, y0 + F - 0.9, 1.2, 2)), hw: HL.tween(FxConfusionFigure.H_MIN), tw: HL.tween(0), phase: ((i * N + j) * 2 * Math.PI) / 7, drawn: NaN, centre: [x0 + F / 2, y0 + F / 2], cls: "" };
    });
    this.byIndex = new Map(this.tiles.map((tile) => [tile.idx, tile]));
    this.dot     = HL.flatDot(this.g, this.C, 1.1, "dot");
  }

  setState(state) {
    const name   = FxConfusionFigure.NAME;
    FxCheck.object(name, "state", state, ["counts"]);
    FxCheck.list(name, "counts", state.counts, FxConfusionFigure.N * FxConfusionFigure.N);
    const counts = Array.from(state.counts, (v, k) => FxCheck.number(name, `counts[${k}]`, v, 0));
    const now    = performance.now();
    this._apply({ counts }, now);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state, now) {
    const N   = FxConfusionFigure.N;
    let max   = 0;
    let best  = -1;
    let top   = 0;
    state.counts.forEach((v, idx) => {
      if (v > max) max = v;
      if (v > state.counts[top]) top = idx;
      const i = Math.floor(idx / N);
      if (i !== idx % N && v > 0 && (best < 0 || v > state.counts[best])) best = idx;
    });
    for (const tile of this.tiles) {
      const h   = max > 0 ? FxConfusionFigure.H_MIN + (FxConfusionFigure.H_MAX - FxConfusionFigure.H_MIN) * Math.sqrt(state.counts[tile.idx] / max) : FxConfusionFigure.H_MIN;
      HL.tset(tile.hw, h, now, (tile.i + tile.j) * 40);
      tile.hull = HL.hull(HL.ringAt(this.P, tile.ring, 0).concat(HL.ringAt(this.P, tile.ring, h)));
      const cls = max === 0 ? "lo dash" : tile.i === tile.j ? "lo" : "sil";
      if (cls !== tile.cls) {
        tile.cls = cls;
        tile.el.sil.setAttribute("class", cls);
      }
    }
    this.rest = best < 0 ? top : best;
    this.best = best;
    if (best >= 0) this.byIndex.get(best).drawn = NaN;
    this.dot.setAttribute("visibility", best < 0 ? "hidden" : "visible");
    if (best >= 0) this.byIndex.get(best).el.g.appendChild(this.dot);
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const wave  = (2 * Math.PI * this.clock) / FxConfusionFigure.BREATH_S;
    let moving  = false;
    for (const tile of this.tiles) {
      const settled = HL.tdone(tile.tw, now) && HL.tdone(tile.hw, now);
      if (!settled) moving = true;
      const h       = HL.tval(tile.hw, now) * (still ? 1 : 1 + FxConfusionFigure.BREATH * Math.sin(wave + tile.phase)) + HL.tval(tile.tw, now);
      if (Math.abs(h - tile.drawn) < 0.004) continue;
      tile.drawn    = h;
      tile.shape.put(tile.el, 0, 0, 0, h);
      if (tile.idx === this.best) HL.place(this.dot, this.P(tile.centre[0], tile.centre[1], h));
    }
    return still ? moving : true;
  }

  _move(p) {
    for (let k = this.tiles.length - 1; k >= 0; k--) {
      if (HairlineGeometry.inside(this.tiles[k].hull, p)) return this._choose(this.tiles[k].idx);
    }
    const [x, y] = HL.unproj(this.C, p[0], p[1], 0);
    const i      = Math.floor(x / FxConfusionFigure.CELL);
    const j      = Math.floor(y / FxConfusionFigure.CELL);
    if (i < 0 || j < 0 || i >= FxConfusionFigure.N || j >= FxConfusionFigure.N) return;
    this._choose(i * FxConfusionFigure.N + j);
  }

  _retarget(now) {
    const centre = this.byIndex.get(this.chosen === null ? this.rest : this.chosen);
    const amount = this.chosen === null ? this.amount * FxConfusionFigure.REST : this.amount;
    for (const tile of this.tiles) {
      const ring = Math.max(Math.abs(tile.i - centre.i), Math.abs(tile.j - centre.j));
      HL.tset(tile.tw, ring === 0 ? amount : 0, now, ring * FxConfusionFigure.STAGGER);
    }
    this.light.set(centre.el.sil);
    this.loop.wake();
  }
}

class FxRankingFigure extends HairlineFigureBase {
  static NAME    = "ranking";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 4.6;
  static PLINTH  = [-30, -12, 30, 12];
  static STEPS   = 5;
  static TREAD   = 10;
  static DEPTH   = 5.5;
  static RISE    = 2.2;
  static CRATE   = 2.4;
  static H       = [1.5, 8.5];
  static TILT    = 8;
  static SWAY    = 0.6;
  static SWAY_S  = 4.2;
  static CLIMB_S = 1.2;
  static TOP_S   = 2;
  static PUCK    = [1.6, 1.1];
  static DEFAULT = { ranks: [3.4, 3.1, 2.8, 2.2, 1.6], rho: -0.4 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxRankingFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxRankingFigure.PLINTH;
    this._camera(this._extent(), 176);
    this._plinth(x0, y0, x1, y1, 6, 3);
    const c    = FxRankingFigure.CRATE;
    this.crateShape = FxParts.shape(this.P, this.front, -c, -c, c, c, 0.7, 0.5);
    this.puckShape  = new FxPrism(this.P, this.front, HL.circ(FxRankingFigure.PUCK[0], 24), HL.circ(FxRankingFigure.PUCK[0] - 0.5, 24));
    this._buildStairs();
    this.ghost = HL.mk("path", { class: "nf lo dash", d: "" }, this.g);
    this.bar   = HL.solid(this.g);
    this.barRing  = HL.rrect(-24, -1.1, 24, 1.1, 0.8, 4);
    this.barInner = HL.rrect(-23.4, -0.6, 23.4, 0.6, 0.5, 4);
    this.puck  = { el: HL.solid(this.g), drawn: "" };
    this.sway  = HL.spring(0, { k: 60, c: 9 });
    this.barDrawn = NaN;
    this._apply(FxRankingFigure.DEFAULT, performance.now());
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxRankingFigure.PLINTH;
    const top = FxRankingFigure.STEPS * FxRankingFigure.RISE + FxRankingFigure.H[1] + 6;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [x0 + 4, 0, top], [x1 - 4, 0, top]];
  }

  _x(k) {
    return -((FxRankingFigure.STEPS - 1) / 2) * FxRankingFigure.TREAD + k * FxRankingFigure.TREAD;
  }

  _buildStairs() {
    const t  = FxRankingFigure.TREAD / 2;
    const d  = FxRankingFigure.DEPTH;
    this.treads = Array.from({ length: FxRankingFigure.STEPS }, (_, k) => {
      const x    = this._x(k);
      const top  = FxRankingFigure.RISE * (k + 1);
      const tread = FxParts.box(this.P, this.front, this.g, x - t, -d, x + t, d, 0, top, 0.8, 0.5);
      const g    = HL.mk("g", {}, this.g);
      const crate = HL.solid(g);
      const led  = HL.mk("circle", { r: 1.2, class: "dot", display: "none" }, g);
      return { k, x, top, tread, crate, led, hw: HL.tween(FxRankingFigure.H[0]), lw: HL.tween(0), drawn: NaN, value: null, rest: this.P(x, 0, top) };
    });
  }

  setState(state) {
    const name  = FxRankingFigure.NAME;
    FxCheck.object(name, "state", state, ["ranks", "rho"]);
    FxCheck.list(name, "ranks", state.ranks, FxRankingFigure.STEPS);
    const ranks = Array.from(state.ranks, (v, k) => (v === null ? null : FxCheck.positive(name, `ranks[${k}]`, v)));
    const now   = performance.now();
    this._apply({ ranks, rho: state.rho === null ? null : FxCheck.number(name, "rho", state.rho, -1, 1) }, now);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state, now) {
    this.data   = state;
    const live  = state.ranks.filter((v) => v !== null);
    const lo    = live.length ? Math.min(...live) : 0;
    const hi    = live.length ? Math.max(...live) : 0;
    const [h0, h1] = FxRankingFigure.H;
    let best    = -1;
    for (const tread of this.treads) {
      const v    = state.ranks[tread.k];
      tread.value = v;
      const h    = v === null ? 0.6 : hi > lo ? h0 + (h1 - h0) * (1 - (v - lo) / (hi - lo)) : (h0 + h1) / 2;
      HL.tset(tread.hw, h, now, tread.k * 40);
      tread.crate.sil.classList.toggle("dash", v === null);
      if (v !== null && (best < 0 || v < state.ranks[best])) best = tread.k;
      tread.drawn = NaN;
    }
    this.best   = best;
    this.treads.forEach((tread) => FxParts.show(tread.led, tread.k === best));
    this.rest   = best < 0 ? FxRankingFigure.STEPS - 1 : best;
    const first = this.treads[0];
    const last  = this.treads[FxRankingFigure.STEPS - 1];
    this.ghost.setAttribute("d", HL.seg(this.P(first.x, 0, first.top + h0), this.P(last.x, 0, last.top + h1)));
  }

  _crateTop(tread, now) {
    return tread.top + HL.tval(tread.hw, now) + HL.tval(tread.lw, now);
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    let moving  = false;
    for (const tread of this.treads) {
      const busy = !HL.tdone(tread.hw, now) || !HL.tdone(tread.lw, now);
      if (busy) moving = true;
      const top  = this._crateTop(tread, now);
      if (Math.abs(top - tread.drawn) < 0.004) continue;
      tread.drawn = top;
      this.crateShape.put(tread.crate, tread.x, 0, tread.top, top);
      HL.place(tread.led, this.P(tread.x, 0, top));
    }
    const rho   = this.data.rho === null ? 0 : this.data.rho;
    const sway  = still ? 0 : HL.rad(FxRankingFigure.SWAY) * Math.sin((2 * Math.PI * this.clock) / FxRankingFigure.SWAY_S);
    this.sway.t = -HL.rad(FxRankingFigure.TILT) * rho + sway;
    moving      = HL.stepS(this.sway, dt) || moving;
    this._drawBar(now);
    this._drawPuck(now, still);
    return still ? moving : true;
  }

  _drawBar(now) {
    const theta = this.sway.x;
    let peak    = 0;
    for (const tread of this.treads) peak = Math.max(peak, this._crateTop(tread, now));
    const zc    = peak + 1.6;
    const key   = `${Math.round(theta * 1e4)}|${Math.round(zc * 100)}`;
    if (key === this.barDrawn) return;
    this.barDrawn = key;
    const c     = Math.cos(theta);
    const s     = Math.sin(theta);
    const at    = (u, v, w) => this.P(u * c - w * s, v, zc + u * s + w * c);
    const top   = this.barRing.map((q) => at(q.u, q.v, 0.6));
    const bot   = this.barRing.map((q) => at(q.u, q.v, -0.6));
    HL.put(this.bar, { sil: HL.poly(HL.hull(top.concat(bot))), crease: HL.open(HL.run(this.barInner, this.front).map((q) => at(q.u, q.v, 0.6))) });
  }

  _drawPuck(now, still) {
    const n      = FxRankingFigure.STEPS;
    const period = n * FxRankingFigure.CLIMB_S + FxRankingFigure.TOP_S;
    const t      = still ? period - 0.01 : this.clock % period;
    const k      = Math.min(n - 1, Math.floor(t / FxRankingFigure.CLIMB_S));
    const s      = t >= n * FxRankingFigure.CLIMB_S ? 1 : FxRank.smooth(((t - k * FxRankingFigure.CLIMB_S) / FxRankingFigure.CLIMB_S) * 1.6);
    const a      = this.treads[Math.max(0, k - 1)];
    const b      = this.treads[k];
    const za     = this._crateTop(a, now);
    const zb     = this._crateTop(b, now);
    const arc    = k === 0 ? 0 : 3 * Math.sin(Math.PI * s);
    const from   = k === 0 ? b : a;
    const pose   = [HL.lerp(from.x, b.x, s), 0, HL.lerp(k === 0 ? zb : za, zb, s) + arc];
    const key    = pose.map((v) => Math.round(v * 100)).join("|");
    this.at      = s > 0.5 ? b : from;
    if (key === this.puck.drawn) return;
    this.puck.drawn = key;
    this.puckShape.put(this.puck.el, pose[0], pose[1], pose[2], pose[2] + FxRankingFigure.PUCK[1]);
  }

  _move(p) {
    this._choose(HairlineFigureBase.nearest(this.treads.map((tread) => tread.rest[0]), p[0]));
  }

  _retarget(now) {
    const lit = this.chosen === null ? this.rest : this.chosen;
    for (const tread of this.treads) HL.tset(tread.lw, tread.k !== lit ? 0 : this.chosen === null ? this.amount * 0.4 : this.amount, now, 0);
    this.light.set(this.treads[lit].crate.sil);
    this.loop.wake();
  }
}

class FxCheckpointFigure extends HairlineFigureBase {
  static NAME    = "checkpoint";
  static RANGE   = [2, 4, 7];
  static SCALE   = 4.75;
  static PLINTH  = [-29, -14, 29, 14];
  static BAYS    = 4;
  static PITCH   = 12;
  static LEVELS  = [0.9, 11.2];
  static SHELF   = 0.9;
  static DEPTH   = 6.2;
  static CART    = [3.6, 4.6];
  static HEIGHT  = { ppo: 7, pretrain: 4.8 };
  static SCAN_S  = 1.1;
  static REST_S  = 4.6;
  static BUMP    = 0.6;
  static STATES  = ["loadable", "incompatible", "none"];
  static KINDS   = ["ppo", "pretrain"];
  static DEFAULT = { slots: [{ state: "loadable", report: true, kind: "ppo" }, { state: "loadable", report: true, kind: "pretrain" }, { state: "loadable", report: false, kind: "ppo" }, { state: "incompatible", report: false, kind: "pretrain" }, { state: "none", report: false, kind: "pretrain" }, { state: "incompatible", report: false, kind: "ppo" }], selected: 0 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxCheckpointFigure.SCALE);
    this.stage = stage;
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxCheckpointFigure.PLINTH;
    this._camera(this._extent(), 168);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.shift = new FxShift(this.P);
    this._buildRack();
    this._apply(FxCheckpointFigure.DEFAULT);
    this._start(stage, svg);
    this.bag.on(stage, "click", (e) => this._pick(e));
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxCheckpointFigure.PLINTH;
    const top = FxCheckpointFigure.LEVELS[1] + FxCheckpointFigure.SHELF + FxCheckpointFigure.HEIGHT.ppo + 2;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [x0 + 3, -FxCheckpointFigure.DEPTH, top], [x1 - 3, -FxCheckpointFigure.DEPTH, top]];
  }

  _bayX(b) {
    return (b - (FxCheckpointFigure.BAYS - 1) / 2) * FxCheckpointFigure.PITCH;
  }

  _buildRack() {
    const half  = (FxCheckpointFigure.BAYS * FxCheckpointFigure.PITCH) / 2 + 1;
    const d     = FxCheckpointFigure.DEPTH;
    const [lo, hi] = FxCheckpointFigure.LEVELS;
    const s     = FxCheckpointFigure.SHELF;
    FxParts.box(this.P, this.front, this.g, -half - 1.6, -d, -half, d, 0, hi + s + 9, 0.6, 0.4);
    const [cx, cy] = FxCheckpointFigure.CART;
    this.cart   = { ppo: FxParts.shape(this.P, this.front, -cx, -cy, cx, cy, 0.8, 0.5), pretrain: FxParts.shape(this.P, this.front, -cx, -cy, cx, cy, 0.8, 0.5) };
    this.slots  = [];
    for (const [level, z] of [[0, lo], [1, hi]]) {
      FxParts.box(this.P, this.front, this.g, -half, -d, half, d, z - s, z, 0.6, 0.4);
      for (let b = 0; b < FxCheckpointFigure.BAYS; b++) {
        const x      = this._bayX(b);
        const socket = HL.mk("path", { class: "nf lo", d: FxParts.flat(this.P, [[x - cx, -cy], [x + cx, -cy], [x + cx, cy], [x - cx, cy]], z) }, this.g);
        const g      = HL.mk("g", {}, this.g);
        const solid  = HL.solid(g);
        const led    = HL.mk("circle", { r: 1.25, class: "dot off" }, g);
        this.slots.push({ index: level * FxCheckpointFigure.BAYS + b, level, b, x, z, socket, g, solid, led, lw: HL.tween(0), at: "", state: "none", kind: "ppo", rest: this.P(x, 0, z + 3) });
      }
    }
    FxParts.box(this.P, this.front, this.g, half, -d, half + 1.6, d, 0, hi + s + 9, 0.6, 0.4);
    this.scanner = HL.mk("path", { class: "nf dash", d: "" }, this.g);
  }

  setState(state) {
    const name  = FxCheckpointFigure.NAME;
    FxCheck.object(name, "state", state, ["slots", "selected"]);
    if (!Array.isArray(state.slots) || state.slots.length > 8) throw new Error(`${name}: slots must be an array of at most 8 slots, got ${state.slots}`);
    const slots = state.slots.map((slot, k) => {
      FxCheck.object(name, `slots[${k}]`, slot, ["state", "report", "kind"]);
      if (!FxCheckpointFigure.STATES.includes(slot.state)) throw new Error(`${name}: slots[${k}].state must be one of ${FxCheckpointFigure.STATES.join(", ")}, got ${slot.state}`);
      if (!FxCheckpointFigure.KINDS.includes(slot.kind)) throw new Error(`${name}: slots[${k}].kind must be ppo or pretrain, got ${slot.kind}`);
      return { state: slot.state, report: FxCheck.boolean(name, `slots[${k}].report`, slot.report), kind: slot.kind };
    });
    const next  = { slots, selected: state.selected === null ? null : FxCheck.integer(name, "selected", state.selected, 0, Math.max(0, slots.length - 1)) };
    const now   = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    for (const slot of this.slots) {
      const src   = state.slots[slot.index] || null;
      slot.state  = src ? src.state : "none";
      slot.kind   = src ? src.kind : "ppo";
      const full  = slot.state !== "none";
      FxParts.show(slot.g, full);
      FxParts.set(slot.socket, src ? "nf lo" : "nf lo dash");
      if (!full) continue;
      const h     = FxCheckpointFigure.HEIGHT[slot.kind];
      this.cart[slot.kind].put(slot.solid, slot.x, 0, slot.z, slot.z + h);
      slot.solid.sil.classList.toggle("dash", slot.state === "incompatible");
      HL.place(slot.led, this.P(slot.x, FxCheckpointFigure.CART[1], slot.z + h - 1.4));
      FxParts.set(slot.led, slot.state === "incompatible" ? "dot off" : src.report ? "dot" : "dot m");
    }
    this.rest = state.selected === null ? 0 : state.selected;
  }

  _tick(dt, now) {
    const still  = HL.reducedMotion();
    if (!still) this.clock += dt;
    const B      = FxCheckpointFigure.BAYS;
    const period = B * FxCheckpointFigure.SCAN_S + FxCheckpointFigure.REST_S;
    const t      = this.clock % period;
    const pos    = still || t > B * FxCheckpointFigure.SCAN_S ? -1 : t / FxCheckpointFigure.SCAN_S - 0.5;
    let moving   = false;
    for (const slot of this.slots) {
      if (!HL.tdone(slot.lw, now)) moving = true;
      const bump = pos < -0.9 ? 0 : FxCheckpointFigure.BUMP * Math.max(0, 1 - Math.abs(pos - slot.b));
      const at   = this.shift.of(0, HL.tval(slot.lw, now), bump);
      if (at === slot.at) continue;
      slot.at    = at;
      slot.g.setAttribute("transform", at);
    }
    const x      = pos < -0.9 ? null : this._bayX(0) + pos * FxCheckpointFigure.PITCH;
    const d      = x === null ? "" : HL.seg(this.P(x, FxCheckpointFigure.DEPTH + 1.5, 0), this.P(x, FxCheckpointFigure.DEPTH + 1.5, FxCheckpointFigure.LEVELS[1] + 9));
    if (d !== this.scanner.getAttribute("d")) this.scanner.setAttribute("d", d);
    return still ? moving : true;
  }

  _nearest(p) {
    let best = 0;
    let gap  = Infinity;
    for (const slot of this.slots) {
      const d = Math.hypot(slot.rest[0] - p[0], slot.rest[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = slot.index;
    }
    return best;
  }

  _move(p) {
    this._choose(this._nearest(p));
  }

  _pick(e) {
    if (!e.isTrusted) return;
    const box  = this.stage.getBoundingClientRect();
    const slot = this._nearest([((e.clientX - box.left) / box.width) * 400, ((e.clientY - box.top) / box.height) * 320]);
    this._choose(slot);
    if (slot >= this.data.slots.length) return;
    this.stage.dispatchEvent(new CustomEvent("figure-pick", { bubbles: true, detail: { slot } }));
  }

  _retarget(now) {
    const lit  = this.chosen === null ? this.rest : this.chosen;
    for (const slot of this.slots) HL.tset(slot.lw, slot.index !== lit ? 0 : this.chosen === null ? this.amount : this.amount * 1.15, now, 0);
    const slot = this.slots[lit];
    this.light.set(slot.state === "none" ? slot.socket : slot.solid.sil);
    this.loop.wake();
  }
}

class FxTrainingFigure extends HairlineFigureBase {
  static NAME    = "training";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 5.5;
  static PLINTH  = [-31, -12, 31, 12];
  static N       = 24;
  static PITCH   = 2.45;
  static SLAT    = [0.9, 5.5];
  static H       = [1.2, 15];
  static NULL_H  = 0.6;
  static STEP_S  = 0.22;
  static REST_S  = 2.2;
  static PUCK    = [1.3, 1];
  static LIVE_S  = 1.2;
  static DEFAULT = { values: Array.from({ length: 24 }, (_, i) => 0.15 + 0.75 * (1 - Math.exp(-i / 7)) + 0.05 * Math.sin(i * 1.7)), best: 21, live: false };

  constructor({ stage, svg }, value) {
    super(svg, value, FxTrainingFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxTrainingFigure.PLINTH;
    this._camera(this._extent(), 176);
    this._plinth(x0, y0, x1, y1, 6, 3);
    const [a, b] = FxTrainingFigure.SLAT;
    this.shape = FxParts.shape(this.P, this.front, -a, -b, a, b, 0.4, 0.25);
    this.puckShape = new FxPrism(this.P, this.front, HL.circ(FxTrainingFigure.PUCK[0], 24), HL.circ(FxTrainingFigure.PUCK[0] - 0.4, 24));
    this.slats = Array.from({ length: FxTrainingFigure.N }, (_, i) => {
      const x     = (i - (FxTrainingFigure.N - 1) / 2) * FxTrainingFigure.PITCH;
      const solid = HL.solid(this.g);
      return { i, x, solid, hw: HL.tween(FxTrainingFigure.H[0]), lw: HL.tween(0), drawn: NaN, value: null, rest: this.P(x, 0, 0)[0] };
    });
    this.dot   = HL.flatDot(this.g, this.C, 0.9, "dot");
    this.puck  = { el: HL.solid(this.g), drawn: "" };
    this._apply(FxTrainingFigure.DEFAULT, performance.now());
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxTrainingFigure.PLINTH;
    const top = FxTrainingFigure.H[1] + FxTrainingFigure.RANGE[2] + 2;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [x0 + 3, 0, top], [x1 - 3, 0, top]];
  }

  setState(state) {
    const name   = FxTrainingFigure.NAME;
    FxCheck.object(name, "state", state, ["values", "best", "live"]);
    FxCheck.list(name, "values", state.values, FxTrainingFigure.N);
    const values = Array.from(state.values, (v, k) => (v === null ? null : FxCheck.number(name, `values[${k}]`, v, 0, 1)));
    const now    = performance.now();
    this._apply({ values, best: state.best === null ? null : FxCheck.integer(name, "best", state.best, 0, FxTrainingFigure.N - 1), live: FxCheck.boolean(name, "live", state.live) }, now);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state, now) {
    this.data = state;
    const [h0, h1] = FxTrainingFigure.H;
    for (const slat of this.slats) {
      const v    = state.values[slat.i];
      slat.value = v;
      HL.tset(slat.hw, v === null ? FxTrainingFigure.NULL_H : h0 + (h1 - h0) * v, now, slat.i * 18);
      FxParts.set(slat.solid.sil, v === null ? "lo dash" : "sil");
      slat.drawn = NaN;
    }
    const best = state.best;
    this.best  = best;
    FxParts.show(this.dot, best !== null);
    if (best !== null) this.slats[best].solid.g.appendChild(this.dot);
    this.last  = this.slats.reduce((last, slat) => (slat.value !== null ? slat.i : last), FxTrainingFigure.N - 1);
  }

  _top(slat, now, breath) {
    return HL.tval(slat.hw, now) * (slat.i === this.last ? breath : 1) + HL.tval(slat.lw, now);
  }

  _tick(dt, now) {
    const still  = HL.reducedMotion();
    if (!still) this.clock += dt;
    const breath = this.data.live && !still ? 1 + 0.08 * (0.5 - 0.5 * Math.cos((2 * Math.PI * this.clock) / FxTrainingFigure.LIVE_S)) : 1;
    let moving   = false;
    for (const slat of this.slats) {
      if (!HL.tdone(slat.hw, now) || !HL.tdone(slat.lw, now)) moving = true;
      const top  = this._top(slat, now, breath);
      if (Math.abs(top - slat.drawn) < 0.004) continue;
      slat.drawn = top;
      this.shape.put(slat.solid, slat.x, 0, 0, top);
      if (slat.i === this.best) HL.place(this.dot, this.P(slat.x, 0, top));
    }
    this._drawPuck(now, still, breath);
    return still ? moving : true;
  }

  _drawPuck(now, still, breath) {
    const n      = FxTrainingFigure.N;
    const period = n * FxTrainingFigure.STEP_S + FxTrainingFigure.REST_S;
    const t      = still ? period - 0.01 : this.clock % period;
    const u      = Math.min(n - 1, t / FxTrainingFigure.STEP_S);
    const i      = Math.floor(u);
    const j      = Math.min(n - 1, i + 1);
    const s      = u - i;
    const a      = this.slats[i];
    const b      = this.slats[j];
    const za     = this._top(a, now, breath);
    const zb     = this._top(b, now, breath);
    const pose   = [HL.lerp(a.x, b.x, s), 0, HL.lerp(za, zb, FxRank.smooth(s)) + 0.6 * Math.sin(Math.PI * s)];
    this.under   = s < 0.5 ? a : b;
    const key    = pose.map((v) => Math.round(v * 100)).join("|");
    if (key === this.puck.drawn) return;
    this.puck.drawn = key;
    this.puckShape.put(this.puck.el, pose[0], pose[1], pose[2], pose[2] + FxTrainingFigure.PUCK[1]);
    if (this.chosen === null) this.light.set(this.under.solid.sil);
  }

  _move(p) {
    this._choose(HairlineFigureBase.nearest(this.slats.map((slat) => slat.rest), p[0]));
  }

  _retarget(now) {
    const lit = this.chosen;
    for (const slat of this.slats) HL.tset(slat.lw, slat.i === lit ? this.amount : 0, now, 0);
    this.light.set(lit === null ? (this.under || this.slats[0]).solid.sil : this.slats[lit].solid.sil);
    this.loop.wake();
  }
}

class FxQueueFigure extends HairlineFigureBase {
  static NAME    = "queue";
  static RANGE   = [1.5, 3, 5];
  static SCALE   = 5.05;
  static PLINTH  = [-34, -13, 34, 13];
  static BELT    = [-30, 8, -3.6, 3.6, 1.4];
  static GATE_X  = 12;
  static PALLET  = [19, 31, -8, 8];
  static CRATE   = 1.6;
  static CRATE_H = 2.8;
  static PITCH   = 5.2;
  static MARKS   = 3.6;
  static SPEED   = 6;
  static LAMP_S  = 0.8;
  static BREATH_S = 5;
  static INCH    = 0.6;
  static DEFAULT = { running: 1, queued: 2, finished: 5, failed: 1 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxQueueFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxQueueFigure.PLINTH;
    this._camera(this._extent(), 168);
    this._plinth(x0, y0, x1, y1, 6, 3);
    const c    = FxQueueFigure.CRATE;
    this.crate = FxParts.shape(this.P, this.front, -c, -c, c, c, 0.5, 0.4);
    const [b0, b1, w0, w1, bh] = FxQueueFigure.BELT;
    FxParts.box(this.P, this.front, this.g, b0, w0, b1, w1, 0, bh, 1, 0.6);
    this.marks = HL.mk("path", { class: "nf lo", d: "" }, this.g);
    const [p0, p1, q0, q1] = FxQueueFigure.PALLET;
    this.depth = new FxDepth(this.g);
    const pallet = this.depth.add(p0 + q0 - 2);
    FxParts.box(this.P, this.front, pallet.group, p0, q0, p1, q1, 0, 1, 1, 0.6);
    this._buildGate();
    this.queued = Array.from({ length: 6 }, (_, k) => {
      const item = this.depth.add(0);
      return { k, item, solid: HL.solid(item.group), drawn: "" };
    });
    this.runner = { item: this.depth.add(FxQueueFigure.GATE_X), solid: null, drawn: "" };
    this.runner.solid = HL.solid(this.runner.item.group);
    this.stacks = [[24, -3.5, "sil", 8], [26.5, 4, "sil dash", 4]].map(([x, y, cls, max], k) => {
      const item  = this.depth.add(x + y);
      const crates = Array.from({ length: max }, () => {
        const solid = HL.solid(item.group);
        solid.sil.setAttribute("class", cls);
        return solid;
      });
      return { k, x, y, crates, count: -1, drawn: "" };
    });
    this._apply(FxQueueFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxQueueFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [FxQueueFigure.GATE_X, -7, 13], [24, -3.5, 2 + 4 * 3.1]];
  }

  _buildGate() {
    const x    = FxQueueFigure.GATE_X;
    const far  = this.depth.add(x - 8);
    FxParts.box(this.P, this.front, far.group, x - 0.9, -7.4, x + 0.9, -5.6, 0, 10.5, 0.5, 0.3);
    const near = this.depth.add(x + 8);
    FxParts.box(this.P, this.front, near.group, x - 0.9, 5.6, x + 0.9, 7.4, 0, 10.5, 0.5, 0.3);
    this.beam  = FxParts.box(this.P, this.front, near.group, x - 0.9, -7.4, x + 0.9, 7.4, 10.5, 12, 0.5, 0.3);
    this.lamp  = FxParts.led(near.group, this.P, x, 0, 12, "dot off");
  }

  setState(state) {
    const name = FxQueueFigure.NAME;
    FxCheck.object(name, "state", state, ["running", "queued", "finished", "failed"]);
    const next = {};
    for (const key of ["running", "queued", "finished", "failed"]) next[key] = FxCheck.integer(name, key, state[key], 0, Infinity);
    const now  = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data  = state;
    this.nQ    = Math.min(6, state.queued);
    const counts = [Math.min(8, state.finished), Math.min(4, state.failed)];
    for (const stack of this.stacks) {
      const count = counts[stack.k];
      if (count === stack.count) continue;
      stack.count = count;
      const pitch = FxQueueFigure.CRATE_H + 0.35;
      stack.crates.forEach((solid, i) => HL.put(solid, i < count ? this.crate.path(stack.x, stack.y, 1 + i * pitch, 1 + i * pitch + FxQueueFigure.CRATE_H) : FxPrism.EMPTY));
    }
    this.stacks.forEach((stack) => (stack.drawn = ""));
    this.rest = state.running > 0 ? this.runner.solid.sil : this.stacks[0].count > 0 ? this.stacks[0].crates[this.stacks[0].count - 1].sil : this.nQ > 0 ? this.queued[0].solid.sil : this.beam.sil;
  }

  _tick(dt, now) {
    const still   = HL.reducedMotion();
    if (!still) this.clock += dt;
    const running = this.data.running > 0;
    const bh      = FxQueueFigure.BELT[4];
    const travel  = running && !still ? (this.clock * FxQueueFigure.SPEED) % FxQueueFigure.MARKS : 0;
    let marks     = "";
    for (let x = FxQueueFigure.BELT[0] + 1 + travel; x < FxQueueFigure.BELT[1] - 0.5; x += FxQueueFigure.MARKS) marks += HL.seg(this.P(x, -2.8, bh), this.P(x, 2.8, bh));
    if (marks !== this.marks.getAttribute("d")) this.marks.setAttribute("d", marks);
    const inch    = running && !still ? FxQueueFigure.INCH * (0.5 - 0.5 * Math.cos((2 * Math.PI * this.clock) / 1.1)) : 0;
    const breath  = !running && !still ? 0.05 * (0.5 - 0.5 * Math.cos((2 * Math.PI * this.clock) / FxQueueFigure.BREATH_S)) : 0;
    this.queued.forEach((crate) => {
      const shown = crate.k < this.nQ;
      const x     = FxQueueFigure.BELT[1] - 3 - crate.k * FxQueueFigure.PITCH + inch;
      const h     = FxQueueFigure.CRATE_H * (crate.k === 0 && !running ? 1 + breath * 3 : 1);
      const key   = shown ? `${Math.round(x * 100)}|${Math.round(h * 100)}` : "";
      if (key === crate.drawn) return;
      crate.drawn = key;
      HL.put(crate.solid, shown ? this.crate.path(x, 0, bh, bh + h) : FxPrism.EMPTY);
      this.depth.move(crate.item, x);
    });
    const rKey    = running ? "on" : "";
    if (rKey !== this.runner.drawn) {
      this.runner.drawn = rKey;
      HL.put(this.runner.solid, running ? this.crate.path(FxQueueFigure.GATE_X, 0, 0, FxQueueFigure.CRATE_H * 1.2) : FxPrism.EMPTY);
    }
    const lamp    = !running ? "dot off" : still || (this.clock / FxQueueFigure.LAMP_S) % 2 < 1 ? "dot" : "dot m";
    FxParts.set(this.lamp, lamp);
    this.light.set(this.chosen === null ? this.rest : this.chosen);
    this.depth.sort();
    return still ? false : true;
  }

  _items() {
    const items = [];
    if (this.data.running > 0) items.push({ el: this.runner.solid.sil, at: this.P(FxQueueFigure.GATE_X, 0, 2) });
    this.queued.forEach((crate) => {
      if (crate.k < this.nQ) items.push({ el: crate.solid.sil, at: this.P(FxQueueFigure.BELT[1] - 3 - crate.k * FxQueueFigure.PITCH, 0, 3) });
    });
    for (const stack of this.stacks) if (stack.count > 0) items.push({ el: stack.crates[stack.count - 1].sil, at: this.P(stack.x, stack.y, 2 + stack.count * 1.6) });
    return items;
  }

  _move(p) {
    let best = null;
    let gap  = Infinity;
    for (const item of this._items()) {
      const d = Math.hypot(item.at[0] - p[0], item.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item.el;
    }
    this._choose(gap < 60 ? best : null);
  }

  _retarget() {
    this.light.set(this.chosen === null ? this.rest : this.chosen);
    this.loop.wake();
  }
}

class FxServicesFigure extends HairlineFigureBase {
  static NAME    = "services";
  static RANGE   = [0.5, 1.5, 3];
  static SCALE   = 5.1;
  static PLINTH  = [-27, -18, 27, 18];
  static CONSOLE = [3, 7, 11, 13, 2.4];
  static TOWERS  = [{ key: "osrm", x: -14, y: -8 }, { key: "vroom", x: 9, y: -11 }];
  static TOWER   = 4;
  static PACKET  = 1.8;
  static BLINK_S = 1.5;
  static DEFAULT = { osrm: true, vroom: true, reachable: true };

  constructor({ stage, svg }, value) {
    super(svg, value, FxServicesFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxServicesFigure.PLINTH;
    this._camera(this._extent(), 166);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.shift  = new FxShift(this.P);
    this.cables = HL.mk("g", {}, this.g);
    this.towers = FxServicesFigure.TOWERS.map((spec, k) => this._buildTower(spec, k));
    const [c0, d0, c1, d1, h] = FxServicesFigure.CONSOLE;
    this.console = FxParts.box(this.P, this.front, this.g, c0, d0, c1, d1, 0, h, 0.9, 0.6);
    this.consoleLed = FxParts.led(this.g, this.P, (c0 + c1) / 2, (d0 + d1) / 2, h, "dot");
    this.packets = this.towers.map(() => HL.flatDot(this.g, this.C, 0.9, "dot"));
    this._apply(FxServicesFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxServicesFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [-14, -8, 21], [9, -11, 21]];
  }

  _buildTower(spec, k) {
    const t     = FxServicesFigure.TOWER;
    const g     = HL.mk("g", {}, this.g);
    const parts = [];
    if (spec.key === "osrm") {
      parts.push(FxParts.box(this.P, this.front, g, spec.x - t, spec.y - t, spec.x + t, spec.y + t, 0, 16, 0.9, 0.6));
      let grid = "";
      for (const f of [-1 / 3, 1 / 3]) {
        grid += HL.seg(this.P(spec.x + f * 2 * t, spec.y - t + 0.8, 16), this.P(spec.x + f * 2 * t, spec.y + t - 0.8, 16));
        grid += HL.seg(this.P(spec.x - t + 0.8, spec.y + f * 2 * t, 16), this.P(spec.x + t - 0.8, spec.y + f * 2 * t, 16));
      }
      HL.mk("path", { class: "nf dash", d: grid }, g);
    } else {
      for (let s = 0; s < 3; s++) parts.push(FxParts.box(this.P, this.front, g, spec.x - t, spec.y - t, spec.x + t, spec.y + t, s * 5.4, s * 5.4 + 4.4, 0.9, 0.6));
    }
    const led   = FxParts.led(g, this.P, spec.x + t, spec.y, 13, "dot");
    const [c0, d0, c1, d1] = FxServicesFigure.CONSOLE;
    const start = [k === 0 ? c0 + 1 : c1 - 1, d0];
    const path  = [start, [start[0], spec.y + t + 5], [spec.x, spec.y + t + 5], [spec.x, spec.y + t]];
    const cable = HL.mk("path", { class: "nf", d: FxParts.line(this.P, path, 0) }, this.cables);
    const short = new FxPolyline(path);
    const cut   = short.at(1 - 4 / short.total);
    const plug  = HL.solid(this.cables);
    HL.put(plug, HL.prism(this.P, this.front, ...HL.rings(cut[0] - 1, cut[1] - 0.8, cut[0] + 1, cut[1] + 0.8, 0.4, 0.25), 0, 1.2));
    return { ...spec, k, g, parts, led, path, line: short, cable, plug, cut, up: true, lw: HL.tween(0), at: "", rest: this.P(spec.x, spec.y, 10) };
  }

  setState(state) {
    const name = FxServicesFigure.NAME;
    FxCheck.object(name, "state", state, ["osrm", "vroom", "reachable"]);
    const next = { osrm: FxCheck.boolean(name, "osrm", state.osrm), vroom: FxCheck.boolean(name, "vroom", state.vroom), reachable: FxCheck.boolean(name, "reachable", state.reachable) };
    const now  = performance.now();
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    for (const tower of this.towers) {
      tower.up    = state.reachable && state[tower.key];
      const cut   = tower.line.total - 4;
      const pts   = tower.up || !state.reachable ? tower.path : this._prefix(tower.line, cut);
      tower.cable.setAttribute("d", FxParts.line(this.P, pts, 0));
      FxParts.set(tower.cable, tower.up ? "nf" : "nf dash");
      FxParts.show(tower.plug.g, !tower.up && state.reachable);
      for (const part of tower.parts) part.sil.classList.toggle("lo", !tower.up);
      FxParts.set(tower.led, tower.up ? "dot" : "dot off");
    }
    this.console.sil.classList.toggle("dash", !state.reachable);
    const down = this.towers.find((tower) => !tower.up);
    this.rest  = down ? down.k : 0;
  }

  _prefix(line, length) {
    const out = [line.points[0]];
    for (let k = 1; k < line.points.length; k++) {
      if (line.lengths[k] >= length) {
        out.push(line.at(length / line.total));
        return out;
      }
      out.push(line.points[k]);
    }
    return out;
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const P2    = 2 * FxServicesFigure.PACKET;
    this.towers.forEach((tower, k) => {
      const dot = this.packets[k];
      if (!tower.up || still) {
        FxParts.show(dot, false);
        return;
      }
      const t   = (this.clock + k * FxServicesFigure.PACKET) % P2;
      const s   = t < FxServicesFigure.PACKET ? t / FxServicesFigure.PACKET : 2 - t / FxServicesFigure.PACKET;
      const [x, y] = tower.line.at(FxRank.smooth(s));
      HL.place(dot, this.P(x, y, 0.3));
      FxParts.show(dot, true);
    });
    const blink = !this.data.reachable && !still ? ((this.clock / FxServicesFigure.BLINK_S) % 2 < 1 ? "dot m" : "dot off") : this.data.reachable ? "dot" : "dot off";
    FxParts.set(this.consoleLed, blink);
    let moving  = false;
    for (const tower of this.towers) {
      if (!HL.tdone(tower.lw, now)) moving = true;
      const at = this.shift.of(0, 0, HL.tval(tower.lw, now));
      if (at === tower.at) continue;
      tower.at = at;
      tower.g.setAttribute("transform", at);
    }
    return still ? moving : true;
  }

  _move(p) {
    this._choose(HairlineFigureBase.nearest(this.towers.map((tower) => tower.rest[0]), p[0]));
  }

  _retarget(now) {
    const lit   = this.chosen === null ? this.rest : this.chosen;
    for (const tower of this.towers) HL.tset(tower.lw, tower.k !== lit ? 0 : this.chosen === null ? this.amount * 0.5 : this.amount, now, 0);
    const tower = this.towers[lit];
    this.light.set(tower.parts[tower.parts.length - 1].sil);
    this.loop.wake();
  }
}

class FxDispatchFigure extends HairlineFigureBase {
  static NAME     = "dispatch";
  static RANGE    = [0.3, 0.45, 0.65];
  static SCALE    = 5.5;
  static PLINTH   = [-23, -15, 25, 15];
  static WALL     = [-19, 17, -12, -9.5, 15];
  static DOOR     = [0, 5.6, 1.4, 10.5];
  static APRON    = [-19, 17, -9.5, -1, 1.4];
  static VAN_Y    = 3.5;
  static STACK    = [10.5, -5.5];
  static MAST     = [-15, 3];
  static VAN      = 1.35;
  static CHASE_S  = 0.5;
  static PUFF_S   = 1.6;
  static PUFF_L   = 2.2;
  static BREATHE  = 5;
  static LEAVE_MS = 900;
  static BACK_MS  = 900;
  static DEFAULT  = { armed: true, loads: 3, queued: false, fired: 0 };

  constructor({ stage, svg }, value) {
    super(svg, value, FxDispatchFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    this.fired = null;
    this.shot  = null;
    const [x0, y0, x1, y1] = FxDispatchFigure.PLINTH;
    this._camera(this._extent(), 166);
    this._plinth(x0, y0, x1, y1, 6, 3);
    const [w0, w1, wy0, wy1, wh] = FxDispatchFigure.WALL;
    FxParts.box(this.P, this.front, this.g, w0, wy0, w1, wy1, 0, wh, 0.8, 0.6);
    const [a0, a1, ay0, ay1, ah] = FxDispatchFigure.APRON;
    FxParts.box(this.P, this.front, this.g, a0, ay0, a1, ay1, 0, ah, 1, 0.7);
    const [dx, half, dz, dh] = FxDispatchFigure.DOOR;
    this.door   = new FxDoor(this.P, this.front, this.g, "y", wy1, dx, half, dz, dh);
    this.crate  = FxParts.shape(this.P, this.front, -1.5, -1.5, 1.5, 1.5, 0.5, 0.4);
    this.vanKit = new FxVan(this.P, this.front, FxDispatchFigure.VAN);
    this.depth  = new FxDepth(this.g);
    const [sx, sy] = FxDispatchFigure.STACK;
    const stack = this.depth.add(sx + sy);
    this.crates = Array.from({ length: 6 }, (_, i) => ({ i, solid: HL.solid(stack.group), x: sx + (i % 2 === 0 ? -1.7 : 1.7), z: ah + Math.floor(i / 2) * 2.9 }));
    this._buildMast();
    const waitItem = this.depth.add(dx + FxDispatchFigure.VAN_Y + 11);
    this.waiting = this.vanKit.make(waitItem.group);
    this.vanKit.dash(this.waiting, true);
    this.vanItem = this.depth.add(dx + FxDispatchFigure.VAN_Y);
    this.van    = this.vanKit.make(this.vanItem.group);
    this.puffs  = [0, 1].map(() => ({ el: HL.mk("circle", { r: 0, class: "dot", display: "none" }, this.g), born: -1e9, at: [0, 0, 0] }));
    this.puffNext = 0.4;
    this._apply(FxDispatchFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxDispatchFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [-19, -12, FxDispatchFigure.WALL[4] + 1], [17, -12, FxDispatchFigure.WALL[4] + 1], [...FxDispatchFigure.MAST, 15]];
  }

  _buildMast() {
    const [x, y] = FxDispatchFigure.MAST;
    const item   = this.depth.add(x + y);
    FxParts.box(this.P, this.front, item.group, x - 0.8, y - 0.8, x + 0.8, y + 0.8, 0, 13.5, 0.4, 0.25);
    FxParts.box(this.P, this.front, item.group, x - 1.4, y - 1.4, x + 1.4, y + 1.4, 0, 1, 0.5, 0.3);
    this.lamps   = [5.5, 8.8, 12.1].map((z) => FxParts.led(item.group, this.P, x + 0.8, y + 0.8, z, "dot off"));
  }

  setState(state) {
    const name = FxDispatchFigure.NAME;
    FxCheck.object(name, "state", state, ["armed", "loads", "queued", "fired"]);
    const next = { armed: FxCheck.boolean(name, "armed", state.armed), loads: FxCheck.integer(name, "loads", state.loads, 0, 6), queued: FxCheck.boolean(name, "queued", state.queued), fired: FxCheck.integer(name, "fired", state.fired, 0, Infinity) };
    const now  = performance.now();
    if (this.fired !== null && next.fired > this.fired && !HL.reducedMotion()) this.shot = { t0: now };
    this.fired = next.fired;
    this._apply(next);
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    this.crates.forEach((crate) => HL.put(crate.solid, crate.i < state.loads ? this.crate.path(crate.x, FxDispatchFigure.STACK[1], crate.z, crate.z + 2.6) : FxPrism.EMPTY));
    this.vanKit.pose(this.waiting, state.queued ? "y-" : "", FxDispatchFigure.DOOR[0], FxDispatchFigure.VAN_Y + 11, 0);
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const armed = this.data.armed;
    const swell = still ? 0 : 0.08 * (0.5 - 0.5 * Math.cos((2 * Math.PI * this.clock) / FxDispatchFigure.BREATHE));
    this.door.draw(armed ? 0.7 : swell);
    const step  = armed && !still ? Math.floor(this.clock / FxDispatchFigure.CHASE_S) % 3 : -1;
    this.lamps.forEach((lamp, k) => FxParts.set(lamp, !armed ? "dot off" : still ? "dot m" : k === step ? "dot" : "dot off"));
    let x       = FxDispatchFigure.DOOR[0];
    let y       = FxDispatchFigure.VAN_Y;
    let heading = "y-";
    let grow    = 1;
    let busy    = false;
    if (this.shot) {
      const age = now - this.shot.t0;
      if (age < FxDispatchFigure.LEAVE_MS) {
        const s = FxRank.smooth(age / FxDispatchFigure.LEAVE_MS);
        heading = "x+";
        x      += 30 * s;
        grow    = 1 - s;
      } else if (age < FxDispatchFigure.LEAVE_MS + FxDispatchFigure.BACK_MS) {
        const s = FxRank.smooth((age - FxDispatchFigure.LEAVE_MS) / FxDispatchFigure.BACK_MS);
        y      += 9 * (1 - s);
        grow    = s;
      } else this.shot = null;
      busy      = true;
    }
    this.vanKit.pose(this.van, heading, x, y, 0, grow);
    this.depth.move(this.vanItem, x + y);
    if (armed && !still && !this.shot) this._puff(x, y);
    else for (const puff of this.puffs) FxParts.show(puff.el, false);
    this.light.set(this.chosen === null ? this.van.body.sil : this.chosen);
    this.depth.sort();
    return still ? busy : true;
  }

  _puff(x, y) {
    if (this.clock >= this.puffNext) {
      this.puffNext += FxDispatchFigure.PUFF_S;
      const puff     = this.puffs[0].born <= this.puffs[1].born ? this.puffs[0] : this.puffs[1];
      puff.born      = this.clock;
      puff.at        = [x + 1.6, y + 4.6 * FxDispatchFigure.VAN, 1.2];
    }
    for (const puff of this.puffs) {
      const a = (this.clock - puff.born) / FxDispatchFigure.PUFF_L;
      if (a < 0 || a >= 1) {
        FxParts.show(puff.el, false);
        continue;
      }
      FxParts.show(puff.el, true);
      const [px, py, pz] = puff.at;
      HL.place(puff.el, this.P(px + 0.8 * a, py + 2.4 * a, pz + 4 * a));
      puff.el.setAttribute("r", String(HL.r2((0.4 + 0.55 * a) * this.C.S)));
      puff.el.setAttribute("class", a < 0.33 ? "dot" : a < 0.66 ? "dot m" : "dot off");
    }
  }

  _move(p) {
    const items = [{ el: this.van.body.sil, at: this.P(FxDispatchFigure.DOOR[0], FxDispatchFigure.VAN_Y, 3) }];
    this.crates.forEach((crate) => {
      if (crate.i < this.data.loads) items.push({ el: crate.solid.sil, at: this.P(crate.x, FxDispatchFigure.STACK[1], crate.z + 1.3) });
    });
    let best = null;
    let gap  = Infinity;
    for (const item of items) {
      const d = Math.hypot(item.at[0] - p[0], item.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item.el;
    }
    this._choose(gap < 70 ? best : null);
  }

  _retarget() {
    this.light.set(this.chosen === null ? this.van.body.sil : this.chosen);
    this.loop.wake();
  }
}

class FxBreakdownFigure extends HairlineFigureBase {
  static NAME    = "breakdown";
  static RANGE   = [1, 2, 3.5];
  static SCALE   = 6.3;
  static PLINTH  = [-24, -14, 24, 14];
  static ROAD    = [-22, 22, -5, 5];
  static VAN_AT  = [7, 3.2];
  static VAN     = 1.45;
  static TRI     = [-3, -11];
  static BLINK_S = 0.6;
  static KINDS   = ["backend", "missing", "failed"];
  static DEFAULT = { kind: "failed" };

  constructor({ stage, svg }, value) {
    super(svg, value, FxBreakdownFigure.SCALE);
    this.light = new FxLight();
    this.clock = 0;
    const [x0, y0, x1, y1] = FxBreakdownFigure.PLINTH;
    this._camera(this._extent(), 168);
    this._plinth(x0, y0, x1, y1, 6, 3);
    this.ground = HL.mk("g", {}, this.g);
    this.vanKit = new FxVan(this.P, this.front, FxBreakdownFigure.VAN);
    this.triangles = FxBreakdownFigure.TRI.map((x) => HL.mk("path", { class: "nf", d: this._triangle(x, 5.4) }, this.g));
    this.van    = this.vanKit.make(this.g);
    const [vx, vy] = FxBreakdownFigure.VAN_AT;
    this.vanKit.pose(this.van, "x+", vx, vy, 0.4);
    const k     = FxBreakdownFigure.VAN;
    this.bonnet = HL.solid(this.g);
    this._drawBonnet(vx + 3.2 * k, vy, k);
    this.hazards = [[-3.2, -1.5], [-3.2, 1.5], [3.2, -1.4], [3.2, 1.4]].map(([dx, dy]) => FxParts.led(this.g, this.P, vx + dx * k, vy + dy * k, 0.4 + (dx > 0 ? 3.2 : 4) * k, "dot"));
    this.rimFlat = HL.mk("ellipse", { class: "nf", rx: 0, ry: 0 }, this.g);
    this._apply(FxBreakdownFigure.DEFAULT);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1] = FxBreakdownFigure.PLINTH;
    return [[x0, y0, -3], [x1, y0, -3], [x0, y1, -3], [x1, y1, -3], [7, 3, 9], [-14, -9, 10]];
  }

  _triangle(x, y) {
    const a = this.P(x, y, 0.2);
    const b = this.P(x, y + 4.4, 0.2);
    const c = this.P(x, y + 2.2, 4);
    return HL.seg(a, b) + HL.seg(b, c) + HL.seg(c, a);
  }

  _drawBonnet(x, y, k) {
    const a     = HL.rad(35);
    const len   = 2.2 * k;
    const z0    = (FxVan.CAB_Z + FxVan.CAB[3]) * k;
    const tip   = [x + len * Math.cos(a), y, z0 + len * Math.sin(a)];
    const half  = 1.3 * k;
    const quad  = [this.P(x, y - half, z0), this.P(tip[0], y - half, tip[2]), this.P(tip[0], y + half, tip[2]), this.P(x, y + half, z0)];
    HL.put(this.bonnet, { sil: HL.poly(quad), crease: "" });
  }

  setState(state) {
    const name = FxBreakdownFigure.NAME;
    FxCheck.object(name, "state", state, ["kind"]);
    if (!FxBreakdownFigure.KINDS.includes(state.kind)) throw new Error(`${name}: kind must be one of ${FxBreakdownFigure.KINDS.join(", ")}, got ${state.kind}`);
    const now = performance.now();
    this._apply({ kind: state.kind });
    this._retarget(now);
    this._tick(0, now);
  }

  _apply(state) {
    this.data = state;
    this.ground.replaceChildren();
    const [r0, r1, w0, w1] = FxBreakdownFigure.ROAD;
    const end = state.kind === "backend" ? 12 : r1;
    FxParts.box(this.P, this.front, this.ground, r0, w0, end, w1, 0, 0.4, 1, 0.6);
    if (state.kind === "backend") HL.mk("path", { class: "nf dash", d: FxParts.flat(this.P, [[end, w0], [r1, w0], [r1, w1], [end, w1]], 0.4) }, this.ground);
    if (state.kind === "missing") {
      FxParts.box(this.P, this.front, this.ground, 15.5, -9.5, 16.5, -8.5, 0, 9, 0.3, 0.2);
      HL.mk("path", { class: "nf dash", d: HL.poly([this.P(16, -12.5, 9.2), this.P(16, -5.5, 9.2), this.P(16, -5.5, 13.4), this.P(16, -12.5, 13.4)]) }, this.ground);
    }
    const [vx, vy] = FxBreakdownFigure.VAN_AT;
    if (state.kind === "failed") {
      const c = this.P(vx - 1, vy + 4.6 * FxBreakdownFigure.VAN, 0.2);
      HL.place(this.rimFlat, c);
      this.rimFlat.setAttribute("rx", String(HL.r2(0.9 * FxBreakdownFigure.VAN * this.C.S)));
      this.rimFlat.setAttribute("ry", String(HL.r2(0.9 * FxBreakdownFigure.VAN * this.C.S * this.C.k)));
    } else {
      this.rimFlat.setAttribute("rx", "0");
      this.rimFlat.setAttribute("ry", "0");
    }
  }

  _tick(dt, now) {
    const still = HL.reducedMotion();
    if (!still) this.clock += dt;
    const phase = still ? 0 : Math.floor(this.clock / FxBreakdownFigure.BLINK_S) % 2;
    this.hazards.forEach((led, k) => FxParts.set(led, still ? "dot m" : (k < 2) === (phase === 0) ? "dot" : "dot off"));
    this.light.set(this.chosen === null ? this.van.body.sil : this.chosen);
    return !still;
  }

  _move(p) {
    const [vx, vy] = FxBreakdownFigure.VAN_AT;
    const items = [{ el: this.van.body.sil, at: this.P(vx, vy, 3) }, ...FxBreakdownFigure.TRI.map((x, k) => ({ el: this.triangles[k], at: this.P(x, 7.7, 1.4) }))];
    let best = null;
    let gap  = Infinity;
    for (const item of items) {
      const d = Math.hypot(item.at[0] - p[0], item.at[1] - p[1]);
      if (d >= gap) continue;
      gap  = d;
      best = item.el;
    }
    this._choose(best);
  }

  _retarget() {
    this.light.set(this.chosen === null ? this.van.body.sil : this.chosen);
    this.loop.wake();
  }
}

class RoutingFigures {
  static entry(Figure) {
    return { mount: (els, value) => new Figure(els, value), range: Figure.RANGE };
  }

  static CATALOG = {
    fleet:      RoutingFigures.entry(FxFleetFigure),
    recording:  RoutingFigures.entry(FxRecordingFigure),
    heads:      RoutingFigures.entry(FxHeadsFigure),
    insert:     RoutingFigures.entry(FxInsertFigure),
    disruption: RoutingFigures.entry(FxDisruptionFigure),
    diverge:    RoutingFigures.entry(FxDivergeFigure),
    confusion:  RoutingFigures.entry(FxConfusionFigure),
    terms:      RoutingFigures.entry(FxTermsFigure),
    ranking:    RoutingFigures.entry(FxRankingFigure),
    balance:    RoutingFigures.entry(FxBalanceFigure),
    checkpoint: RoutingFigures.entry(FxCheckpointFigure),
    training:   RoutingFigures.entry(FxTrainingFigure),
    queue:      RoutingFigures.entry(FxQueueFigure),
    services:   RoutingFigures.entry(FxServicesFigure),
    dispatch:   RoutingFigures.entry(FxDispatchFigure),
    breakdown:  RoutingFigures.entry(FxBreakdownFigure),
  };

  static LABELS = {
    fleet:      "Vehicles lapping their routes around the depot, with the unassigned jobs waiting beside it.",
    recording:  "One film strip per agent lane, filling frame by frame as the episode records.",
    heads:      "The three policy heads as columns of probability, with the chosen operator, vehicle and job.",
    insert:     "A job moving between the unassigned tray and a vehicle route.",
    disruption: "A disruption arriving at the depot while the clock advances.",
    diverge:    "The model and the teacher leaving the same road and ending at different rewards.",
    confusion:  "Model operators against teacher operators as a three by three block of columns.",
    terms:      "Reward terms ranked on a podium.",
    ranking:    "Mean job rank on a staircase of priorities.",
    balance:    "Model reward weighed against the teacher.",
    checkpoint: "Checkpoints on a rack, lit when they load and carry a report.",
    training:   "A training curve as a row of slats.",
    queue:      "Jobs on a belt through the running gate to the finished stack.",
    services:   "The console wired to the OSRM and VROOM towers.",
    dispatch:   "A van at the depot door, waiting for the launch.",
    breakdown:  "A van stopped on the shoulder with its hazard lights on.",
    terminal:   "A terminal printing a log.",
    cabinet:    "A filing cabinet of run files.",
    riffle:     "A stack of figures.",
  };
}

window.RoutingFigures = RoutingFigures;
