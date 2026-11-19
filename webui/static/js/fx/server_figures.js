"use strict";

class GpuShelfFigure extends HairlineFigureBase {
  static RANGE     = [8, 14, 22];
  static THICK     = 4;
  static GAP       = 3.2;
  static DEPTH     = 34;
  static HEIGHT    = 26;
  static FAN       = 6;
  static BLADES    = 3;
  static DEVICES   = [["busy", 70], ["busy", 55], ["idle", 0], ["busy", 85], ["idle", 0], ["idle", 0]];
  static LIGHT     = { mine: "dot", others: "dot m", stale: "dot m", busy: "dot m", idle: "dot off" };
  static SPIN      = [0.5, 10];
  static REACH     = 1.6;
  static STAGGER   = 45;
  static REST      = 2;
  static REST_LIFT = 0.45;

  constructor({ stage, svg }, value) {
    super(svg, value, 3.3);
    this.devices = GpuShelfFigure.DEVICES.map(([holder, util]) => ({ holder, util }));
    this.rest    = GpuShelfFigure.REST;
    this._layout();
    this._start(stage, svg);
  }

  _scale(count) {
    const span = count * (GpuShelfFigure.THICK + GpuShelfFigure.GAP) - GpuShelfFigure.GAP + 12;
    return 3.3 * Math.min(1.4, (96 / (span + 44)) ** 0.8);
  }

  _layout() {
    this.g.replaceChildren();
    this.C      = HL.Cam(45, 0.5, this._scale(this.devices.length));
    this.length = this.devices.length * (GpuShelfFigure.THICK + GpuShelfFigure.GAP) - GpuShelfFigure.GAP;
    this._camera(this._extent(), 166);
    this._plinth(-6, -5, this.length + 6, GpuShelfFigure.DEPTH + 5, 5, 4);
    this._buildCards();
  }

  _extent() {
    const length = this.length;
    const top    = GpuShelfFigure.HEIGHT + GpuShelfFigure.RANGE[2];
    return [[-6, -5, -4], [length + 6, GpuShelfFigure.DEPTH + 5, -4], [length + 6, -5, -4], [-6, GpuShelfFigure.DEPTH + 5, -4], [0, 0, top], [length, GpuShelfFigure.DEPTH, top]];
  }

  _buildCards() {
    this.cards = this.devices.map((device, k) => {
      const x0            = k * (GpuShelfFigure.THICK + GpuShelfFigure.GAP);
      const [ring, inner] = HL.rings(x0, 0, x0 + GpuShelfFigure.THICK, GpuShelfFigure.DEPTH, 1.3, 0.6);
      const group         = HL.mk("g", {}, this.g);
      const el            = HL.solid(group);
      const fans          = HL.mk("path", { class: "nf lo" }, group);
      const led           = HL.flatDot(group, this.C, 0.9, GpuShelfFigure.LIGHT[device.holder]);
      const lit           = k === this.rest;
      el.sil.classList.toggle("hi", lit);
      return { k, x0, ring, inner, el, fans, led, phase: k * 0.7, spin: HL.spring(this._spin(device.util), { k: 6, c: 5, eps: 0.01 }), tw: HL.tween(lit ? this.amount * GpuShelfFigure.REST_LIFT : 0), drawn: NaN, restX: this.P(x0 + GpuShelfFigure.THICK / 2, GpuShelfFigure.DEPTH / 2, 0)[0] };
    });
  }

  _spin(util) {
    const [idle, full] = GpuShelfFigure.SPIN;
    return idle + (full - idle) * HL.clamp((util || 0) / 100, 0, 1);
  }

  _tick(dt, now) {
    this._advance(this.cards, now);
    for (const card of this.cards) {
      HL.stepS(card.spin, dt);
      card.phase = (card.phase + card.spin.x * dt) % (Math.PI * 2);
      this._drawFans(card, Number.isFinite(card.drawn) ? card.drawn : 0);
    }
    return true;
  }

  _draw(card, lift) {
    HL.put(card.el, HL.prism(this.P, this.front, card.ring, card.inner, lift, lift + GpuShelfFigure.HEIGHT));
    HL.place(card.led, this.P(card.x0 + GpuShelfFigure.THICK / 2, GpuShelfFigure.DEPTH - 3, lift + GpuShelfFigure.HEIGHT));
  }

  _drawFans(card, lift) {
    const face = card.x0 + GpuShelfFigure.THICK;
    const cz   = lift + GpuShelfFigure.HEIGHT * 0.55;
    const r    = GpuShelfFigure.FAN;
    const at   = (cy, a, radius) => this.P(face, cy + radius * Math.cos(a), cz + radius * Math.sin(a));
    let d      = "";
    for (const share of [0.3, 0.7]) {
      const cy = GpuShelfFigure.DEPTH * share;
      d       += HairlineGeometry.faceCircle(this.P, face, cy, cz, r);
      for (let j = 0; j < GpuShelfFigure.BLADES; j++) {
        const a = card.phase + (j * Math.PI * 2) / GpuShelfFigure.BLADES;
        d      += HL.seg(at(cy, a, 1.2), at(cy, a + 0.5, r * 0.82));
      }
    }
    card.fans.setAttribute("d", d);
  }

  _move(p) {
    this._choose(HairlineFigureBase.nearest(this.cards.map((card) => card.restX), p[0]));
  }

  _restCard() {
    const rank = (device) => (device.holder === "mine" ? 1000 : device.holder === "idle" ? 0 : 500) + (device.util || 0);
    let best = 0;
    this.devices.forEach((device, k) => { if (rank(device) > rank(this.devices[best])) best = k; });
    return best;
  }

  setState(state) {
    const devices = (state && state.devices) || [];
    if (!devices.length) throw new Error("gpus figure: state.devices must list at least one device");
    const recount = devices.length !== this.devices.length;
    this.devices  = devices.map((device) => ({ holder: GpuShelfFigure.LIGHT[device.holder] ? device.holder : "idle", util: device.util || 0 }));
    this.rest     = this._restCard();
    if (recount) {
      this.chosen = null;
      this._layout();
    }
    this.cards.forEach((card, k) => {
      card.led.setAttribute("class", GpuShelfFigure.LIGHT[this.devices[k].holder]);
      card.spin.t = this._spin(this.devices[k].util);
    });
    this._retarget(performance.now());
  }

  _retarget(now) {
    const anchor = this.chosen === null ? this.rest : this.chosen;
    const amount = this.cards.length === 1 ? this.amount * GpuShelfFigure.REST_LIFT : this.amount;
    for (const card of this.cards) {
      const distance = Math.abs(card.k - anchor);
      const reach    = Math.max(0, 1 - distance / GpuShelfFigure.REACH) ** 2;
      const target   = this.chosen === null ? (card.k === this.rest ? amount * GpuShelfFigure.REST_LIFT : 0) : amount * reach;
      HL.tset(card.tw, target, now, distance * GpuShelfFigure.STAGGER);
      card.el.sil.classList.toggle("hi", card.k === anchor);
    }
    this.loop.wake();
  }
}


class SchedulerFigure extends HairlineFigureBase {
  static RANGE   = [3, 6, 10];
  static RACK    = [22, 30, 34];
  static STEPS   = 5;
  static STEP_W  = 11;
  static STEP_Y  = [3, 27];
  static TOP     = 28;
  static DROP    = 5.5;
  static FLOOR   = 82.5;
  static CRATE   = 3.6;
  static CRATE_Z = 7;
  static HOP     = 7;
  static SOCKETS = [[6, 8], [16, 8], [6, 22], [16, 22]];
  static BLADES  = [7, 14, 21, 28];
  static REST    = 0.35;

  constructor({ stage, svg }, value) {
    super(svg, value, 3.3);
    this.crates = new Map();
    this.lead   = null;
    this._camera(this._extent(), 168);
    this._plinth(-6, -5, SchedulerFigure.FLOOR + 6.5, SchedulerFigure.RACK[1] + 5, 6, 4);
    this._buildRack();
    this._buildSteps();
    this.ring = HL.rings(-SchedulerFigure.CRATE, -SchedulerFigure.CRATE, SchedulerFigure.CRATE, SchedulerFigure.CRATE, 1.5, 0.7);
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [, depth, height] = SchedulerFigure.RACK;
    const top               = height + SchedulerFigure.CRATE_Z + SchedulerFigure.RANGE[2];
    return [[-6, -5, -4], [SchedulerFigure.FLOOR + 6.5, depth + 5, -4], [SchedulerFigure.FLOOR + 6.5, -5, -4], [-6, depth + 5, -4], [0, 0, top], [22, depth, top]];
  }

  _buildRack() {
    const [width, depth, height] = SchedulerFigure.RACK;
    const [ring, inner]          = HL.rings(0, 0, width, depth, 3, 1.4);
    this.rack                    = HL.solid(this.g);
    HL.put(this.rack, HL.prism(this.P, this.front, ring, inner, 0, height));
    const sockets = HL.mk("path", { class: "nf lo" }, this.g);
    sockets.setAttribute("d", SchedulerFigure.SOCKETS.map(([x, y]) => HL.poly(HL.ringAt(this.P, HL.rrect(x - 4.4, y - 4.4, x + 4.4, y + 4.4, 1.6, 4), height))).join(""));
    const blades = HL.mk("path", { class: "nf lo" }, this.g);
    blades.setAttribute("d", SchedulerFigure.BLADES.map((z) => HL.seg(this.P(2.5, depth, z), this.P(width - 2.5, depth, z))).join(""));
    this.leds = SchedulerFigure.BLADES.map((z) => {
      const led = HL.mk("circle", { r: 1.1, class: "dot off" }, this.g);
      HL.place(led, this.P(width - 5, depth, z - 3.2));
      return led;
    });
    this.rackLayer = HL.mk("g", {}, this.g);
    this.rackX     = this.P(width / 2, depth / 2, height)[0];
  }

  _buildSteps() {
    const [y0, y1] = SchedulerFigure.STEP_Y;
    this.steps     = Array.from({ length: SchedulerFigure.STEPS }, (_, i) => {
      const x0            = SchedulerFigure.RACK[0] + i * SchedulerFigure.STEP_W;
      const top           = SchedulerFigure.TOP - i * SchedulerFigure.DROP;
      const [ring, inner] = HL.rings(x0, y0, x0 + SchedulerFigure.STEP_W, y1, 1.6, 0.8);
      const layer         = HL.mk("g", {}, this.g);
      const el            = HL.solid(layer);
      HL.put(el, HL.prism(this.P, this.front, ring, inner, 0, top));
      const crates        = HL.mk("g", {}, layer);
      return { i, el, crates, top, centre: [x0 + SchedulerFigure.STEP_W / 2, (y0 + y1) / 2], screenX: this.P(x0 + SchedulerFigure.STEP_W / 2, (y0 + y1) / 2, top)[0] };
    });
    this.floorLayer = HL.mk("g", {}, this.g);
    this.floorX     = this.P(SchedulerFigure.FLOOR, 15, 0)[0];
  }

  _pose(slot) {
    if (slot.kind === "socket") {
      const [x, y] = SchedulerFigure.SOCKETS[slot.index];
      return [x, y, SchedulerFigure.RACK[2]];
    }
    if (slot.kind === "step") return [...this.steps[slot.index].centre, this.steps[slot.index].top];
    return [SchedulerFigure.FLOOR, 15, 0];
  }

  _rank(slot) {
    if (slot.kind === "socket") return -1;
    if (slot.kind === "step") return slot.index;
    return SchedulerFigure.STEPS;
  }

  _layer(rank) {
    if (rank < 0) return this.rackLayer;
    if (rank >= SchedulerFigure.STEPS) return this.floorLayer;
    return this.steps[rank].crates;
  }

  _slots(jobs) {
    const slots   = new Map();
    const running = jobs.filter((job) => job.running).slice(0, SchedulerFigure.SOCKETS.length);
    running.forEach((job, index) => slots.set(job.id, { kind: "socket", index }));

    const waiting = jobs.filter((job) => !job.running).map((job) => ({ id: job.id, share: job.position == null ? 0 : 1 - (job.position - 1) / Math.max(1, job.of - 1) }));
    waiting.sort((a, b) => b.share - a.share);
    let last = -1;
    for (const job of waiting) {
      const own  = Math.round((1 - HL.clamp(job.share, 0, 1)) * (SchedulerFigure.STEPS - 1));
      const step = Math.max(own, last + 1);
      if (step > SchedulerFigure.STEPS) break;
      last = step;
      slots.set(job.id, step === SchedulerFigure.STEPS ? { kind: "floor", index: 0 } : { kind: "step", index: step });
    }
    this.lead = waiting.length && slots.has(waiting[0].id) ? waiting[0].id : null;
    return slots;
  }

  setState(state) {
    const now   = performance.now();
    const slots = this._slots(state.jobs);
    for (const [id, crate] of this.crates) {
      if (slots.has(id) || crate.leaving) continue;
      crate.leaving = true;
      HL.tset(crate.grow, 0, now, 0);
    }
    for (const [id, slot] of slots) {
      const crate = this.crates.get(id) || this._crate(id, slot, now);
      if (crate.leaving) {
        crate.leaving = false;
        HL.tset(crate.grow, 1, now, 0);
      }
      this._send(crate, slot, now);
    }
    const running = state.jobs.filter((job) => job.running).length;
    this.leds.forEach((led, k) => led.setAttribute("class", k < running ? "dot" : "dot off"));
    this._order();
    this._retarget(now);
  }

  _crate(id, slot, now) {
    const group = HL.mk("g", {}, this._layer(this._rank(slot)));
    const pose  = this._pose(slot);
    const crate = { id, group, el: HL.solid(group), slot, rank: this._rank(slot), from: pose, to: pose, travel: HL.tween(1), grow: HL.tween(0), lift: HL.tween(0), leaving: false, drawn: "" };
    HL.tset(crate.grow, 1, now, 0);
    this.crates.set(id, crate);
    return crate;
  }

  _send(crate, slot, now) {
    const to = this._pose(slot);
    if (to.every((v, k) => v === crate.to[k])) return;
    crate.from   = this._at(crate, now);
    crate.to     = to;
    crate.slot   = slot;
    crate.travel = HL.tween(0);
    HL.tset(crate.travel, 1, now, 0);
    crate.rank   = Math.max(crate.rank, this._rank(slot));
  }

  _at(crate, now) {
    const t = HL.tval(crate.travel, now);
    return crate.from.map((v, k) => HL.lerp(v, crate.to[k], t));
  }

  _order() {
    const sockets = [...this.crates.values()].filter((crate) => crate.rank < 0).sort((a, b) => a.to[0] + a.to[1] - (b.to[0] + b.to[1]));
    for (const crate of sockets) this.rackLayer.appendChild(crate.group);
    for (const crate of this.crates.values()) if (crate.rank >= 0) this._layer(crate.rank).appendChild(crate.group);
  }

  _tick(dt, now) {
    let moving = false;
    for (const crate of [...this.crates.values()]) {
      const t    = HL.tval(crate.travel, now);
      const grow = HL.tval(crate.grow, now);
      const lift = HL.tval(crate.lift, now);
      if (!HL.tdone(crate.travel, now) || !HL.tdone(crate.grow, now) || !HL.tdone(crate.lift, now)) moving = true;
      if (HL.tdone(crate.travel, now) && crate.rank !== this._rank(crate.slot)) {
        crate.rank = this._rank(crate.slot);
        this._order();
      }
      if (crate.leaving && HL.tdone(crate.grow, now)) {
        crate.group.remove();
        this.crates.delete(crate.id);
        continue;
      }
      const key = `${t}|${grow}|${lift}`;
      if (key === crate.drawn) continue;
      crate.drawn = key;
      this._draw(crate, t, grow, lift);
    }
    return moving;
  }

  _draw(crate, t, grow, lift) {
    const [x, y, z]     = crate.from.map((v, k) => HL.lerp(v, crate.to[k], t));
    const hop           = crate.from === crate.to ? 0 : SchedulerFigure.HOP * Math.sin(Math.PI * t);
    const base          = z + hop + lift;
    const height        = SchedulerFigure.CRATE_Z * grow;
    const [ring, inner] = this.ring;
    HL.put(crate.el, height < 0.2 ? { sil: "", crease: "" } : HL.prism(this._shifted(x, y, 0), this.front, ring, inner, base, base + height));
  }

  _move(p) {
    const targets = [this.rackX, ...this.steps.map((step) => step.screenX), this.floorX];
    this._choose(HairlineFigureBase.nearest(targets, p[0]) - 1);
  }

  _onTarget(crate, target) {
    if (target < 0) return crate.slot.kind === "socket";
    if (target >= SchedulerFigure.STEPS) return crate.slot.kind === "floor";
    return crate.slot.kind === "step" && crate.slot.index === target;
  }

  _retarget(now) {
    let lit = null;
    if (this.chosen === null) {
      lit = this.lead !== null ? this.crates.get(this.lead).el : this.rack;
      for (const crate of this.crates.values()) HL.tset(crate.lift, crate.id === this.lead ? this.amount * SchedulerFigure.REST : 0, now, 0);
    } else {
      const picked = [...this.crates.values()].filter((crate) => !crate.leaving && this._onTarget(crate, this.chosen));
      for (const crate of this.crates.values()) HL.tset(crate.lift, picked.includes(crate) ? this.amount * (this.chosen < 0 ? 0.6 : 1) : 0, now, 0);
      if (this.chosen < 0) lit = this.rack;
      else if (picked.length) lit = picked[0].el;
      else if (this.chosen < SchedulerFigure.STEPS) lit = this.steps[this.chosen].el;
    }
    this.rack.sil.classList.toggle("hi", lit === this.rack);
    for (const step of this.steps) step.el.sil.classList.toggle("hi", lit === step.el);
    for (const crate of this.crates.values()) crate.el.sil.classList.toggle("hi", lit === crate.el);
    this.loop.wake();
  }
}


class WatchdogFigure extends HairlineFigureBase {
  static RANGE    = [0.5, 0.9, 1.4];
  static RING     = 29;
  static PLINTH   = 40;
  static PUCK     = 5.2;
  static RISE     = [1.5, 5, 1.3];
  static SPOT     = 6.5;
  static POLE     = 36;
  static PIVOT    = 2.6;
  static BODY     = [6.4, 5.6, 5, 8];
  static LENS     = [2.2, 2.2];
  static DROOP    = 1.25;
  static LINGER   = [0.3, 0.45];
  static CATCH    = 0.05;
  static HOLD_MS  = 2600;
  static SAMPLE   = [0.42, 0.55, 0.62, 0.58];
  static LIMITS   = [0.9, 0.8, 0.8, 0.9];
  static ANGLES   = [-0.75 * Math.PI, -0.25 * Math.PI, 0.25 * Math.PI, 0.75 * Math.PI];
  static LEVEL_MS = 700;

  constructor({ stage, svg }, value) {
    super(svg, value, 3.5);
    this.armed  = true;
    this.alerts = WatchdogFigure.SAMPLE.map(() => false);
    this.seen   = null;
    this.tip    = [0, 0];
    this.hold   = null;
    this.passed = null;
    this._camera(this._extent(), 178);
    this._buildPlinth();
    this._buildTargets();
    this._buildCamera();
    this.turn  = HL.spring(WatchdogFigure.ANGLES[0] - 0.6, { k: 40, c: 12, eps: 0.001 });
    this.pitch = HL.spring(this._aimPitch(), { k: 40, c: 12, eps: 0.001 });
    this._start(stage, svg);
    this._retarget();
  }

  _extent() {
    const r = WatchdogFigure.PLINTH;
    return WatchdogFigure.ANGLES.map((a) => [r * Math.cos(a), r * Math.sin(a), -3]).concat([[0, 0, WatchdogFigure.POLE + WatchdogFigure.PIVOT + 5]]);
  }

  _buildPlinth() {
    const ring  = WatchdogFigure.RING;
    const patrol = Array.from({ length: 97 }, (_, k) => this.P(ring * Math.cos((2 * Math.PI * k) / 96), ring * Math.sin((2 * Math.PI * k) / 96), 0));
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, HL.circ(WatchdogFigure.PLINTH, 96), HL.circ(WatchdogFigure.PLINTH - 2.2, 96), -3, 0));
    HL.mk("path", { d: HL.open(patrol), class: "dash nf" }, this.g);
    this.spot = HL.mk("path", { class: "nf" }, this.g);
  }

  _buildTargets() {
    this.ring    = HL.circ(WatchdogFigure.PUCK, 48);
    this.inner   = HL.circ(WatchdogFigure.PUCK - 1, 48);
    this.targets = WatchdogFigure.ANGLES.map((a, k) => {
      const x = WatchdogFigure.RING * Math.cos(a);
      const y = WatchdogFigure.RING * Math.sin(a);
      return { k, a, x, y, P: this._shifted(x, y, 0), level: HL.tween(WatchdogFigure.SAMPLE[k] / WatchdogFigure.LIMITS[k], WatchdogFigure.LEVEL_MS), drawn: NaN };
    });
    for (const target of [...this.targets].sort((p, q) => p.x + p.y - (q.x + q.y))) {
      const group  = HL.mk("g", {}, this.g);
      target.puck  = HL.solid(group);
      target.rings = HL.mk("path", { class: "nf lo" }, group);
      target.dot   = HL.flatDot(group, this.C, 1, "dot off");
    }
  }

  _buildCamera() {
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, HL.circ(3.6, 40), HL.circ(2.6, 40), 0, 1.2));
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, HL.rrect(-1, -1, 1, 1, 0.4, 2), null, 1.2, WatchdogFigure.POLE));
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, HL.circ(1.8, 24), null, WatchdogFigure.POLE - 0.4, WatchdogFigure.POLE + 0.8));

    this.lensBack  = HL.mk("g", {}, this.g);
    this.body      = HL.solid(this.g);
    this.lensFront = HL.mk("g", {}, this.g);
    this.lensGroup = HL.mk("g", {}, this.lensFront);
    this.cone      = HL.mk("path", { class: "dash nf" }, this.lensGroup);
    this.lens      = HL.solid(this.lensGroup);
    this.eye       = HL.flatDot(this.lensGroup, this.C, 0.9, "dot");
  }

  _aimPitch() {
    return Math.atan2(WatchdogFigure.POLE + WatchdogFigure.PIVOT, WatchdogFigure.RING);
  }

  _tick(dt, now) {
    for (const target of this.targets) {
      const level = HL.tval(target.level, now);
      if (level === target.drawn) continue;
      target.drawn = level;
      this._drawTarget(target, level);
    }
    this._steer(dt, now);
    HL.stepS(this.pitch, dt);
    this._drawCamera(this.turn.x, this.pitch.x);
    this._drawBeam();
    this._spotTargets();
    return true;
  }

  _drawTarget(target, level) {
    const [base, rise, cap] = WatchdogFigure.RISE;
    const h                 = base + rise * Math.min(cap, level);
    const rings             = [3.2, 1.6].map((radius) => HL.poly(HL.circ(radius, 32).map((q) => target.P(q.u, q.v, h)))).join("");
    HL.put(target.puck, HL.prism(target.P, this.front, this.ring, this.inner, 0, h));
    target.rings.setAttribute("d", rings);
    HL.place(target.dot, target.P(0, 0, h));
  }

  _steer(dt, now) {
    this._catch(now);
    const goal = this._goal();
    if (goal !== null) {
      this.turn.t = this.turn.x + this._wrap(goal - this.turn.x);
      HL.stepS(this.turn, dt);
      return;
    }

    const [floor, width] = WatchdogFigure.LINGER;
    const near           = Math.min(...this.targets.map((target) => Math.abs(this._wrap(target.a - this.turn.x))));
    const speed          = this.amount * (floor + (1 - floor) * HL.clamp(near / width, 0, 1));
    this.turn.x         += speed * dt;
    this.turn.t          = this.turn.x;
    this.turn.v          = speed;
  }

  _catch(now) {
    if (this.hold !== null && (now >= this.hold.until || !this.alerts[this.hold.k])) {
      this.passed = this.hold.k;
      this.hold   = null;
    }
    if (this.passed !== null && Math.abs(this._wrap(this.targets[this.passed].a - this.turn.x)) > WatchdogFigure.LINGER[1]) this.passed = null;
    if (this.hold !== null || this.chosen !== null || !this.armed) return;

    const caught = this.targets.find((target) => this.alerts[target.k] && target.k !== this.passed && Math.abs(this._wrap(target.a - this.turn.x)) < WatchdogFigure.CATCH);
    if (caught) this.hold = { k: caught.k, until: now + WatchdogFigure.HOLD_MS };
  }

  _goal() {
    if (!this.armed) return this.turn.x;
    if (this.chosen !== null) return this.targets[this.chosen].a;
    if (this.hold !== null) return this.targets[this.hold.k].a;
    return null;
  }

  _wrap(angle) {
    return Math.atan2(Math.sin(angle), Math.cos(angle));
  }

  _drawCamera(yaw, pitch) {
    const at                  = this._cameraAt(yaw, pitch);
    const [w, h, back, front] = WatchdogFigure.BODY;
    const [radius, length]    = WatchdogFigure.LENS;
    const home                = Math.cos(yaw) + Math.sin(yaw) < 0 ? this.lensBack : this.lensFront;
    HL.put(this.body, HL.prism(at, this.front, HL.rrect(-w / 2, -h / 2, w / 2, h / 2, 1.1, 3), null, -back, front));
    HL.put(this.lens, HL.prism(at, this.front, HL.circ(radius, 24), null, front, front + length));
    this.tip = at(0, 0, front + length);
    HL.place(this.eye, this.tip);
    if (this.lensGroup.parentNode !== home) home.appendChild(this.lensGroup);
  }

  _cameraAt(yaw, pitch) {
    const forward = [Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch)];
    const right   = [-Math.sin(yaw), Math.cos(yaw), 0];
    const up      = [Math.cos(yaw) * Math.sin(pitch), Math.sin(yaw) * Math.sin(pitch), Math.cos(pitch)];
    const z0      = WatchdogFigure.POLE + WatchdogFigure.PIVOT;
    return (u, v, w) => this.P(u * right[0] + v * up[0] + w * forward[0], u * right[1] + v * up[1] + w * forward[1], z0 + v * up[2] + w * forward[2]);
  }

  _drawBeam() {
    if (!this.armed) {
      this.spot.setAttribute("d", "");
      this.cone.setAttribute("d", "");
      return;
    }

    const cx     = WatchdogFigure.RING * Math.cos(this.turn.x);
    const cy     = WatchdogFigure.RING * Math.sin(this.turn.x);
    const spot   = Array.from({ length: 33 }, (_, k) => this.P(cx + WatchdogFigure.SPOT * Math.cos((2 * Math.PI * k) / 32), cy + WatchdogFigure.SPOT * Math.sin((2 * Math.PI * k) / 32), 0));
    const angles = spot.map((p) => Math.atan2(p[1] - this.tip[1], p[0] - this.tip[0]));
    const left   = spot[angles.indexOf(Math.min(...angles))];
    const right  = spot[angles.indexOf(Math.max(...angles))];
    const locked = this.chosen === null && this.hold !== null;
    this.spot.setAttribute("d", HL.open(spot));
    this.cone.setAttribute("d", HL.seg(this.tip, left) + HL.seg(this.tip, right));
    this.spot.setAttribute("class", locked ? "nf hi" : "nf");
    this.cone.setAttribute("class", locked ? "nf hi" : "dash nf");
  }

  _spotTargets() {
    const reach = (WatchdogFigure.SPOT + 0.6 * WatchdogFigure.PUCK) / WatchdogFigure.RING;
    const hit   = this.targets.find((target) => Math.abs(this._wrap(target.a - this.turn.x)) < reach);
    const seen  = this.armed && hit ? hit.k : null;
    if (seen === this.seen) return;
    this.seen = seen;
    this._paint();
  }

  _paint() {
    for (const target of this.targets) {
      target.puck.sil.classList.toggle("hi", target.k === this.seen);
      target.dot.setAttribute("class", this.alerts[target.k] ? "dot" : target.k === this.seen ? "dot m" : "dot off");
    }
    this.eye.setAttribute("class", this.armed ? "dot" : "dot off");
  }

  _move(p) {
    const [x, y] = HL.unproj(this.C, p[0], p[1], 0);
    if (Math.hypot(x, y) < 8) return;
    const angle = Math.atan2(y, x);
    this._choose(HairlineFigureBase.nearest(this.targets.map((target) => Math.abs(this._wrap(target.a - angle))), 0));
  }

  setState(state) {
    if (!Array.isArray(state.gauges) || state.gauges.length !== this.targets.length) throw new Error(`watchdog figure: state.gauges must hold ${this.targets.length} gauges`);
    const now   = performance.now();
    this.armed  = !!state.armed;
    this.alerts = state.gauges.map((reading) => !!reading.alert);
    state.gauges.forEach((reading, k) => HL.tset(this.targets[k].level, Math.max(0, reading.level) / reading.limit, now, 0));
    this._retarget();
  }

  _retarget() {
    this.pitch.t = this.armed ? this._aimPitch() : WatchdogFigure.DROOP;
    this._paint();
    this.loop.wake();
  }
}


class NukeFigure extends HairlineFigureBase {
  static RANGE      = [3, 5, 8];
  static PLINTH     = [-17, -11, 12, 11];
  static BOX        = [-10, 4, 4, 3.5, 6];
  static ROD        = [0.9, 8];
  static HANDLE     = [4, 0.9, 1.5];
  static DRUM       = [3, -2, 6.5, 12];
  static BANDS      = [0.42, 0.56];
  static TREFOIL    = [1.1, 2, 4.8, 30];
  static CLOUD      = [2.6, 8.5, 11, 16];
  static WAVE       = [6.5, 24];
  static WIRE       = [[-6, 5], [-1.5, 8.5], [3, 4.6]];
  static PRESS_MS   = 260;
  static PULSE_MS   = 450;
  static WAVE_MS    = 1000;
  static HOLD_MS    = 2200;
  static BLINK_MS   = 1600;
  static ids        = 0;

  constructor({ stage, svg }, value) {
    super(svg, value, 7.4);
    this.firing   = false;
    this.fired    = null;
    this.released = null;
    this.down     = false;
    this.wave     = null;
    this.cloud    = HL.tween(0, 700);
    this.plunger  = HL.spring(1, { k: 50, c: 11, eps: 0.002 });
    const pts = this._extent();
    this._camera(pts, 320 - this._height(pts) / 2 - 6);
    this._buildPlinth(svg);
    this._buildWire();
    this._buildDrum();
    this._buildDetonator();
    this._buildCloud();
    this.parts = [this.det, this.drum];
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [r0, d0, r1, d1]  = NukeFigure.PLINTH;
    const [br, bd, , , bh]  = NukeFigure.BOX;
    const [dr, dd, R, H]    = NukeFigure.DRUM;
    const lift              = NukeFigure.RANGE[2];
    const top               = bh + 1.5 + NukeFigure.ROD[1] + NukeFigure.HANDLE[2] + lift;
    const cap               = NukeFigure.CLOUD[1];
    const crown             = H + NukeFigure.CLOUD[3];
    const floor             = [[r0, d0], [r1, d0], [r0, d1], [r1, d1]].map(([r, d]) => [...this._onScreen(r, d), -3]);
    const cloud             = [[dr - cap, dd], [dr + cap, dd], [dr, dd - cap], [dr, dd + cap]].map(([r, d]) => [...this._onScreen(r, d), crown]);
    return floor.concat([[...this._onScreen(br, bd), top], [...this._onScreen(dr, dd), H + lift]], cloud);
  }

  _buildPlinth(svg) {
    const [r0, d0, r1, d1] = NukeFigure.PLINTH;
    const ring             = this._screenBox(r0, d0, r1, d1, 6);
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ring, this._screenBox(r0 + 2.2, d0 + 2.2, r1 - 2.2, d1 - 2.2, 4), -3, 0));
    const id   = `nk-top-${++NukeFigure.ids}`;
    const clip = HL.mk("clipPath", { id }, HL.mk("defs", {}, svg));
    HL.mk("path", { d: HL.poly(HL.ringAt(this.P, ring, 0)) }, clip);
    this.clip = `url(#${id})`;
  }

  _buildWire() {
    const [a, m, b] = NukeFigure.WIRE;
    const pts       = Array.from({ length: 17 }, (_, k) => {
      const f = k / 16;
      const w = 1 - f;
      return [w * w * a[0] + 2 * w * f * m[0] + f * f * b[0], w * w * a[1] + 2 * w * f * m[1] + f * f * b[1]];
    });
    const cum = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
    this.wire = { pts, cum };
    HL.mk("path", { d: HL.open(pts.map(([r, d]) => this.P(...this._onScreen(r, d), 0.3))), class: "nf lo" }, this.g);
    this.ripple = HL.mk("path", { class: "nf hi", "clip-path": this.clip }, this.g);
    this.pulse  = HL.flatDot(this.g, this.C, 1.2, "dot");
    this.pulse.setAttribute("display", "none");
  }

  _alongWire(f) {
    const { pts, cum } = this.wire;
    const s            = HL.clamp(f, 0, 1) * cum[cum.length - 1];
    let i              = 1;
    while (i < cum.length - 1 && cum[i] < s) i++;
    const t = (s - cum[i - 1]) / (cum[i] - cum[i - 1]);
    return [HL.lerp(pts[i - 1][0], pts[i][0], t), HL.lerp(pts[i - 1][1], pts[i][1], t)];
  }

  _buildDrum() {
    const [r, d, R] = NukeFigure.DRUM;
    const [x, y]    = this._onScreen(r, d);
    const group     = HL.mk("g", {}, this.g);
    const el        = HL.solid(group);
    this.drum = { k: 1, x, y, el, sils: [el], ring: HL.circ(R, 48), inner: HL.circ(R - 1, 48), bands: HL.mk("path", { class: "dash nf" }, group), trefoil: HL.mk("path", { class: "nf" }, group), hub: HL.mk("path", { class: "nf" }, group), live: false, tw: HL.tween(0), drawn: null, restX: this.P(x, y, 0)[0] };
  }

  _buildDetonator() {
    const [r, d] = NukeFigure.BOX;
    const [x, y] = this._onScreen(r, d);
    const group  = HL.mk("g", {}, this.g);
    const el     = HL.solid(group);
    const rod    = HL.solid(group);
    const handle = HL.solid(group);
    this.det = { k: 0, x, y, el, rod, handle, sils: [el, rod, handle], lamp: HL.flatDot(group, this.C, 0.8, "dot m"), cls: "dot m", tw: HL.tween(0), drawn: null, restX: this.P(x, y, 0)[0] };
  }

  _buildCloud() {
    const group = HL.mk("g", {}, this.g);
    this.stem   = HL.solid(group);
    this.cap    = HL.solid(group);
    this.cloudDrawn = null;
  }

  _tick(dt, now) {
    this._sequence(now);
    HL.stepS(this.plunger, dt);
    this._drawDrum(now);
    this._drawDetonator(now);
    this._drawCloud(now);
    this._drawWave(now);
    return true;
  }

  _sequence(now) {
    if (this.fired === null) return;
    const t   = now - this.fired;
    const p0  = NukeFigure.PRESS_MS;
    const p1  = p0 + NukeFigure.PULSE_MS;
    const fly = !this.down && t >= p0 && t < p1 && !HL.reducedMotion();
    if (fly) {
      HL.place(this.pulse, this.P(...this._onScreen(...this._alongWire((t - p0) / NukeFigure.PULSE_MS)), 0.3));
      this.pulse.removeAttribute("display");
    } else {
      this.pulse.setAttribute("display", "none");
    }
    if (!this.down && t >= p1) {
      this.down = true;
      this.wave = now;
      HL.tset(this.cloud, 1, now, 0);
    }
    const settled = this.down && t >= p1 + NukeFigure.HOLD_MS && this.released !== null && now - this.released >= 400;
    if (settled) {
      this.fired     = null;
      this.down      = false;
      this.plunger.k = 50;
      this.plunger.c = 11;
      this.plunger.t = 1;
      HL.tset(this.cloud, 0, now, 0);
    }
  }

  _drawDrum(now) {
    const drum = this.drum;
    const lift = HL.tval(drum.tw, now);
    const key  = lift.toFixed(3);
    if (key !== drum.drawn) {
      drum.drawn = key;
      const [, , , H]          = NukeFigure.DRUM;
      const [hub, ri, ro, half] = NukeFigure.TREFOIL;
      const P                   = this._shifted(drum.x, drum.y, lift);
      const face                = HL.run(drum.ring, this.front);
      HL.put(drum.el, HL.prism(P, this.front, drum.ring, drum.inner, 0, H));
      drum.bands.setAttribute("d", NukeFigure.BANDS.map((at) => HL.open(HL.ringAt(P, face, H * at))).join(""));
      drum.trefoil.setAttribute("d", [90, 210, 330].map((a0) => {
        const pts = [];
        for (let k = 0; k <= 6; k++) {
          const a = HL.rad(a0 - half + (2 * half * k) / 6);
          pts.push(P(ri * Math.cos(a), ri * Math.sin(a), H));
        }
        for (let k = 6; k >= 0; k--) {
          const a = HL.rad(a0 - half + (2 * half * k) / 6);
          pts.push(P(ro * Math.cos(a), ro * Math.sin(a), H));
        }
        return HL.poly(pts);
      }).join(""));
      drum.hub.setAttribute("d", HL.poly(HL.ringAt(P, HL.circ(hub, 16), H)));
    }
    const live = this.fired !== null;
    if (live !== drum.live) {
      drum.live = live;
      drum.trefoil.setAttribute("class", live ? "nf hi" : "nf");
      drum.hub.setAttribute("class", live ? "nf hi" : "nf");
    }
  }

  _drawDetonator(now) {
    const det   = this.det;
    const lift  = HL.tval(det.tw, now);
    const press = Math.max(0, this.plunger.x);
    const key   = `${lift.toFixed(3)}:${press.toFixed(3)}`;
    if (key !== det.drawn) {
      det.drawn = key;
      const [, , w, dd, h] = NukeFigure.BOX;
      const [rr, stroke]   = NukeFigure.ROD;
      const [hl, ht, hz]   = NukeFigure.HANDLE;
      const P              = this._shifted(det.x, det.y, lift);
      const top            = h + 1.5 + stroke * press;
      HL.put(det.el, HL.prism(P, this.front, this._screenBox(-w, -dd, w, dd, 1.2), this._screenBox(-w + 0.8, -dd + 0.8, w - 0.8, dd - 0.8, 0.8), 0, h));
      HL.put(det.rod, HL.prism(P, this.front, HL.circ(rr, 24), null, h, top));
      HL.put(det.handle, HL.prism(P, this.front, this._screenBox(-hl, -ht, hl, ht, 0.7), this._screenBox(-hl + 0.4, -ht + 0.4, hl - 0.4, ht - 0.4, 0.4), top, top + hz));
      HL.place(det.lamp, P(...this._onScreen(w - 1.6, dd - 1.6), h));
    }
    const cls = this.fired !== null ? "dot" : now % NukeFigure.BLINK_MS < NukeFigure.BLINK_MS * 0.7 ? "dot m" : "dot off";
    if (cls !== det.cls) {
      det.cls = cls;
      det.lamp.setAttribute("class", cls);
    }
  }

  _drawCloud(now) {
    const g    = HL.tval(this.cloud, now);
    const lift = HL.tval(this.drum.tw, now);
    const key  = `${g.toFixed(3)}:${lift.toFixed(3)}`;
    if (key === this.cloudDrawn) return;
    this.cloudDrawn = key;
    if (g < 0.03) {
      HL.put(this.stem, { sil: "", crease: "" });
      HL.put(this.cap, { sil: "", crease: "" });
      return;
    }
    const [, , , H]                 = NukeFigure.DRUM;
    const [stemR, capR, rise0, rise1] = NukeFigure.CLOUD;
    const P                         = this._shifted(this.drum.x, this.drum.y, lift);
    const swell                     = 0.35 + 0.65 * g;
    const z0                        = H + rise0 * g;
    const z1                        = H + rise1 * g;
    HL.put(this.stem, HL.prism(P, this.front, HL.circ(stemR * swell, 24), null, H, z0 + 1));
    HL.put(this.cap, HL.prism(P, this.front, HL.circ(capR * swell, 40), HL.circ((capR - 1.5) * swell, 40), z0, z1));
  }

  _drawWave(now) {
    if (this.wave === null) return;
    const s = (now - this.wave) / NukeFigure.WAVE_MS;
    if (s >= 1 || HL.reducedMotion()) {
      this.wave = null;
      this.ripple.setAttribute("d", "");
      return;
    }
    const [w0, w1] = NukeFigure.WAVE;
    const P        = this._shifted(this.drum.x, this.drum.y, 0);
    this.ripple.setAttribute("d", HL.poly(HL.ringAt(P, HL.circ(HL.lerp(w0, w1, s), 72), 0.2)));
  }

  _move(p) {
    this._choose(this.parts[HairlineFigureBase.nearest(this.parts.map((part) => part.restX), p[0])].k);
  }

  setState(state) {
    const firing = !!state.firing;
    if (firing === this.firing) return;
    const now   = performance.now();
    this.firing = firing;
    if (firing) {
      this.released = null;
      if (this.fired === null) {
        this.fired     = now;
        this.down      = false;
        this.plunger.k = 160;
        this.plunger.c = 14;
        this.plunger.t = 0;
      }
    } else {
      this.released = now;
    }
    this.loop.wake();
  }

  _retarget(now) {
    for (const part of this.parts) {
      const on = this.chosen === part.k;
      HL.tset(part.tw, on ? this.amount : 0, now, 0);
      for (const el of part.sils) el.sil.classList.toggle("hi", on);
    }
    this.loop.wake();
  }
}


class CoresFigure extends HairlineFigureBase {
  static RANGE  = [2, 4, 7];
  static CELL   = 10;
  static FOOT   = 7.4;
  static BASE   = 1.6;
  static RISE   = 26;
  static BUSY   = 50;
  static SPAN   = [372, 280];
  static SAMPLE = [35, 80, 12, 60, 95, 20, 48, 70];

  constructor({ stage, svg }, value) {
    super(svg, value, 2.7);
    this.loads = CoresFigure.SAMPLE.slice();
    this.count = this.loads.length;
    this._layout();
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _height(load) {
    return CoresFigure.BASE + (CoresFigure.RISE * HL.clamp(load, 0, 100)) / 100;
  }

  _layout() {
    this.g.replaceChildren();
    const n     = this.count;
    this.cols   = Math.max(1, Math.ceil(Math.sqrt(n * 2)));
    this.rows   = Math.max(1, Math.ceil(n / this.cols));
    this.width  = this.cols * CoresFigure.CELL;
    this.depth  = this.rows * CoresFigure.CELL;
    const peak  = CoresFigure.BASE + CoresFigure.RISE + CoresFigure.RANGE[2];
    const pts   = [[-4, -4, -4], [this.width + 4, this.depth + 4, -4], [this.width + 4, -4, -4], [-4, this.depth + 4, -4], [0, 0, peak], [this.width, this.depth, peak]];
    this.C.S    = 1;
    const unit  = HL.proj(this.C);
    const xs    = pts.map(([x, y, z]) => unit(x, y, z)[0]);
    const ys    = pts.map(([x, y, z]) => unit(x, y, z)[1]);
    this.C.S    = Math.min(CoresFigure.SPAN[0] / (Math.max(...xs) - Math.min(...xs)), CoresFigure.SPAN[1] / (Math.max(...ys) - Math.min(...ys)));
    this._camera(pts, 166);
    this._plinth(-4, -4, this.width + 4, this.depth + 4, 7, 4);
    const z       = this._height(CoresFigure.BUSY);
    const corners = [[-2, -2], [this.width + 2, -2], [this.width + 2, this.depth + 2], [-2, this.depth + 2]];
    HL.mk("path", { d: HL.poly(corners.map(([x, y]) => this.P(x, y, z))), class: "dash nf" }, this.g);
    const inset = (CoresFigure.CELL - CoresFigure.FOOT) / 2;
    this.pillars = [];
    for (let k = 0; k < n; k++) {
      const i             = k % this.cols;
      const j             = Math.floor(k / this.cols);
      const x0            = i * CoresFigure.CELL + inset;
      const y0            = j * CoresFigure.CELL + inset;
      const [ring, inner] = HL.rings(x0, y0, x0 + CoresFigure.FOOT, y0 + CoresFigure.FOOT, 1.8, 0.7);
      const at            = this.P(x0 + CoresFigure.FOOT / 2, y0 + CoresFigure.FOOT / 2, 0);
      this.pillars.push({ k, i, j, ring, inner, at, h: HL.tween(this._height(this.loads[k] || 0), 520), tw: HL.tween(0), drawn: null, busy: null, el: null });
    }
    this.pillars.sort((a, b) => a.i + a.j - (b.i + b.j));
    for (const pillar of this.pillars) pillar.el = HL.solid(this.g);
  }

  _tick(dt, now) {
    let moving = false;
    const busyAt = this._height(CoresFigure.BUSY) - 1e-6;
    for (const pillar of this.pillars) {
      const h    = HL.tval(pillar.h, now);
      const lift = HL.tval(pillar.tw, now);
      if (!HL.tdone(pillar.h, now) || !HL.tdone(pillar.tw, now)) moving = true;
      const key = `${h.toFixed(2)}:${lift.toFixed(2)}`;
      if (key !== pillar.drawn) {
        pillar.drawn = key;
        HL.put(pillar.el, HL.prism(this.P, this.front, pillar.ring, pillar.inner, lift, lift + h));
      }
      const busy = pillar.h.to >= busyAt;
      if (busy !== pillar.busy) {
        pillar.busy = busy;
        pillar.el.cr.setAttribute("class", busy ? "nf hi" : "nf lo");
      }
    }
    return moving;
  }

  _move(p) {
    let best = null;
    let near = Infinity;
    for (const pillar of this.pillars) {
      const d = Math.hypot(pillar.at[0] - p[0], pillar.at[1] - p[1]);
      if (d < near) {
        near = d;
        best = pillar;
      }
    }
    if (best) this._choose(best.k);
  }

  setState(state) {
    if (!Array.isArray(state.cores)) throw new Error("cores figure: state.cores must be an array of busy percentages");
    const loads = state.cores.map((u) => HL.clamp(Number(u) || 0, 0, 100));
    const now   = performance.now();
    this.loads  = loads;
    if (loads.length !== this.count) {
      this.count  = Math.max(1, loads.length);
      this.chosen = null;
      this._layout();
    } else {
      for (const pillar of this.pillars) HL.tset(pillar.h, this._height(loads[pillar.k]), now, pillar.k * 12);
    }
    this.loop.wake();
  }

  _retarget(now) {
    for (const pillar of this.pillars) {
      const on = this.chosen === pillar.k;
      HL.tset(pillar.tw, on ? this.amount : 0, now, 0);
      pillar.el.sil.classList.toggle("hi", on);
    }
    this.loop.wake();
  }
}


class GpuWireFigure extends HairlineFigureBase {
  static RANGE    = [3, 5, 8];
  static PLINTH   = [2, -21, 18, 21];
  static SOCKET   = [6.5, -18, 13.5, 18, 1.4];
  static CARD     = [8, -16, 12, 16, 16];
  static FANS     = [[-8, 8, 5.4], [8, 8, 5.4]];
  static HEIGHT   = 32;
  static INLET    = -40;
  static SPIN     = [0.6, 14];
  static RATE     = [2.6, 0.22];
  static SPEED    = [0.35, 1.5];
  static FLASH_MS = 160;
  static SAMPLE   = { util: 55, power: 0.55 };

  constructor({ stage, svg }, value) {
    super(svg, value, 5.6);
    this.util    = GpuWireFigure.SAMPLE.util / 100;
    this.energy  = GpuWireFigure.SAMPLE.power;
    this.spin    = HL.spring(this._spinFor(this.util), { k: 6, c: 5, eps: 0.01 });
    this.flow    = HL.spring(this.energy, { k: 10, c: 6, eps: 0.002 });
    this.angle   = 0;
    this.packets = [];
    this.next    = 0;
    this.flash   = 0;
    this.lit     = null;
    this._camera(this._extent(), 160);
    const [x0, y0, x1, y1] = GpuWireFigure.PLINTH;
    this._plinth(x0, y0, x1, y1, 5, 3);
    this._buildWire();
    this._buildCard();
    this.parts = [this.card];
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _spinFor(util) {
    const [idle, full] = GpuWireFigure.SPIN;
    return idle + (full - idle) * util;
  }

  _extent() {
    const [x0, y0, x1, y1]         = GpuWireFigure.PLINTH;
    const [cx0, cy0, cx1, cy1, ch] = GpuWireFigure.CARD;
    const lift                     = GpuWireFigure.RANGE[2];
    const mid                      = (cx0 + cx1) / 2;
    return [[x0, y0, -3], [x1, y1, -3], [x1, y0, -3], [x0, y1, -3], [cx0, cy0, ch + lift], [cx1, cy1, ch + lift], [mid, GpuWireFigure.INLET, GpuWireFigure.HEIGHT], [mid, 0, GpuWireFigure.HEIGHT]];
  }

  _buildWire() {
    const [cx0, , cx1, , ch] = GpuWireFigure.CARD;
    const mid                = (cx0 + cx1) / 2;
    const pts                = [[mid, GpuWireFigure.INLET, GpuWireFigure.HEIGHT], [mid, 0, GpuWireFigure.HEIGHT], [mid, 0, ch + 0.5]];
    const cum                = [0];
    for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]));
    this.wire = { pts, cum };
    HL.mk("path", { d: HL.open(pts.map(([x, y, z]) => this.P(x, y, z))), class: "nf lo" }, this.g);
    this.packetLayer = HL.mk("g", {}, this.g);
  }

  _alongWire(f) {
    const { pts, cum } = this.wire;
    const s            = HL.clamp(f, 0, 1) * cum[cum.length - 1];
    let i              = 1;
    while (i < cum.length - 1 && cum[i] < s) i++;
    const t = (s - cum[i - 1]) / (cum[i] - cum[i - 1]);
    return [0, 1, 2].map((k) => HL.lerp(pts[i - 1][k], pts[i][k], t));
  }

  _buildCard() {
    const [sx0, sy0, sx1, sy1, sh] = GpuWireFigure.SOCKET;
    const [x0, y0, x1, y1]         = GpuWireFigure.CARD;
    const group                    = HL.mk("g", {}, this.g);
    HL.put(HL.solid(group), HL.prism(this.P, this.front, ...HL.rings(sx0, sy0, sx1, sy1, 0.9, 0.5), 0, sh));
    const [ring, inner] = HL.rings(x0, y0, x1, y1, 1.1, 0.5);
    const el            = HL.solid(group);
    this.card = { k: 0, ring, inner, el, fans: HL.mk("path", { class: "nf lo" }, group), led: HL.flatDot(group, this.C, 0.9, "dot off"), lift: 0, tw: HL.tween(0), drawn: NaN, restX: this.P((x0 + x1) / 2, (y0 + y1) / 2, 0)[0] };
  }

  _tick(dt, now) {
    HL.stepS(this.spin, dt);
    HL.stepS(this.flow, dt);
    this.angle = (this.angle + this.spin.x * dt) % (Math.PI * 2);
    this._advance(this.parts, now);
    this._drawFans();
    this._flow(dt, now);
    return true;
  }

  _draw(card, lift) {
    const [x0, y0, x1, y1, h] = GpuWireFigure.CARD;
    card.lift                 = lift;
    HL.put(card.el, HL.prism(this.P, this.front, card.ring, card.inner, lift, lift + h));
    HL.place(card.led, this.P((x0 + x1) / 2, y1 - 2.4, lift + h));
  }

  _drawFans() {
    const card           = this.card;
    const [, y0, x1, y1] = GpuWireFigure.CARD;
    const mid            = (y0 + y1) / 2;
    const at             = (cy, cz, a, radius) => this.P(x1, cy + radius * Math.cos(a), cz + radius * Math.sin(a));
    let d                = "";
    for (const [fy, fz, R] of GpuWireFigure.FANS) {
      const cy = mid + fy;
      const cz = card.lift + fz;
      d       += HairlineGeometry.faceCircle(this.P, x1, cy, cz, R);
      for (let j = 0; j < 3; j++) {
        const a = this.angle + (j * Math.PI * 2) / 3;
        d      += HL.seg(at(cy, cz, a, 1.1), at(cy, cz, a + 0.5, R * 0.82));
      }
    }
    card.fans.setAttribute("d", d);
  }

  _flow(dt, now) {
    const e     = HL.clamp(this.flow.x, 0, 1);
    const speed = HL.lerp(GpuWireFigure.SPEED[0], GpuWireFigure.SPEED[1], e);
    let arrived = false;
    for (const packet of this.packets) {
      packet.s += speed * dt;
      if (packet.s >= 1) {
        arrived = true;
        packet.el.remove();
      }
    }
    if (arrived) this.flash = now + GpuWireFigure.FLASH_MS;
    this.packets = this.packets.filter((packet) => packet.s < 1);
    if (now >= this.next) {
      this.next = now + HL.lerp(GpuWireFigure.RATE[0], GpuWireFigure.RATE[1], e) * 1000;
      this.packets.push({ s: 0, el: HL.flatDot(this.packetLayer, this.C, 1.1, "dot") });
    }
    for (const packet of this.packets) HL.place(packet.el, this.P(...this._alongWire(packet.s)));
    const lit = now < this.flash ? "dot" : this.util >= 0.5 ? "dot m" : "dot off";
    if (lit !== this.lit) {
      this.lit = lit;
      this.card.led.setAttribute("class", lit);
    }
  }

  _move() {
    this._choose(0);
  }

  setState(state) {
    const util = HL.clamp(Number(state.util) || 0, 0, 100) / 100;
    if (!Number.isFinite(util)) throw new Error("gpuwire figure: state.util must be a percentage");
    this.util   = util;
    this.energy = Number.isFinite(state.power) ? HL.clamp(state.power, 0, 1) : util;
    this.spin.t = this._spinFor(util);
    this.flow.t = this.energy;
    this.loop.wake();
  }

  _retarget(now) {
    const on = this.chosen === 0;
    HL.tset(this.card.tw, on ? this.amount : 0, now, 0);
    this.card.el.sil.classList.toggle("hi", on);
    this.loop.wake();
  }
}


class GuillotineFigure extends HairlineFigureBase {
  static RANGE    = [2, 4, 7];
  static PLINTH   = [-16, -15, 16, 15];
  static SERVER   = [-4.5, -3.8, 4.5, 3.8, 15];
  static POSTS    = [-9.5, 9.5];
  static POST     = [0.9, 30];
  static BAR      = 2;
  static BLADE    = [0.6, 7, 8, 3.2];
  static TOP      = 20;
  static APART    = 2.4;
  static RISE_MS  = 1800;
  static HOLD_MS  = 900;
  static DROP_MS  = 260;
  static SPLIT_MS = 420;
  static OPEN_MS  = 1100;
  static JOIN_MS  = 500;

  constructor({ stage, svg }, value) {
    super(svg, value, 6);
    this.phase = "rise";
    this.t0    = performance.now();
    this.blade = 0;
    this.down  = false;
    this.gap   = HL.tween(0, GuillotineFigure.SPLIT_MS);
    this.drawn = null;
    this._camera(this._extent(), 162);
    const [x0, y0, x1, y1] = GuillotineFigure.PLINTH;
    this._plinth(x0, y0, x1, y1, 5, 3);
    this._buildFrame();
    this.parts = [this.server];
    this._start(stage, svg);
    this._retarget(performance.now());
  }

  _extent() {
    const [x0, y0, x1, y1]  = GuillotineFigure.PLINTH;
    const [sx0, sy0, sx1, sy1, sh] = GuillotineFigure.SERVER;
    const [pw, ph]          = GuillotineFigure.POST;
    const [pa, pb]          = GuillotineFigure.POSTS;
    const crown             = ph + GuillotineFigure.BAR;
    const lift              = GuillotineFigure.RANGE[2];
    return [[x0, y0, -3], [x1, y1, -3], [x1, y0, -3], [x0, y1, -3], [-pw, pa - pw, crown], [pw, pb + pw, crown], [sx0, sy0, sh + lift], [sx1, sy1, sh + lift]];
  }

  _buildFrame() {
    const [pw, ph] = GuillotineFigure.POST;
    const [pa, pb] = GuillotineFigure.POSTS;
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ...HL.rings(-pw, pa - pw, pw, pa + pw, 0.4, 0.2), 0, ph));
    const group  = HL.mk("g", {}, this.g);
    const left   = HL.solid(group);
    const bladeG = HL.mk("g", {}, group);
    const right  = HL.solid(group);
    this.server  = { k: 0, left, right, vents: HL.mk("path", { class: "nf lo" }, group), led: HL.flatDot(group, this.C, 0.9, "dot m"), tw: HL.tween(0), drawn: NaN, restX: this.P(0, 0, 0)[0] };
    this.server.el = right;
    this.bladeEl   = HL.solid(bladeG);
    this.bladeEdge = HL.mk("path", { class: "nf lo" }, bladeG);
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ...HL.rings(-pw, pb - pw, pw, pb + pw, 0.4, 0.2), 0, ph));
    HL.put(HL.solid(this.g), HL.prism(this.P, this.front, ...HL.rings(-pw, pa - pw, pw, pb + pw, 0.4, 0.2), ph, ph + GuillotineFigure.BAR));
  }

  _tick(dt, now) {
    this._phase(now);
    const lift = HL.tval(this.server.tw, now);
    const gap  = HL.tval(this.gap, now);
    const key  = `${lift.toFixed(2)}:${gap.toFixed(3)}:${this.blade.toFixed(2)}`;
    if (key !== this.drawn) {
      this.drawn = key;
      this._drawServer(lift, gap);
      this._drawBlade(this.blade);
    }
    return true;
  }

  _phase(now) {
    const t     = now - this.t0;
    const quick = HL.reducedMotion();
    switch (this.phase) {
      case "rise":
        this.blade = GuillotineFigure.TOP * (quick ? 1 : HL.EASE_LIFT(t / GuillotineFigure.RISE_MS));
        if (t >= GuillotineFigure.RISE_MS || quick) { this.phase = "hold"; this.t0 = now; this.blade = GuillotineFigure.TOP; }
        break;
      case "hold":
        if (t >= GuillotineFigure.HOLD_MS) { this.phase = "drop"; this.t0 = now; }
        break;
      case "drop": {
        const s    = quick ? 1 : Math.min(1, t / GuillotineFigure.DROP_MS);
        this.blade = GuillotineFigure.TOP * (1 - s * s);
        if (s >= 1) { this.phase = "open"; this.t0 = now; HL.tset(this.gap, 1, now, 0); }
        break;
      }
      case "open":
        if (t >= GuillotineFigure.OPEN_MS && !this.down) { this.phase = "join"; this.t0 = now; HL.tset(this.gap, 0, now, 0); }
        break;
      case "join":
        if (t >= GuillotineFigure.JOIN_MS) { this.phase = "rise"; this.t0 = now; }
        break;
      default:
        break;
    }
  }

  _drawServer(lift, gap) {
    const server              = this.server;
    const [x0, y0, x1, y1, h] = GuillotineFigure.SERVER;
    const a                   = GuillotineFigure.APART * gap;
    HL.put(server.left, HL.prism(this.P, this.front, ...HL.rings(x0 - a, y0, -a, y1, 1, 0.5), lift, lift + h));
    HL.put(server.right, HL.prism(this.P, this.front, ...HL.rings(a, y0, x1 + a, y1, 1, 0.5), lift, lift + h));
    server.vents.setAttribute("d", [3.4, 6.4, 9.4].map((z) => HL.seg(this.P(x0 - a + 1.4, y1, lift + z), this.P(-a - 0.4, y1, lift + z)) + HL.seg(this.P(a + 0.4, y1, lift + z), this.P(x1 + a - 1.4, y1, lift + z))).join(""));
    HL.place(server.led, this.P(x1 + a - 2, y1 - 2, lift + h));
  }

  _drawBlade(z) {
    const [bt, bw, bh, slant] = GuillotineFigure.BLADE;
    const face = (x) => [[x, -bw, z + slant], [x, bw, z], [x, bw, z + bh], [x, -bw, z + bh]].map(([px, py, pz]) => this.P(px, py, pz));
    const far  = face(-bt);
    const near = face(bt);
    HL.put(this.bladeEl, { sil: HL.poly(HL.hull(far.concat(near))), crease: HL.poly(near) });
    this.bladeEdge.setAttribute("d", HL.seg(near[0], near[1]));
  }

  _move() {
    this._choose(0);
  }

  setState(state) {
    this.down = !!state.down;
    this.loop.wake();
  }

  _retarget(now) {
    const on = this.chosen === 0;
    HL.tset(this.server.tw, on ? this.amount : 0, now, 0);
    this.server.left.sil.classList.toggle("hi", on);
    this.server.right.sil.classList.toggle("hi", on);
    this.loop.wake();
  }
}


class DeskCalendarFigure extends HairlineFigureBase {
  static RANGE   = [8, 14, 24];
  static SLOTS   = 6;
  static WIDTH   = 32;
  static LENGTH  = 24;
  static SLOPE   = (22 * Math.PI) / 180;
  static BOARD   = 1.2;
  static MARGIN  = 1.4;
  static PAGE    = 0.12;
  static BARS    = [5.4, 3.1, 1.5];
  static INSET   = 3;
  static RINGS   = [-10.5, -3.5, 3.5, 10.5];
  static STEP_MS = 2600;
  static HOLD_MS = 6000;

  static sample() {
    return Array.from({ length: 30 }, (_, day) => ({
      slots: Array.from({ length: DeskCalendarFigure.SLOTS }, (_, slot) => {
        const weekday = day % 7 < 5;
        const mine    = weekday && slot >= 2 && slot <= 4 ? 0.55 + 0.3 * Math.sin(day + slot) : 0.25 + 0.2 * Math.sin(day * 1.7 + slot);
        const others  = 0.15 + 0.15 * Math.cos(day * 0.9 + slot * 1.3);
        return { mine: HL.clamp(mine, 0, 0.85), others: HL.clamp(others, 0, 1 - HL.clamp(mine, 0, 0.85)) };
      }),
    }));
  }

  constructor({ stage, svg }, value) {
    super(svg, value, 5.0);
    this.days  = DeskCalendarFigure.sample();
    this.ridge = (DeskCalendarFigure.LENGTH + DeskCalendarFigure.MARGIN) * Math.cos(DeskCalendarFigure.SLOPE);
    this.pos   = HL.spring(0, { k: this.amount, c: 2 * Math.sqrt(this.amount), eps: 0.002 });
    this.next  = 0;
    this.drawn = NaN;
    this._camera(this._extent(), 160);
    this._build();
    this._start(stage, svg);
    this._retarget();
  }

  _extent() {
    const half  = DeskCalendarFigure.WIDTH / 2 + 3;
    const depth = this.ridge * Math.tan(DeskCalendarFigure.SLOPE) + 4;
    const box   = [[-half, -depth], [half, -depth], [-half, depth], [half, depth]].map(([x, y]) => [x, y, -2.5]);
    return box.concat([[-half, 0, this.ridge + DeskCalendarFigure.LENGTH], [half, 0, this.ridge + DeskCalendarFigure.LENGTH]]);
  }

  _build() {
    const half  = DeskCalendarFigure.WIDTH / 2 + 3;
    const depth = this.ridge * Math.tan(DeskCalendarFigure.SLOPE) + 4;
    this._plinth(-half, -depth, half, depth, 3, 2.5);

    this.backBoard  = HL.mk("path", { class: "sil" }, this.g);
    this.backStack  = HL.mk("path", { class: "sil" }, this.g);
    this.flipBack   = HL.mk("g", {}, this.g);
    this.frontBoard = HL.mk("path", { class: "sil" }, this.g);
    this.frontStack = HL.mk("path", { class: "sil" }, this.g);
    this.face       = this._buildFace(this.g);
    this.flipFront  = HL.mk("g", {}, this.g);
    this.flip       = HL.mk("g", {}, this.flipFront);
    this.flipPage   = HL.mk("path", { class: "sil" }, this.flip);
    this.flipFace   = this._buildFace(this.flip);
    this.rings      = HL.mk("path", { class: "nf" }, this.g);
  }

  _buildFace(parent) {
    const group  = HL.mk("g", {}, parent);
    const header = HL.mk("path", { class: "nf lo" }, group);
    const bars   = Array.from({ length: DeskCalendarFigure.SLOTS }, () => ({
      track  : HL.mk("path", { class: "nf lo" }, group),
      others : HL.mk("path", { class: "nf" }, group),
      mine   : HL.mk("path", { class: "nf hi" }, group),
    }));
    return { group, header, bars };
  }

  _tick(dt, now) {
    this._ambient(now);
    HL.stepS(this.pos, dt);
    if (this.pos.x !== this.drawn) {
      this.drawn = this.pos.x;
      this._draw(this.pos.x);
    }
    return true;
  }

  _ambient(now) {
    if (this.chosen !== null || now < this.next) return;
    const last = this.days.length - 1;
    if (this.pos.t >= last) {
      this.pos.t = 0;
      this.next  = now + DeskCalendarFigure.HOLD_MS;
      return;
    }
    this.pos.t = Math.round(this.pos.t) + 1;
    this.next  = now + (this.pos.t >= last ? DeskCalendarFigure.HOLD_MS : DeskCalendarFigure.STEP_MS);
  }

  _draw(pos) {
    const count    = this.days.length;
    const day      = HL.clamp(Math.floor(pos), 0, count - 1);
    const turn     = pos - day;
    const flipping = turn > 0.002 && day < count - 1;
    const top      = flipping ? day + 1 : day;
    const front    = (count - top) * DeskCalendarFigure.PAGE;
    const back     = day * DeskCalendarFigure.PAGE;
    const slope    = DeskCalendarFigure.SLOPE;
    const half     = DeskCalendarFigure.WIDTH / 2;
    const edge     = half + DeskCalendarFigure.MARGIN;
    const length   = DeskCalendarFigure.LENGTH + DeskCalendarFigure.MARGIN;
    const fore     = this._pageAt(slope);
    const rear     = this._pageAt(2 * Math.PI - slope);

    this.backBoard.setAttribute("d", this._box(rear, -edge, edge, 0, length, 0, DeskCalendarFigure.BOARD));
    this.backStack.setAttribute("d", back > 0 ? this._box(rear, -half, half, 0.6, DeskCalendarFigure.LENGTH, -back, 0) : "");
    this.frontBoard.setAttribute("d", this._box(fore, -edge, edge, 0, length, -DeskCalendarFigure.BOARD, 0));
    this.frontStack.setAttribute("d", this._box(fore, -half, half, 0.6, DeskCalendarFigure.LENGTH, 0, front));
    this._drawFace(this.face, fore, front, this.days[top]);

    const theta  = slope + turn * (2 * Math.PI - 2 * slope);
    const page   = this._pageAt(theta);
    const home   = theta < Math.PI ? this.flipFront : this.flipBack;
    const offset = HL.lerp(front, -back - DeskCalendarFigure.PAGE, turn);
    this.flipPage.setAttribute("d", flipping ? this._box(page, -half, half, 0.6, DeskCalendarFigure.LENGTH, offset, offset + DeskCalendarFigure.PAGE) : "");
    if (flipping && Math.cos(theta) + 0.816 * Math.sin(theta) > 0) this._drawFace(this.flipFace, page, offset + DeskCalendarFigure.PAGE, this.days[day]);
    else this._clearFace(this.flipFace);
    if (this.flip.parentNode !== home) home.appendChild(this.flip);

    this.rings.setAttribute("d", DeskCalendarFigure.RINGS.map((u) => HL.poly(Array.from({ length: 20 }, (_, k) => this.P(u, 1.3 * Math.cos((2 * Math.PI * k) / 20), this.ridge + 0.4 + 1.3 * Math.sin((2 * Math.PI * k) / 20))))).join(""));
  }

  _pageAt(theta) {
    const down   = [Math.sin(theta), -Math.cos(theta)];
    const normal = [Math.cos(theta), Math.sin(theta)];
    return (u, v, n) => this.P(u, v * down[0] + n * normal[0], this.ridge + v * down[1] + n * normal[1]);
  }

  _box(at, u0, u1, v0, v1, n0, n1) {
    const corners = [];
    for (const u of [u0, u1]) for (const v of [v0, v1]) for (const n of [n0, n1]) corners.push(at(u, v, n));
    return HL.poly(HL.hull(corners));
  }

  _drawFace(face, at, n, day) {
    const half             = DeskCalendarFigure.WIDTH / 2;
    const [first, pitch, h] = DeskCalendarFigure.BARS;
    const u0               = -half + DeskCalendarFigure.INSET;
    const span             = 2 * (half - DeskCalendarFigure.INSET);
    const rect             = (a, b, v) => (b - a < 0.3 ? "" : HL.poly([at(a, v, n), at(b, v, n), at(b, v + h, n), at(a, v + h, n)]));
    face.header.setAttribute("d", HL.seg(at(-half + 1.4, 3, n), at(half - 1.4, 3, n)));
    day.slots.forEach((slot, s) => {
      const v     = first + s * pitch;
      const mine  = u0 + span * HL.clamp(slot.mine, 0, 1);
      const both  = u0 + span * HL.clamp(slot.mine + slot.others, 0, 1);
      face.bars[s].track.setAttribute("d", rect(u0, u0 + span, v));
      face.bars[s].mine.setAttribute("d", rect(u0, mine, v));
      face.bars[s].others.setAttribute("d", rect(mine, both, v));
    });
  }

  _clearFace(face) {
    face.header.setAttribute("d", "");
    for (const bar of face.bars) {
      bar.track.setAttribute("d", "");
      bar.mine.setAttribute("d", "");
      bar.others.setAttribute("d", "");
    }
  }

  _move(p) {
    const half = DeskCalendarFigure.WIDTH / 2;
    const x0   = this.P(-half, 0, this.ridge)[0];
    const x1   = this.P(half, 0, this.ridge)[0];
    this._choose(Math.round(HL.clamp((p[0] - x0) / (x1 - x0), 0, 1) * (this.days.length - 1)));
  }

  setState(state) {
    if (!Array.isArray(state.days) || !state.days.length) throw new Error("calendar figure: state.days must list at least one day");
    this.days  = state.days;
    this.drawn = NaN;
    if (this.pos.t > this.days.length - 1) this.pos.t = this.days.length - 1;
    this._retarget();
  }

  _retarget() {
    this.pos.k = this.amount;
    this.pos.c = 2 * Math.sqrt(this.amount);
    if (this.chosen !== null) this.pos.t = this.chosen;
    this.loop.wake();
  }
}


class ServerFigures {
  static register() {
    Object.assign(RoutingFigures.CATALOG, {
      gpus:       RoutingFigures.entry(GpuShelfFigure),
      scheduler:  RoutingFigures.entry(SchedulerFigure),
      watchdog:   RoutingFigures.entry(WatchdogFigure),
      nuke:       RoutingFigures.entry(NukeFigure),
      cores:      RoutingFigures.entry(CoresFigure),
      gpuwire:    RoutingFigures.entry(GpuWireFigure),
      guillotine: RoutingFigures.entry(GuillotineFigure),
      calendar:   RoutingFigures.entry(DeskCalendarFigure),
    });
    Object.assign(RoutingFigures.LABELS, {
      gpus:       "The GPU cards standing in a chassis, each light showing who holds it.",
      scheduler:  "A rack with running jobs in its sockets and waiting jobs on the steps up to it.",
      watchdog:   "A searchlight sweeping four targets for CPU, load, RAM and disk, each standing taller the nearer it is to its limit.",
      nuke:       "A charge with a plunger detonator, fired when every job is stopped at once.",
      cores:      "One pillar per processor core, as tall as that core is busy.",
      gpuwire:    "A graphics card fed by a power cable, its fans as fast as the GPU is busy.",
      guillotine: "A server tower under a guillotine, split when the front end is killed.",
      calendar:   "A desk calendar of thirty days, each page carrying six four-hour slots of GPU activity.",
      drawer:     "A drawer of saved runs.",
      patch:      "A patch panel of GPU activity cables.",
      phosphor:   "A phosphor screen of host load.",
      router:     "A router passing network traffic.",
    });
  }
}

ServerFigures.register();
