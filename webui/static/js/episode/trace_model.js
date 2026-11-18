import { isNum } from "../format.js";

export class Position {
  static parse(text) {
    if (text === undefined || text === null || text === "") return null;
    if (/^\d+$/.test(text)) return { seq: Number(text) };
    const match = /^r(\d+)(?:e(\d+))?$/.exec(text);
    if (!match) return null;
    return { round: Number(match[1]), edit: match[2] === undefined ? null : Number(match[2]) };
  }

  static of(frame) {
    return frame.kind === "round_open" ? `r${frame.round}` : `r${frame.round}e${frame.edit}`;
  }
}

export class TraceModel {
  constructor(agentKey) {
    this.agentKey = agentKey;
    this.meta = null;
    this.depot = null;
    this.entities = { jobs: {}, vehicles: {} };
    this.paths = {};
    this.frames = [];
    this.summary = null;
    this.expectation = null;
    this.findings = [];
    this.batch = 0;
  }

  get complete() {
    return Boolean(this.meta && this.meta.complete);
  }

  get length() {
    return this.frames.length;
  }

  merge(payload) {
    this.batch += 1;
    const prefix = `b${this.batch}_`;
    this.meta = payload.meta;
    this.depot = payload.depot;
    this.entities = payload.entities;
    this.summary = payload.summary;
    this.expectation = payload.expectation;
    this.findings = payload.findings;
    for (const [id, path] of Object.entries(payload.paths)) this.paths[prefix + id] = path;
    for (const frame of payload.frames) {
      for (const route of frame.plan.routes) route.path_id = prefix + route.path_id;
      this.frames[frame.seq] = frame;
    }
  }

  frame(seq) {
    return this.frames[seq] || null;
  }

  resolve(position) {
    if (!position) return null;
    if ("seq" in position) return this.frames[position.seq] ? position.seq : null;
    const match = this.frames.find((frame) => frame && frame.round === position.round && (position.edit === null ? frame.kind === "round_open" : frame.kind === "decision" && frame.edit === position.edit));
    return match ? match.seq : null;
  }

  nearest(round, edit) {
    const inRound = this.frames.filter((frame) => frame && frame.round === round);
    if (!inRound.length) return this.frames.length ? this.frames[this.frames.length - 1].seq : null;
    if (edit === null) return inRound[0].seq;
    const decisions = inRound.filter((frame) => frame.kind === "decision");
    if (!decisions.length) return inRound[0].seq;
    return decisions[Math.min(edit, decisions.length - 1)].seq;
  }

  roundOpen(round) {
    return this.frames.find((frame) => frame && frame.kind === "round_open" && frame.round === round) || null;
  }

  rounds() {
    const out = [];
    for (const frame of this.frames) {
      if (!frame) continue;
      if (!out[frame.round]) out[frame.round] = { round: frame.round, open: null, decisions: [] };
      if (frame.kind === "round_open") out[frame.round].open = frame;
      else out[frame.round].decisions.push(frame);
    }
    return out;
  }

  commitOf(round) {
    const decisions = this.frames.filter((frame) => frame && frame.kind === "decision" && frame.round === round);
    return decisions.length ? decisions[decisions.length - 1] : null;
  }

  path(route) {
    return route.path || this.paths[route.path_id] || [];
  }

  cumulative(seq) {
    for (let index = seq; index >= 0; index -= 1) {
      const frame = this.frames[index];
      if (frame && frame.decision) return frame.decision.cumulative;
    }
    return 0;
  }

  kpiDelta(seq, key, against) {
    const frame = this.frames[seq];
    const other = this.frames[against];
    if (!frame || !other) return null;
    const now = frame.plan.kpis[key];
    const then = other.plan.kpis[key];
    return isNum(now) && isNum(then) ? now - then : null;
  }

  routeOf(seq, vehicleId) {
    const frame = this.frames[seq];
    return frame ? frame.plan.routes.find((route) => route.vehicle_id === vehicleId) || null : null;
  }

  vehicleOfJob(frame, jobId) {
    for (const route of frame.plan.routes) {
      if (route.stops.some((stop) => stop.job_id === jobId)) return route.vehicle_id;
    }
    return null;
  }

  vehicleIds(frame) {
    return Object.keys(frame.plan.vehicles).map(Number).sort((a, b) => a - b);
  }
}
