import { INK } from "./gauges.js";

export const SERIES = {
  cpu: INK.accent,
  ram: INK.live,
  rx: "240, 171, 252",
  tx: "251, 146, 60",
  temp: INK.warn,
  power: "240, 171, 252",
  mine: INK.accent,
  others: INK.warn,
  unknown: INK.live,
};

export class CanvasFrame {
  static fit(canvas) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  static font(weight, size) {
    return `${weight} ${size}px "JetBrains Mono", ui-monospace, monospace`;
  }

  static pad(n) {
    return String(n).padStart(2, "0");
  }

  static clock(date) {
    return `${CanvasFrame.pad(date.getHours())}:${CanvasFrame.pad(date.getMinutes())}`;
  }

  static day(date) {
    return `${CanvasFrame.pad(date.getDate())}.${CanvasFrame.pad(date.getMonth() + 1)}`;
  }

  static tick(ctx, text, x, w, y, last, align) {
    const width = ctx.measureText(text).width;
    const right = Math.min(w - 1, align === "left" ? x + width : x + width / 2);
    const left = right - width;
    if (left < last + 8) return last;
    ctx.textAlign = "left";
    ctx.fillText(text, left, y);
    return right;
  }

  static empty(ctx, w, h, text) {
    ctx.fillStyle = `rgb(${INK.hint})`;
    ctx.font = CanvasFrame.font(600, 12);
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 2, h / 2);
  }
}

export class Spark {
  static draw(canvas, series, maxSamples) {
    const frame = CanvasFrame.fit(canvas);
    if (!frame) return;
    const { ctx, w, h } = frame;

    ctx.fillStyle = "rgba(9, 13, 17, 0.6)";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = `rgba(${INK.ink}, 0.1)`;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    ctx.setLineDash([1, 4]);
    ctx.strokeStyle = `rgba(${INK.ink}, 0.14)`;
    [0.25, 0.5, 0.75].forEach((f) => {
      const y = Math.round(h * f) + 0.5;
      ctx.beginPath();
      ctx.moveTo(2, y);
      ctx.lineTo(w - 2, y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    const step = w / Math.max(1, maxSamples - 1);
    series.forEach((entry) => {
      const data = entry.data;
      if (data.length < 2) return;
      const x0 = w - (data.length - 1) * step;
      const py = (v) => h - 2.5 - (Math.max(0, Math.min(100, v)) / 100) * (h - 5);

      ctx.beginPath();
      data.forEach((v, i) => {
        if (i === 0) ctx.moveTo(x0 + i * step, py(v));
        else ctx.lineTo(x0 + i * step, py(v));
      });
      ctx.strokeStyle = `rgba(${entry.color}, 0.95)`;
      ctx.lineWidth = 1.4;
      ctx.stroke();

      ctx.lineTo(w, h);
      ctx.lineTo(x0, h);
      ctx.closePath();
      ctx.fillStyle = `rgba(${entry.color}, ${entry.fill})`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(w - 2.5, py(data[data.length - 1]), 2.2, 0, Math.PI * 2);
      ctx.fillStyle = `rgb(${entry.color})`;
      ctx.fill();
    });
  }
}

export class TimeStrip {
  static AXIS_H = 18;
  static GAP_BUCKETS = 2.5;

  static points(samples, key) {
    return samples.map((sample) => ({ t: sample.t, v: sample[key] }));
  }

  static niceMax(value) {
    const exp = Math.pow(10, Math.floor(Math.log10(value)));
    const step = [1, 2, 2.5, 5, 10].find((m) => m * exp >= value);
    return step * exp;
  }

  static format(value) {
    if (value >= 10) return String(Math.round(value));
    if (value >= 1) return value.toFixed(1).replace(/\.0$/, "");
    return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  static ticks(view, plotW) {
    const start = view.now - view.window_s;
    const ticks = [];
    if (view.window_s > 86400) {
      const cursor = new Date(start * 1000);
      cursor.setHours(24, 0, 0, 0);
      const days = [];
      while (cursor.getTime() / 1000 <= view.now) {
        days.push(cursor.getTime() / 1000);
        cursor.setDate(cursor.getDate() + 1);
      }
      const every = Math.max(1, Math.ceil((days.length * 46) / Math.max(1, plotW)));
      days.forEach((t, i) => ticks.push({ t, label: i % every === 0 ? CanvasFrame.day(new Date(t * 1000)) : null }));
      return ticks;
    }
    const step = view.window_s > 7200 ? 3 * 3600 : 900;
    for (let t = Math.ceil(start / step) * step; t <= view.now; t += step) ticks.push({ t, label: CanvasFrame.clock(new Date(t * 1000)) });
    return ticks;
  }

  static draw(canvas, view, tracks, opts) {
    canvas.strip = { view, tracks, opts };
    TimeStrip.wire(canvas);
    const frame = CanvasFrame.fit(canvas);
    if (!frame) return;
    const { ctx, w, h } = frame;

    const values = tracks.flatMap((track) => track.points.map((point) => point.v)).filter((v) => Number.isFinite(v));
    const yMax = opts.yMax > 0 ? opts.yMax : TimeStrip.niceMax(Math.max(1, ...values));
    const unit = opts.unit || "";
    const gapS = (view.bucket_s || 300) * TimeStrip.GAP_BUCKETS;

    ctx.font = CanvasFrame.font(600, 11);
    const ticksY = [0, yMax / 4, yMax / 2, (3 * yMax) / 4, yMax];
    const left = Math.ceil(Math.max(...ticksY.map((v) => ctx.measureText(TimeStrip.format(v)).width))) + 10;
    const top = 8;
    const plotH = h - TimeStrip.AXIS_H - top;
    const start = view.now - view.window_s;
    const px = (t) => left + ((t - start) / view.window_s) * (w - left - 4);
    const py = (v) => top + plotH - Math.min(1, Math.max(0, v / yMax)) * plotH;

    ctx.fillStyle = "rgba(9, 13, 17, 0.6)";
    ctx.fillRect(left, top, w - left - 2, plotH);

    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    ticksY.forEach((v, i) => {
      const y = Math.round(py(v)) + 0.5;
      ctx.strokeStyle = i === 0 ? `rgba(${INK.ink}, 0.22)` : `rgba(${INK.ink}, 0.08)`;
      ctx.setLineDash(i === 0 ? [] : [2, 4]);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - 2, y);
      ctx.stroke();
      if (i % 2 === 1) return;
      ctx.fillStyle = `rgb(${INK.faint})`;
      ctx.fillText(TimeStrip.format(v), left - 6, y);
    });
    ctx.setLineDash([]);

    ctx.textBaseline = "top";
    let last = -Infinity;
    TimeStrip.ticks(view, w - left).forEach(({ t, label }) => {
      const x = Math.round(px(t)) + 0.5;
      ctx.strokeStyle = `rgba(${INK.ink}, 0.07)`;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + plotH);
      ctx.stroke();
      if (!label) return;
      ctx.fillStyle = `rgb(${INK.faint})`;
      last = CanvasFrame.tick(ctx, label, x, w, top + plotH + 5, last, "center");
    });

    ctx.save();
    ctx.beginPath();
    ctx.rect(left, top - 2, w - left, plotH + 4);
    ctx.clip();
    tracks.forEach((track) => TimeStrip.trace(ctx, track, px, py, top + plotH, gapS));
    ctx.restore();

    if (!values.length) CanvasFrame.empty(ctx, w, h, "nothing recorded in this window yet");
    if (canvas.cursor != null && values.length) TimeStrip.readout(ctx, canvas.cursor, view, tracks, unit, { left, top, plotH, w, px, start });
  }

  static wire(canvas) {
    if (canvas.stripWired) return;
    canvas.stripWired = true;
    const redraw = () => TimeStrip.draw(canvas, canvas.strip.view, canvas.strip.tracks, canvas.strip.opts);
    canvas.addEventListener("pointermove", (event) => {
      canvas.cursor = event.clientX - canvas.getBoundingClientRect().left;
      redraw();
    });
    canvas.addEventListener("pointerleave", () => {
      canvas.cursor = null;
      redraw();
    });
  }

  static nearest(points, t, reach) {
    let best = null;
    let gap = Infinity;
    for (const point of points) {
      if (!Number.isFinite(point.v)) continue;
      const d = Math.abs(point.t - t);
      if (d < gap) {
        gap = d;
        best = point.v;
      }
    }
    return gap <= reach ? best : null;
  }

  static readout(ctx, cursorX, view, tracks, unit, frame) {
    if (cursorX < frame.left) return;
    const t = frame.start + ((cursorX - frame.left) / (frame.w - frame.left - 4)) * view.window_s;
    const x = Math.round(frame.px(t)) + 0.5;
    ctx.strokeStyle = `rgba(${INK.ink}, 0.35)`;
    ctx.beginPath();
    ctx.moveTo(x, frame.top);
    ctx.lineTo(x, frame.top + frame.plotH);
    ctx.stroke();

    const date = new Date(t * 1000);
    const stamp = view.window_s > 86400 ? `${CanvasFrame.day(date)} ${CanvasFrame.clock(date)}` : CanvasFrame.clock(date);
    const lines = [{ text: stamp, color: INK.ink }];
    for (const track of tracks) {
      const near = TimeStrip.nearest(track.points, t, (view.bucket_s || 300) * TimeStrip.GAP_BUCKETS);
      lines.push({ text: `${track.label} ${near == null ? "no record" : TimeStrip.format(near) + unit}`, color: track.color });
    }
    ctx.font = CanvasFrame.font(600, 11);
    const boxW = Math.max(...lines.map((line) => ctx.measureText(line.text).width)) + 14;
    const boxH = lines.length * 15 + 8;
    const boxX = x + 8 + boxW > frame.w ? x - 8 - boxW : x + 8;
    ctx.fillStyle = "rgba(9, 13, 17, 0.94)";
    ctx.fillRect(boxX, frame.top + 2, boxW, boxH);
    ctx.strokeStyle = `rgba(${INK.ink}, 0.2)`;
    ctx.strokeRect(boxX + 0.5, frame.top + 2.5, boxW - 1, boxH - 1);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    lines.forEach((line, i) => {
      ctx.fillStyle = `rgb(${line.color})`;
      ctx.fillText(line.text, boxX + 7, frame.top + 7 + i * 15);
    });
  }

  static segments(points, gapS) {
    const runs = [];
    let run = [];
    for (const point of points) {
      if (!Number.isFinite(point.v)) {
        run = [];
        continue;
      }
      const last = run[run.length - 1];
      if (last && point.t - last.t > gapS) run = [];
      if (!run.length) runs.push(run);
      run.push(point);
    }
    return runs;
  }

  static trace(ctx, track, px, py, base, gapS) {
    for (const segment of TimeStrip.segments(track.points, gapS)) {
      if (segment.length > 1) {
        ctx.beginPath();
        ctx.moveTo(px(segment[0].t), base);
        segment.forEach((point) => ctx.lineTo(px(point.t), py(point.v)));
        ctx.lineTo(px(segment[segment.length - 1].t), base);
        ctx.closePath();
        ctx.fillStyle = `rgba(${track.color}, 0.1)`;
        ctx.fill();
      }
      ctx.beginPath();
      segment.forEach((point, i) => {
        if (i === 0) ctx.moveTo(px(point.t), py(point.v));
        else ctx.lineTo(px(point.t), py(point.v));
      });
      ctx.strokeStyle = `rgba(${track.color}, 0.92)`;
      ctx.lineWidth = 1.4;
      ctx.stroke();
      if (segment.length === 1) {
        ctx.beginPath();
        ctx.arc(px(segment[0].t), py(segment[0].v), 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgb(${track.color})`;
        ctx.fill();
      }
    }
  }
}

export class ActivityTimeline {
  static ROW_MIN = 30;
  static ROW_MAX = 80;
  static ROW_GAP = 8;
  static AXIS_H = 18;
  static LABEL_W = 48;

  constructor(canvas, readout) {
    this.canvas = canvas;
    this.readout = readout;
    this.idle = readout.textContent;
    this.state = null;
    this.geom = null;
    this.rows = [];
    canvas.addEventListener("pointermove", (event) => this.hover(event));
    canvas.addEventListener("pointerleave", () => this.blur());
  }

  render(state) {
    this.state = state;
    const rows = Math.max(1, state.gpus.length);
    const gaps = ActivityTimeline.AXIS_H + (rows - 1) * ActivityTimeline.ROW_GAP;
    this.canvas.parentElement.style.minHeight = `${gaps + rows * ActivityTimeline.ROW_MIN}px`;
    this.canvas.parentElement.style.maxHeight = `${gaps + rows * ActivityTimeline.ROW_MAX}px`;
    this.draw();
  }

  segments(bucket) {
    const segs = [];
    if (bucket.mine > 0) segs.push({ color: SERIES.mine, share: bucket.mine });
    if (bucket.others > 0) segs.push({ color: SERIES.others, share: bucket.others });
    if (!segs.length && bucket.busy > 0) segs.push({ color: SERIES.unknown, share: bucket.busy });
    return segs;
  }

  draw() {
    if (!this.state) return;
    const frame = CanvasFrame.fit(this.canvas);
    if (!frame) return;
    const { ctx, w, h } = frame;
    const state = this.state;
    const plotX = ActivityTimeline.LABEL_W;
    this.geom = { start: state.now - state.window_s, now: state.now, window: state.window_s, bucket: state.bucket_s, plotX, plotW: Math.max(1, w - plotX - 2) };
    this.rows = [];
    if (!state.gpus.length) {
      CanvasFrame.empty(ctx, w, h, "no GPU activity recorded yet");
      return;
    }

    const gaps = (state.gpus.length - 1) * ActivityTimeline.ROW_GAP;
    const rowH = Math.min(ActivityTimeline.ROW_MAX, Math.max(ActivityTimeline.ROW_MIN, (h - ActivityTimeline.AXIS_H - gaps) / state.gpus.length));
    state.gpus.forEach((gpu, index) => {
      const y = index * (rowH + ActivityTimeline.ROW_GAP);
      this.rows.push({ y, h: rowH, gpu });
      ctx.fillStyle = `rgba(${INK.ink}, 0.05)`;
      ctx.fillRect(plotX, y, this.geom.plotW, rowH);
      ctx.fillStyle = `rgb(${INK.faint})`;
      ctx.font = CanvasFrame.font(600, 11);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`gpu ${gpu.index}`, 0, y + rowH / 2);
      for (const bucket of gpu.buckets) {
        if (bucket.t + this.geom.bucket < this.geom.start) continue;
        const x0 = plotX + ((bucket.t - this.geom.start) / this.geom.window) * this.geom.plotW;
        const bw = Math.max(1.5, (this.geom.bucket / this.geom.window) * this.geom.plotW);
        const segs = this.segments(bucket);
        segs.forEach((seg, s) => {
          const sh = rowH / segs.length;
          ctx.fillStyle = `rgba(${seg.color}, ${(0.3 + 0.65 * seg.share).toFixed(3)})`;
          ctx.fillRect(x0, y + s * sh, bw, sh);
        });
      }
      ctx.strokeStyle = `rgba(${INK.ink}, 0.1)`;
      ctx.lineWidth = 1;
      ctx.strokeRect(plotX + 0.5, y + 0.5, this.geom.plotW - 1, rowH - 1);
    });
    this.axis(ctx, w, state.gpus.length * rowH + gaps);
  }

  hourTicks() {
    const g = this.geom;
    const ticks = [];
    for (let t = Math.ceil(g.start / 3600) * 3600; t <= g.now; t += 3600) {
      const date = new Date(t * 1000);
      if (date.getHours() % 3 === 0) ticks.push({ t, label: CanvasFrame.clock(date) });
    }
    return ticks;
  }

  axis(ctx, w, bottom) {
    const g = this.geom;
    ctx.font = CanvasFrame.font(600, 11);
    ctx.textBaseline = "top";
    const ticks = g.window > 86400 ? TimeStrip.ticks({ now: g.now, window_s: g.window }, g.plotW) : this.hourTicks();
    let last = -Infinity;
    for (const { t, label } of ticks) {
      const x = Math.round(g.plotX + ((t - g.start) / g.window) * g.plotW) + 0.5;
      ctx.strokeStyle = `rgba(${INK.ink}, 0.12)`;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      if (!label) continue;
      ctx.fillStyle = `rgb(${INK.faint})`;
      last = CanvasFrame.tick(ctx, label, x, w, bottom + 5, last, "center");
    }
  }

  hover(event) {
    if (!this.geom || !this.rows.length) return;
    const box = this.canvas.getBoundingClientRect();
    const x = event.clientX - box.left;
    const y = event.clientY - box.top;
    const row = this.rows.find((r) => y >= r.y && y <= r.y + r.h);
    if (!row || x < this.geom.plotX) {
      this.blur();
      return;
    }
    const when = this.geom.start + ((x - this.geom.plotX) / this.geom.plotW) * this.geom.window;
    const bucket = row.gpu.buckets.find((b) => when >= b.t && when < b.t + this.geom.bucket);
    const date = new Date(Math.round(when) * 1000);
    const clock = this.geom.window > 86400 ? `${CanvasFrame.day(date)} ${CanvasFrame.clock(date)}` : CanvasFrame.clock(date);
    if (!bucket) {
      this.readout.textContent = `${clock} · gpu ${row.gpu.index} · no record`;
      return;
    }
    const who = bucket.holders.join(", ");
    this.readout.textContent = bucket.busy > 0 ? `${clock} · gpu ${row.gpu.index} · busy ${Math.round(100 * bucket.busy)}% · util ${Math.round(bucket.util)}% · ${who || "owner unknown"}` : `${clock} · gpu ${row.gpu.index} · idle`;
  }

  blur() {
    this.readout.textContent = this.idle;
  }
}

export class HeatmapCalendar {
  static SLOT_H = 10;
  static ROW_GAP = 16;
  static LABEL_W = 48;
  static AXIS_H = 18;
  static SLOTS = 6;

  constructor(canvas, readout) {
    this.canvas = canvas;
    this.readout = readout;
    this.idle = readout.textContent;
    this.state = null;
    this.cells = [];
    canvas.addEventListener("pointermove", (event) => this.hover(event));
    canvas.addEventListener("pointerleave", () => this.blur());
  }

  render(state) {
    this.state = state;
    const rows = Math.max(1, state.gpus.length);
    const tall = HeatmapCalendar.AXIS_H + rows * (HeatmapCalendar.SLOTS * HeatmapCalendar.SLOT_H + HeatmapCalendar.ROW_GAP);
    this.canvas.style.height = `${Math.max(96, tall)}px`;
    this.draw();
  }

  days() {
    const first = new Date((this.state.now - this.state.window_s) * 1000);
    first.setHours(0, 0, 0, 0);
    const days = [];
    const cursor = new Date(first);
    while (cursor.getTime() / 1000 <= this.state.now) {
      days.push(cursor.getTime() / 1000);
      cursor.setDate(cursor.getDate() + 1);
    }
    return days;
  }

  slot(t) {
    return Math.min(HeatmapCalendar.SLOTS - 1, Math.floor(new Date(t * 1000).getHours() / 4));
  }

  draw() {
    if (!this.state) return;
    const frame = CanvasFrame.fit(this.canvas);
    if (!frame) return;
    const { ctx, w, h } = frame;
    this.cells = [];
    const gpus = this.state.gpus;
    if (!gpus.length) {
      CanvasFrame.empty(ctx, w, h, "no GPU activity recorded yet");
      return;
    }

    const days = this.days();
    const plotX = HeatmapCalendar.LABEL_W;
    const plotW = Math.max(1, w - plotX - 2);
    const cellW = plotW / days.length;
    const slotH = HeatmapCalendar.SLOT_H;
    const rowH = HeatmapCalendar.SLOTS * slotH;
    gpus.forEach((gpu, row) => {
      const y0 = row * (rowH + HeatmapCalendar.ROW_GAP);
      ctx.fillStyle = `rgb(${INK.faint})`;
      ctx.font = CanvasFrame.font(600, 11);
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`gpu ${gpu.index}`, 0, y0 + rowH / 2);
      ctx.fillStyle = `rgba(${INK.ink}, 0.04)`;
      ctx.fillRect(plotX, y0, plotW, rowH);
      for (const bucket of gpu.buckets) {
        const day = Math.floor((bucket.t - days[0]) / 86400);
        if (day < 0 || day >= days.length) continue;
        const x = plotX + day * cellW;
        const y = y0 + this.slot(bucket.t) * slotH;
        if (bucket.busy > 0) {
          const color = bucket.others > bucket.mine ? SERIES.others : SERIES.mine;
          ctx.fillStyle = `rgba(${color}, ${(0.14 + 0.8 * bucket.busy).toFixed(3)})`;
          ctx.fillRect(x + 0.5, y + 0.5, Math.max(1, cellW - 1), slotH - 1);
        }
        this.cells.push({ x, y, w: cellW, h: slotH, gpu, bucket });
      }
      ctx.strokeStyle = `rgba(${INK.ink}, 0.1)`;
      ctx.lineWidth = 1;
      ctx.strokeRect(plotX + 0.5, y0 + 0.5, plotW - 1, rowH - 1);
    });

    const bottom = gpus.length * (rowH + HeatmapCalendar.ROW_GAP) - HeatmapCalendar.ROW_GAP;
    ctx.font = CanvasFrame.font(600, 11);
    ctx.textBaseline = "top";
    const every = Math.max(1, Math.ceil((days.length * 46) / plotW));
    let last = -Infinity;
    days.forEach((t, index) => {
      if (index % every !== 0) return;
      const x = plotX + index * cellW;
      ctx.fillStyle = `rgb(${INK.faint})`;
      last = CanvasFrame.tick(ctx, CanvasFrame.day(new Date(t * 1000)), x, w, bottom + 5, last, "left");
    });
  }

  figureDays() {
    const gpus = this.state.gpus;
    const first = this.days();
    const days = first.map(() => Array.from({ length: HeatmapCalendar.SLOTS }, () => ({ mine: 0, others: 0 })));
    for (const gpu of gpus) {
      for (const bucket of gpu.buckets) {
        const day = Math.floor((bucket.t - first[0]) / 86400);
        if (day < 0 || day >= days.length) continue;
        const slot = days[day][this.slot(bucket.t)];
        slot.mine += bucket.mine / gpus.length;
        slot.others += bucket.others / gpus.length;
      }
    }
    return days.map((slots) => ({ slots: slots.map((slot) => ({ mine: Math.round(slot.mine * 100) / 100, others: Math.round(slot.others * 100) / 100 })) }));
  }

  hover(event) {
    if (!this.cells.length) return;
    const box = this.canvas.getBoundingClientRect();
    const x = event.clientX - box.left;
    const y = event.clientY - box.top;
    const cell = this.cells.find((c) => x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h);
    if (!cell) {
      this.blur();
      return;
    }
    const start = new Date(cell.bucket.t * 1000);
    start.setHours(this.slot(cell.bucket.t) * 4, 0, 0, 0);
    const end = new Date(start.getTime() + 4 * 3600 * 1000);
    const label = `${CanvasFrame.day(start)} ${CanvasFrame.clock(start)} to ${CanvasFrame.clock(end)}`;
    const who = cell.bucket.holders.join(", ");
    this.readout.textContent = cell.bucket.busy > 0 ? `${label} · gpu ${cell.gpu.index} · busy ${Math.round(100 * cell.bucket.busy)}% · you ${Math.round(100 * cell.bucket.mine)}% · others ${Math.round(100 * cell.bucket.others)}%${who ? ` · ${who}` : ""}` : `${label} · gpu ${cell.gpu.index} · idle`;
  }

  blur() {
    this.readout.textContent = this.idle;
  }
}
