import { esc, fmtFixed, fmtSigned, isNum } from "../format.js";
import { tooltip, Tooltip } from "./tooltip.js";

export class Scale {
  constructor(domain, range, log) {
    this.log = Boolean(log);
    this.d0 = this.log ? Math.log10(domain[0]) : domain[0];
    this.d1 = this.log ? Math.log10(domain[1]) : domain[1];
    if (this.d1 === this.d0) this.d1 = this.d0 + 1;
    this.r0 = range[0];
    this.r1 = range[1];
  }

  map(value) {
    const v = this.log ? Math.log10(value) : value;
    return this.r0 + ((v - this.d0) / (this.d1 - this.d0)) * (this.r1 - this.r0);
  }

  invert(pixel) {
    const v = this.d0 + ((pixel - this.r0) / (this.r1 - this.r0)) * (this.d1 - this.d0);
    return this.log ? 10 ** v : v;
  }

  static extent(values, includeZero) {
    const finite = values.filter(isNum);
    if (!finite.length) return [0, 1];
    let lo = Math.min(...finite);
    let hi = Math.max(...finite);
    if (includeZero) {
      lo = Math.min(lo, 0);
      hi = Math.max(hi, 0);
    }
    if (lo === hi) {
      lo -= 1;
      hi += 1;
    }
    return [lo, hi];
  }

  static ticks(lo, hi, count = 5) {
    const span = hi - lo;
    const raw = span / Math.max(1, count);
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * power).find((candidate) => span / candidate <= count);
    const out = [];
    for (let tick = Math.ceil(lo / step) * step; tick <= hi + step * 1e-9; tick += step) out.push(Math.abs(tick) < step * 1e-9 ? 0 : tick);
    return out;
  }

  static tickText(tick, ticks) {
    const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : Math.abs(tick) || 1;
    const digits = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
    if (Math.abs(tick) >= 10000) return `${(tick / 1000).toFixed(step >= 1000 ? 0 : 1)}k`;
    return tick.toFixed(digits);
  }
}

class Frame {
  constructor(width, height, margin) {
    this.width = width;
    this.height = height;
    this.m = margin;
    this.x0 = margin.left;
    this.x1 = width - margin.right;
    this.y0 = height - margin.bottom;
    this.y1 = margin.top;
  }

  axes(x, y, xTicks, yTicks, xLabel, yLabel) {
    const parts = [];
    for (const tick of yTicks) {
      const py = y.map(tick);
      parts.push(`<line class="ch-grid" x1="${this.x0}" x2="${this.x1}" y1="${py}" y2="${py}"></line><text class="ch-tick" x="${this.x0 - 6}" y="${py + 4}" text-anchor="end">${Scale.tickText(tick, yTicks)}</text>`);
    }
    for (const tick of xTicks) {
      const px = x.map(tick);
      parts.push(`<line class="ch-tickmark" x1="${px}" x2="${px}" y1="${this.y0}" y2="${this.y0 + 4}"></line><text class="ch-tick" x="${px}" y="${this.y0 + 16}" text-anchor="middle">${Scale.tickText(tick, xTicks)}</text>`);
    }
    parts.push(`<line class="ch-axis" x1="${this.x0}" x2="${this.x1}" y1="${this.y0}" y2="${this.y0}"></line>`);
    if (xLabel) parts.push(`<text class="ch-label" x="${(this.x0 + this.x1) / 2}" y="${this.height - 4}" text-anchor="middle">${esc(xLabel)}</text>`);
    if (yLabel) parts.push(`<text class="ch-label" x="${this.x0 - this.m.left + 2}" y="${this.y1 - 8}" text-anchor="start">${esc(yLabel)}</text>`);
    return parts.join("");
  }

  open(extra) {
    return `<svg class="chart" viewBox="0 0 ${this.width} ${this.height}" width="${this.width}" height="${this.height}" role="img"${extra || ""}>`;
  }
}

function measure(container, fallback) {
  return Math.max(160, Math.floor(container.clientWidth) || fallback);
}

function emptyFrame(container, height, text) {
  container.innerHTML = `<div class="chart-empty" style="height:${height}px">${esc(text)}</div>`;
}

export class LineChart {
  constructor(container, options = {}) {
    this.container = container;
    this.options = options;
    this.data = [];
    container.addEventListener("pointermove", (event) => this.hover(event));
    container.addEventListener("pointerleave", () => this.leave());
  }

  static smooth(values, alpha) {
    if (!alpha) return values;
    const out = [];
    let state = null;
    for (const value of values) {
      state = state === null ? value : alpha * state + (1 - alpha) * value;
      out.push(state);
    }
    return out;
  }

  setOptions(options) {
    Object.assign(this.options, options);
    this.render(this.data);
  }

  render(series) {
    this.data = series;
    const { height = 200, xLabel = "step", yLabel = "", smoothing = 0, log = false, emptyText = "not logged" } = this.options;
    const drawn = series.filter((entry) => entry.values.length).map((entry) => ({ ...entry, shown: LineChart.smooth(entry.values, smoothing) }));
    if (!drawn.length) {
      emptyFrame(this.container, height, emptyText);
      this.state = null;
      return;
    }

    const width = measure(this.container, 480);
    const labelRoom = drawn.length > 1 ? Math.min(140, 12 + 7 * Math.max(...drawn.map((entry) => entry.label.length))) : 16;
    const frame = new Frame(width, height, { left: 52, right: labelRoom, top: 24, bottom: 34 });
    const allX = drawn.flatMap((entry) => entry.steps);
    let allY = drawn.flatMap((entry) => entry.shown).filter(isNum);
    const useLog = log && allY.every((value) => value > 0);
    const [xLo, xHi] = Scale.extent(allX);
    let [yLo, yHi] = Scale.extent(allY);
    if (!useLog) {
      const pad = (yHi - yLo) * 0.06;
      yLo -= pad;
      yHi += pad;
    }
    const x = new Scale([xLo, xHi], [frame.x0, frame.x1]);
    const y = new Scale([yLo, yHi], [frame.y0, frame.y1], useLog);
    const yTicks = useLog ? LineChart.logTicks(yLo, yHi) : Scale.ticks(yLo, yHi, 4);

    const lines = drawn.map((entry, index) => {
      const points = entry.steps.map((step, i) => (isNum(entry.shown[i]) ? `${x.map(step).toFixed(1)},${y.map(entry.shown[i]).toFixed(1)}` : null)).filter((point) => point);
      const lastIndex = entry.shown.length - 1;
      const endY = y.map(entry.shown[lastIndex]);
      const label = drawn.length > 1 ? `<text class="ch-direct" x="${frame.x1 + 6}" y="${endY + 4}" data-end="${index}">${esc(entry.label)}</text>` : "";
      return `<polyline class="ch-line ch-line--${entry.style || "solid"} ${entry.className || ""}" points="${points.join(" ")}"></polyline>${label}`;
    }).join("");

    this.container.innerHTML = `${frame.open()}${frame.axes(x, y, Scale.ticks(xLo, xHi, 5), yTicks, xLabel, yLabel)}${lines}<line class="ch-cross" y1="${frame.y1}" y2="${frame.y0}" x1="0" x2="0" visibility="hidden"></line></svg>`;
    LineChart.spreadLabels(this.container, frame);
    this.state = { frame, x, drawn };
  }

  static logTicks(lo, hi) {
    const out = [];
    for (let power = Math.floor(Math.log10(lo)); power <= Math.ceil(Math.log10(hi)); power += 1) {
      if (10 ** power >= lo && 10 ** power <= hi) out.push(10 ** power);
    }
    return out.length ? out : [lo, hi];
  }

  static spreadLabels(container, frame) {
    const labels = [...container.querySelectorAll(".ch-direct")].sort((a, b) => Number(a.getAttribute("y")) - Number(b.getAttribute("y")));
    let floor = frame.y1;
    for (const label of labels) {
      const wanted = Math.max(floor, Number(label.getAttribute("y")));
      label.setAttribute("y", wanted);
      floor = wanted + 13;
    }
  }

  hover(event) {
    if (!this.state) return;
    const { frame, x, drawn } = this.state;
    const svg = this.container.querySelector("svg");
    const box = svg.getBoundingClientRect();
    const px = (event.clientX - box.left) * (frame.width / box.width);
    if (px < frame.x0 || px > frame.x1) {
      this.leave();
      return;
    }
    const step = x.invert(px);
    const rows = [];
    let snapped = null;
    for (const entry of drawn) {
      let best = 0;
      for (let i = 1; i < entry.steps.length; i += 1) if (Math.abs(entry.steps[i] - step) < Math.abs(entry.steps[best] - step)) best = i;
      if (snapped === null) snapped = entry.steps[best];
      rows.push([esc(entry.label), fmtFixed(entry.shown[best], 4)]);
    }
    const cross = svg.querySelector(".ch-cross");
    cross.setAttribute("x1", x.map(snapped));
    cross.setAttribute("x2", x.map(snapped));
    cross.setAttribute("visibility", "visible");
    tooltip.show(Tooltip.rows(`${esc(this.options.xLabel || "step")} ${snapped}`, rows), event.clientX, event.clientY);
    if (this.options.onHover) this.options.onHover(snapped);
  }

  leave() {
    tooltip.hide();
    const cross = this.container.querySelector(".ch-cross");
    if (cross) cross.setAttribute("visibility", "hidden");
  }
}

export class BarChart {
  static html(rows, { digits = 2, unit = "", max } = {}) {
    const extent = max || Math.max(1e-9, ...rows.map((row) => Math.abs(row.value)).filter(isNum));
    return `<div class="sbar" role="table">${rows.map((row) => {
      const share = isNum(row.value) ? Math.min(1, Math.abs(row.value) / extent) * 50 : 0;
      const side = row.value >= 0 ? "gain" : "loss";
      const style = row.value >= 0 ? `left:50%;width:${share.toFixed(2)}%` : `right:50%;width:${share.toFixed(2)}%`;
      return `
        <div class="sbar__row" role="row"${row.attrs || ""}>
          <div class="sbar__label">${row.labelHtml || esc(row.label)}</div>
          <div class="sbar__track"><i class="sbar__zero"></i><i class="sbar__fill sbar__fill--${side}${row.hatched ? " is-hatched" : ""}" style="${style}"></i></div>
          <div class="sbar__value ${isNum(row.value) && Math.abs(row.value) > 5e-7 ? side : ""}">${fmtSigned(row.value, digits)}${unit ? ` ${esc(unit)}` : ""}</div>
          ${row.note ? `<div class="sbar__note">${row.note}</div>` : ""}
        </div>`;
    }).join("")}</div>`;
  }
}

export class IntervalPlot {
  static render(container, rows, { xLabel = "", height, zeroLine = false, digits = 2 } = {}) {
    const width = measure(container, 420);
    const rowHeight = 26;
    const total = height || rows.length * rowHeight + 46;
    const labelWidth = Math.min(150, 16 + 7 * Math.max(...rows.map((row) => row.label.length)));
    const frame = new Frame(width, total, { left: labelWidth, right: 64, top: 6, bottom: 34 });
    const values = rows.flatMap((row) => [row.mean, row.lo, row.hi]).filter(isNum);
    let [lo, hi] = Scale.extent(values, zeroLine);
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    const x = new Scale([lo, hi], [frame.x0, frame.x1]);
    const ticks = Scale.ticks(lo, hi, 5);

    const grid = ticks.map((tick) => `<line class="ch-grid" x1="${x.map(tick)}" x2="${x.map(tick)}" y1="${frame.y1}" y2="${frame.y0}"></line><text class="ch-tick" x="${x.map(tick)}" y="${frame.y0 + 16}" text-anchor="middle">${Scale.tickText(tick, ticks)}</text>`).join("");
    const zero = zeroLine && lo < 0 && hi > 0 ? `<line class="ch-zero" x1="${x.map(0)}" x2="${x.map(0)}" y1="${frame.y1}" y2="${frame.y0}"></line>` : "";
    const marks = rows.map((row, index) => {
      const cy = frame.y1 + rowHeight * index + rowHeight / 2;
      const whisker = isNum(row.lo) && isNum(row.hi) ? `<line class="ch-whisker ch-agent--${row.style || "model"}" x1="${x.map(row.lo)}" x2="${x.map(row.hi)}" y1="${cy}" y2="${cy}"></line><line class="ch-cap" x1="${x.map(row.lo)}" x2="${x.map(row.lo)}" y1="${cy - 4}" y2="${cy + 4}"></line><line class="ch-cap" x1="${x.map(row.hi)}" x2="${x.map(row.hi)}" y1="${cy - 4}" y2="${cy + 4}"></line>` : "";
      const dot = isNum(row.mean) ? `<circle class="ch-dot ch-dot--${row.style || "model"}" cx="${x.map(row.mean)}" cy="${cy}" r="4.5"></circle>` : "";
      const tip = Tooltip.rows(esc(row.label), [["mean", fmtFixed(row.mean, digits)], isNum(row.lo) ? ["95% interval", `${fmtFixed(row.lo, digits)} to ${fmtFixed(row.hi, digits)}`] : null, row.n ? ["n", row.n] : null]);
      return `<g data-tip="${esc(tip)}"><rect class="ch-hit" x="0" y="${cy - rowHeight / 2}" width="${width}" height="${rowHeight}"></rect><text class="ch-rowlabel" x="${frame.x0 - 10}" y="${cy + 4}" text-anchor="end">${esc(row.label)}</text>${whisker}${dot}<text class="ch-value" x="${frame.x1 + 8}" y="${cy + 4}">${fmtFixed(row.mean, digits)}</text></g>`;
    }).join("");

    container.innerHTML = `${frame.open()}${grid}${zero}${marks}<text class="ch-label" x="${(frame.x0 + frame.x1) / 2}" y="${total - 4}" text-anchor="middle">${esc(xLabel)}</text></svg>`;
    bindTips(container);
  }
}

export class Scatter {
  static render(container, points, { xLabel = "", yLabel = "", height = 280, diagonal = true, bins = 0, onClick, digits = 2 } = {}) {
    const usable = points.filter((point) => isNum(point.x) && isNum(point.y));
    if (!usable.length) {
      emptyFrame(container, height, "no points");
      return;
    }
    const width = measure(container, 420);
    const frame = new Frame(width, height, { left: 52, right: 16, top: 24, bottom: 36 });
    let [lo, hi] = Scale.extent(usable.flatMap((point) => [point.x, point.y]));
    const pad = (hi - lo) * 0.06;
    lo -= pad;
    hi += pad;
    const x = new Scale([lo, hi], [frame.x0, frame.x1]);
    const y = new Scale([lo, hi], [frame.y0, frame.y1]);
    const ticks = Scale.ticks(lo, hi, 5);
    const line = diagonal ? `<line class="ch-diagonal" x1="${x.map(lo)}" y1="${y.map(lo)}" x2="${x.map(hi)}" y2="${y.map(hi)}"></line>` : "";

    let binned = "";
    if (bins > 1) {
      const [xLo, xHi] = Scale.extent(usable.map((point) => point.x));
      const sums = Array.from({ length: bins }, () => ({ x: 0, y: 0, n: 0 }));
      for (const point of usable) {
        const slot = sums[Math.min(bins - 1, Math.floor(((point.x - xLo) / (xHi - xLo)) * bins))];
        slot.x += point.x;
        slot.y += point.y;
        slot.n += 1;
      }
      const path = sums.filter((slot) => slot.n).map((slot) => `${x.map(slot.x / slot.n).toFixed(1)},${y.map(slot.y / slot.n).toFixed(1)}`).join(" ");
      binned = `<polyline class="ch-line ch-binned" points="${path}"></polyline>`;
    }

    const dots = usable.map((point, index) => `<circle class="ch-point${onClick ? " is-link" : ""}" cx="${x.map(point.x).toFixed(1)}" cy="${y.map(point.y).toFixed(1)}" r="4" data-index="${index}" data-tip="${esc(Tooltip.rows(esc(point.title || ""), [[esc(xLabel), fmtFixed(point.x, digits)], [esc(yLabel), fmtFixed(point.y, digits)]]))}"${onClick ? ` tabindex="0" role="link"` : ""}></circle>`).join("");

    container.innerHTML = `${frame.open()}${frame.axes(x, y, ticks, ticks, xLabel, yLabel)}${line}${binned}${dots}<text class="ch-note" x="${frame.x1}" y="${frame.y1 - 8}" text-anchor="end">n = ${usable.length}</text></svg>`;
    bindTips(container);
    if (onClick) {
      container.onclick = (event) => {
        const dot = event.target.closest(".ch-point");
        if (dot) onClick(usable[Number(dot.dataset.index)]);
      };
      container.onkeydown = (event) => {
        const dot = event.target.closest(".ch-point");
        if (dot && event.key === "Enter") onClick(usable[Number(dot.dataset.index)]);
      };
    }
  }
}

export class Histogram {
  static render(container, values, { xLabel = "", bins = 10, height = 180, domain, integer = false } = {}) {
    const usable = values.filter(isNum);
    if (!usable.length) {
      emptyFrame(container, height, "no values");
      return;
    }
    const width = measure(container, 360);
    const frame = new Frame(width, height, { left: 44, right: 12, top: 24, bottom: 36 });
    const [lo, hi] = domain || Scale.extent(usable);
    const count = integer ? Math.round(hi - lo) + 1 : bins;
    const span = integer ? count : hi - lo;
    const start = integer ? lo - 0.5 : lo;
    const counts = new Array(count).fill(0);
    for (const value of usable) counts[Math.max(0, Math.min(count - 1, Math.floor(((value - start) / span) * count)))] += 1;
    const x = new Scale([start, start + span], [frame.x0, frame.x1]);
    const y = new Scale([0, Math.max(...counts)], [frame.y0, frame.y1]);
    const yTicks = Scale.ticks(0, Math.max(...counts), 3).filter((tick) => Number.isInteger(tick));
    const xTicks = integer ? counts.map((_, index) => lo + index) : Scale.ticks(lo, hi, 5);
    const bars = counts.map((value, index) => {
      const left = x.map(start + (span * index) / count) + 1;
      const right = x.map(start + (span * (index + 1)) / count) - 1;
      const from = start + (span * index) / count;
      const label = integer ? String(lo + index) : `${fmtFixed(from, 2)} to ${fmtFixed(from + span / count, 2)}`;
      return `<rect class="ch-bar" x="${left}" y="${y.map(value)}" width="${Math.max(1, right - left)}" height="${frame.y0 - y.map(value)}" data-tip="${esc(Tooltip.rows(esc(xLabel), [[label, `${value} of ${usable.length}`]]))}"></rect>`;
    }).join("");
    container.innerHTML = `${frame.open()}${frame.axes(x, y, xTicks, yTicks, xLabel, "count")}${bars}<text class="ch-note" x="${frame.x1}" y="${frame.y1 - 8}" text-anchor="end">n = ${usable.length}</text></svg>`;
    bindTips(container);
  }
}

export class StackBar {
  static NAMES = ["INSERT", "REMOVE", "DO_NOTHING"];

  static html(counts, { showWords = true } = {}) {
    const total = counts.reduce((sum, value) => sum + value, 0);
    if (!total) return `<span class="faint">no decisions</span>`;
    const segments = counts.map((value, index) => (value ? `<i class="stack__seg stack__seg--op${index}" style="flex:${value}" title="${StackBar.NAMES[index]} ${value} of ${total}"></i>` : "")).join("");
    const words = showWords ? `<span class="stack__words">${counts.map((value, index) => `<span><i class="opmark opmark--op${index}"></i>${StackBar.NAMES[index]} ${((value / total) * 100).toFixed(0)}%</span>`).join("")}</span>` : "";
    return `<span class="stack"><span class="stack__bar">${segments}</span>${words}</span>`;
  }
}

export class Sparkline {
  static svg(values, { width = 80, height = 20, zero = false } = {}) {
    const usable = values.filter(isNum);
    if (usable.length < 2) return "";
    const [lo, hi] = Scale.extent(usable, zero);
    const x = new Scale([0, values.length - 1], [1, width - 1]);
    const y = new Scale([lo, hi], [height - 2, 2]);
    const points = values.map((value, index) => (isNum(value) ? `${x.map(index).toFixed(1)},${y.map(value).toFixed(1)}` : null)).filter((point) => point).join(" ");
    const base = zero ? `<line class="ch-zero" x1="0" x2="${width}" y1="${y.map(0)}" y2="${y.map(0)}"></line>` : "";
    return `<svg class="spark" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" aria-hidden="true">${base}<polyline class="ch-line" points="${points}"></polyline></svg>`;
  }
}

export class StepProfile {
  static svg(profile, capacity, { width = 96, height = 22 } = {}) {
    if (!profile || profile.length < 2) return "";
    const top = Math.max(capacity || 0, ...profile, 1);
    const x = new Scale([0, profile.length - 1], [1, width - 1]);
    const y = new Scale([0, top], [height - 2, 3]);
    let path = `M${x.map(0).toFixed(1)},${y.map(profile[0]).toFixed(1)}`;
    for (let index = 1; index < profile.length; index += 1) path += ` H${x.map(index).toFixed(1)} V${y.map(profile[index]).toFixed(1)}`;
    const cap = capacity ? `<line class="ch-capline" x1="0" x2="${width}" y1="${y.map(capacity)}" y2="${y.map(capacity)}"></line>` : "";
    return `<svg class="spark spark--step" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="Load after each stop, peak ${Math.max(...profile)} of ${capacity}">${cap}<path class="ch-line" d="${path}"></path></svg>`;
  }
}

export function bindTips(container) {
  if (container.dataset.tipsBound) return;
  container.dataset.tipsBound = "1";
  tooltip.bind(container, "[data-tip]", (target) => target.dataset.tip);
}
