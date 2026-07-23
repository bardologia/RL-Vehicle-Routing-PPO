"use strict";

class LineChart {
  constructor(host, options) {
    this.host = host;
    this.options = options || {};
    this.series = [];
    this.width = 0;
    this.height = this.options.height || 200;
    this.pad = { top: 10, right: 12, bottom: 24, left: 46 };

    this.root = document.createElement("div");
    this.root.className = "linechart";
    this.tip = document.createElement("div");
    this.tip.className = "linechart__tip";
    this.root.appendChild(this.tip);
    this.host.appendChild(this.root);

    this._resize = () => this._render();
    window.addEventListener("resize", this._resize);
  }

  static palette(i) {
    const colors = ["#1d4fd8", "#0f766e", "#b91c1c", "#a16207", "#7c3aed", "#0369a1", "#be185d", "#4d7c0f", "#c2410c", "#334155"];
    return colors[i % colors.length];
  }

  destroy() {
    window.removeEventListener("resize", this._resize);
    this.root.remove();
  }

  setSeries(series) {
    this.series = series.filter((s) => s.points.length > 0);
    this._render();
  }

  _extent() {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const s of this.series) {
      for (const p of s.points) {
        if (p.x < x0) x0 = p.x;
        if (p.x > x1) x1 = p.x;
        if (p.y < y0) y0 = p.y;
        if (p.y > y1) y1 = p.y;
      }
    }
    if (x0 === x1) { x0 -= 0.5; x1 += 0.5; }
    if (y0 === y1) { y0 -= 0.5 || 0.5; y1 += 0.5; }
    const yPad = (y1 - y0) * 0.06;
    return { x0, x1, y0: y0 - yPad, y1: y1 + yPad };
  }

  _ticks(lo, hi, count) {
    const span = hi - lo;
    const step = Math.pow(10, Math.floor(Math.log10(span / count)));
    const err = (span / count) / step;
    const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const inc = step * mult;
    const ticks = [];
    for (let v = Math.ceil(lo / inc) * inc; v <= hi + inc * 1e-9; v += inc) ticks.push(v);
    return ticks;
  }

  _fmt(value) {
    const abs = Math.abs(value);
    if (abs >= 1e6) return (value / 1e6).toFixed(1) + "M";
    if (abs >= 1e4) return (value / 1e3).toFixed(0) + "k";
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 1) return +value.toFixed(2) + "";
    if (abs === 0) return "0";
    return +value.toFixed(4) + "";
  }

  _render() {
    this.root.querySelectorAll("svg").forEach((el) => el.remove());

    this.width = this.root.clientWidth || this.host.clientWidth || 400;
    const W = this.width, H = this.height, P = this.pad;
    const iw = W - P.left - P.right, ih = H - P.top - P.bottom;
    if (iw <= 0 || !this.series.length) return;

    const ext = this._extent();
    const sx = (x) => P.left + ((x - ext.x0) / (ext.x1 - ext.x0)) * iw;
    const sy = (y) => P.top + ih - ((y - ext.y0) / (ext.y1 - ext.y0)) * ih;
    this._sx = sx; this._ext = ext;

    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);

    const axis = document.createElementNS(ns, "g");
    axis.setAttribute("class", "axis");

    for (const t of this._ticks(ext.y0, ext.y1, 4)) {
      const line = document.createElementNS(ns, "line");
      line.setAttribute("x1", P.left); line.setAttribute("x2", W - P.right);
      line.setAttribute("y1", sy(t)); line.setAttribute("y2", sy(t));
      line.setAttribute("class", "gridline");
      svg.appendChild(line);

      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", P.left - 6);
      label.setAttribute("y", sy(t) + 3);
      label.setAttribute("text-anchor", "end");
      label.textContent = this._fmt(t);
      axis.appendChild(label);
    }

    for (const t of this._ticks(ext.x0, ext.x1, 5)) {
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", sx(t));
      label.setAttribute("y", H - 6);
      label.setAttribute("text-anchor", "middle");
      label.textContent = this._fmt(t);
      axis.appendChild(label);
    }

    svg.appendChild(axis);

    this.series.forEach((s, i) => {
      const path = document.createElementNS(ns, "path");
      const d = s.points.map((p, j) => `${j ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
      path.setAttribute("d", d);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", s.color || LineChart.palette(i));
      path.setAttribute("stroke-width", "1.6");
      path.setAttribute("stroke-linejoin", "round");
      svg.appendChild(path);
    });

    this.cursor = document.createElementNS(ns, "line");
    this.cursor.setAttribute("y1", P.top);
    this.cursor.setAttribute("y2", P.top + ih);
    this.cursor.setAttribute("stroke", "#9aa196");
    this.cursor.setAttribute("stroke-width", "1");
    this.cursor.setAttribute("visibility", "hidden");
    svg.appendChild(this.cursor);

    svg.addEventListener("mousemove", (e) => this._hover(e, svg));
    svg.addEventListener("mouseleave", () => {
      this.tip.style.display = "none";
      this.cursor.setAttribute("visibility", "hidden");
    });

    this.root.appendChild(svg);
  }

  _hover(e, svg) {
    const rect = svg.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * this.width;
    const x = this._ext.x0 + ((px - this.pad.left) / (this.width - this.pad.left - this.pad.right)) * (this._ext.x1 - this._ext.x0);

    const lines = [];
    let anchorX = null;
    for (const s of this.series) {
      let best = null, bestDist = Infinity;
      for (const p of s.points) {
        const d = Math.abs(p.x - x);
        if (d < bestDist) { bestDist = d; best = p; }
      }
      if (best) {
        lines.push(`${s.name}: ${this._fmt(best.y)}`);
        anchorX = best.x;
      }
    }
    if (!lines.length) return;

    this.cursor.setAttribute("x1", this._sx(anchorX));
    this.cursor.setAttribute("x2", this._sx(anchorX));
    this.cursor.setAttribute("visibility", "visible");

    this.tip.style.display = "block";
    this.tip.textContent = `step ${this._fmt(anchorX)} · ${lines.join(" · ")}`;
    const tipX = Math.min(Math.max(px + 12, 0), this.width - 140);
    this.tip.style.left = `${(tipX / this.width) * 100}%`;
    this.tip.style.top = "6px";
  }
}

window.LineChart = LineChart;
