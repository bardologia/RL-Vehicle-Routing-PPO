import { esc, fmtFixed, fmtSigned, icon, signClass } from "../format.js";
import { statusChip } from "../ui/status.js";
import { tooltip, Tooltip } from "../ui/tooltip.js";
import { Agents, OPERATORS, Sentences, TERMS } from "./encoding.js";

export class Timeline {
  static LABEL_W = 208;
  static VALUE_W = 64;
  static HEAD_H = 26;
  static GAP = 10;
  static LANE_H = 28;
  static LANE_H_SHORT = 20;
  static COMPACT = "(max-width: 1500px), (max-height: 860px)";

  constructor(container, { onSelect }) {
    this.container = container;
    this.onSelect = onSelect;
    this.state = null;
    container.classList.add("timeline");
    container.addEventListener("click", (event) => {
      const cell = event.target instanceof Element ? event.target.closest("[data-seq]") : null;
      if (cell) {
        this.onSelect(cell.dataset.lane, Number(cell.dataset.seq));
        return;
      }
      const lane = event.target instanceof Element ? event.target.closest("[data-lanehead]") : null;
      if (lane) this.onSelect(lane.dataset.lanehead, null);
    });
    tooltip.bind(container, "[data-seq]", (target) => this.describe(target.dataset.lane, Number(target.dataset.seq)));
    this.observer = new ResizeObserver(() => {
      if (this.state) this.render(this.state);
    });
    this.observer.observe(container);
  }

  static workLeft(frame) {
    return frame.decision.is_commit && frame.plan.kpis.unassigned > 0 && frame.decision.edits_remaining > 0;
  }

  describe(laneKey, seq) {
    const lane = this.state.lanes.find((entry) => entry.key === laneKey);
    const frame = lane.trace.frame(seq);
    if (!frame) return null;
    if (frame.kind === "round_open") {
      const open = frame.round_open;
      return Tooltip.rows(`${esc(Agents.label(laneKey))}: round ${frame.round + 1} opens`, [
        ["gap", `${Math.round(open.gap_s / 60)} min`],
        ["holding charged", fmtSigned(open.holding_charged, 2)],
        ["executed", esc(Sentences.executed(open.executed))],
        ["event", esc(Sentences.event(open.event))],
      ]);
    }
    const decision = frame.decision;
    const flags = [];
    if (decision.wasted) flags.push("WASTED edit");
    if (Timeline.workLeft(frame)) flags.push("commit with unassigned jobs and edits remaining");
    return Tooltip.rows(`${esc(Agents.label(laneKey))}: ${esc(Sentences.action(decision.action))}`, [
      ["round, edit", `${frame.round + 1}, ${frame.edit + 1}`],
      ...TERMS.map((term) => [term.label.toLowerCase(), fmtSigned(decision.rewards[term.key], 2)]),
      ["step reward", fmtSigned(decision.reward, 2)],
      ["cumulative", fmtSigned(decision.cumulative, 2)],
      flags.length ? ["flag", esc(flags.join("; "))] : null,
    ]);
  }

  layout(lanes, width) {
    const slots = [];
    for (const lane of lanes) {
      for (const round of lane.trace.rounds()) {
        if (round) slots[round.round] = Math.max(slots[round.round] || 4, 1 + round.decisions.length);
      }
    }
    const total = slots.reduce((sum, value) => sum + (value || 0), 0);
    const room = width - Timeline.LABEL_W - Timeline.VALUE_W - Timeline.GAP * Math.max(0, slots.length - 1);
    const cell = Math.max(9, Math.min(64, room / Math.max(1, total)));
    const roundX = [];
    let x = Timeline.LABEL_W;
    slots.forEach((count, round) => {
      roundX[round] = x;
      x += (count || 0) * cell + Timeline.GAP;
    });
    return { slots, cell, roundX, end: x - Timeline.GAP };
  }

  headerCell(round, lanes, divergence, selected, x, width) {
    const open = lanes.map((lane) => lane.trace.roundOpen(round)).find((frame) => frame);
    if (!open) return "";
    const selectedLane = lanes.find((lane) => lane.key === selected.lane);
    const own = (selectedLane && selectedLane.trace.roundOpen(round)) || open;
    const info = own.round_open;
    const parts = [`<b>Round ${round + 1}</b>`];
    if (round > 0) parts.push(`gap ${Math.round(info.gap_s / 60)} min`);
    parts.push(esc(Sentences.event(info.event)));
    if (round > 0) parts.push(esc(Sentences.executed(info.executed)));
    const diverges = divergence.filter((entry) => entry.round === round);
    const mark = diverges.length ? `<span class="timeline__diverge" title="${esc(diverges.map((entry) => `${entry.kind === "event_ids" ? "event ids" : "pickup outcomes"} differ between ${entry.agents.map(Agents.label).join(", ")}`).join("; "))}">${icon("warning")}differs</span>` : "";
    return `<div class="timeline__round" style="left:${x}px;width:${width}px" title="${esc(parts.join(", ").replace(/<\/?b>/g, ""))}">${mark}<span>${parts.join('<i class="timeline__sep"></i>')}</span></div>`;
  }

  render(state) {
    this.state = state;
    const { lanes, selected, divergence } = state;
    const width = Math.max(600, Math.floor(this.container.clientWidth));
    const { slots, cell, roundX, end } = this.layout(lanes, width);
    const laneH = window.matchMedia(Timeline.COMPACT).matches ? Timeline.LANE_H_SHORT : Timeline.LANE_H;
    const svgH = laneH * lanes.length;
    this.container.style.height = `${Timeline.HEAD_H + laneH * Math.max(3, lanes.length) + 2}px`;

    const magnitudes = [0.5];
    const totals = [0];
    for (const lane of lanes) {
      for (const frame of lane.trace.frames) {
        if (!frame) continue;
        if (frame.decision) {
          magnitudes.push(Math.abs(frame.decision.reward));
          totals.push(frame.decision.cumulative);
        } else {
          magnitudes.push(Math.abs(frame.round_open.holding_charged));
        }
      }
    }
    const extent = Math.max(...magnitudes);
    const cumLo = Math.min(...totals);
    const cumHi = Math.max(...totals);

    const header = slots.map((count, round) => this.headerCell(round, lanes, divergence, selected, roundX[round], count * cell)).join("");
    const legend = `<div class="timeline__legend">${OPERATORS.map((name, index) => `<span><i class="opmark opmark--op${index}"></i>${name}</span>`).join("")}</div>`;

    const bands = slots.map((count, round) => `<rect class="tl-round${selected.round === round ? " is-current" : ""}" x="${roundX[round]}" y="0" width="${count * cell}" height="${svgH}"></rect>`).join("");

    const rows = lanes.map((lane, index) => {
      const top = index * laneH;
      const mid = top + laneH * 0.62;
      const half = Math.min(laneH * 0.34, 12);
      const glyph = Math.min(laneH >= Timeline.LANE_H ? 10 : 9, cell - 3);
      const parts = [`<line class="tl-zero" x1="${Timeline.LABEL_W}" x2="${end}" y1="${mid}" y2="${mid}"></line>`];
      const cumPoints = [];
      let previousCommitX = null;

      for (const round of lane.trace.rounds()) {
        if (!round) continue;
        const frames = [round.open, ...round.decisions].filter((frame) => frame);
        frames.forEach((frame) => {
          const slot = frame.kind === "round_open" ? 0 : 1 + frame.edit;
          const x = roundX[frame.round] + slot * cell;
          const cx = x + cell / 2;
          const isSelected = selected.lane === lane.key && selected.seq === frame.seq;

          if (frame.kind === "round_open") {
            const charged = frame.round_open.holding_charged;
            if (Math.abs(charged) > 1e-9) {
              const size = Math.max(2, (Math.abs(charged) / extent) * half);
              parts.push(`<rect class="tl-holding" x="${cx - Math.max(2, Math.min(9, cell * 0.28))}" y="${mid}" width="${Math.max(4, Math.min(18, cell * 0.56))}" height="${size}"></rect>`);
              if (previousCommitX !== null) parts.push(`<line class="tl-connector" x1="${previousCommitX}" x2="${cx}" y1="${mid + half + 3}" y2="${mid + half + 3}"></line><line class="tl-connector" x1="${previousCommitX}" x2="${previousCommitX}" y1="${mid + half}" y2="${mid + half + 3}"></line><line class="tl-connector" x1="${cx}" x2="${cx}" y1="${mid + size}" y2="${mid + half + 3}"></line>`);
            }
            parts.push(`<line class="tl-open" x1="${x + 0.5}" x2="${x + 0.5}" y1="${top + 3}" y2="${top + laneH - 3}"></line>`);
          } else {
            const decision = frame.decision;
            const size = Math.max(1, (Math.abs(decision.reward) / extent) * half);
            const barW = Math.max(3, Math.min(18, cell * 0.56));
            const side = decision.reward >= 0 ? "gain" : "loss";
            parts.push(`<rect class="tl-bar tl-bar--${side}" x="${cx - barW / 2}" y="${decision.reward >= 0 ? mid - size : mid}" width="${barW}" height="${size}"></rect>`);
            parts.push(`<rect class="tl-op tl-op--op${decision.action.operator}" x="${cx - glyph / 2}" y="${top + 3}" width="${glyph}" height="${glyph}"></rect>`);
            if (decision.wasted || Timeline.workLeft(frame)) parts.push(`<path class="tl-warn" d="M${cx - 4} ${top + laneH - 2}h8l-4 -7z"></path>`);
            cumPoints.push([cx, decision.cumulative]);
            if (decision.ends_round || decision.is_commit) previousCommitX = cx;
          }
          parts.push(`<rect class="tl-hit${isSelected ? " is-selected" : ""}" x="${x}" y="${top + 1}" width="${cell}" height="${laneH - 2}" data-lane="${esc(lane.key)}" data-seq="${frame.seq}"></rect>`);
        });
      }

      if (cumPoints.length > 1 && cumHi > cumLo) {
        const y = (value) => top + laneH - 3 - ((value - cumLo) / (cumHi - cumLo)) * (laneH - 6);
        parts.unshift(`<polyline class="tl-cum tl-cum--${Agents.style(lane.key)}" points="${cumPoints.map(([px, value]) => `${px.toFixed(1)},${y(value).toFixed(1)}`).join(" ")}"></polyline>`);
      }
      if (index > 0) parts.unshift(`<line class="tl-lane" x1="0" x2="${width}" y1="${top}" y2="${top}"></line>`);
      return parts.join("");
    }).join("");

    const heads = lanes.map((lane, index) => {
      const info = lane.info;
      let state;
      if (info.status === "failed") state = statusChip("FAIL");
      else if (info.status === "running") state = `<span class="muted">round ${info.round + 1} of ${this.state.caps.rounds}, decision ${info.decisions}, ${Math.round(info.elapsed_s)} s</span>`;
      else if (info.status === "queued") state = `<span class="muted">queued</span>`;
      else if (info.status === "cancelled") state = `<span class="muted">cancelled</span>`;
      else state = "";
      const ended = lane.trace.summary && lane.trace.length;
      const total = ended ? fmtFixed(lane.trace.cumulative(lane.trace.length - 1), 2) : "";
      const sign  = ended ? signClass(lane.trace.cumulative(lane.trace.length - 1)) : "";
      const current = selected.lane === lane.key;
      return `
        <button type="button" class="timeline__lane timeline__lane--${Agents.style(lane.key)}${current ? " is-current" : ""}" data-lanehead="${esc(lane.key)}" style="top:${Timeline.HEAD_H + index * laneH}px;height:${laneH}px" aria-pressed="${current}" title="${esc(Agents.label(lane.key))}">
          <i class="timeline__stroke"></i><span class="timeline__name">${esc(Agents.label(lane.key))}</span>${state}
        </button>
        <div class="timeline__total ${sign}" style="top:${Timeline.HEAD_H + index * laneH}px;height:${laneH}px;left:${end + 8}px" title="Cumulative reward at the end of the lane">${total}</div>`;
    }).join("");

    this.container.innerHTML = `
      <div class="timeline__inner" style="height:${Timeline.HEAD_H + svgH}px">
        ${legend}${header}
        <svg class="timeline__svg" width="${width}" height="${svgH}" style="top:${Timeline.HEAD_H}px" role="img" aria-label="Decisions of every recorded agent, grouped by round">${bands}${rows}</svg>
        ${heads}
        <div class="timeline__axis" style="left:${end + 8}px">return</div>
      </div>`;
  }
}
