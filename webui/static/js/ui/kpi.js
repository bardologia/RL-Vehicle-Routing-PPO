import { esc } from "../format.js";
import { statusChip } from "./status.js";

export function kpiTile({ key, label, value, unit, full, detail, sub, subs, note, deltaPrev, deltaStart, status, title, tone, ch }) {
  const lines = [];
  if (subs) for (const line of subs) lines.push(typeof line === "string" ? `<span>${esc(line)}</span>` : line.html);
  if (note) lines.push(`<span>${esc(note)}</span>`);
  const deltas = [];
  if (deltaPrev !== undefined) deltas.push(`<span><i>step</i>${esc(deltaPrev)}</span>`);
  if (deltas.length) lines.push(deltas.join(""));
  const hint = [title, deltaStart !== undefined ? `Since the episode start: ${deltaStart}.` : null].filter((part) => part).join(" ");
  if (status) lines.push(statusChip(status));
  const side = detail || sub;

  return `
    <div class="kpi"${key ? ` data-kpi="${esc(key)}"` : ""}${hint ? ` data-hint="${esc(hint)}" data-hint-title="${esc(full || label)}"` : ""}>
      <div class="kpi__head"><span class="kpi__label">${esc(label)}</span>${side ? `<span class="kpi__detail">${esc(side)}</span>` : ""}</div>
      <div class="kpi__value${tone ? ` ${tone}` : ""}"><span${ch ? ` style="--ch:${ch}"` : ""}>${esc(value)}</span>${unit ? `<small>${esc(unit)}</small>` : ""}</div>
      ${lines.map((line) => `<div class="kpi__sub">${line}</div>`).join("")}
    </div>`;
}

export function kpiRow(tiles, { strip, columns } = {}) {
  return `<div class="kpirow${strip ? " kpirow--strip" : ""}" style="--n:${columns || Math.min(6, Math.max(1, tiles.length))}">${tiles.map(kpiTile).join("")}</div>`;
}
