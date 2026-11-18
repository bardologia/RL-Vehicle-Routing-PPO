import { esc } from "../format.js";
import { statusChip } from "./status.js";

export function kpiTile({ key, label, value, unit, sub, note, deltaPrev, deltaStart, status, title }) {
  const deltas = [];
  if (deltaPrev !== undefined) deltas.push(`<span class="kpi__delta" title="Change against the previous frame"><i>step</i>${esc(deltaPrev)}</span>`);
  if (deltaStart !== undefined) deltas.push(`<span class="kpi__delta" title="Change against the episode start"><i>start</i>${esc(deltaStart)}</span>`);

  return `
    <div class="kpi"${key ? ` data-kpi="${esc(key)}"` : ""}${title ? ` title="${esc(title)}"` : ""}>
      <div class="kpi__label">${esc(label)}</div>
      <div class="kpi__value"><span>${esc(value)}</span>${unit ? `<small>${esc(unit)}</small>` : ""}${sub ? `<em>${esc(sub)}</em>` : ""}${status ? statusChip(status) : ""}</div>
      ${note ? `<div class="kpi__deltas"><span class="kpi__delta"><i>${esc(note)}</i></span></div>` : deltas.length ? `<div class="kpi__deltas">${deltas.join("")}</div>` : ""}
    </div>`;
}

export function kpiRow(tiles) {
  return `<div class="kpirow">${tiles.map(kpiTile).join("")}</div>`;
}
