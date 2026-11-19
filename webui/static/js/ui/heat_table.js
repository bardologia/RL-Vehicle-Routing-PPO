import { esc, isNum } from "../format.js";
import { tooltip } from "./tooltip.js";

export class HeatTable {
  constructor(container, { onHover, onClick }) {
    this.container = container;
    this.onHover = onHover;
    this.onClick = onClick;
    container.classList.add("heat");
    tooltip.bind(container, "[data-tip]", (target) => target.dataset.tip);
    container.addEventListener("pointerover", (event) => {
      const cell = event.target instanceof Element ? event.target.closest("[data-row]") : null;
      if (this.onHover) this.onHover(cell ? { row: cell.dataset.row, column: cell.dataset.column || null } : null);
    });
    container.addEventListener("pointerleave", () => this.onHover && this.onHover(null));
    container.addEventListener("click", (event) => {
      const cell = event.target instanceof Element ? event.target.closest("[data-row]") : null;
      if (cell && this.onClick) this.onClick({ row: cell.dataset.row, column: cell.dataset.column || null });
    });
  }

  static rampStyle(share) {
    const clamped = Math.max(0, Math.min(1, share));
    const step = Math.min(3, Math.floor(clamped * 4));
    const mix = clamped * 4 - step;
    const ink = clamped >= 0.5 ? "var(--heat-on-light)" : "var(--ink)";
    return `background:color-mix(in oklab, var(--ramp-${step + 1}) ${(mix * 100).toFixed(0)}%, var(--ramp-${step}));color:${ink}`;
  }

  static divergingStyle(value, extent) {
    const share = Math.max(0, Math.min(1, Math.abs(value) / (extent || 1)));
    const tone = value >= 0 ? "var(--gain)" : "var(--loss)";
    const ink = share > 0.55 ? "var(--heat-on-light)" : "var(--ink)";
    return `background:color-mix(in oklab, ${tone} ${(share * 70).toFixed(0)}%, var(--heat-neutral));color:${ink}`;
  }

  render({ rows, columns, cells, rowTotals, columnTotals, mode = "ramp", format, cornerLabel, totalLabel = "sum", marks = {} }) {
    const flat = cells.flat().filter((cell) => cell && isNum(cell.value)).map((cell) => Math.abs(cell.value));
    const extent = Math.max(1e-9, ...flat);
    const showNumbers = columns.length <= 14;

    const head = `<tr><th class="heat__corner">${esc(cornerLabel || "")}</th>${columns.map((column) => `<th class="heat__col${column.selected ? " is-selected" : ""}" data-row="" data-column="${esc(column.key)}" title="${esc(column.title || "")}">${column.html}</th>`).join("")}${rowTotals ? `<th class="heat__total num">${esc(totalLabel)}</th>` : ""}</tr>`;

    const body = rows.map((row, r) => {
      const tds = columns.map((column, c) => {
        const cell = cells[r][c];
        const key = `${row.key}:${column.key}`;
        const flags = [];
        if (marks.chosen === key) flags.push("is-chosen");
        if (marks.argmax === key) flags.push("is-argmax");
        if (marks.teacher === key) flags.push("is-teacher");
        if (!cell || cell.state === "masked") return `<td class="heat__cell is-masked ${flags.join(" ")}" data-row="${esc(row.key)}" data-column="${esc(column.key)}" data-tip="${esc(cell && cell.tip ? cell.tip : "masked")}"></td>`;
        if (cell.state === "infeasible") return `<td class="heat__cell is-infeasible ${flags.join(" ")}" data-row="${esc(row.key)}" data-column="${esc(column.key)}" data-tip="${esc(cell.tip || "infeasible")}"><span>inf.</span></td>`;
        const style = mode === "diverging" ? HeatTable.divergingStyle(cell.value, extent) : HeatTable.rampStyle(cell.value / extent);
        return `<td class="heat__cell ${flags.join(" ")}" style="${style}" data-row="${esc(row.key)}" data-column="${esc(column.key)}" data-tip="${esc(cell.tip || "")}">${showNumbers ? `<span>${esc(format(cell.value))}</span>` : ""}</td>`;
      }).join("");
      const total = rowTotals ? `<td class="heat__total num">${esc(format(rowTotals[r]))}</td>` : "";
      return `<tr><th class="heat__row" data-row="${esc(row.key)}">${row.html}</th>${tds}${total}</tr>`;
    }).join("");

    const foot = columnTotals ? `<tr class="heat__foot"><th class="heat__row muted">${esc(totalLabel)}</th>${columnTotals.map((value) => `<td class="num">${showNumbers && isNum(value) ? esc(format(value)) : ""}</td>`).join("")}${rowTotals ? "<td></td>" : ""}</tr>` : "";

    this.container.innerHTML = `<div class="heat__scroll"><table style="--heat-cols:${columns.length}"><thead>${head}</thead><tbody>${body}${foot}</tbody></table></div>`;
  }
}
