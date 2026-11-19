import { esc, icon, isNum, NONE, Store } from "../format.js";
import { Popover } from "./popover.js";

export class DataTable {
  constructor(container, { columns, rowKey, onSelect, onOpen, onHover, dense, storeKey, sort, emptyText, rowClass, rowLink, sets, menuHost }) {
    this.container = container;
    this.columns = columns;
    this.rowKey = rowKey;
    this.onSelect = onSelect;
    this.onOpen = onOpen;
    this.onHover = onHover;
    this.storeKey = storeKey;
    this.emptyText = emptyText || "No rows.";
    this.rowClass = rowClass;
    this.rowLink = rowLink;
    this.sets = sets || null;
    this.set = sets ? Store.read(`${storeKey}.set`, sets[0].value) : null;
    this.rows = [];
    this.selected = null;
    this.sort = sort || null;
    this.hidden = new Set(storeKey ? Store.read(`${storeKey}.hidden`, columns.filter((column) => column.hidden).map((column) => column.key)) : columns.filter((column) => column.hidden).map((column) => column.key));

    container.classList.add("dtable");
    if (dense !== false) container.classList.add("dtable--dense");
    container.tabIndex = 0;
    container.addEventListener("click", (event) => this.onClick(event));
    container.addEventListener("dblclick", (event) => this.onDouble(event));
    container.addEventListener("keydown", (event) => this.onKey(event));
    container.addEventListener("pointerover", (event) => this.onPointer(event));
    container.addEventListener("pointerleave", () => this.onHover && this.onHover(null));
    this.menuHost = menuHost && columns.some((column) => column.hideable) ? menuHost : null;
    if (this.menuHost) {
      this.menuHost.insertAdjacentHTML("afterbegin", this.menuButton());
      this.menuHost.querySelector("[data-columns]").addEventListener("click", (event) => this.toggleColumns(event.currentTarget));
    }
  }

  get visibleColumns() {
    return this.columns.filter((column) => !this.hidden.has(column.key) && (!this.set || !column.sets || column.sets.includes(this.set)));
  }

  setColumnSet(value) {
    this.set = value;
    if (this.storeKey) Store.write(`${this.storeKey}.set`, value);
    this.render();
  }

  setRows(rows) {
    this.rows = rows;
    this.render();
  }

  setSelected(key, scroll) {
    this.selected = key === null || key === undefined ? null : String(key);
    for (const row of this.container.querySelectorAll("tbody tr[data-key]")) {
      const on = row.dataset.key === this.selected;
      row.setAttribute("aria-selected", String(on));
      if (on && scroll) row.scrollIntoView({ block: "nearest" });
    }
  }

  sortedRows() {
    if (!this.sort) return this.rows;
    const column = this.columns.find((entry) => entry.key === this.sort.key);
    const value = column.sortValue || ((row) => row[column.key]);
    const sign = this.sort.descending ? -1 : 1;
    return [...this.rows].sort((a, b) => {
      const left = value(a);
      const right = value(b);
      const leftMissing = left === null || left === undefined;
      const rightMissing = right === null || right === undefined;
      if (leftMissing || rightMissing) return leftMissing === rightMissing ? 0 : leftMissing ? 1 : -1;
      if (isNum(left) && isNum(right)) return sign * (left - right);
      return sign * String(left).localeCompare(String(right));
    });
  }

  cell(column, row) {
    const raw = row[column.key];
    const classes = [];
    if (column.numeric) classes.push("num");
    if (column.className) classes.push(column.className);
    let inner;
    if (column.render) inner = column.render(row);
    else if (raw === null || raw === undefined) inner = NONE;
    else inner = esc(column.format ? column.format(raw, row) : raw);

    if (column.bar && isNum(column.bar.value(row))) {
      const share = Math.max(0, Math.min(1, column.bar.value(row) / column.bar.max(this.rows)));
      inner = `<span class="dtable__bar"><i style="width:${(share * 100).toFixed(1)}%"></i></span><span>${inner}</span>`;
      classes.push("dtable__barcell");
    }
    return `<td class="${classes.join(" ")}">${inner}</td>`;
  }

  render() {
    const columns = this.visibleColumns;
    const head = columns.map((column) => {
      const sorted = this.sort && this.sort.key === column.key;
      const label = `<span>${esc(column.label)}</span>${column.unit ? `<small>${esc(column.unit)}</small>` : ""}`;
      const title = column.title || column.full ? ` title="${esc(column.title || column.full)}"` : "";
      if (column.sortable === false) return `<th class="${column.numeric ? "num" : ""}"${title}>${label}</th>`;
      return `<th class="${column.numeric ? "num" : ""}" aria-sort="${sorted ? (this.sort.descending ? "descending" : "ascending") : "none"}"${title}><button type="button" data-sort="${esc(column.key)}">${label}${sorted ? icon(this.sort.descending ? "chevron-down" : "chevron-up") : ""}</button></th>`;
    }).join("");

    const rows = this.sortedRows();
    const body = rows.length ? rows.map((row) => {
      const key = String(this.rowKey(row));
      const extra = this.rowClass ? this.rowClass(row) : "";
      return `<tr data-key="${esc(key)}" aria-selected="${key === this.selected}"${extra ? ` class="${extra}"` : ""}>${columns.map((column) => this.cell(column, row)).join("")}</tr>`;
    }).join("") : `<tr class="dtable__empty"><td colspan="${columns.length}">${esc(this.emptyText)}</td></tr>`;

    const inline = !this.menuHost && this.columns.some((column) => column.hideable);
    const menu = inline ? this.menuButton(" dtable__menu") : "";
    const reopen = inline && this.menuOpen();

    this.container.innerHTML = `${menu}<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    if (reopen) Popover.reanchor(this.container.querySelector("[data-columns]"));
  }

  menuButton(extra = "") {
    return `<button type="button" class="tbtn tbtn--sm${extra}" data-columns aria-haspopup="dialog" aria-expanded="false" aria-label="Choose columns" title="Columns">${icon("columns")}</button>`;
  }

  menuOpen() {
    return Boolean(Popover.current && this.container.contains(Popover.current.anchor));
  }

  columnsHtml() {
    const hideable = this.columns.filter((column) => column.hideable);
    return `<div class="dtable__menulist">${hideable.map((column) => `<label><input type="checkbox" data-column="${esc(column.key)}"${this.hidden.has(column.key) ? "" : " checked"}><span>${esc(column.full || column.label)}</span></label>`).join("")}</div>`;
  }

  toggleColumns(anchor) {
    const node = Popover.toggle(anchor, this.columnsHtml(), { label: "Columns", className: "popover--menu" });
    if (node) node.addEventListener("change", (event) => this.onColumn(event));
  }

  onColumn(event) {
    const toggle = event.target instanceof Element ? event.target.closest("input[data-column]") : null;
    if (!toggle) return;
    if (toggle.checked) this.hidden.delete(toggle.dataset.column);
    else this.hidden.add(toggle.dataset.column);
    if (this.storeKey) Store.write(`${this.storeKey}.hidden`, [...this.hidden]);
    this.render();
  }

  rowOf(event) {
    const row = event.target instanceof Element ? event.target.closest("tbody tr[data-key]") : null;
    return row ? row.dataset.key : null;
  }

  find(key) {
    return this.rows.find((row) => String(this.rowKey(row)) === key);
  }

  onClick(event) {
    const sortButton = event.target.closest("button[data-sort]");
    if (sortButton) {
      const key = sortButton.dataset.sort;
      this.sort = this.sort && this.sort.key === key ? { key, descending: !this.sort.descending } : { key, descending: Boolean(this.columns.find((column) => column.key === key).numeric) };
      this.render();
      return;
    }
    const menuButton = event.target.closest("[data-columns]");
    if (menuButton) {
      this.toggleColumns(menuButton);
      return;
    }
    if (event.target.closest("a, button, summary, label")) return;
    const key = this.rowOf(event);
    if (key === null) return;
    this.setSelected(key);
    if (this.onSelect) this.onSelect(this.find(key));
    if (this.rowLink) window.location.hash = this.rowLink(this.find(key));
  }

  onDouble(event) {
    const key = this.rowOf(event);
    if (key !== null && this.onOpen) this.onOpen(this.find(key));
  }

  onPointer(event) {
    if (!this.onHover) return;
    const key = this.rowOf(event);
    this.onHover(key === null ? null : this.find(key));
  }

  onKey(event) {
    const rows = this.sortedRows();
    if (!rows.length) return;
    const step = event.key === "ArrowDown" || event.key === "j" ? 1 : event.key === "ArrowUp" || event.key === "k" ? -1 : 0;
    if (step) {
      const index = rows.findIndex((row) => String(this.rowKey(row)) === this.selected);
      const next = rows[Math.max(0, Math.min(rows.length - 1, index + step))];
      this.setSelected(this.rowKey(next), true);
      if (this.onSelect) this.onSelect(next);
      event.preventDefault();
      event.stopPropagation();
      return;
    }
    if (event.key === "Enter" && this.selected !== null && (this.onOpen || this.rowLink)) {
      if (this.onOpen) this.onOpen(this.find(this.selected));
      else window.location.hash = this.rowLink(this.find(this.selected));
      event.preventDefault();
      event.stopPropagation();
    }
  }
}

export function needsTraces() {
  return `<span class="faint" data-hint="Needs traces">${NONE}</span>`;
}

export function kvTable(pairs) {
  return `<table class="kv"><tbody>${pairs.map(([key, value]) => `<tr><th>${esc(key)}</th><td>${value}</td></tr>`).join("")}</tbody></table>`;
}
