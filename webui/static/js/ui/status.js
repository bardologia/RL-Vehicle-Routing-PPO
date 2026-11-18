import { esc, icon } from "../format.js";

const TONE = {
  PASS: "pass", UP: "pass", SAME: "pass", LOADABLE: "pass", ACTIVE: "pass", finished: "pass",
  WARN: "warn", WASTED: "warn", DIFFERENT: "warn", "SHIFT OVER": "warn", cancelled: "warn",
  FAIL: "fail", DOWN: "fail", INCOMPATIBLE: "fail", failed: "fail",
};

const ICON = { pass: "check", warn: "warning", fail: "cross" };

export function statusTone(word) {
  return TONE[word] || "plain";
}

export function statusChip(word, detail, prefix) {
  const tone = statusTone(word);
  const mark = ICON[tone] ? icon(ICON[tone]) : "";
  const tail = detail ? `<span class="chip__detail">${esc(detail)}</span>` : "";
  return `<span class="chip chip--${tone}">${mark}<span class="chip__word">${prefix ? esc(prefix) + " " : ""}${esc(word)}</span>${tail}</span>`;
}

export function plainChip(text, title) {
  return `<span class="chip chip--plain"${title ? ` title="${esc(title)}"` : ""}><span>${esc(text)}</span></span>`;
}

export function errorBlock(cause, details) {
  const more = details ? `<details class="errorblock__more"><summary>Details</summary><pre>${esc(details)}</pre></details>` : "";
  return `<div class="errorblock" role="alert">${statusChip("FAIL")}<span class="errorblock__cause">${esc(cause)}</span>${more}</div>`;
}

export function emptyBlock(sentence, actionHtml) {
  return `<div class="emptyblock"><p>${esc(sentence)}</p>${actionHtml || ""}</div>`;
}

export function skeleton(rows = 3) {
  return `<div class="skeleton" aria-busy="true">${"<i></i>".repeat(rows)}</div>`;
}

export class Toast {
  constructor(element) {
    this.element = element;
    this.timer = null;
  }

  confirm(message) {
    this.element.innerHTML = `${icon("check")}<span>${esc(message)}</span>`;
    this.element.classList.add("is-shown");
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.element.classList.remove("is-shown"), 2600);
  }
}
