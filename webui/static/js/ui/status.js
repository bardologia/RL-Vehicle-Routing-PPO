import { esc } from "../format.js";
import { Orb } from "../fx/orb.js";

const TONE = {
  PASS: "pass", UP: "pass", SAME: "pass", LOADABLE: "pass", ACTIVE: "pass", finished: "pass",
  WARN: "warn", WASTED: "warn", DIFFERENT: "warn", "SHIFT OVER": "warn", cancelled: "warn", stopped: "warn",
  FAIL: "fail", DOWN: "fail", INCOMPATIBLE: "fail", failed: "fail",
  running: "live", recording: "live",
  queued: "none", pending: "none", "NO DATA": "none", "NO CHECKPOINT": "none",
};

export function statusTone(word) {
  return TONE[word] || "none";
}

export function statusChip(word, detail, prefix) {
  const text = [prefix, word].filter((part) => part).join(" ").toLowerCase();
  return `<span class="state state--${statusTone(word)}"><i></i>${esc(text)}${detail ? ` ${esc(detail)}` : ""}</span>`;
}

export function plainChip(text, title) {
  return `<span class="state state--plain"${title ? ` title="${esc(title)}"` : ""}>${esc(text)}</span>`;
}

export function errorBlock(cause, details) {
  const more = details ? `<details class="errorblock__more"><summary>Details</summary><pre>${esc(details)}</pre></details>` : "";
  return `<div class="errorblock" role="alert"><span class="state state--fail"><i></i></span><span class="errorblock__cause">${esc(cause)}</span>${more}</div>`;
}

export function emptyBlock(sentence, actionHtml) {
  return `<div class="figure-empty"><p class="figure-empty__text">${esc(sentence)}</p>${actionHtml || ""}</div>`;
}

export function skeleton(rows = 3) {
  const large = rows >= 4;
  return `<div class="skeleton${large ? " skeleton--panel" : ""}" aria-busy="true">${Orb.html("searching", large ? 64 : 20)}<span>Loading</span></div>`;
}

export class Toast {
  constructor(element) {
    this.element = element;
    this.timer = null;
  }

  confirm(message) {
    this.element.innerHTML = `<span>${esc(message)}</span>`;
    this.element.classList.add("is-shown");
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.element.classList.remove("is-shown"), 2600);
  }
}
