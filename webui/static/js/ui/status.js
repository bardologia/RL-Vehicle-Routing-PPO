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

export class FigureState {
  static html(name, state, attrs) {
    const data = state === undefined || state === null ? "" : ` data-state="${esc(JSON.stringify(state))}"`;
    return `<hairline-figure name="${name}"${data}${attrs ? ` ${attrs}` : ""}></hairline-figure>`;
  }

  static apply(node) {
    const found = node.matches("hairline-figure[data-state]") ? [node] : node.querySelectorAll("hairline-figure[data-state]");
    for (const figure of found) {
      figure.state = JSON.parse(figure.dataset.state);
      figure.removeAttribute("data-state");
    }
  }

  static watch(root) {
    const observer = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.addedNodes) if (node.nodeType === 1) FigureState.apply(node);
      }
    });
    observer.observe(root, { childList: true, subtree: true });
    FigureState.apply(root);
  }
}

export const EMPTY_FIGURES = {
  checkpoints: { name: "checkpoint", state: { slots: [], selected: null } },
  report:      { name: "diverge", state: { rounds: 3, split: null, gap: 0, agree: null, teacher: false } },
  events:      { name: "training", state: { values: Array(24).fill(null), best: null, live: false } },
  jobs:        { name: "queue", state: { running: 0, queued: 0, finished: 0, failed: 0 } },
  log:         { name: "terminal", state: null },
  images:      { name: "riffle", state: null },
  files:       { name: "cabinet", state: null },
};

export function emptyBlock(sentence, actionHtml, figure) {
  const art = figure ? FigureState.html(figure.name, figure.state, 'aria-hidden="true"') : "";
  return `<div class="figure-empty">${art}<p class="figure-empty__text">${esc(sentence)}</p>${actionHtml || ""}</div>`;
}

export function failView(cause, details, error) {
  const status = error && typeof error.status === "number" ? error.status : null;
  const figure = status === 0 ? FigureState.html("services", { osrm: false, vroom: false, reachable: false }, 'aria-hidden="true"') : FigureState.html("breakdown", { kind: status === 404 ? "missing" : "failed" }, 'aria-hidden="true"');
  return `<div class="figure-empty figure-empty--fail">${figure}${errorBlock(cause, details)}</div>`;
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
