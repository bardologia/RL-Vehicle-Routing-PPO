import { el, esc } from "../format.js";
import { Popover } from "./popover.js";
import { emptyBlock, errorBlock, skeleton } from "./status.js";

export class Panel {
  constructor({ title, note, info, actions, flush, className, attrs } = {}) {
    this.title = title || "";
    this.root = el(panelHtml(this.title, "", { note, info, actions, flush, className, attrs }));
    this.body = this.root.querySelector(".panel__body");
    this.noteNode = this.root.querySelector(".panel__note");
    this.actionsNode = this.root.querySelector("[data-role=panel-actions]");
    this.infoNode = this.root.querySelector("[data-role=panel-info]");
  }

  setNote(text) {
    this.noteNode.textContent = text;
  }

  setInfo(html) {
    this.infoNode.innerHTML = Popover.infoButton(this.title, html);
  }

  setActions(html) {
    this.actionsNode.innerHTML = html;
  }

  setContent(html) {
    this.body.innerHTML = html;
  }

  setLoading(rows) {
    this.body.innerHTML = skeleton(rows);
  }

  setEmpty(sentence, actionHtml) {
    this.body.innerHTML = emptyBlock(sentence, actionHtml);
  }

  setError(cause, details) {
    this.body.innerHTML = errorBlock(cause, details);
  }
}

export function infoText(...sentences) {
  return sentences.filter((sentence) => sentence).map((sentence) => `<p>${esc(sentence)}</p>`).join("");
}

export function panelHead(title, { note, info, actions, titleClass, titleAttrs } = {}) {
  return `
    <header class="panel__head">
      <h3 class="panel__title${titleClass ? ` ${titleClass}` : ""}"${titleAttrs || ""}>${esc(title)}</h3>
      <span class="panel__note" data-role="panel-note">${esc(note || "")}</span>
      <span class="panel__actions"><span data-role="panel-actions" class="panel__actions">${actions || ""}</span><span data-role="panel-info" class="panel__actions">${Popover.infoButton(title, info)}</span></span>
    </header>`;
}

export function panelHtml(title, bodyHtml, { note, info, actions, flush, className, attrs } = {}) {
  return `
    <section class="panel${flush ? " panel--flush" : ""}${className ? " " + className : ""}"${attrs ? ` ${attrs}` : ""}>
      ${title ? panelHead(title, { note, info, actions }) : ""}
      <div class="panel__body">${bodyHtml}</div>
    </section>`;
}
