import { el, esc } from "../format.js";
import { emptyBlock, errorBlock, skeleton } from "./status.js";

export class Panel {
  constructor({ title, help, actions, flush, className } = {}) {
    this.root = el(`
      <section class="panel${flush ? " panel--flush" : ""}${className ? " " + className : ""}">
        <header class="panel__head${title ? "" : " hidden"}">
          <h3 class="panel__title">${esc(title || "")}</h3>
          <span class="panel__help">${esc(help || "")}</span>
          <span class="panel__actions">${actions || ""}</span>
        </header>
        <div class="panel__body"></div>
      </section>`);
    this.body = this.root.querySelector(".panel__body");
    this.helpNode = this.root.querySelector(".panel__help");
    this.actionsNode = this.root.querySelector(".panel__actions");
  }

  setHelp(text) {
    this.helpNode.textContent = text;
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

export function panelHtml(title, bodyHtml, { help, actions, flush, className } = {}) {
  return `
    <section class="panel${flush ? " panel--flush" : ""}${className ? " " + className : ""}">
      <header class="panel__head">
        <h3 class="panel__title">${esc(title)}</h3>
        <span class="panel__help">${esc(help || "")}</span>
        <span class="panel__actions">${actions || ""}</span>
      </header>
      <div class="panel__body">${bodyHtml}</div>
    </section>`;
}
