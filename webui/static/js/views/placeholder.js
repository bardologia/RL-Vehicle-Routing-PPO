import { el, esc } from "../format.js";

export class PlaceholderView {
  constructor(app, root, title, sentence) {
    this.app = app;
    this.root = root;
    root.appendChild(el(`
      <div class="view__pad">
        <header class="pagehead"><h2>${esc(title)}</h2></header>
        <section class="panel"><div class="panel__body"><div class="emptyblock"><p>${esc(sentence)}</p><a class="btn" href="#/episode">Open Episode</a></div></div></section>
      </div>`));
  }

  enter() {}

  leave() {}
}
