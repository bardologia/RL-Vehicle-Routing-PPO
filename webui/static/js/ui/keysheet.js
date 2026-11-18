import { esc } from "../format.js";

export class KeySheet {
  constructor(node, keys, scopeLabel) {
    this.node = node;
    this.keys = keys;
    this.scopeLabel = scopeLabel;
    this.restore = null;
    node.addEventListener("click", (event) => {
      if (event.target === node || event.target.closest("[data-action=close]")) this.close();
    });
    window.addEventListener("keydown", (event) => {
      if (!this.isOpen || event.key !== "Escape") return;
      event.stopImmediatePropagation();
      event.preventDefault();
      this.close();
    }, true);
  }

  get isOpen() {
    return !this.node.hidden;
  }

  static NAMES = { ArrowRight: "Right", ArrowLeft: "Left", ArrowUp: "Up", ArrowDown: "Down", Escape: "Esc" };

  static label(key) {
    return key.split("+").map((part) => `<kbd>${esc(KeySheet.NAMES[part] || (part.length === 1 ? part.toUpperCase() : part))}</kbd>`).join(" ");
  }

  static rows(bindings) {
    const seen = new Set();
    return bindings.filter((binding) => {
      if (seen.has(binding.key)) return false;
      seen.add(binding.key);
      return true;
    }).map((binding) => `<tr><th>${KeySheet.label(binding.key)}</th><td>${esc(binding.description)}</td></tr>`).join("");
  }

  open() {
    const { global, scoped } = this.keys.list();
    const tabs = global.filter((binding) => /^[1-9]$/.test(binding.key));
    const rest = global.filter((binding) => !/^[1-9]$/.test(binding.key));
    this.node.innerHTML = `
      <div class="keysheet__box">
        <header class="keysheet__head"><h2>Keys</h2><button type="button" class="btn btn--sm" data-action="close">Close</button></header>
        <div class="keysheet__cols">
          <section><h3>Everywhere</h3><table class="kv">${KeySheet.rows(rest)}<tr><th><kbd>${esc(tabs[0].key)}</kbd> to <kbd>${esc(tabs[tabs.length - 1].key)}</kbd></th><td>Switch tab, in the order of the top bar</td></tr><tr><th><kbd>Tab</kbd></th><td>Move to the next control</td></tr></table></section>
          <section><h3>${esc(this.scopeLabel())}</h3>${scoped.length ? `<table class="kv">${KeySheet.rows(scoped)}</table>` : `<p class="muted">This view has no keys of its own. Tables take the arrow keys, J and K, and Enter when focused.</p>`}</section>
        </div>
      </div>`;
    this.restore = document.activeElement;
    this.node.hidden = false;
    this.node.querySelector("[data-action=close]").focus();
  }

  close() {
    this.node.hidden = true;
    if (this.restore instanceof HTMLElement) this.restore.focus();
  }

  toggle() {
    if (this.isOpen) this.close();
    else this.open();
  }
}
