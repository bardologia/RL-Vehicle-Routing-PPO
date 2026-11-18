export class Keys {
  constructor() {
    this.global = [];
    this.scoped = new Map();
    this.scope = null;
    window.addEventListener("keydown", (event) => this.handle(event));
  }

  static label(event) {
    const parts = [];
    if (event.shiftKey && event.key.length > 1) parts.push("Shift");
    parts.push(event.key === " " ? "Space" : event.key.length === 1 ? event.key.toLowerCase() : event.key);
    return parts.join("+");
  }

  bindGlobal(key, description, handler) {
    this.global.push({ key, description, handler });
  }

  bind(scope, key, description, handler) {
    if (!this.scoped.has(scope)) this.scoped.set(scope, []);
    this.scoped.get(scope).push({ key, description, handler });
  }

  setScope(scope) {
    this.scope = scope;
  }

  list() {
    return { global: this.global, scoped: this.scoped.get(this.scope) || [] };
  }

  handle(event) {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    const typing = target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
    const label = Keys.label(event);
    if (typing && label !== "Escape") return;

    const bindings = [...(this.scoped.get(this.scope) || []), ...this.global];
    const match = bindings.find((binding) => binding.key === label);
    if (!match) return;

    const handled = match.handler(event);
    if (handled !== false) event.preventDefault();
  }
}

export const keys = new Keys();
