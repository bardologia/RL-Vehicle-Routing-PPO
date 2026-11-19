import { esc, icon } from "../format.js";

export class Popover {
  static current = null;

  static bound = false;

  static bind() {
    if (Popover.bound) return;
    Popover.bound = true;
    document.addEventListener("pointerdown", (event) => {
      const open = Popover.current;
      if (!open || !(event.target instanceof Node)) return;
      if (open.node.contains(event.target) || open.anchor.contains(event.target)) return;
      Popover.close();
    }, true);
    window.addEventListener("keydown", (event) => {
      if (!Popover.current || event.key !== "Escape") return;
      event.stopImmediatePropagation();
      event.preventDefault();
      Popover.close(true);
    }, true);
    window.addEventListener("resize", () => Popover.close());
    window.addEventListener("hashchange", () => Popover.close());
    document.addEventListener("click", (event) => {
      const button = event.target instanceof Element ? event.target.closest("[data-info]") : null;
      if (!button) return;
      const source = button.parentElement.querySelector("template[data-info-body]");
      if (!source) return;
      event.preventDefault();
      Popover.toggle(button, source.innerHTML, { label: button.getAttribute("aria-label") });
    });
  }

  static isOpenFor(anchor) {
    return Boolean(Popover.current && Popover.current.anchor === anchor);
  }

  static place(node, anchor) {
    const box = anchor.getBoundingClientRect();
    const size = node.getBoundingClientRect();
    const margin = 8;
    let left = box.right - size.width;
    if (left < margin) left = Math.min(box.left, window.innerWidth - size.width - margin);
    let top = box.bottom + 8;
    if (top + size.height > window.innerHeight - margin && box.top - 8 - size.height >= margin) top = box.top - 8 - size.height;
    node.style.left = `${Math.max(margin, left)}px`;
    node.style.top = `${Math.max(margin, Math.min(top, window.innerHeight - size.height - margin))}px`;
  }

  static open(anchor, html, { label, className, onClose } = {}) {
    Popover.bind();
    Popover.close();
    const node = document.createElement("div");
    node.className = `popover glass${className ? ` ${className}` : ""}`;
    node.setAttribute("role", "dialog");
    if (label) node.setAttribute("aria-label", label);
    node.tabIndex = -1;
    node.innerHTML = html;
    document.body.appendChild(node);
    Popover.place(node, anchor);
    anchor.setAttribute("aria-expanded", "true");
    Popover.current = { node, anchor, onClose };
    return node;
  }

  static close(restoreFocus) {
    const open = Popover.current;
    if (!open) return;
    Popover.current = null;
    open.node.remove();
    open.anchor.setAttribute("aria-expanded", "false");
    if (restoreFocus && open.anchor.isConnected) open.anchor.focus();
    if (open.onClose) open.onClose();
  }

  static toggle(anchor, html, options) {
    if (Popover.isOpenFor(anchor)) {
      Popover.close();
      return null;
    }
    return Popover.open(anchor, html, options);
  }

  static reflow() {
    if (Popover.current) Popover.place(Popover.current.node, Popover.current.anchor);
  }

  static infoButton(title, html) {
    if (!html) return "";
    return `<button type="button" class="tbtn tbtn--sm" data-info aria-haspopup="dialog" aria-expanded="false" aria-label="About ${esc(title)}" title="About ${esc(title)}">${icon("info")}</button><template data-info-body>${html}</template>`;
  }

  static menu(items) {
    return `<div class="popover__menu">${items.filter((item) => item).map((item) => (item.href ? `<a href="${esc(item.href)}"${item.title ? ` title="${esc(item.title)}"` : ""}>${esc(item.label)}</a>` : `<button type="button" data-action="${esc(item.action)}">${esc(item.label)}</button>`)).join("")}</div>`;
  }
}

Popover.bind();
