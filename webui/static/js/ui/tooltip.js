export class Tooltip {
  constructor() {
    this.node = document.createElement("div");
    this.node.className = "tooltip";
    this.node.setAttribute("role", "tooltip");
    document.body.appendChild(this.node);
  }

  show(html, x, y) {
    this.node.innerHTML = html;
    this.node.classList.add("is-shown");
    this.move(x, y);
  }

  move(x, y) {
    const box = this.node.getBoundingClientRect();
    let left = x + 14;
    let top = y + 14;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - 14;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - 14;
    this.node.style.left = `${Math.max(8, left)}px`;
    this.node.style.top = `${Math.max(8, top)}px`;
  }

  hide() {
    this.node.classList.remove("is-shown");
  }

  bind(container, selector, describe) {
    container.addEventListener("pointermove", (event) => {
      const target = event.target instanceof Element ? event.target.closest(selector) : null;
      if (!target || !container.contains(target)) {
        this.hide();
        return;
      }
      const html = describe(target, event);
      if (html) this.show(html, event.clientX, event.clientY);
      else this.hide();
    });
    container.addEventListener("pointerleave", () => this.hide());
  }

  static rows(title, pairs) {
    const body = pairs.filter((pair) => pair).map(([label, value]) => `<div class="tooltip__row"><span>${label}</span><b>${value}</b></div>`).join("");
    return `${title ? `<div class="tooltip__title">${title}</div>` : ""}${body}`;
  }
}

export const tooltip = new Tooltip();
