export class Tooltip {
  constructor() {
    this.node = document.createElement("div");
    this.node.className = "viz-tip glass";
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
    let left = x + 16;
    let top = y + 16;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - 16;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - 16;
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
    const shown = pairs.filter((pair) => pair);
    const keyed = shown.some((pair) => pair[2]);
    const body = shown.map(([label, value, swatch]) => `<span class="viz-tip__row${keyed ? "" : " viz-tip__row--plain"}">${keyed ? `<i${swatch ? ` style="--sw:${swatch}"` : ""}></i>` : ""}<span>${label}</span><b>${value}</b></span>`).join("");
    return `${title ? `<span class="viz-tip__title">${title}</span>` : ""}${body}`;
  }

  static text(title, sentence) {
    return `${title ? `<span class="viz-tip__title">${title}</span>` : ""}<span class="viz-tip__text">${sentence}</span>`;
  }
}

export const tooltip = new Tooltip();
