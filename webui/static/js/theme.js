import { Store } from "./format.js";

export class Theme {
  constructor(button) {
    this.button = button;
    this.listeners = [];
    const stored = Store.read("theme", null);
    this.value = stored || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    this.apply();
    button.addEventListener("click", () => this.toggle());
  }

  onChange(listener) {
    this.listeners.push(listener);
  }

  apply() {
    document.documentElement.dataset.theme = this.value;
    const next = this.value === "dark" ? "light" : "dark";
    this.button.innerHTML = `<svg class="icon" aria-hidden="true"><use href="#i-${next === "dark" ? "moon" : "sun"}"></use></svg><span>${next === "dark" ? "Dark" : "Light"} theme</span>`;
    this.button.setAttribute("aria-label", `Switch to the ${next} theme`);
  }

  toggle() {
    this.value = this.value === "dark" ? "light" : "dark";
    Store.write("theme", this.value);
    this.apply();
    for (const listener of this.listeners) listener(this.value);
  }
}
