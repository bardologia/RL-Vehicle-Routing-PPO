import { esc } from "../format.js";

export class Segmented {
  constructor(container, options, value, onChange, label) {
    this.container = container;
    this.options = options;
    this.value = value;
    this.onChange = onChange;
    container.classList.add("segmented");
    container.setAttribute("role", "radiogroup");
    if (label) container.setAttribute("aria-label", label);
    container.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button || button.disabled) return;
      this.set(button.dataset.value, true);
    });
    this.render();
  }

  render() {
    this.container.innerHTML = this.options.map((option) => `
      <button type="button" role="radio" data-value="${esc(option.value)}" aria-checked="${option.value === this.value}"${option.disabled ? " disabled" : ""}${option.title ? ` title="${esc(option.title)}"` : ""}>${option.html || esc(option.label)}</button>`).join("");
  }

  setOptions(options) {
    this.options = options;
    this.render();
  }

  set(value, notify) {
    if (value === this.value) return;
    this.value = value;
    this.render();
    if (notify) this.onChange(value);
  }
}

export class Tabs {
  constructor(container, tabs, value, onChange) {
    this.container = container;
    this.tabs = tabs;
    this.value = value;
    this.onChange = onChange;
    container.classList.add("tabs");
    container.setAttribute("role", "tablist");
    container.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-tab]");
      if (button) this.set(button.dataset.tab, true);
    });
    this.render();
  }

  render() {
    this.container.innerHTML = this.tabs.map((tab) => `
      <button type="button" role="tab" data-tab="${esc(tab.value)}" aria-selected="${tab.value === this.value}">${esc(tab.label)}${tab.count !== undefined && tab.count !== null ? `<span class="tabs__count">${esc(tab.count)}</span>` : ""}</button>`).join("");
  }

  setCount(value, count) {
    const tab = this.tabs.find((entry) => entry.value === value);
    if (tab.count === count) return;
    tab.count = count;
    this.render();
  }

  set(value, notify) {
    const changed = value !== this.value;
    this.value = value;
    this.render();
    if (notify) this.onChange(value, changed);
  }
}
