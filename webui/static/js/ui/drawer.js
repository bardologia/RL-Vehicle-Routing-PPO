import { el, icon, Store } from "../format.js";
import { Tabs } from "./tabs.js";

export class Drawer {
  constructor(container, { tabs, storeKey, startCollapsed, onTab }) {
    this.min = 160;
    this.max = 420;
    this.storeKey = storeKey;
    this.onTab = onTab;
    this.height = Store.read(`${storeKey}.height`, 240);
    this.collapsed = Store.read(`${storeKey}.collapsed`, startCollapsed);

    this.root = el(`
      <div class="drawer">
        <div class="drawer__grip" role="separator" aria-orientation="horizontal" aria-label="Resize the drawer" tabindex="0"></div>
        <div class="drawer__bar">
          <div class="drawer__tabs"></div>
          <div class="drawer__actions"></div>
          <button type="button" class="btn btn--sm drawer__toggle"></button>
        </div>
        <div class="drawer__body"></div>
      </div>`);
    container.appendChild(this.root);

    this.body = this.root.querySelector(".drawer__body");
    this.actions = this.root.querySelector(".drawer__actions");
    this.toggle = this.root.querySelector(".drawer__toggle");
    this.tabs = new Tabs(this.root.querySelector(".drawer__tabs"), tabs, Store.read(`${storeKey}.tab`, tabs[0].value), (value, changed) => {
      Store.write(`${storeKey}.tab`, value);
      if (this.collapsed) this.setCollapsed(false);
      else if (!changed) this.setCollapsed(true);
      this.onTab(value);
    });

    this.toggle.addEventListener("click", () => this.setCollapsed(!this.collapsed));
    this.bindGrip(this.root.querySelector(".drawer__grip"));
    this.apply();
  }

  get tab() {
    return this.tabs.value;
  }

  bindGrip(grip) {
    grip.addEventListener("pointerdown", (event) => {
      if (this.collapsed) return;
      const startY = event.clientY;
      const startHeight = this.height;
      grip.setPointerCapture(event.pointerId);
      const move = (moveEvent) => {
        this.height = Math.max(this.min, Math.min(this.max, startHeight + startY - moveEvent.clientY));
        this.apply();
      };
      const up = () => {
        grip.removeEventListener("pointermove", move);
        grip.removeEventListener("pointerup", up);
        Store.write(`${this.storeKey}.height`, this.height);
      };
      grip.addEventListener("pointermove", move);
      grip.addEventListener("pointerup", up);
    });
    grip.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
      this.height = Math.max(this.min, Math.min(this.max, this.height + (event.key === "ArrowUp" ? 20 : -20)));
      Store.write(`${this.storeKey}.height`, this.height);
      this.apply();
      event.preventDefault();
      event.stopPropagation();
    });
  }

  setCollapsed(collapsed) {
    this.collapsed = collapsed;
    Store.write(`${this.storeKey}.collapsed`, collapsed);
    this.apply();
  }

  apply() {
    this.root.classList.toggle("is-collapsed", this.collapsed);
    this.root.style.setProperty("--drawer-h", `${this.height}px`);
    this.toggle.innerHTML = `${icon(this.collapsed ? "chevron-up" : "chevron-down")}<span>${this.collapsed ? "Open" : "Collapse"}</span>`;
    this.toggle.setAttribute("aria-expanded", String(!this.collapsed));
  }
}
