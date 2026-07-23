"use strict";

class Router {
  constructor(onChange) {
    this.onChange = onChange;
    this.pages = {};
    document.querySelectorAll(".page").forEach((p) => {
      this.pages[p.dataset.page] = p;
    });
    this.links = [...document.querySelectorAll("[data-route]")];
    this.current = null;

    window.addEventListener("hashchange", () => this._sync());
  }

  start() {
    this._sync();
  }

  go(route) {
    window.location.hash = `#/${route}`;
  }

  _parse() {
    const raw = (window.location.hash || "").replace(/^#\/?/, "").trim();
    const [page, ...rest] = raw.split("/");
    if (this.pages[page]) return { page, param: rest.join("/") || null };
    return { page: "home", param: null };
  }

  _sync() {
    const { page, param } = this._parse();
    const key = `${page}/${param || ""}`;
    if (key === this.current) return;
    this.current = key;

    Object.entries(this.pages).forEach(([id, el]) => {
      el.classList.toggle("is-active", id === page);
    });

    this.links.forEach((a) => a.classList.toggle("is-current", a.dataset.route === page));

    window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });

    if (this.onChange) this.onChange(page, param);
  }
}

window.Router = Router;
