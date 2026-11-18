export class Router {
  constructor(tabs, onRoute) {
    this.tabs = tabs;
    this.onRoute = onRoute;
    this.current = null;
    window.addEventListener("hashchange", () => this.sync());
  }

  static build(parts, query) {
    const path = parts.filter((part) => part !== null && part !== undefined && part !== "").map((part) => encodeURIComponent(part)).join("/");
    const pairs = Object.entries(query || {}).map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
    return `#/${path}${pairs.length ? "?" + pairs.join("&") : ""}`;
  }

  parse() {
    const raw = (window.location.hash || "").replace(/^#\/?/, "");
    const [path, search] = raw.split("?");
    const parts = path.split("/").filter((part) => part.length).map((part) => decodeURIComponent(part));
    const query = {};
    for (const [key, value] of new URLSearchParams(search || "")) query[key] = value;
    const known = this.tabs.includes(parts[0]);
    return { known, tab: known ? parts[0] : this.tabs[0], parts: known ? parts.slice(1) : [], query: known ? query : {} };
  }

  go(hash) {
    if (window.location.hash === hash) this.sync(true);
    else window.location.hash = hash;
  }

  replace(hash) {
    history.replaceState(null, "", hash);
    this.sync();
  }

  silent(hash) {
    history.replaceState(null, "", hash);
    this.current = hash;
  }

  sync(force) {
    const route = this.parse();
    if (!route.known) history.replaceState(null, "", `#/${route.tab}`);
    const hash = window.location.hash;
    if (!force && hash === this.current) return;
    this.current = hash;
    this.onRoute(route);
  }
}
