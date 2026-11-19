import { cssVar, Store } from "../format.js";

export class BuildingsLayer {
  static MIN_ZOOM = 15;
  static TILE_ZOOM = 16;
  static BLOCK_ZOOM = 14;
  static NEAR_ZOOM = 17;
  static CONCURRENCY = 4;
  static RETRY_MS = 60000;

  constructor(map, onState) {
    this.map = map;
    this.onState = onState;
    this.enabled = Store.read("buildings", true);
    this.layer = null;
    this.attached = false;
    this.suspended = false;
    this.tiles = new Map();
    this.cachedBlocks = new Set();
    this.queue = [];
    this.active = 0;
    this.probing = new Set();
    this.failed = new Map();
    this.wanted = [];
    this.abort = null;
    this.flushTimer = null;
    this.retryTimer = null;
    this.shown = null;
    this.flushed = null;

    map.on("zoomend", () => this.sync());
    map.on("moveend", () => {
      if (this.suspended) {
        this.suspended = false;
        this.sync();
      } else if (this.attached) {
        this.request();
      }
    });
    this.sync();
  }

  get near() {
    return this.map.getZoom() >= BuildingsLayer.NEAR_ZOOM;
  }

  get missing() {
    return this.wanted.filter((key) => !this.tiles.has(key));
  }

  get lost() {
    return this.missing.filter((key) => this.failure(BuildingsLayer.block(key)) !== null);
  }

  get detail() {
    const missing = this.missing;
    const lost = this.lost;
    if (lost.length && lost.length === missing.length) return `OpenStreetMap building data did not arrive for ${lost.length} of ${this.wanted.length} tiles in view: ${this.failure(BuildingsLayer.block(lost[0]))}. It is requested again after a minute, or when the layer is switched off and on.`;
    if (missing.length) return `Loading building data for ${missing.length} of ${this.wanted.length} tiles in view. A block that is not cached yet is fetched from OpenStreetMap and can take up to a minute.`;
    return "Extruded building footprints from OpenStreetMap, shown from zoom 15.";
  }

  get pending() {
    return this.queue.length + this.active;
  }

  get label() {
    if (!this.enabled) return "Buildings: off";
    if (this.map.getZoom() < BuildingsLayer.MIN_ZOOM) return "Buildings: zoom in";
    const missing = this.missing.length;
    const loaded = this.wanted.length - missing;
    if (missing === 0) return "Buildings: on";
    if (this.lost.length < missing) return loaded ? "Buildings: on, loading more" : "Buildings: loading";
    return loaded ? "Buildings: on, part unavailable" : "Buildings: unavailable";
  }

  failure(block) {
    const entry = this.failed.get(block);
    return entry && Date.now() - entry.at < BuildingsLayer.RETRY_MS ? entry.message : null;
  }

  static block(key) {
    const [zoom, x, y] = key.split("/").map(Number);
    const shift = zoom - BuildingsLayer.BLOCK_ZOOM;
    return `${x >> shift}/${y >> shift}`;
  }

  toggle() {
    this.enabled = !this.enabled;
    Store.write("buildings", this.enabled);
    this.failed.clear();
    this.sync();
  }

  suspend() {
    if (!this.attached) return;
    this.suspended = true;
    this.detach();
  }

  restyle() {
    const suffix = this.near ? "-near" : "";
    if (this.layer) this.layer.style({ wallColor: cssVar(`--bld-wall${suffix}`), roofColor: cssVar(`--bld-roof${suffix}`), shadows: false });
  }

  attach() {
    if (!this.layer) this.layer = new OSMBuildings();
    this.layer.addTo(this.map);
    this.restyle();
    this.attached = true;
    this.flushed = null;
    this.abort = new AbortController();
    this.request();
  }

  detach() {
    if (this.layer && this.attached) this.map.removeLayer(this.layer);
    this.attached = false;
    this.queue = [];
    this.probing.clear();
    if (this.abort) this.abort.abort();
    this.abort = null;
    clearTimeout(this.flushTimer);
    clearTimeout(this.retryTimer);
  }

  visibleTiles() {
    const zoom = BuildingsLayer.TILE_ZOOM;
    const bounds = this.map.getBounds();
    const scale = 2 ** zoom;
    const column = (lon) => Math.floor(((lon + 180) / 360) * scale);
    const row = (lat) => Math.floor(((1 - Math.asinh(Math.tan((lat * Math.PI) / 180)) / Math.PI) / 2) * scale);
    const out = [];
    for (let x = column(bounds.getWest()); x <= column(bounds.getEast()); x += 1) {
      for (let y = row(bounds.getNorth()); y <= row(bounds.getSouth()); y += 1) out.push(`${zoom}/${x}/${y}`);
    }
    return out;
  }

  request() {
    if (!this.attached) return;
    this.wanted = this.visibleTiles();
    this.queue = this.missing.filter((key) => this.failure(BuildingsLayer.block(key)) === null);
    this.flush();
    this.pump();
    this.notify();
  }

  next() {
    const at = this.queue.findIndex((key) => this.cachedBlocks.has(BuildingsLayer.block(key)) || !this.probing.has(BuildingsLayer.block(key)));
    return at >= 0 ? this.queue.splice(at, 1)[0] : null;
  }

  pump() {
    while (this.attached && this.active < BuildingsLayer.CONCURRENCY) {
      const key = this.next();
      if (key === null) return;
      this.fetchTile(key);
    }
  }

  async fetchTile(key) {
    const block = BuildingsLayer.block(key);
    const probe = !this.cachedBlocks.has(block);
    if (probe) this.probing.add(block);
    this.active += 1;
    const signal = this.abort.signal;
    try {
      const response = await fetch(`/api/buildings/${key}.json`, { signal });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
      this.tiles.set(key, body.features);
      this.cachedBlocks.add(block);
      this.failed.delete(block);
      this.scheduleFlush();
    } catch (error) {
      if (!signal.aborted) {
        this.failed.set(block, { at: Date.now(), message: error.message });
        this.queue = this.queue.filter((other) => BuildingsLayer.block(other) !== block);
        clearTimeout(this.retryTimer);
        this.retryTimer = setTimeout(() => this.request(), BuildingsLayer.RETRY_MS + 100);
      }
    }
    if (probe) this.probing.delete(block);
    this.active -= 1;
    this.pump();
    this.notify();
  }

  notify() {
    const shown = `${this.label}|${this.detail}`;
    if (shown === this.shown) return;
    this.shown = shown;
    this.onState(this);
  }

  scheduleFlush() {
    clearTimeout(this.flushTimer);
    this.flushTimer = setTimeout(() => this.flush(), 120);
  }

  flush() {
    if (!this.attached) return;
    const loaded = this.wanted.filter((key) => this.tiles.has(key));
    const signature = loaded.join(" ");
    if (signature === this.flushed) return;
    this.flushed = signature;
    const features = [];
    for (const key of loaded) features.push(...this.tiles.get(key));
    this.layer.set({ type: "FeatureCollection", features });
  }

  sync() {
    const wanted = this.enabled && !this.suspended && this.map.getZoom() >= BuildingsLayer.MIN_ZOOM;
    if (wanted && !this.attached) this.attach();
    else if (!wanted && this.attached) this.detach();
    else if (wanted) {
      this.restyle();
      this.request();
    }
    this.shown = `${this.label}|${this.detail}`;
    this.onState(this);
  }
}
