import { cssVar, Store } from "../format.js";

export class BuildingsLayer {
  static MIN_ZOOM = 15;
  static TILE_ZOOM = 16;
  static MAX_TILES = 60;
  static CONCURRENCY = 2;
  static RETRY_MS = 60000;

  constructor(map, onState) {
    this.map = map;
    this.onState = onState;
    this.enabled = Store.read("buildings", true);
    this.layer = null;
    this.attached = false;
    this.suspended = false;
    this.tiles = new Map();
    this.queue = [];
    this.active = 0;
    this.wanted = [];
    this.failedAt = 0;
    this.failure = null;
    this.abort = null;
    this.flushTimer = null;

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

  get unavailable() {
    return this.failure !== null && Date.now() - this.failedAt < BuildingsLayer.RETRY_MS;
  }

  get detail() {
    if (this.wanted.length === 0 && this.attached) return "Zoom in further to load building data for this view.";
    const missing = this.wanted.filter((key) => !this.tiles.has(key)).length;
    if (this.unavailable && missing) return `OpenStreetMap building data did not arrive for ${missing} of ${this.wanted.length} tiles in view: ${this.failure}`;
    return "Extruded building footprints from OpenStreetMap, shown from zoom 15.";
  }

  get pending() {
    return this.queue.length + this.active;
  }

  get label() {
    if (!this.enabled) return "Buildings: off";
    if (this.map.getZoom() < BuildingsLayer.MIN_ZOOM) return "Buildings: zoom in";
    if (this.pending) return "Buildings: loading";
    if (this.unavailable && !this.wanted.some((key) => this.tiles.has(key))) return "Buildings: unavailable";
    return "Buildings: on";
  }

  toggle() {
    this.enabled = !this.enabled;
    Store.write("buildings", this.enabled);
    this.failure = null;
    this.sync();
  }

  suspend() {
    if (!this.attached) return;
    this.suspended = true;
    this.detach();
  }

  restyle() {
    if (this.layer) this.layer.style({ wallColor: cssVar("--bld-wall"), roofColor: cssVar("--bld-roof"), shadows: false });
  }

  attach() {
    if (!this.layer) this.layer = new OSMBuildings();
    this.layer.addTo(this.map);
    this.restyle();
    this.attached = true;
    this.abort = new AbortController();
    this.request();
  }

  detach() {
    if (this.layer && this.attached) this.map.removeLayer(this.layer);
    this.attached = false;
    this.queue = [];
    if (this.abort) this.abort.abort();
    this.abort = null;
    clearTimeout(this.flushTimer);
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
    const visible = this.visibleTiles();
    this.wanted = visible.length > BuildingsLayer.MAX_TILES ? [] : visible;
    this.queue = this.unavailable ? [] : this.wanted.filter((key) => !this.tiles.has(key));
    this.flush();
    this.pump();
    this.onState(this);
  }

  pump() {
    while (this.attached && this.active < BuildingsLayer.CONCURRENCY && this.queue.length) this.fetchTile(this.queue.shift());
  }

  async fetchTile(key) {
    this.active += 1;
    const signal = this.abort.signal;
    try {
      const response = await fetch(`/api/buildings/${key}.json`, { signal });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
      this.tiles.set(key, body.features);
      this.failure = null;
      this.scheduleFlush();
    } catch (error) {
      if (!signal.aborted) {
        this.failure = error.message;
        this.failedAt = Date.now();
        this.queue = [];
      }
    }
    this.active -= 1;
    this.pump();
    if (!this.pending || this.failure) this.onState(this);
  }

  scheduleFlush() {
    clearTimeout(this.flushTimer);
    this.flushTimer = setTimeout(() => this.flush(), 120);
  }

  flush() {
    if (!this.attached) return;
    const features = [];
    for (const key of this.wanted) {
      if (this.tiles.has(key)) features.push(...this.tiles.get(key));
    }
    this.layer.set({ type: "FeatureCollection", features });
  }

  sync() {
    const wanted = this.enabled && !this.suspended && this.map.getZoom() >= BuildingsLayer.MIN_ZOOM;
    if (wanted && !this.attached) this.attach();
    else if (!wanted && this.attached) this.detach();
    else if (wanted) this.request();
    this.onState(this);
  }
}
