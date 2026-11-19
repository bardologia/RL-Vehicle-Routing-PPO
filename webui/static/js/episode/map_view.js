import { el, esc, fmtClock, fmtMin, icon, Store } from "../format.js";
import { tooltip, Tooltip } from "../ui/tooltip.js";
import { BuildingsLayer } from "./buildings_layer.js";
import { Vehicles } from "./encoding.js";

const PANES = { routes: 450, changes: 460, stops: 600, vehicles: 620, depot: 630, labels: 640 };
const CENTER = [-23.55, -46.63];

export class Geometry {
  static crossing(a, b, c, d) {
    const r = [b[0] - a[0], b[1] - a[1]];
    const s = [d[0] - c[0], d[1] - c[1]];
    const denominator = r[0] * s[1] - r[1] * s[0];
    if (Math.abs(denominator) < 1e-14) return null;
    const t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denominator;
    const u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / denominator;
    if (t <= 1e-9 || t >= 1 - 1e-9 || u <= 1e-9 || u >= 1 - 1e-9) return null;
    return [a[0] + t * r[0], a[1] + t * r[1]];
  }

  static legs(route, start) {
    const points = [start, ...route.stops.map((stop) => stop.location)];
    const out = [];
    for (let index = 1; index < points.length; index += 1) {
      if (points[index][0] !== points[index - 1][0] || points[index][1] !== points[index - 1][1]) out.push([points[index - 1], points[index]]);
    }
    return out;
  }

  static crossings(routes, starts) {
    const out = [];
    for (let i = 0; i < routes.length; i += 1) {
      for (let j = i + 1; j < routes.length; j += 1) {
        for (const [a, b] of Geometry.legs(routes[i], starts[routes[i].vehicle_id])) {
          for (const [c, d] of Geometry.legs(routes[j], starts[routes[j].vehicle_id])) {
            const point = Geometry.crossing(a, b, c, d);
            if (point) out.push(point);
          }
        }
      }
    }
    return out;
  }

  static nearestIndex(path, lonLat) {
    let best = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < path.length; index += 1) {
      const distance = (path[index][0] - lonLat[1]) ** 2 + (path[index][1] - lonLat[0]) ** 2;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = index;
      }
    }
    return best;
  }
}

export class MapView {
  constructor(container, { onSelect, onIsolate, onMapClick, onDrag, onToggle, controls } = {}) {
    this.onSelect = onSelect || (() => {});
    this.onIsolate = onIsolate || (() => {});
    this.onMapClick = onMapClick || null;
    this.onDrag = onDrag || null;
    this.onToggle = onToggle || (() => {});
    this.scene = null;
    this.isolated = null;
    this.playing = false;
    this.fitted = false;
    this.cover = 0;
    this.animation = null;
    this.byVehicle = new Map();
    this.jobMarkers = new Map();
    this.vehicleMarkers = new Map();
    this.toggles = { follow: Store.read("map.follow", true), ghost: false, crossings: false, teacher: true };
    this.available = { ghost: false, teacher: false };
    this.editable = false;

    this.root = el(`
      <div class="map">
        <div class="map__caption glass" role="status" aria-live="polite"></div>
        <div class="map__canvas"></div>
        <div class="map__controls" role="toolbar" aria-label="Map controls"></div>
        <div class="map__legend"></div>
      </div>`);
    container.appendChild(this.root);
    this.captionNode = this.root.querySelector(".map__caption");
    this.controlsNode = this.root.querySelector(".map__controls");
    this.legendNode = this.root.querySelector(".map__legend");
    this.controlSet = controls || ["fit", "follow", "buildings", "teacher", "ghost", "crossings"];

    this.map = L.map(this.root.querySelector(".map__canvas"), { zoomControl: false, attributionControl: true, zoomSnap: 1, maxZoom: 19 }).setView(CENTER, 11);
    this.map.attributionControl.setPrefix(false);
    L.control.zoom({ position: "bottomright" }).addTo(this.map);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "(c) OpenStreetMap contributors" }).addTo(this.map);

    for (const [name, zIndex] of Object.entries(PANES)) this.map.createPane(name).style.zIndex = zIndex;
    this.map.getPane("labels").style.pointerEvents = "none";

    this.groups = {};
    for (const name of ["ghost", "routes", "chevrons", "changes", "teacher", "crossings", "stops", "vehicles", "depot", "labels"]) this.groups[name] = L.layerGroup().addTo(this.map);

    this.buildings = new BuildingsLayer(this.map, () => this.renderControls());
    this.legendOpen = Store.read("map.legend", window.innerHeight >= 900);

    this.map.on("zoomend", () => this.drawChevrons());
    this.map.on("click", (event) => {
      if (this.onMapClick) this.onMapClick([Number(event.latlng.lng.toFixed(6)), Number(event.latlng.lat.toFixed(6))]);
    });
    this.controlsNode.addEventListener("click", (event) => this.onControl(event));
    this.legendNode.addEventListener("click", (event) => {
      if (event.target.closest("button")) this.toggleLegend();
    });

    this.observer = new ResizeObserver(() => this.map.invalidateSize({ animate: false }));
    this.observer.observe(this.root);
    this.renderControls();
    this.renderLegend();
  }

  destroy() {
    this.cancelAnimation();
    this.observer.disconnect();
    this.map.remove();
  }

  setCaption(html) {
    this.captionNode.innerHTML = html;
    this.captionNode.classList.toggle("hidden", !html);
  }

  setEditable(editable) {
    this.editable = editable;
  }

  setCursor(mode) {
    this.root.dataset.cursor = mode || "";
  }

  setAvailable(available) {
    Object.assign(this.available, available);
    this.renderControls();
  }

  renderControls() {
    const defs = {
      fit: { label: "Fit plan", iconName: "fit", key: "F" },
      follow: { label: "Follow", iconName: "follow", pressed: this.toggles.follow, title: "Pan to the changed job when it is off screen. Never zooms. Paused during playback." },
      buildings: { label: this.buildings ? this.buildings.label : "Buildings", iconName: "buildings", pressed: this.buildings ? this.buildings.enabled : false, key: "B", title: this.buildings ? this.buildings.detail : "" },
      teacher: { label: "Teacher alternative", iconName: "teacher", pressed: this.toggles.teacher && this.available.teacher, disabled: !this.available.teacher, title: this.available.teacher ? "The insertion the teacher would make, as a dashed leg." : "Ask the teacher on this decision first." },
      ghost: { label: "Ghost lane", iconName: "ghost", pressed: this.toggles.ghost && this.available.ghost, disabled: !this.available.ghost, key: "G", title: this.available.ghost ? "Routes of another agent at the same round, as thin grey lines." : "Needs a second recorded agent." },
      crossings: { label: "Crossings", iconName: "crossing", pressed: this.toggles.crossings, title: "Points where straight stop-to-stop legs of different routes intersect." },
    };
    this.controlsNode.innerHTML = this.controlSet.map((name) => {
      const def = defs[name];
      const pressed = def.pressed === undefined ? "" : ` aria-pressed="${def.pressed}"`;
      return `<button type="button" class="btn btn--sm" data-control="${name}"${pressed}${def.disabled ? " disabled" : ""} title="${esc(def.title || def.label)}${def.key ? ` (key ${def.key})` : ""}">${icon(def.iconName)}<span>${esc(def.label)}</span></button>`;
    }).join("");
  }

  onControl(event) {
    const button = event.target.closest("button[data-control]");
    if (!button || button.disabled) return;
    this.control(button.dataset.control);
  }

  control(name) {
    if (name === "fit") this.fit(true);
    else if (name === "buildings") this.buildings.toggle();
    else {
      if (name in this.available && !this.available[name]) return;
      this.toggles[name] = !this.toggles[name];
      if (name === "follow") Store.write("map.follow", this.toggles.follow);
      this.onToggle(name, this.toggles[name]);
      if (name === "crossings") this.drawCrossings();
    }
    this.renderControls();
  }

  toggleLegend() {
    this.legendOpen = !this.legendOpen;
    Store.write("map.legend", this.legendOpen);
    this.renderLegend();
  }

  renderLegend() {
    const stop = (shape, size, extra) => `<span class="mk-stop mk-stop--${shape} sz-${size} veh-1 ${extra || ""}">1</span>`;
    const rows = [
      [`<span class="lg-line veh-1"></span>`, "route on roads, colour and number by vehicle id"],
      [`<span class="lg-line lg-line--dashed veh-1"></span>`, "vehicles 4 to 7: same hues, dashed, hollow badge"],
      [`<span class="mk-vehicle veh-1"><svg viewBox="0 0 26 26"><path d="M13 2l10 7.5-3.8 12H6.800L3 9.5z"/><text x="13" y="17.500" text-anchor="middle">0</text></svg></span>`, "vehicle position, numbered by id"],
      [`<span class="mk-depot"></span>`, "depot"],
      [stop("circle", 2), "support job, number is the visit order"],
      [stop("square", 2), "repossession pickup"],
      [`<span class="mk-drop veh-1"></span>`, "depot drops of a route"],
      [`${stop("circle", 1)}${stop("circle", 2)}${stop("circle", 3)}`, "priority 1 to 2, 3, 4 to 5"],
      [`<span class="mk-stop mk-stop--circle sz-2 is-unassigned">14</span>`, "unassigned job, number is the job id"],
      [`<span class="lg-line lg-line--ghost"></span>`, "removed leg or ghost lane"],
      [`<span class="lg-line lg-line--teacher"></span>`, "teacher alternative"],
    ];
    this.legendNode.innerHTML = `
      <button type="button" class="btn btn--sm" aria-expanded="${this.legendOpen}" title="Legend (key L)">${icon("legend")}<span>Legend</span>${icon(this.legendOpen ? "chevron-down" : "chevron-up")}</button>
      ${this.legendOpen ? `<div class="map__legendbody glass">${rows.map(([mark, text]) => `<div class="lg-row"><span class="lg-mark">${mark}</span><span>${text}</span></div>`).join("")}</div>` : ""}`;
  }

  latLng(lonLat) {
    return [lonLat[1], lonLat[0]];
  }

  marker(lonLat, html, size, pane, options = {}) {
    return L.marker(this.latLng(lonLat), {
      icon: L.divIcon({ className: "mk", html, iconSize: [size, size], iconAnchor: [size / 2, size / 2] }),
      pane,
      interactive: options.interactive !== false,
      draggable: Boolean(options.draggable),
      keyboard: false,
      zIndexOffset: options.z || 0,
    });
  }

  bindTip(marker, describe) {
    marker.on("mouseover", (event) => tooltip.show(describe(), event.originalEvent.clientX, event.originalEvent.clientY));
    marker.on("mousemove", (event) => tooltip.move(event.originalEvent.clientX, event.originalEvent.clientY));
    marker.on("mouseout", () => tooltip.hide());
  }

  bindDrag(marker, kind, id) {
    marker.on("dragstart", () => tooltip.hide());
    marker.on("dragend", () => {
      const position = marker.getLatLng();
      this.onDrag(kind, id, [Number(position.lng.toFixed(6)), Number(position.lat.toFixed(6))]);
    });
  }

  track(vehicleId) {
    if (!this.byVehicle.has(vehicleId)) this.byVehicle.set(vehicleId, { lines: [], markers: [] });
    return this.byVehicle.get(vehicleId);
  }

  sizeStep(priority) {
    return priority >= 4 ? 3 : priority === 3 ? 2 : 1;
  }

  setScene(scene) {
    this.cancelAnimation();
    this.scene = scene;
    for (const group of Object.values(this.groups)) group.clearLayers();
    this.byVehicle.clear();
    this.jobMarkers.clear();
    this.vehicleMarkers.clear();
    tooltip.hide();
    if (!scene) return;

    this.drawRoutes();
    this.drawStops();
    this.drawVehicles();
    this.drawDepot();
    this.drawChevrons();
    this.drawCrossings();
    this.applyIsolation();
    if (!this.fitted) this.fit(false);
  }

  drawRoutes() {
    const { plan, pathOf } = this.scene;
    for (const route of plan.routes) {
      const id = route.vehicle_id;
      const path = pathOf(route);
      if (path.length < 2) continue;
      const slot = Vehicles.slot(id);
      const casing = L.polyline(path, { pane: "routes", className: "route-casing", weight: 7, interactive: true, bubblingMouseEvents: false });
      const line = L.polyline(path, { pane: "routes", className: `route-line veh-${slot}${Vehicles.dashed(id) ? " is-dashed" : ""}`, weight: 3, interactive: false });
      casing.on("click", () => this.onIsolate(id));
      casing.addTo(this.groups.routes);
      line.addTo(this.groups.routes);
      this.track(id).lines.push(casing, line);
      this.track(id).line = line;

      const middle = this.badgePoint(route, path);
      const badge = this.marker([middle[1], middle[0]], `<span class="mk-routebadge veh-${slot}${Vehicles.dashed(id) ? " is-hollow" : ""}">vehicle ${id}</span>`, 18, "labels", { interactive: false });
      badge.addTo(this.groups.labels);
      this.track(id).markers.push(badge);
    }
  }

  badgePoint(route, path) {
    const anchors = [this.scene.vehicles[route.vehicle_id].start, this.scene.depot, ...route.stops.map((stop) => stop.location)];
    const stride = Math.max(1, Math.floor(path.length / 40));
    let best = path[Math.floor(path.length / 2)];
    let bestDistance = -1;
    for (let index = stride; index < path.length - 1; index += stride) {
      const point = path[index];
      let nearest = Infinity;
      for (const anchor of anchors) nearest = Math.min(nearest, (point[0] - anchor[1]) ** 2 + (point[1] - anchor[0]) ** 2);
      if (nearest > bestDistance) {
        bestDistance = nearest;
        best = point;
      }
    }
    return best;
  }

  drawStops() {
    const { plan, jobs, onboardOnly } = this.scene;
    for (const route of plan.routes) {
      const id = route.vehicle_id;
      const slot = Vehicles.slot(id);
      let order = 0;
      let drops = 0;
      for (const stop of route.stops) {
        if (stop.kind === "delivery") {
          drops += 1;
          continue;
        }
        order += 1;
        const job = jobs[stop.job_id];
        const shape = stop.kind === "pickup" ? "square" : "circle";
        const html = `<span class="mk-stop mk-stop--${shape} sz-${this.sizeStep(job.priority)} veh-${slot}${Vehicles.dashed(id) ? " is-hollow" : ""}" data-job="${job.id}">${order}</span>`;
        const marker = this.marker(stop.location, html, 26, "stops", { draggable: this.editable, z: job.priority * 10 });
        const visit = order;
        this.bindTip(marker, () => Tooltip.rows(`Job ${job.id}`, [
          ["kind", job.kind], ["priority", job.priority], ["vehicle", id], ["visit order", visit],
          ["arrival", fmtClock(stop.arrival)], ["service", `${fmtMin(job.service)} min`], ["load after stop", stop.load],
        ]));
        marker.on("click", () => this.onSelect({ kind: "job", id: job.id, vehicleId: id }));
        if (this.editable) this.bindDrag(marker, "job", job.id);
        marker.addTo(this.groups.stops);
        this.track(id).markers.push(marker);
        this.jobMarkers.set(job.id, marker);
      }
      if (drops && !onboardOnly) this.drawDrops(route, drops);
    }

    for (const entry of plan.unassigned) {
      const job = jobs[entry.job_id];
      const shape = job.kind === "repossession" ? "square" : "circle";
      const html = `<span class="mk-stop mk-stop--${shape} sz-${this.sizeStep(job.priority)} is-unassigned" data-job="${job.id}">${job.id}</span>`;
      const marker = this.marker(job.location, html, 26, "stops", { draggable: this.editable, z: 100 + job.priority * 10 });
      this.bindTip(marker, () => Tooltip.rows(`Job ${job.id}`, [["state", "unassigned"], ["kind", job.kind], ["priority", job.priority], ["waiting", `${fmtMin(entry.waiting_s)} min`], ["service", `${fmtMin(job.service)} min`]]));
      marker.on("click", () => this.onSelect({ kind: "job", id: job.id, vehicleId: null }));
      if (this.editable) this.bindDrag(marker, "job", job.id);
      marker.addTo(this.groups.stops);
      this.jobMarkers.set(job.id, marker);
    }
  }

  drawDrops(route, drops) {
    const depotPoint = this.map.latLngToLayerPoint(this.latLng(this.scene.depot));
    const angle = (route.vehicle_id % 8) * (Math.PI / 4) - Math.PI / 2;
    const marker = L.marker(this.latLng(this.scene.depot), {
      icon: L.divIcon({ className: "mk", html: `<span class="mk-drop veh-${Vehicles.slot(route.vehicle_id)}"></span>`, iconSize: [10, 10], iconAnchor: [5 - Math.cos(angle) * 15, 5 - Math.sin(angle) * 15] }),
      pane: "stops",
      keyboard: false,
    });
    this.bindTip(marker, () => Tooltip.rows(`Vehicle ${route.vehicle_id}`, [["depot drops on this route", drops]]));
    marker.addTo(this.groups.stops);
    this.track(route.vehicle_id).markers.push(marker);
    return depotPoint;
  }

  drawVehicles() {
    const { plan, vehicles } = this.scene;
    const idle = new Set(plan.idle_vehicle_ids);
    for (const [key, vehicle] of Object.entries(vehicles)) {
      const id = Number(key);
      const classes = `mk-vehicle veh-${Vehicles.slot(id)}${idle.has(id) ? " is-idle" : ""}${Vehicles.dashed(id) ? " is-hollow" : ""}`;
      const html = `<span class="${classes}" data-vehicle="${id}"><svg viewBox="0 0 26 26"><path d="M13 2l10 7.5-3.8 12H6.800L3 9.5z"/><text x="13" y="17.500" text-anchor="middle">${id}</text></svg></span>`;
      const marker = this.marker(vehicle.start, html, 26, "vehicles", { draggable: this.editable });
      const route = plan.routes.find((entry) => entry.vehicle_id === id);
      this.bindTip(marker, () => Tooltip.rows(`Vehicle ${id}`, [
        ["state", idle.has(id) ? "IDLE" : route ? route.kpis.state : "ACTIVE"],
        ["onboard", `${vehicle.onboard} bikes`],
        route ? ["stops", route.kpis.stops] : null,
        route ? ["drive time", `${fmtMin(route.drive_s)} min`] : null,
      ]));
      marker.on("click", () => {
        this.onSelect({ kind: "vehicle", id });
        this.onIsolate(id);
      });
      if (this.editable) this.bindDrag(marker, "vehicle", id);
      marker.addTo(this.groups.vehicles);
      this.track(id).markers.push(marker);
      this.vehicleMarkers.set(id, marker);
    }
  }

  drawDepot() {
    const marker = this.marker(this.scene.depot, `<span class="mk-depot"></span>`, 16, "depot", { draggable: this.editable });
    this.bindTip(marker, () => Tooltip.rows("Depot", [["role", "repossessed bikes are dropped here"]]));
    if (this.editable) this.bindDrag(marker, "depot", null);
    marker.addTo(this.groups.depot);
  }

  drawChevrons() {
    this.groups.chevrons.clearLayers();
    if (!this.scene) return;
    const spacing = 120;
    for (const route of this.scene.plan.routes) {
      const path = this.scene.pathOf(route);
      if (path.length < 2) continue;
      const slot = Vehicles.slot(route.vehicle_id);
      let travelled = spacing / 2;
      let previous = this.map.latLngToLayerPoint(path[0]);
      let placed = 0;
      for (let index = 1; index < path.length && placed < 60; index += 1) {
        const point = this.map.latLngToLayerPoint(path[index]);
        const length = previous.distanceTo(point);
        while (travelled <= length && length > 0 && placed < 60) {
          const share = travelled / length;
          const at = this.map.layerPointToLatLng([previous.x + (point.x - previous.x) * share, previous.y + (point.y - previous.y) * share]);
          const angle = (Math.atan2(point.y - previous.y, point.x - previous.x) * 180) / Math.PI;
          const chevron = L.marker(at, {
            icon: L.divIcon({ className: "mk", html: `<svg class="mk-chevron veh-${slot}" viewBox="0 0 12 12" style="transform:rotate(${angle.toFixed(0)}deg)"><path d="M2.500 1.500L10 6l-7.500 4.500z"/></svg>`, iconSize: [12, 12], iconAnchor: [6, 6] }),
            pane: "routes",
            interactive: false,
            keyboard: false,
          });
          chevron.addTo(this.groups.chevrons);
          this.track(route.vehicle_id).chevrons = this.track(route.vehicle_id).chevrons || [];
          this.track(route.vehicle_id).chevrons.push(chevron);
          travelled += spacing;
          placed += 1;
        }
        travelled -= length;
        previous = point;
      }
    }
    this.applyIsolation();
  }

  drawCrossings() {
    this.groups.crossings.clearLayers();
    if (!this.scene || !this.toggles.crossings) return;
    const starts = {};
    for (const [id, vehicle] of Object.entries(this.scene.vehicles)) starts[id] = vehicle.start;
    for (const point of Geometry.crossings(this.scene.plan.routes, starts)) {
      this.marker(point, `<span class="mk-crossing"></span>`, 14, "changes", { interactive: false }).addTo(this.groups.crossings);
    }
  }

  setGhost(routes, pathOf) {
    this.groups.ghost.clearLayers();
    if (!routes || !this.toggles.ghost) return;
    for (const route of routes) {
      const path = pathOf(route);
      if (path.length > 1) L.polyline(path, { pane: "routes", className: "route-ghost", weight: 3, interactive: false }).addTo(this.groups.ghost);
    }
  }

  setTeacherAlternative(alternative) {
    this.groups.teacher.clearLayers();
    if (!alternative || !this.toggles.teacher || !this.scene) return;
    const job = this.scene.jobs[alternative.job_id];
    const vehicle = this.scene.vehicles[alternative.vehicle_id];
    if (!job || !vehicle) return;
    const route = this.scene.plan.routes.find((entry) => entry.vehicle_id === alternative.vehicle_id);
    const anchors = [vehicle.start, ...(route ? route.stops.map((stop) => stop.location) : [])];
    let nearest = anchors[0];
    for (const anchor of anchors) {
      if ((anchor[0] - job.location[0]) ** 2 + (anchor[1] - job.location[1]) ** 2 < (nearest[0] - job.location[0]) ** 2 + (nearest[1] - job.location[1]) ** 2) nearest = anchor;
    }
    L.polyline([this.latLng(nearest), this.latLng(job.location)], { pane: "changes", className: "route-teacher", weight: 3, interactive: false }).addTo(this.groups.teacher);
    this.label(job.location, `teacher: job ${alternative.job_id} into vehicle ${alternative.vehicle_id}`, this.groups.teacher, "below", "teacher");
  }

  label(lonLat, text, group, placement, tone) {
    L.marker(this.latLng(lonLat), {
      icon: L.divIcon({ className: "mk", html: `<span class="mk-label mk-label--${placement || "above"}${tone ? ` mk-label--${tone}` : ""}">${esc(text)}</span>`, iconSize: [0, 0], iconAnchor: [0, 0] }),
      pane: "labels",
      interactive: false,
      keyboard: false,
    }).addTo(group || this.groups.changes);
  }

  showChange(change) {
    this.groups.changes.clearLayers();
    this.drawCrossings();
    if (!change || !this.scene) return;

    if (change.inserted) {
      const marker = this.jobMarkers.get(change.inserted.jobId);
      const track = this.byVehicle.get(change.inserted.vehicleId);
      if (track && track.line) track.line.setStyle({ weight: 5 });
      if (marker) {
        this.pulse(marker);
        this.label(this.scene.jobs[change.inserted.jobId].location, change.inserted.wasted ? `job ${change.inserted.jobId}: no feasible insertion` : `job ${change.inserted.jobId} inserted`);
        this.follow(marker.getLatLng());
      }
    }

    if (change.removed) {
      if (change.removed.oldPath && change.removed.oldPath.length > 1) {
        L.polyline(change.removed.oldPath, { pane: "changes", className: `route-removed veh-${Vehicles.slot(change.removed.vehicleId)}`, weight: 3, interactive: false }).addTo(this.groups.changes);
      }
      const marker = this.jobMarkers.get(change.removed.jobId);
      if (marker) {
        this.pulse(marker);
        this.label(this.scene.jobs[change.removed.jobId].location, `job ${change.removed.jobId} removed`);
        this.follow(marker.getLatLng());
      }
    }

    for (const outcome of change.outcomes || []) {
      const job = this.scene.jobs[outcome.jobId];
      this.marker(job.location, `<span class="mk-outcome mk-outcome--${outcome.word}"></span>`, 12, "changes", { interactive: false }).addTo(this.groups.changes);
      this.label(job.location, `job ${outcome.jobId} ${outcome.word}`, undefined, undefined, outcome.word);
    }

    for (const jobId of change.arrived || []) {
      const marker = this.jobMarkers.get(jobId);
      if (marker) {
        this.pulse(marker);
        this.label(this.scene.jobs[jobId].location, `new job ${jobId}`, undefined, undefined, "new");
      }
    }

    if (change.moves && change.moves.length) this.animate(change.moves);
  }

  pulse(marker) {
    const node = marker.getElement();
    if (!node) return;
    const inner = node.firstElementChild;
    inner.classList.remove("is-pulse");
    void inner.offsetWidth;
    inner.classList.add("is-pulse");
  }

  follow(latLng) {
    if (!this.toggles.follow || this.playing) return;
    if (this.map.getBounds().pad(-0.08).contains(latLng)) return;
    this.buildings.suspend();
    this.map.panTo(latLng, { animate: true, duration: 0.3 });
  }

  animate(moves) {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const usable = moves.filter((move) => this.vehicleMarkers.has(move.vehicleId) && move.path.length > 1);
    if (reduced || !usable.length) return;
    const started = performance.now();
    const duration = 1200;
    const tick = (now) => {
      const share = Math.min(1, (now - started) / duration);
      const eased = share < 0.5 ? 2 * share * share : 1 - (-2 * share + 2) ** 2 / 2;
      for (const move of usable) {
        const position = eased * (move.path.length - 1);
        const index = Math.min(move.path.length - 2, Math.floor(position));
        const rest = position - index;
        const a = move.path[index];
        const b = move.path[index + 1];
        this.vehicleMarkers.get(move.vehicleId).setLatLng([a[0] + (b[0] - a[0]) * rest, a[1] + (b[1] - a[1]) * rest]);
      }
      this.animation = share < 1 ? requestAnimationFrame(tick) : null;
    };
    this.animation = requestAnimationFrame(tick);
  }

  cancelAnimation() {
    if (this.animation) cancelAnimationFrame(this.animation);
    this.animation = null;
  }

  isolate(vehicleId) {
    this.isolated = vehicleId;
    this.applyIsolation();
  }

  applyIsolation() {
    for (const [id, track] of this.byVehicle) {
      const dim = this.isolated !== null && id !== this.isolated;
      for (const line of track.lines) {
        const node = line.getElement();
        if (node) node.classList.toggle("is-dim", dim);
      }
      for (const marker of [...track.markers, ...(track.chevrons || [])]) {
        const node = marker.getElement();
        if (node) node.classList.toggle("is-dim", dim);
      }
    }
  }

  highlight(target) {
    for (const node of this.root.querySelectorAll(".is-hot")) node.classList.remove("is-hot");
    if (!target) return;
    if (target.jobId !== undefined && target.jobId !== null && this.jobMarkers.has(target.jobId)) {
      const node = this.jobMarkers.get(target.jobId).getElement();
      if (node) node.firstElementChild.classList.add("is-hot");
    }
    if (target.vehicleId !== undefined && target.vehicleId !== null) {
      const track = this.byVehicle.get(target.vehicleId);
      if (track && track.line && track.line.getElement()) track.line.getElement().classList.add("is-hot");
      const marker = this.vehicleMarkers.get(target.vehicleId);
      if (marker && marker.getElement()) marker.getElement().firstElementChild.classList.add("is-hot");
    }
  }

  bounds() {
    if (!this.scene) return null;
    const points = [this.latLng(this.scene.depot)];
    for (const vehicle of Object.values(this.scene.vehicles)) points.push(this.latLng(vehicle.start));
    for (const route of this.scene.plan.routes) for (const stop of route.stops) points.push(this.latLng(stop.location));
    for (const entry of this.scene.plan.unassigned) points.push(this.latLng(this.scene.jobs[entry.job_id].location));
    for (const route of this.scene.plan.routes) {
      const path = this.scene.pathOf(route);
      const stride = Math.max(1, Math.floor(path.length / 60));
      for (let index = 0; index < path.length; index += stride) points.push(path[index]);
    }
    return L.latLngBounds(points);
  }

  fit(animate) {
    const bounds = this.bounds();
    if (!bounds) return;
    this.map.invalidateSize({ animate: false });
    if (this.map.getSize().y < 50) return;
    this.buildings.suspend();
    const size = this.map.getSize();
    const legendRoom = this.legendOpen && size.x > 900 ? this.legendNode.offsetWidth + 24 : 48;
    this.map.fitBounds(bounds, { paddingTopLeft: [legendRoom, 64], paddingBottomRight: [Math.min(190, size.x * 0.16), this.cover ? this.cover + 12 : 48], maxZoom: 16, animate: Boolean(animate) });
    this.fitted = true;
  }

  setCover(bottom) {
    if (bottom === this.cover) return;
    this.cover = bottom;
    if (this.fitted) this.fit(false);
  }

  resetFit() {
    this.fitted = false;
  }

  restyle() {
    this.buildings.restyle();
  }
}
