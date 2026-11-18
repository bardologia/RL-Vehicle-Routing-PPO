import { fmtDur, fmtFixed, fmtKm, fmtMin, fmtPct, fmtSigned, fmtSignedDur, fmtSignedInt, isNum, NONE } from "../format.js";
import { kpiRow } from "../ui/kpi.js";

export class KpiStrip {
  static DEFS = [
    { key: "drive_s", label: "Plan drive time", unit: "h:mm", value: fmtDur, delta: fmtSignedDur, title: "Sum of the VROOM travel time of every route. This is what the reward calls distance cost." },
    { key: "distance_m", label: "Road distance", unit: "km", value: (v) => fmtKm(v, 0), delta: (v) => fmtSigned(v / 1000, 0), title: "OSRM road distance over each route's stop order." },
    { key: "unassigned", label: "Unassigned", unit: "jobs", value: String, delta: fmtSignedInt, sub: (k) => `prio ${k.unassigned_priority}`, title: "Jobs on no route, and the sum of their priorities." },
    { key: "oldest_unassigned_s", label: "Oldest unassigned", unit: "min", value: (v) => fmtMin(v), delta: (v) => fmtSigned(v / 60, 0), title: "Longest time an unassigned job has been known." },
    { key: "idle_vehicles", label: "Idle vehicles", unit: "", value: String, delta: fmtSignedInt, title: "Vehicles in the pool without a route." },
    { key: "standing_cost", label: "Standing cost", unit: "pts", value: (v) => fmtFixed(v, 1), delta: (v) => fmtSigned(v, 1), title: "w_distance x drive time / 1000 + w_unassigned x unassigned + w_idle x idle vehicles + w_priority x unassigned priority. Reward points." },
    { key: "holding_rate", label: "Holding rate", unit: "pts/h", value: (v) => fmtFixed(v, 1), delta: (v) => fmtSigned(v, 1), title: "What this plan costs per hour of gap if committed: unassigned and priority penalty per 3600 s." },
    { key: "served", label: "Executed", unit: "", value: (v, k) => `${k.served} / ${k.failed} / ${k.dropped}`, note: "served / failed / dropped", title: "Cumulative jobs served, pickups failed and bikes dropped at the depot up to this frame." },
    { key: "drive_per_served_s", label: "Drive per served", unit: "min", value: (v) => fmtMin(v), delta: (v) => fmtSigned(v / 60, 0), title: "Current plan drive time divided by the jobs served so far." },
    { key: "load_use", label: "Load use", unit: "", value: (v) => fmtPct(v), delta: (v) => `${fmtSigned(v * 100, 0)} pt`, title: "Sum of peak load over sum of capacity on active routes." },
    { key: "crossings", label: "Crossings", unit: "", value: String, delta: fmtSignedInt, title: "Intersections between straight stop-to-stop legs of different routes." },
    { key: "return", label: "Return so far", unit: "pts", value: (v) => fmtSigned(v, 2), delta: (v) => fmtSigned(v, 2), title: "Cumulative reward up to this frame; holding is the part charged for plans left waiting." },
  ];

  static tiles(kpis, previous, start) {
    return KpiStrip.DEFS.map((def) => {
      const raw = kpis[def.key];
      const tile = { key: def.key, label: def.label, unit: def.unit, title: def.title, value: isNum(raw) ? def.value(raw, kpis) : NONE };
      if (def.sub) tile.sub = def.sub(kpis);
      if (def.note) {
        tile.note = def.note;
        return tile;
      }
      if (previous !== undefined) tile.deltaPrev = previous && isNum(raw) && isNum(previous[def.key]) ? def.delta(raw - previous[def.key]) : NONE;
      if (start !== undefined) tile.deltaStart = start && isNum(raw) && isNum(start[def.key]) ? def.delta(raw - start[def.key]) : NONE;
      return tile;
    });
  }

  static html(trace, seq) {
    const frame = trace.frame(seq);
    const previous = seq > 0 ? trace.frame(seq - 1) : null;
    const start = trace.frame(0);
    return kpiRow(KpiStrip.tiles(frame.plan.kpis, previous ? previous.plan.kpis : null, start ? start.plan.kpis : null));
  }

  static planHtml(kpis) {
    return kpiRow(KpiStrip.tiles(kpis).filter((tile) => !["served", "drive_per_served_s", "return", "oldest_unassigned_s"].includes(tile.key)));
  }
}
