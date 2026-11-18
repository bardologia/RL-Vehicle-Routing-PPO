import { fmtClock, fmtDur, fmtFixed, fmtKm, fmtPct, fmtSignedDur, isNum, NONE } from "../format.js";
import { StepProfile } from "../ui/charts.js";
import { statusChip } from "../ui/status.js";
import { DataTable } from "../ui/table.js";
import { Vehicles } from "./encoding.js";

export class RouteTable {
  constructor(container, { onIsolate, onHover }) {
    this.table = new DataTable(container, {
      rowKey: (row) => row.vehicle_id,
      storeKey: "routes",
      onSelect: (row) => onIsolate(row.vehicle_id),
      onHover: (row) => onHover(row ? row.vehicle_id : null),
      emptyText: "No vehicles in this frame.",
      columns: [
        { key: "vehicle_id", label: "Vehicle", render: (row) => Vehicles.swatch(row.vehicle_id, row.state === "IDLE") },
        { key: "state", label: "State", render: (row) => (row.state === "ACTIVE" ? `<span>ACTIVE</span>` : statusChip(row.state)) },
        { key: "stops", label: "Stops", numeric: true, title: "Job and pickup stops on the route" },
        { key: "stops_support", label: "Support", numeric: true, hideable: true, hidden: true },
        { key: "stops_repossession", label: "Pickups", numeric: true, hideable: true, hidden: true },
        { key: "depot_drops", label: "Depot drops", numeric: true, hideable: true },
        { key: "drive_s", label: "Drive", unit: "h:mm", numeric: true, format: fmtDur, bar: { value: (row) => row.drive_s, max: (rows) => Math.max(1, ...rows.map((row) => row.drive_s)) } },
        { key: "drive_delta_s", label: "Step change", unit: "h:mm", numeric: true, render: (row) => (row.drive_delta_s ? `<span class="strong">${fmtSignedDur(row.drive_delta_s)}</span>` : `<span class="faint">0:00</span>`), title: "Change of drive time caused by this frame" },
        { key: "drive_share", label: "Share", numeric: true, format: (value) => fmtPct(value), title: "Share of the plan drive time" },
        { key: "distance_m", label: "Distance", unit: "km", numeric: true, format: (value) => fmtKm(value, 1) },
        { key: "service_s", label: "Service", unit: "h:mm", numeric: true, format: fmtDur },
        { key: "end_clock", label: "Ends", numeric: true, format: fmtClock, title: "Completion of the last stop" },
        { key: "slack_s", label: "Shift slack", unit: "h:mm", numeric: true, render: (row) => (!isNum(row.slack_s) ? NONE : row.slack_s < 0 ? `${statusChip("FAIL")} ${fmtDur(row.slack_s)}` : fmtDur(row.slack_s)), title: "Shift end minus the completion of the last stop" },
        { key: "peak_load", label: "Peak load", numeric: true, render: (row) => `${row.peak_load} of ${row.capacity}`, title: "Highest load against capacity, in bikes" },
        { key: "load_profile", label: "Load along stops", sortable: false, render: (row) => StepProfile.svg(row.load_profile, row.capacity) || `<span class="faint">${NONE}</span>` },
        { key: "onboard", label: "Onboard", numeric: true, hideable: true, hidden: true },
        { key: "priority_sum", label: "Priority", unit: "pts", numeric: true, title: "Sum of job priorities on the route" },
        { key: "detour", label: "Detour", numeric: true, format: (value) => fmtFixed(value, 2), title: "Road distance over straight-line distance through the stops" },
        { key: "crossings", label: "Crossings", numeric: true },
        { key: "reward_weight", label: "Reward weight", unit: "pts", numeric: true, format: (value) => fmtFixed(value, 1), title: "w_distance x drive time / 1000" },
        { key: "served_last", label: "Served", numeric: true, title: "Jobs of this vehicle served in the latest advance", hideable: true, hidden: true },
        { key: "failed_last", label: "Failed", numeric: true, title: "Pickups of this vehicle that failed in the latest advance", hideable: true, hidden: true },
      ],
    });
  }

  static rows(plan) {
    const rows = plan.routes.map((route) => ({ vehicle_id: route.vehicle_id, ...route.kpis }));
    for (const idle of plan.idle || []) rows.push({ vehicle_id: idle.vehicle_id, ...idle.kpis });
    return rows.sort((a, b) => a.vehicle_id - b.vehicle_id);
  }

  render(plan, isolated) {
    this.table.setRows(RouteTable.rows(plan));
    this.table.setSelected(isolated);
  }

  setIsolated(vehicleId) {
    this.table.setSelected(vehicleId);
  }
}
