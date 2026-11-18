import { esc, fmtFixed, isNum } from "../format.js";
import { statusChip } from "../ui/status.js";
import { Vehicles } from "./encoding.js";

export class Inputs {
  static UNITS = { drive_s: "s", clock_s: "s", service_s: "s", setup_s: "s", shift_left_s: "s", depot_distance_m: "m", lon: "deg", lat: "deg" };

  static raw(name, value) {
    if (!isNum(value)) return String(value);
    const text = Number.isInteger(value) ? String(value) : fmtFixed(value, Math.abs(value) < 100 ? 4 : 1);
    return Inputs.UNITS[name] ? `${text} ${Inputs.UNITS[name]}` : text;
  }

  static rows(names, raw, normalised, flagRange) {
    return names.map((name, index) => {
      const value = normalised[index];
      const outside = flagRange && !["lon", "lat"].includes(name) && (value < 0 || value > 1.5);
      return `<tr><td><code>${esc(name)}</code></td><td class="num">${esc(Inputs.raw(name, raw[index]))}</td><td class="num">${fmtFixed(value, 3)}</td><td>${outside ? statusChip("WARN", "outside 0 to 1.5") : ""}</td></tr>`;
    }).join("");
  }

  static table(title, names, raw, normalised, flagRange) {
    return `
      <div class="inputs__block">
        <h4>${title}</h4>
        <div class="dtable dtable--dense dtable--static"><table>
          <thead><tr><th>Feature</th><th class="num">Raw value</th><th class="num">Normalised input</th><th></th></tr></thead>
          <tbody>${Inputs.rows(names, raw, normalised, flagRange)}</tbody>
        </table></div>
      </div>`;
  }

  static html(frame, selection) {
    if (!frame || frame.kind !== "decision") return `<p class="drawer__empty muted">Model inputs are recorded on decision frames. Step to a decision.</p>`;
    const features = frame.decision.features;
    if (!features) return `<p class="drawer__empty muted">Rule-based agent: no model inputs were computed on this frame.</p>`;

    const context = Inputs.table("Context, 8 values the policy sees for the whole state", features.context.names, features.context.raw, features.context.normalised, true);
    let entity = `<div class="inputs__block"><h4>Entity features</h4><p class="muted">Click a job or a vehicle on the map to see its nine input features.</p></div>`;
    if (selection && selection.kind === "job") {
      const row = features.job.rows[selection.id];
      entity = row ? Inputs.table(`Job ${selection.id}, 9 features`, features.job.names, row.raw, row.normalised, true) : `<div class="inputs__block"><h4>Job ${selection.id}</h4><p class="muted">This job is not in the graph on this frame.</p></div>`;
    }
    if (selection && selection.kind === "vehicle") {
      const row = features.vehicle.rows[selection.id];
      entity = row ? Inputs.table(`Vehicle ${Vehicles.swatch(selection.id)}, 9 features`, features.vehicle.names, row.raw, row.normalised, true) : `<div class="inputs__block"><h4>Vehicle ${selection.id}</h4><p class="muted">This vehicle is not in the graph on this frame.</p></div>`;
    }
    return `<div class="inputs">${context}${entity}</div>`;
  }
}
