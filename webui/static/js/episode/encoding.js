import { esc, listIds, plural } from "../format.js";

export const OPERATORS = ["INSERT", "REMOVE", "DO_NOTHING"];
export const OPERATOR_ICONS = ["insert", "remove", "hold"];

export const TERMS = [
  { key: "distance_reward", label: "Distance", cost: "distance_cost" },
  { key: "unassigned_reward", label: "Unassigned", cost: "unassigned_cost" },
  { key: "idle_reward", label: "Idle", cost: "idle_cost" },
  { key: "priority_reward", label: "Priority", cost: "priority_cost" },
  { key: "action_reward", label: "Action", cost: null },
  { key: "holding_reward", label: "Holding", cost: null },
];

export class Vehicles {
  static slot(vehicleId) {
    return (Number(vehicleId) % 4) + 1;
  }

  static dashed(vehicleId) {
    return Math.floor(Number(vehicleId) / 4) % 2 === 1;
  }

  static swatch(vehicleId, idle) {
    const variant = idle ? " swatch--idle" : Vehicles.dashed(vehicleId) ? " swatch--hollow" : "";
    return `<span class="swatch veh-${Vehicles.slot(vehicleId)}${variant}" title="Vehicle ${esc(vehicleId)}">${esc(vehicleId)}</span>`;
  }
}

export class Agents {
  static parse(agentKey) {
    const parts = agentKey.split("~");
    if (parts[0] === "model" && parts.length === 3) return { agent: "model", run: parts[1], sampled: parts[2] === "sampled" };
    if (agentKey === "model_sampled") return { agent: "model", run: null, sampled: true };
    return { agent: agentKey, run: null, sampled: false };
  }

  static label(agentKey) {
    const parsed = Agents.parse(agentKey);
    if (parsed.agent === "model") return `model${parsed.run ? " " + parsed.run : ""}${parsed.sampled ? " (sampled)" : ""}`;
    return { teacher: "teacher", insertion_only: "insertion only", do_nothing: "do nothing" }[parsed.agent] || agentKey;
  }

  static style(agentKey) {
    return Agents.parse(agentKey).agent;
  }

  static isModel(agentKey) {
    return Agents.parse(agentKey).agent === "model";
  }

  static order(agentKey) {
    return ["model", "teacher", "insertion_only", "do_nothing"].indexOf(Agents.parse(agentKey).agent);
  }
}

export class Sentences {
  static action(action) {
    if (action.operator === 0) return `INSERT job ${action.job_id} into vehicle ${action.vehicle_id}`;
    if (action.operator === 1) return `REMOVE job ${action.job_id} from vehicle ${action.vehicle_id}`;
    return "DO_NOTHING (commit the plan)";
  }

  static event(event) {
    if (!event) return "no disruption";
    if (event.type === "new_job") return `new ${event.added_job_ids.length === 1 ? "job" : "jobs"} ${listIds(event.added_job_ids)}`;
    if (event.type === "remove_job") return `${event.removed_job_ids.length === 1 ? "job" : "jobs"} ${listIds(event.removed_job_ids)} cancelled`;
    if (event.type === "new_vehicle") return `new ${event.added_vehicle_ids.length === 1 ? "vehicle" : "vehicles"} ${listIds(event.added_vehicle_ids)}`;
    if (event.type === "remove_vehicle") {
      const orphaned = event.orphaned_job_ids.length ? `, ${plural(event.orphaned_job_ids.length, "job")} ${listIds(event.orphaned_job_ids)} orphaned` : "";
      return `${event.removed_vehicle_ids.length === 1 ? "vehicle" : "vehicles"} ${listIds(event.removed_vehicle_ids)} removed${orphaned}`;
    }
    return `${event.type} (${event.count})`;
  }

  static eventKind(type) {
    return { new_job: "new job", remove_job: "job cancelled", new_vehicle: "new vehicle", remove_vehicle: "vehicle removed" }[type] || type;
  }

  static executed(executed) {
    const dropped = executed.dropped.reduce((sum, entry) => sum + entry.count, 0);
    return `served ${executed.served.length}, failed ${executed.failed.length}, dropped ${dropped}`;
  }

  static frame(frame, caps) {
    const round = `Round ${frame.round + 1}`;
    if (frame.kind === "round_open") {
      const open = frame.round_open;
      if (frame.round === 0) return `${round} opens: ${Sentences.event(open.event)}`;
      return `${round} opens after ${Math.round(open.gap_s / 60)} min: ${Sentences.executed(open.executed)}; ${Sentences.event(open.event)}`;
    }
    return `${round}, edit ${frame.edit + 1} of ${caps.edits}: ${Sentences.action(frame.decision.action)}`;
  }
}
