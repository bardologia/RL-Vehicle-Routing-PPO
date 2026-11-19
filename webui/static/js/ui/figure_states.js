import { Agents } from "../episode/encoding.js";
import { isNum } from "../format.js";

export class FigureStates {
  static write(figure, state) {
    if (!figure) return;
    const key = JSON.stringify(state);
    if (figure.dataset.stateKey === key && figure.state !== undefined) return;
    figure.dataset.stateKey = key;
    figure.state = state;
  }

  static balance(report) {
    if (!report || !report.agents.model) return { mine: 0, others: 0, alarm: false };
    const means = Object.values(report.agents).map((agent) => agent.mean_reward).filter(isNum);
    const floor = Math.min(...means);
    const model = report.agents.model.mean_reward;
    const other = report.agents.teacher ? report.agents.teacher.mean_reward : Math.max(...Object.entries(report.agents).filter(([key]) => key !== "model").map(([, agent]) => agent.mean_reward));
    const paired = report.paired.teacher;
    return { mine: model - floor, others: isNum(other) ? other - floor : 0, alarm: Boolean(paired && paired.ci95[1] < 0) };
  }

  static checkpoints(runs, selected) {
    const ordered = [...runs].sort((a, b) => b.mtime - a.mtime);
    let picked = ordered.slice(0, 8);
    const chosen = ordered.find((run) => run.name === selected);
    if (chosen && !picked.includes(chosen)) picked = [...picked.slice(0, 7), chosen];
    const slots = picked.map((run) => ({ state: !run.has_checkpoint ? "none" : run.loadable ? "loadable" : "incompatible", report: run.reports.length > 0, kind: run.kind === "ppo" ? "ppo" : "pretrain" }));
    const index = picked.findIndex((run) => run.name === selected);
    return { names: picked.map((run) => run.name), state: { slots, selected: index < 0 ? null : index } };
  }

  static queue(jobs) {
    const count = (...states) => jobs.filter((job) => states.includes(job.status)).length;
    return { running: count("running"), queued: count("queued", "pending"), finished: count("finished"), failed: count("failed", "stopped", "cancelled", "error") };
  }

  static diverge(report, index) {
    if (!report) return { rounds: 3, split: null, gap: 0, agree: null, teacher: false };
    const rounds = Math.max(1, Math.min(10, index ? index.meta.caps.rounds : report.meta.caps.max_disruptions_per_episode));
    const rows = index ? index.episodes.flatMap((episode) => episode.divergent_rounds) : [];
    const split = rows.length ? Math.min(rounds - 1, Math.min(...rows)) : null;
    const paired = report.paired.teacher;
    const scale = paired ? Math.max(Math.abs(paired.ci95[0]), Math.abs(paired.ci95[1]), 1e-9) : 1;
    const gap = paired ? Math.max(-1, Math.min(1, paired.mean_diff / scale)) : 0;
    const agreement = index && index.aggregates.agreement ? index.aggregates.agreement.operator : null;
    return { rounds, split, gap, agree: isNum(agreement) ? Math.max(0, Math.min(1, agreement)) : null, teacher: "teacher" in report.agents };
  }

  static confusion(index) {
    const agreement = index && index.aggregates.agreement ? index.aggregates.agreement : null;
    return { counts: agreement ? agreement.confusion.flat().map((value) => Math.max(0, value)) : Array(9).fill(0) };
  }

  static heads(report) {
    const share = (agent) => (agent && agent.operator_frequency ? [0, 1, 2].map((op) => Math.max(0, Math.min(1, agent.operator_frequency[`op${op}`] || 0))) : null);
    const operator = share(report && report.agents.model) || [0, 0, 0];
    const chosen = operator.indexOf(Math.max(...operator));
    return { operator, reference: share(report && report.agents.teacher), masked: [false, false, false], vehicle: null, job: null, chosen: [chosen, null, null] };
  }

  static decisionHeads(decision) {
    const policy = decision.policy;
    const clamp = (value) => Math.max(0, Math.min(1, isNum(value) ? value : 0));
    const operator = policy.operator.masked.map((masked, k) => (masked ? 0 : clamp(policy.operator.probs[k])));
    const o = decision.action.operator;
    let vehicle = null;
    let job = null;
    let v = null;
    let j = null;
    if (o < 2 && policy.vehicle_probs && policy.vehicle_probs[o]) {
      const at = policy.vehicle_ids.indexOf(decision.action.vehicle_id);
      vehicle = policy.vehicle_probs[o].slice(0, 4).map(clamp);
      if (at >= 0 && at < 4) v = at;
      const row = at >= 0 && policy.job_probs && policy.job_probs[o] ? policy.job_probs[o][at] : null;
      if (row) {
        const entries = row.map((p, k) => ({ p, id: policy.job_ids[k] })).filter((entry) => isNum(entry.p)).sort((a, b) => b.p - a.p);
        let top = entries.slice(0, 8);
        const taken = entries.find((entry) => entry.id === decision.action.job_id);
        if (taken && !top.includes(taken)) top = [...top.slice(0, 7), taken];
        if (top.length) {
          job = top.map((entry) => clamp(entry.p));
          const k = top.indexOf(taken);
          j = k < 0 ? null : k;
        }
      }
    }
    return { operator, reference: policy.operator.raw_probs.map(clamp), masked: policy.operator.masked.map(Boolean), vehicle: vehicle && vehicle.length ? vehicle : null, job, chosen: [o, vehicle && vehicle.length ? v : null, job ? j : null] };
  }

  static insert(frame) {
    const decision = frame.decision;
    const action = decision.action;
    const route = frame.plan.routes.find((entry) => entry.vehicle_id === action.vehicle_id);
    return { operator: action.operator, stops: Math.min(40, route ? route.stops.length : 0), unassigned: frame.plan.kpis.unassigned, wasted: Boolean(decision.wasted), commit: Boolean(decision.is_commit) };
  }

  static disruption(frame) {
    const open = frame.round_open;
    const event = open.event;
    return { type: event ? event.type : null, count: event ? Math.max(0, Math.min(6, event.count)) : 0, gap: Math.max(0, open.gap_s || 0), served: open.executed.served.length, failed: open.executed.failed.length, dropped: open.executed.dropped.length };
  }

  static terms(index, termKeys) {
    if (!index) return { values: Array(6).fill(null) };
    const rows = index.episodes.map((episode) => episode.agents.model).filter((agent) => agent);
    return { values: termKeys.map((key) => (rows.length ? rows.reduce((sum, agent) => sum + agent.reward_by_term[key], 0) / rows.length : null)) };
  }

  static ranking(index) {
    if (!index) return { ranks: Array(5).fill(null), rho: null };
    const ranks = [1, 2, 3, 4, 5].map((priority) => {
      const values = index.decisions.map((decision) => (decision.job_rank_by_priority ? decision.job_rank_by_priority[priority] : undefined)).filter(isNum);
      return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    });
    const rho = index.aggregates.job_ranking ? index.aggregates.job_ranking.spearman : null;
    return { ranks: ranks.map((value) => (value !== null && value > 0 ? value : null)), rho: isNum(rho) ? Math.max(-1, Math.min(1, rho)) : null };
  }

  static recordingFromEpisode(episode) {
    if (!episode) return { lanes: [], rounds: null };
    const keys = Object.keys(episode.agents).sort((a, b) => Agents.order(a) - Agents.order(b)).slice(0, 4);
    return { lanes: keys.map((key) => FigureStates.lane(episode.agents[key].status, episode.agents[key].status === "finished" ? 12 : 0, null)), rounds: null };
  }

  static recordingFromStatus(status) {
    const keys = Object.keys(status.lanes).sort((a, b) => Agents.order(a) - Agents.order(b)).slice(0, 4);
    const total = status.caps && status.caps.rounds ? status.caps.rounds * ((status.caps.edits || 0) + 1) : null;
    const rounds = status.caps && status.caps.rounds ? Math.max(1, Math.min(10, status.caps.rounds)) : null;
    return { lanes: keys.map((key) => FigureStates.lane(status.lanes[key].status, Math.max(0, status.lanes[key].frames || 0), total)), rounds };
  }

  static lane(status, frames, total) {
    const known = ["queued", "running", "finished", "failed", "cancelled"];
    return { frames: Math.round(frames), total: total && total > 0 ? Math.round(total) : null, status: known.includes(status) ? status : "queued" };
  }

  static training(values, live) {
    const finite = (values || []).filter(isNum);
    if (!finite.length) return { values: Array(24).fill(null), best: null, live: Boolean(live) };
    const n = values.length;
    const bins = Array.from({ length: 24 }, (_, b) => {
      const from = Math.floor((b * n) / 24);
      const to = Math.max(from + 1, Math.floor(((b + 1) * n) / 24));
      const slice = values.slice(from, Math.min(n, to)).filter(isNum);
      if (slice.length) return slice.reduce((a, c) => a + c, 0) / slice.length;
      const at = Math.min(n - 1, Math.round(((b + 0.5) * n) / 24 - 0.5));
      return isNum(values[at]) ? values[at] : null;
    });
    const present = bins.filter(isNum);
    const lo = Math.min(...present);
    const hi = Math.max(...present);
    const scaled = bins.map((value) => (value === null ? null : hi > lo ? (value - lo) / (hi - lo) : 0.5));
    let best = null;
    scaled.forEach((value, k) => {
      if (value !== null && (best === null || value > scaled[best])) best = k;
    });
    return { values: scaled, best, live: Boolean(live) };
  }

  static fleet(plan, scenario, solving) {
    if (plan) {
      return { routes: plan.routes.slice(0, 4).map((route) => ({ stops: route.stops.length, drive: Math.max(0, Math.min(1, route.kpis && isNum(route.kpis.drive_share) ? route.kpis.drive_share : 0)) })), unassigned: plan.unassigned.length, idle: Math.min(4, plan.idle_vehicle_ids.length), solving };
    }
    const vehicles = scenario ? scenario.vehicles : [];
    return { routes: vehicles.slice(0, 4).map(() => ({ stops: 0, drive: 0 })), unassigned: scenario ? scenario.jobs.length : 0, idle: 0, solving };
  }
}
