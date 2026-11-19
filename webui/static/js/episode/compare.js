import { esc, fmtDur, fmtFixed, fmtSigned, isNum, NONE, signClass } from "../format.js";
import { StackBar } from "../ui/charts.js";
import { infoText } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { statusChip } from "../ui/status.js";
import { DataTable } from "../ui/table.js";
import { Agents, Sentences } from "./encoding.js";

export class Compare {
  constructor(container, { actions }) {
    this.container = container;
    this.actions = actions;
    container.innerHTML = `<div class="cmp"><p class="drawer__empty muted" data-role="empty" hidden>No recorded lane to compare yet.</p><div data-role="table" class="dtable--static"></div><div class="cmp__grid" data-role="grid"></div></div>`;
    this.emptyNode = container.querySelector("[data-role=empty]");
    this.gridNode = container.querySelector("[data-role=grid]");
    this.infoNode = document.createElement("span");
    this.infoNode.className = "drawer__info";
    actions.appendChild(this.infoNode);
    this.table = new DataTable(container.querySelector("[data-role=table]"), {
      rowKey: (row) => row.key,
      storeKey: "episode.compare",
      menuHost: actions,
      columns: [
        { key: "key", label: "Agent", sortValue: (row) => Agents.order(row.key), render: (row) => `${Compare.agentCell(row.key)}${row.partial ? ` <span class="muted">partial</span>` : ""}` },
        { key: "total", label: "Return", unit: "pts", numeric: true, render: (row) => `<span class="strong ${signClass(row.total)}">${fmtSigned(row.total, 2)}</span>` },
        { key: "holding", label: "Holding", unit: "pts", numeric: true, format: (value) => fmtSigned(value, 2), title: "Sum of the holding term" },
        { key: "action", label: "Action", unit: "pts", numeric: true, format: (value) => fmtSigned(value, 2), title: "Sum of the action term" },
        { key: "plan", label: "Plan change", unit: "pts", numeric: true, format: (value) => fmtSigned(value, 2), title: "Sum of the distance, unassigned, idle and priority terms" },
        { key: "unassigned", label: "Unassigned", full: "Final unassigned", numeric: true, title: "Jobs unassigned at the end of the episode" },
        { key: "mix", label: "Mix", full: "Operator mix", sortable: false, render: (row) => StackBar.html(row.counts, { showWords: false }), title: "INSERT, REMOVE and DO_NOTHING shares of the decisions" },
        { key: "wasted", label: "Wasted", full: "Wasted edits", numeric: true, render: (row) => (row.wasted ? `${row.wasted} ${statusChip("WASTED")}` : "0") },
        { key: "drive", label: "Drive", full: "Final drive", unit: "h:mm", numeric: true, format: fmtDur, hideable: true, hidden: true },
        { key: "served", label: "Served", numeric: true, hideable: true, hidden: true },
        { key: "failed", label: "Failed", numeric: true, hideable: true, hidden: true },
        { key: "dropped", label: "Dropped", numeric: true, hideable: true, hidden: true },
        { key: "counts_text", label: "Ins / rem / hold", full: "Decisions, INSERT / REMOVE / DO_NOTHING", numeric: true, sortable: false, hideable: true, hidden: true },
        { key: "edits", label: "Edits / round", full: "Edits per round", numeric: true, format: (value) => fmtFixed(value, 1), hideable: true, hidden: true },
      ],
    });
    this.table.container.classList.add("dtable--static");
  }

  static sumTerms(summary, keys) {
    return keys.reduce((total, key) => total + summary.reward_by_term[key], 0);
  }

  static roundRewards(trace) {
    return trace.rounds().map((round) => (round ? round.decisions.reduce((total, frame) => total + frame.decision.reward, 0) : null));
  }

  static agentCell(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  static row(lane) {
    const summary = lane.trace.summary;
    const rounds = summary.edits_per_round.length;
    return {
      key: lane.key,
      partial: !lane.trace.complete,
      total: summary.total_reward,
      holding: summary.reward_by_term.holding_reward,
      action: summary.reward_by_term.action_reward,
      plan: Compare.sumTerms(summary, ["distance_reward", "unassigned_reward", "idle_reward", "priority_reward"]),
      unassigned: summary.final_unassigned,
      counts: summary.operator_counts,
      counts_text: summary.operator_counts.join(" / "),
      wasted: summary.wasted_edits,
      drive: summary.final_drive_s,
      served: summary.served,
      failed: summary.failed,
      dropped: summary.dropped,
      edits: rounds ? summary.edits_per_round.reduce((a, b) => a + b, 0) / rounds : null,
    };
  }

  static matrix(title, unit, lanes, values, signed) {
    const rounds = Math.max(0, ...values.map((list) => list.length));
    if (!rounds) return "";
    const extent = Math.max(1e-9, ...values.flat().filter(isNum).map(Math.abs));
    const head = Array.from({ length: rounds }, (_, round) => `<th class="num">Round ${round + 1}</th>`).join("");
    const body = lanes.map((lane, index) => {
      const cells = Array.from({ length: rounds }, (_, round) => {
        const value = values[index][round];
        if (!isNum(value)) return `<td class="num faint">${NONE}</td>`;
        const share = (Math.abs(value) / extent) * (signed ? 50 : 100);
        const style = signed ? (value >= 0 ? `left:50%;width:${share.toFixed(1)}%` : `right:50%;width:${share.toFixed(1)}%`) : `left:0;width:${share.toFixed(1)}%`;
        const tone = signed ? (value >= 0 ? "gain" : "loss") : "plain";
        return `<td class="num cmp__cell"><span class="cmp__bar"><i class="cmp__fill cmp__fill--${tone}" style="${style}"></i>${signed ? `<i class="cmp__zero"></i>` : ""}</span><span class="cmp__num">${signed ? fmtSigned(value, 2) : value}</span></td>`;
      }).join("");
      return `<tr><td>${Compare.agentCell(lane.key)}</td>${cells}</tr>`;
    }).join("");
    return `
      <div class="cmp__matrix">
        <h4>${esc(title)} <span class="muted">${esc(unit)}</span></h4>
        <div class="dtable dtable--dense dtable--static"><table><thead><tr><th>Agent</th>${head}</tr></thead><tbody>${body}</tbody></table></div>
      </div>`;
  }

  static sharedInfo(status) {
    const gaps = status.shared.gaps.map((gap, round) => `round ${round + 1}: ${round === 0 ? "start" : `${Math.round(gap / 60)} min gap`}${status.shared.events[round] ? `, ${Sentences.eventKind(status.shared.events[round].type)} x ${status.shared.events[round].count}` : ", no event"}`).join("; ");
    const clean = status.divergence.length ? null : "No round where entity ids or pickup outcomes differ between the recorded agents.";
    return infoText("Every lane starts from the same jobs, vehicles and depot and sees the same gaps and the same event type and count per round. Which ids an event hits and which pickups fail can differ once the plans differ.", gaps ? `${gaps.charAt(0).toUpperCase()}${gaps.slice(1)}.` : null, clean);
  }

  static flags(status) {
    if (!status.divergence.length) return "";
    return `<ul class="cmp__flags">${status.divergence.map((entry) => `<li>${statusChip("WARN", `round ${entry.round + 1}`)} ${entry.kind === "event_ids" ? "the disruption hit different entity ids" : "pickup outcomes differ"} for ${entry.agents.map((key) => esc(Agents.label(key))).join(", ")}</li>`).join("")}</ul>`;
  }

  render(lanes, status) {
    const usable = lanes.filter((lane) => lane.trace.summary && lane.trace.length);
    this.emptyNode.hidden = usable.length > 0;
    this.table.container.hidden = !usable.length;
    this.infoNode.innerHTML = usable.length ? Popover.infoButton("What the agents share", Compare.sharedInfo(status)) : "";
    if (!usable.length) {
      this.gridNode.innerHTML = "";
      return;
    }
    this.table.setRows(usable.map((lane) => Compare.row(lane)));
    const unassigned = usable.map((lane) => lane.trace.summary.unassigned_at_commit);
    const rewards = usable.map((lane) => Compare.roundRewards(lane.trace));
    this.gridNode.innerHTML = `
      ${Compare.matrix("Unassigned jobs at commit", "jobs", usable, unassigned, false)}
      ${Compare.matrix("Reward per round", "points", usable, rewards, true)}
      ${Compare.flags(status)}`;
  }
}
