import { esc, fmtDur, fmtFixed, fmtSigned, isNum, NONE, signClass } from "../format.js";
import { StackBar } from "../ui/charts.js";
import { statusChip } from "../ui/status.js";
import { Agents, Sentences } from "./encoding.js";

export class Compare {
  static sumTerms(summary, keys) {
    return keys.reduce((total, key) => total + summary.reward_by_term[key], 0);
  }

  static roundRewards(trace) {
    return trace.rounds().map((round) => (round ? round.decisions.reduce((total, frame) => total + frame.decision.reward, 0) : null));
  }

  static agentCell(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  static summaryTable(lanes) {
    const rows = lanes.map((lane) => {
      const summary = lane.trace.summary;
      const rounds = summary.edits_per_round.length;
      const meanEdits = rounds ? summary.edits_per_round.reduce((a, b) => a + b, 0) / rounds : null;
      const cells = [
        `<td class="num strong ${signClass(summary.total_reward)}">${fmtSigned(summary.total_reward, 2)}</td>`,
        `<td class="num">${fmtSigned(summary.reward_by_term.holding_reward, 2)}</td>`,
        `<td class="num">${fmtSigned(summary.reward_by_term.action_reward, 2)}</td>`,
        `<td class="num">${fmtSigned(Compare.sumTerms(summary, ["distance_reward", "unassigned_reward", "idle_reward", "priority_reward"]), 2)}</td>`,
        `<td class="num">${summary.final_unassigned}</td>`,
        `<td class="num">${fmtDur(summary.final_drive_s)}</td>`,
        `<td class="num">${summary.served}</td><td class="num">${summary.failed}</td><td class="num">${summary.dropped}</td>`,
        `<td>${StackBar.html(summary.operator_counts, { showWords: false })}</td>`,
        `<td class="num">${summary.operator_counts.join(" / ")}</td>`,
        `<td class="num">${isNum(meanEdits) ? fmtFixed(meanEdits, 1) : NONE}</td>`,
        `<td class="num">${summary.wasted_edits ? `${summary.wasted_edits} ${statusChip("WASTED")}` : "0"}</td>`,
      ];
      return `<tr data-lane="${esc(lane.key)}"><td>${Compare.agentCell(lane.key)}${lane.trace.complete ? "" : ` <span class="muted">(partial)</span>`}</td>${cells.join("")}</tr>`;
    }).join("");

    return `
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr>
          <th>Agent</th><th class="num">Return<small>pts</small></th><th class="num" title="Sum of the holding term">Holding<small>pts</small></th><th class="num" title="Sum of the action term">Action<small>pts</small></th>
          <th class="num" title="Sum of the distance, unassigned, idle and priority terms">Plan change<small>pts</small></th><th class="num">Final unassigned</th><th class="num">Final drive<small>h:mm</small></th>
          <th class="num">Served</th><th class="num">Failed</th><th class="num">Dropped</th><th>Operator mix</th><th class="num" title="INSERT / REMOVE / DO_NOTHING">Ins / rem / hold</th><th class="num">Edits per round</th><th class="num">Wasted edits</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
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

  static shared(status) {
    const gaps = status.shared.gaps.map((gap, round) => `round ${round + 1}: ${round === 0 ? "start" : `${Math.round(gap / 60)} min gap`}${status.shared.events[round] ? `, ${Sentences.eventKind(status.shared.events[round].type)} x ${status.shared.events[round].count}` : ", no event"}`).join("; ");
    const flags = status.divergence.length
      ? status.divergence.map((entry) => `<li>${statusChip("WARN")} Round ${entry.round + 1}: ${entry.kind === "event_ids" ? "the disruption hit different entity ids" : "pickup outcomes differ"} for ${entry.agents.map((key) => esc(Agents.label(key))).join(", ")}.</li>`).join("")
      : `<li>${statusChip("PASS")} No round where entity ids or pickup outcomes differ between the recorded agents.</li>`;
    return `
      <div class="cmp__shared">
        <h4>What the agents share</h4>
        <p>Every lane starts from the same jobs, vehicles and depot and sees the same gaps and the same event type and count per round. Which ids an event hits and which pickups fail can differ once the plans differ.</p>
        <p class="muted">${esc(gaps)}</p>
        <ul class="cmp__flags">${flags}</ul>
      </div>`;
  }

  static html(lanes, status) {
    const usable = lanes.filter((lane) => lane.trace.summary && lane.trace.length);
    if (!usable.length) return `<p class="drawer__empty muted">No recorded lane to compare yet.</p>`;
    const unassigned = usable.map((lane) => lane.trace.summary.unassigned_at_commit);
    const rewards = usable.map((lane) => Compare.roundRewards(lane.trace));
    return `
      <div class="cmp">
        ${Compare.summaryTable(usable)}
        <div class="cmp__grid">
          ${Compare.matrix("Unassigned jobs at commit", "jobs", usable, unassigned, false)}
          ${Compare.matrix("Reward per round", "points", usable, rewards, true)}
          ${Compare.shared(status)}
        </div>
      </div>`;
  }
}
