import { esc, fmtDur, fmtFixed, fmtSigned, isNum, NONE, signClass } from "../format.js";
import { errorBlock, statusChip } from "../ui/status.js";
import { kvTable } from "../ui/table.js";
import { Sentences, Vehicles } from "./encoding.js";

export class VerdictPanel {
  static chips(verdict) {
    if (!verdict) return "";
    const word = (same) => (same ? "SAME" : "DIFFERENT");
    const parts = [statusChip(word(verdict.match.operator), "operator")];
    if (verdict.match.job !== null) parts.push(statusChip(word(verdict.match.job), "job"));
    if (verdict.match.vehicle !== null) parts.push(statusChip(word(verdict.match.vehicle), "vehicle"));
    return parts.join("");
  }

  static buttons(state, hasInsertions) {
    const busy = state.status === "running";
    return `
      <div class="verdict__buttons">
        ${hasInsertions ? "" : `<button type="button" class="btn" data-verdict="insertions"${busy ? " disabled" : ""}>Ask the teacher (insertions, about 1 to 3 s)</button>`}
        <button type="button" class="btn" data-verdict="full"${busy ? " disabled" : ""}>${hasInsertions ? "Ask again with removals (up to about 20 s)" : "with removals (up to about 20 s)"}</button>
      </div>
      ${busy ? `<p class="muted" role="status">The teacher is scoring ${state.scope === "full" ? "insertions and removals" : "every feasible insertion"} on this state.</p>` : ""}
      ${state.status === "error" ? errorBlock("The teacher verdict failed.", state.error) : ""}`;
  }

  static signed(value) {
    return `<span class="${signClass(value)}">${fmtSigned(value, 2)}</span>`;
  }

  static result(verdict) {
    const best = verdict.best ? `job ${verdict.best.job_id} into vehicle ${verdict.best.vehicle_id}, ${VerdictPanel.signed(verdict.best.reward)}${isNum(verdict.best.regret) ? `, regret-2 ${fmtFixed(verdict.best.regret, 2)}` : ""}` : "no feasible insertion";
    const facts = kvTable([
      ["Teacher action", `<b>${esc(Sentences.action(verdict.teacher_action))}</b>`],
      ["No-op baseline", VerdictPanel.signed(verdict.baseline)],
      ["Best insertion", best],
      ["Plan value", `${VerdictPanel.signed(verdict.plan.value)} via ${esc(Sentences.action(verdict.plan.action))}`],
      ["Taken action", `${VerdictPanel.signed(verdict.taken_reward)} <span class="muted">(realised, without holding)</span>`],
      ["One-step regret", `<b>${fmtFixed(verdict.one_step_regret, 2)}</b> <span class="muted">points left on the table by this action</span>`],
      ["Profitable insertions not taken", `${verdict.profitable_left.count}, worth ${VerdictPanel.signed(verdict.profitable_left.reward_sum)} in sum`],
    ]);

    const insertions = [...verdict.insertions].sort((a, b) => b.reward - a.reward);
    const insertionRows = insertions.slice(0, 8).map((entry) => `
      <tr><td>job ${entry.job_id}</td><td>${Vehicles.swatch(entry.vehicle_id)}</td><td class="num">${VerdictPanel.signed(entry.reward)}</td><td class="num">${fmtDur(entry.added_drive_s)}</td><td class="num">${isNum(verdict.break_even[entry.job_id]) ? fmtDur(verdict.break_even[entry.job_id]) : NONE}</td></tr>`).join("");
    const insertionTable = insertions.length ? `
      <h4 class="decision__sub">One-step reward of each feasible insertion</h4>
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr><th>Job</th><th>Vehicle</th><th class="num">Reward<small>pts</small></th><th class="num">Added drive<small>h:mm</small></th><th class="num">Break-even<small>h:mm</small></th></tr></thead>
        <tbody>${insertionRows}</tbody>
      </table></div>
      ${insertions.length > 8 ? `<p class="muted">${insertions.length - 8} more in the grid above (switch it to teacher reward).</p>` : ""}
      ${verdict.infeasible.length ? `<p class="muted">${verdict.infeasible.length} job and vehicle pairs are infeasible.</p>` : ""}` : `<p class="muted">No feasible insertion on this state.</p>`;

    let removalTable = "";
    if (verdict.removals) {
      const removals = [...verdict.removals].sort((a, b) => b.value - a.value);
      removalTable = removals.length ? `
        <h4 class="decision__sub">Removals: one-step reward and continuation</h4>
        <div class="dtable dtable--dense dtable--static"><table>
          <thead><tr><th>Job</th><th>Vehicle</th><th class="num">Reward<small>pts</small></th><th class="num">Continuation<small>pts</small></th><th class="num">Value<small>pts</small></th></tr></thead>
          <tbody>${removals.slice(0, 8).map((entry) => `<tr><td>job ${entry.job_id}</td><td>${Vehicles.swatch(entry.vehicle_id)}</td><td class="num">${VerdictPanel.signed(entry.reward)}</td><td class="num">${VerdictPanel.signed(entry.continuation)}</td><td class="num">${VerdictPanel.signed(entry.value)}</td></tr>`).join("")}</tbody>
        </table></div>
        ${removals.length > 8 ? `<p class="muted">Best 8 of ${removals.length} removals by value.</p>` : ""}` : `<p class="muted">No removal is available on this state.</p>`;
    }

    return `<div class="verdict__chips">${VerdictPanel.chips(verdict)}<span class="muted">scope: ${verdict.scope === "full" ? "insertions and removals" : "insertions only"}</span></div>${facts}${insertionTable}${removalTable}`;
  }

  static html(frame, state) {
    const verdict = frame.decision.verdict;
    if (!verdict) return `<p class="muted">Not computed for this decision. The teacher scores every alternative on a copy of this state.</p>${VerdictPanel.buttons(state, false)}`;
    return `${VerdictPanel.result(verdict)}${verdict.scope === "full" ? "" : VerdictPanel.buttons(state, true)}`;
  }
}
