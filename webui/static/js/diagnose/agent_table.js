import { esc, fmtFixed, fmtInt, fmtMin, fmtSigned, isNum, NONE } from "../format.js";
import { Agents } from "../episode/encoding.js";
import { bindTips, IntervalStrip, StackBar } from "../ui/charts.js";
import { plainChip } from "../ui/status.js";

export class AgentTable {
  static ORDER = ["model", "teacher", "insertion_only", "do_nothing"];

  static agentCell(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  static render(container, report) {
    container.innerHTML = AgentTable.html(container, report, false);
    const wrapper = container.querySelector(".atable");
    if (wrapper.scrollWidth > wrapper.clientWidth) container.innerHTML = AgentTable.html(container, report, true);
    bindTips(container);
  }

  static html(container, report, compact) {
    const present = AgentTable.ORDER.filter((key) => report.agents[key]);
    const n = report.meta.episodes;
    const rewardRows = present.map((key) => ({ key, label: Agents.label(key), style: Agents.style(key), mean: report.agents[key].mean_reward, lo: report.agents[key].ci95[0], hi: report.agents[key].ci95[1], n: report.agents[key].episodes }));
    const pairedRows = present.filter((key) => report.paired[key]).map((key) => ({ key, label: `model minus ${Agents.label(key)}`, style: Agents.style(key), mean: report.paired[key].mean_diff, lo: report.paired[key].ci95[0], hi: report.paired[key].ci95[1], n }));
    const width = compact ? 132 : container.clientWidth >= 1500 ? 168 : container.clientWidth >= 1250 ? 120 : 96;
    const rewardStrip = IntervalStrip.scale(rewardRows, width, true);
    const pairedStrip = pairedRows.length ? IntervalStrip.scale(pairedRows, width, true) : null;
    const cell = (strip, figures) => (compact ? `<td class="atable__strip">${strip}<div class="atable__under">${figures}</div></td>` : `<td class="atable__strip">${strip}</td><td class="num">${figures}</td>`);

    const body = present.map((key) => {
      const agent = report.agents[key];
      const reward = rewardRows.find((row) => row.key === key);
      const paired = pairedRows.find((row) => row.key === key);
      const pair = report.paired[key];
      const separable = pair && (pair.ci95[0] > 0 || pair.ci95[1] < 0);
      const chip = agent.identical_to ? plainChip(`identical to ${agent.identical_to}`, `Per-episode rewards equal ${agent.identical_to} in all ${n} episodes`) : "";
      const mix = agent.operator_frequency;
      const pairedCell = pair
        ? `${cell(IntervalStrip.cell(paired, pairedStrip), `<b class="${separable ? (pair.mean_diff > 0 ? "gain" : "loss") : "muted"}" data-hint="${separable ? "separable" : "not separable"}: 95% interval ${fmtSigned(pair.ci95[0], 1)} to ${fmtSigned(pair.ci95[1], 1)}">${fmtSigned(pair.mean_diff)}</b>`)}<td class="num" title="Episodes where the model is ahead, level and behind">${pair.wins} / ${pair.ties} / ${pair.losses}</td>`
        : `${cell("", `<span class="faint">reference</span>`)}<td class="num faint">${NONE}</td>`;
      return `
        <tr data-agent="${esc(key)}">
          <td class="atable__agent">${AgentTable.agentCell(key)} ${chip}</td>
          ${cell(IntervalStrip.cell(reward, rewardStrip), `<b data-hint="95% interval ${fmtSigned(agent.ci95[0], 1)} to ${fmtSigned(agent.ci95[1], 1)}">${fmtSigned(agent.mean_reward)}</b>`)}
          ${pairedCell}
          <td class="num">${fmtMin(agent.mean_final_cost, 0)}</td>
          <td class="num">${fmtFixed(agent.mean_final_unassigned, 1)}</td>
          <td class="num">${fmtFixed(agent.mean_served, 1)} / ${fmtFixed(agent.mean_failed, 1)} / ${fmtFixed(agent.mean_dropped, 1)}</td>
          <td class="num">${isNum(agent.cost_per_served) ? fmtInt(agent.cost_per_served) : NONE}</td>
          <td><span class="mixcell">${StackBar.html([mix.op0, mix.op1, mix.op2], { shares: true, showWords: false })}<span class="atable__mix">${[mix.op0, mix.op1, mix.op2].map((share) => (share * 100).toFixed(0)).join(" / ")}</span></span></td>
        </tr>`;
    }).join("");

    const missing = report.missing_agents.length ? `<p class="noteblock atable__missing">Missing from this report: ${report.missing_agents.map((key) => esc(Agents.label(key))).join(", ")}. The table shows the agents present.</p>` : "";
    const head = (title, full, axis, figures) => `<th class="atable__strip" data-hint="${full}">${title}<small>points</small><div>${axis}</div></th>${compact ? "" : `<th class="num">${figures}</th>`}`;

    return `
      ${missing}
      <div class="dtable dtable--static atable${compact ? " atable--compact" : ""}">
        <table>
          <thead>
            <tr>
              <th>Agent</th>
              ${head("Reward", "Mean reward, 95% interval", IntervalStrip.axis(rewardStrip), "Mean")}
              ${head("Vs model", "Model minus agent, paired", pairedStrip ? IntervalStrip.axis(pairedStrip) : "", "Difference")}
              <th class="num" data-hint="Wins / ties / losses of the model against this agent">W / T / L</th>
              <th class="num" data-hint="Final drive: plan drive time at the end of the episode">Drive<small>min</small></th>
              <th class="num" data-hint="Jobs left unassigned at the end of the episode">Unassigned<small>jobs</small></th>
              <th class="num" data-hint="Served / failed / dropped">S / F / D</th>
              <th class="num" data-hint="Drive per served job">Drive / job<small>s</small></th>
              <th data-hint="Operator mix in percent. Filled = INSERT, hatched = REMOVE, outline = DO_NOTHING">Mix<small>%</small></th>
            </tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;
  }

  static info(report) {
    return `<p class="atable__foot">One row per agent on the same ${report.meta.episodes} episodes, seed ${report.meta.seed}. Reward is the undiscounted episode return in reward points; higher is better.</p><p class="atable__foot">Interval = mean plus or minus 1.96 standard errors. A paired difference is not separable when its interval contains 0.</p><p class="atable__foot">Final drive = plan drive time at the end of the episode; unassigned = jobs left unassigned at the end. Operator mix in percent, in the order INSERT (filled), REMOVE (hatched), DO_NOTHING (outline).</p>`;
  }
}
