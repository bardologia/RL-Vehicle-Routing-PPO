import { esc, fmtFixed, fmtInt, fmtMin, fmtSigned, isNum, NONE, Store } from "../format.js";
import { Agents } from "../episode/encoding.js";
import { bindTips, IntervalStrip, StackBar } from "../ui/charts.js";
import { plainChip } from "../ui/status.js";
import { Segmented } from "../ui/tabs.js";

export class AgentTable {
  static ORDER = ["model", "teacher", "insertion_only", "do_nothing"];

  static SETS = [
    { value: "outcome", label: "Outcome", title: "Reward, paired difference, record, unassigned jobs and operator mix" },
    { value: "execution", label: "Execution", title: "Final drive, served / failed / dropped, drive per served job, unassigned jobs and operator mix in percent" },
  ];

  static MIN = 96;

  static MAX = { brief: 360, outcome: 420, execution: 0 };

  static get set() {
    const value = Store.read("agents.set", "outcome");
    return AgentTable.SETS.some((entry) => entry.value === value) ? value : "outcome";
  }

  static switcher(host, onChange) {
    return new Segmented(host, AgentTable.SETS, AgentTable.set, (value) => {
      Store.write("agents.set", value);
      onChange(value);
    }, "Agent columns");
  }

  static agentCell(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  static render(container, report, { set = AgentTable.set, tail = "" } = {}) {
    container.innerHTML = AgentTable.html(report, { set, tail, width: AgentTable.MIN, compact: false });
    if (set !== "execution") {
      const wrapper = container.querySelector(".atable");
      const table = wrapper.querySelector("table");
      table.style.width = "max-content";
      const spare = wrapper.clientWidth - table.offsetWidth;
      table.style.width = "";
      if (spare < 0) container.innerHTML = AgentTable.html(report, { set, tail, width: 120, compact: true });
      else container.innerHTML = AgentTable.html(report, { set, tail, width: Math.round(Math.min(AgentTable.MAX[set], AgentTable.MIN + spare / 2)), compact: false });
    }
    bindTips(container);
  }

  static mixHint(mix) {
    return `INSERT ${(mix.op0 * 100).toFixed(0)}%, REMOVE ${(mix.op1 * 100).toFixed(0)}%, DO_NOTHING ${(mix.op2 * 100).toFixed(0)}%`;
  }

  static html(report, { set, tail, width, compact }) {
    const present = AgentTable.ORDER.filter((key) => report.agents[key]);
    const n = report.meta.episodes;
    const rewardRows = present.map((key) => ({ key, label: Agents.label(key), style: Agents.style(key), mean: report.agents[key].mean_reward, lo: report.agents[key].ci95[0], hi: report.agents[key].ci95[1], n: report.agents[key].episodes }));
    const pairedRows = present.filter((key) => report.paired[key]).map((key) => ({ key, label: `model minus ${Agents.label(key)}`, style: Agents.style(key), mean: report.paired[key].mean_diff, lo: report.paired[key].ci95[0], hi: report.paired[key].ci95[1], n }));
    const rewardStrip = IntervalStrip.scale(rewardRows, width, true);
    const pairedStrip = pairedRows.length ? IntervalStrip.scale(pairedRows, width, true) : null;
    const forest = set !== "execution";
    const cell = (strip, figures) => (compact ? `<td class="atable__strip">${strip}<div class="atable__under">${figures}</div></td>` : `<td class="atable__strip" style="width:${width + 12}px">${strip}</td><td class="num atable__value">${figures}</td>`);

    const body = present.map((key) => {
      const agent = report.agents[key];
      const reward = rewardRows.find((row) => row.key === key);
      const paired = pairedRows.find((row) => row.key === key);
      const pair = report.paired[key];
      const separable = pair && (pair.ci95[0] > 0 || pair.ci95[1] < 0);
      const chip = agent.identical_to ? plainChip(`same as ${Agents.label(agent.identical_to)}`, `Per-episode rewards equal ${agent.identical_to} in all ${n} episodes`) : "";
      const mix = agent.operator_frequency;
      const shares = [mix.op0, mix.op1, mix.op2];
      const cells = [`<td class="atable__agent">${AgentTable.agentCell(key)} ${chip}</td>`];
      if (forest) {
        cells.push(cell(IntervalStrip.cell(reward, rewardStrip), `<b data-hint="95% interval ${fmtSigned(agent.ci95[0], 1)} to ${fmtSigned(agent.ci95[1], 1)}" data-hint-title="Mean reward">${fmtSigned(agent.mean_reward)}</b>`));
        cells.push(pair ? cell(IntervalStrip.cell(paired, pairedStrip), `<b class="${separable ? "" : "muted"}" data-hint="${separable ? "separable" : "not separable"}: 95% interval ${fmtSigned(pair.ci95[0], 1)} to ${fmtSigned(pair.ci95[1], 1)}" data-hint-title="Model minus ${esc(Agents.label(key))}, paired">${fmtSigned(pair.mean_diff)}</b>`) : cell("", `<span class="faint">reference</span>`));
      }
      if (set === "outcome") cells.push(pair ? `<td class="num" data-hint="Episodes where the model is ahead, level and behind">${pair.wins} / ${pair.ties} / ${pair.losses}</td>` : `<td class="num faint">${NONE}</td>`);
      if (set === "execution") {
        cells.push(`<td class="num">${fmtMin(agent.mean_final_cost, 0)}</td>`);
        cells.push(`<td class="num">${fmtFixed(agent.mean_served, 1)} / ${fmtFixed(agent.mean_failed, 1)} / ${fmtFixed(agent.mean_dropped, 1)}</td>`);
        cells.push(`<td class="num">${isNum(agent.cost_per_served) ? fmtInt(agent.cost_per_served) : NONE}</td>`);
      }
      if (set !== "brief") {
        cells.push(`<td class="num">${fmtFixed(agent.mean_final_unassigned, 1)}</td>`);
        cells.push(set === "outcome" ? `<td class="atable__mix" data-hint="${AgentTable.mixHint(mix)}" data-hint-title="Operator mix">${StackBar.html(shares, { shares: true, showWords: false })}</td>` : `<td class="num" data-hint="${AgentTable.mixHint(mix)}" data-hint-title="Operator mix">${shares.map((share) => (share * 100).toFixed(0)).join(" / ")}</td>`);
      }
      if (tail) cells.push(`<td class="atable__tail"></td>`);
      return `<tr data-agent="${esc(key)}">${cells.join("")}</tr>`;
    }).join("");

    const missing = report.missing_agents.length ? `<p class="noteblock atable__missing">Missing from this report: ${report.missing_agents.map((key) => esc(Agents.label(key))).join(", ")}. The table shows the agents present.</p>` : "";
    const strip = (title, full, axis) => `<th class="atable__strip"${compact ? "" : ' colspan="2"'} data-hint="${full}"><span>${title}</span><small>pts</small><div class="atable__axis">${axis}</div></th>`;
    const heads = ["<th>Agent</th>"];
    if (forest) {
      heads.push(strip("Reward", "Mean reward, 95% interval", IntervalStrip.axis(rewardStrip)));
      heads.push(strip("Vs model", "Model minus agent, paired, 95% interval", pairedStrip ? IntervalStrip.axis(pairedStrip) : ""));
    }
    if (set === "outcome") heads.push(`<th class="num" data-hint="Wins / ties / losses of the model against this agent">W / T / L</th>`);
    if (set === "execution") {
      heads.push(`<th class="num" data-hint="Final drive: plan drive time at the end of the episode">Drive<small>min</small></th>`);
      heads.push(`<th class="num" data-hint="Served / failed / dropped">S / F / D</th>`);
      heads.push(`<th class="num" data-hint="Drive per served job">Drive / job<small>s</small></th>`);
    }
    if (set !== "brief") {
      heads.push(`<th class="num" data-hint="Jobs left unassigned at the end of the episode">Unassigned<small>jobs</small></th>`);
      heads.push(`<th class="${set === "outcome" ? "atable__mix" : "num"}" data-hint="Operator mix in percent, INSERT / REMOVE / DO_NOTHING">Mix${set === "execution" ? "<small>%</small>" : ""}</th>`);
    }
    if (tail) heads.push(`<th class="atable__tail">${tail}</th>`);

    return `
      ${missing}
      <div class="dtable dtable--static atable atable--${set}${compact ? " atable--compact" : ""}">
        <table>
          <thead><tr>${heads.join("")}</tr></thead>
          <tbody>${body}</tbody>
        </table>
      </div>`;
  }

  static info(report) {
    return `<p class="atable__foot">One row per agent on the same ${report.meta.episodes} episodes, seed ${report.meta.seed}. Reward is the undiscounted episode return in reward points; higher is better.</p><p class="atable__foot">Interval = mean plus or minus 1.96 standard errors. A paired difference is not separable when its interval contains 0.</p><p class="atable__foot">Final drive = plan drive time at the end of the episode; unassigned = jobs left unassigned at the end. Operator mix in percent, in the order INSERT (filled), REMOVE (hatched), DO_NOTHING (outline).</p>`;
  }
}
