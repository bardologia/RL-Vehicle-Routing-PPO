import { el, esc, fmtFixed, fmtPct, fmtSigned, isNum, NONE, plural } from "../format.js";
import { Agents, OPERATORS, TERMS } from "../episode/encoding.js";
import { BarChart, bindTips, ColumnChart, Histogram, Scatter, StackBar } from "../ui/charts.js";
import { HeatTable } from "../ui/heat_table.js";
import { kpiRow } from "../ui/kpi.js";
import { infoText, panelHead } from "../ui/panel.js";
import { emptyBlock } from "../ui/status.js";
import { kvTable } from "../ui/table.js";
import { Links } from "./links.js";

const PANELS = [
  { key: "operator-mix", title: "Operator mix", help: "Share of INSERT, REMOVE and DO_NOTHING per agent, then by edit position inside a round.", more: "The number beside each bar is the count of decisions taken at that edit position. Filled = INSERT, hatched = REMOVE, outline = DO_NOTHING." },
  { key: "burst-profile", title: "Burst profile", help: "How many edits the model spends per round, and how often a round ends with work left." },
  { key: "reward-by-term", title: "Reward by term", help: "Mean episode sum of each reward term per agent, on one scale." },
  { key: "critic-calibration", title: "Critic calibration", help: "V(s) against the realised discounted return-to-go of every model decision. Click a point to open the decision.", more: "Dashed diagonal = perfect critic. Solid line = mean return in six bins of V(s)." },
  { key: "confidence", title: "Confidence", help: "Joint probability the policy gave to the action it took." },
  { key: "job-ranking", title: "Job ranking", help: "Mean rank the job head gives a job, by job priority. Rank 1 is the most probable job.", more: "A head that favours urgent jobs gives a clearly negative rho. Bars of equal height mean the job head ignores priority. n per bar is the number of decisions that offered a job of that priority." },
  { key: "teacher-agreement", title: "Teacher agreement", help: "Model action against the teacher verdict on the same state." },
  { key: "execution", title: "Execution", help: "What the vehicles did between rounds. Pickup outcomes are sampled, so this is luck, not policy." },
];

export class Aggregate {
  constructor(index) {
    this.index = index;
    this.keys = Object.keys(index.coverage).sort((a, b) => Agents.order(a) - Agents.order(b));
  }

  agents(key) {
    return this.index.episodes.map((episode) => episode.agents[key]).filter((agent) => agent);
  }

  sum(key, pick) {
    return this.agents(key).reduce((total, agent) => total + pick(agent), 0);
  }

  mean(key, pick) {
    const rows = this.agents(key);
    return rows.length ? this.sum(key, pick) / rows.length : null;
  }

  operatorCounts(key) {
    return [0, 1, 2].map((op) => this.sum(key, (agent) => agent.operator_counts[op]));
  }

  operatorByEdit(key) {
    const edits = this.index.meta.caps.edits;
    return Array.from({ length: edits }, (_, edit) => [0, 1, 2].map((op) => this.sum(key, (agent) => (agent.operator_by_edit[edit] ? agent.operator_by_edit[edit][op] : 0))));
  }

  rounds(key) {
    return this.sum(key, (agent) => agent.unassigned_at_commit.length);
  }

  roundsWithUnassigned(key) {
    return this.sum(key, (agent) => agent.unassigned_at_commit.filter((count) => count > 0).length);
  }

  termMean(key, term) {
    return this.mean(key, (agent) => agent.reward_by_term[term]);
  }

  holdingShare(key) {
    const holding = Math.abs(this.sum(key, (agent) => agent.reward_by_term.holding_reward));
    const total = TERMS.reduce((sum, term) => sum + Math.abs(this.sum(key, (agent) => agent.reward_by_term[term.key])), 0);
    return total ? holding / total : null;
  }
}

export class BehaviourPanels {
  constructor(container, { run, seed, recordHash, onNavigate }) {
    this.container = container;
    this.run = run;
    this.seed = seed;
    this.recordHash = recordHash;
    this.onNavigate = onNavigate;
    this.bodies = {};
    container.classList.add("diag__grid");
    container.innerHTML = "";
    this.panels = [];
    for (const panel of PANELS) {
      const node = el(`
        <section class="panel" data-panel="${panel.key}">
          ${panelHead(panel.title, { info: infoText(panel.help, panel.more) })}
          <div class="panel__body"><div data-role="body"></div></div>
        </section>`);
      this.panels.push(node);
      this.bodies[panel.key] = { body: node.querySelector("[data-role=body]"), n: node.querySelector("[data-role=panel-note]") };
    }
  }

  layout() {
    const count = window.innerWidth > 1500 ? 3 : 2;
    if (this.container.children.length !== count) {
      this.container.innerHTML = "";
      for (let index = 0; index < count; index += 1) this.container.appendChild(el(`<div class="diag__col"></div>`));
    }
    this.panels.forEach((panel, index) => this.container.children[index % count].appendChild(panel));
  }

  needsTraces() {
    this.layout();
    for (const { body, n } of Object.values(this.bodies)) {
      n.textContent = "";
      body.innerHTML = emptyBlock("Needs traces.", `<a class="btn btn--sm" href="${esc(this.recordHash)}">Record traces</a>`);
    }
  }

  agentLabel(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  operatorMix(agg) {
    const { body, n } = this.bodies["operator-mix"];
    const totals = agg.keys.map((key) => ({ key, counts: agg.operatorCounts(key) }));
    n.textContent = totals.map((row) => `${Agents.label(row.key)} n = ${row.counts.reduce((a, b) => a + b, 0)}`).join(", ");
    const overall = totals.map((row) => `<tr><th>${this.agentLabel(row.key)}</th><td>${StackBar.html(row.counts)}</td></tr>`).join("");
    const byEdit = agg.keys.map((key) => ({ key, edits: agg.operatorByEdit(key) }));
    const edits = agg.index.meta.caps.edits;
    const rows = Array.from({ length: edits }, (_, edit) => `<tr><th class="num">${edit + 1}</th>${byEdit.map((row) => {
      const total = row.edits[edit].reduce((a, b) => a + b, 0);
      return `<td>${total ? `<span class="mixcell">${StackBar.html(row.edits[edit], { showWords: false })}<small>${total}</small></span>` : `<span class="faint">${NONE}</span>`}</td>`;
    }).join("")}</tr>`).join("");
    body.innerHTML = `
      <table class="kv kv--wide">${overall}</table>
      <h4 class="ctl__label behav__sub">By edit position</h4>
      <table class="mixgrid">
        <thead><tr><th class="num">Edit</th>${byEdit.map((row) => `<th>${esc(Agents.label(row.key))}</th>`).join("")}</tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  }

  burst(agg) {
    const { body, n } = this.bodies["burst-profile"];
    const edits = agg.agents("model").flatMap((agent) => agent.edits_per_round);
    n.textContent = `model, n = ${plural(edits.length, "round")}`;
    const rows = agg.keys.map((key) => {
      const rounds = agg.rounds(key);
      const left = agg.roundsWithUnassigned(key);
      const early = agg.sum(key, (agent) => agent.commits_with_work_left);
      return `<tr><th>${this.agentLabel(key)}</th><td class="num">${left} of ${rounds} (${fmtPct(rounds ? left / rounds : null)})</td><td class="num">${early} of ${rounds} (${fmtPct(rounds ? early / rounds : null)})</td></tr>`;
    }).join("");
    body.innerHTML = `
      <div data-role="chart"></div>
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr><th>Agent</th><th class="num" data-hint="Rounds ended with unassigned jobs">Ended unassigned</th><th class="num" data-hint="Of those rounds, the ones committed with edits remaining">Edits remaining</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
    Histogram.render(body.querySelector("[data-role=chart]"), edits, { xLabel: "edits used in a round (model)", integer: true, domain: [1, agg.index.meta.caps.edits], height: 170 });
  }

  rewardByTerm(agg) {
    const { body, n } = this.bodies["reward-by-term"];
    n.textContent = `n = ${plural(agg.index.episodes.length, "episode")}`;
    const values = TERMS.map((term) => agg.keys.map((key) => ({ key, value: agg.termMean(key, term.key) })));
    const max = Math.max(1e-9, ...values.flat().map((row) => Math.abs(row.value)).filter(isNum));
    const blocks = TERMS.map((term, index) => `
      <div class="termblock">
        <h4 class="ctl__label behav__sub">${esc(term.label)}</h4>
        ${BarChart.html(values[index].map((row) => ({ label: Agents.label(row.key), labelHtml: this.agentLabel(row.key), value: row.value })), { max })}
      </div>`).join("");
    const shares = agg.keys.map((key) => `${esc(Agents.label(key))} ${fmtPct(agg.holdingShare(key))}`).join(", ");
    body.innerHTML = `<div class="termgrid">${blocks}</div><p class="statline"><span>Holding share of the summed absolute terms</span><b>${shares}</b></p>`;
  }

  critic(agg) {
    const { body, n } = this.bodies["critic-calibration"];
    const critic = agg.index.aggregates.critic;
    n.textContent = `n = ${plural(critic.n, "decision")}`;
    body.innerHTML = `
      ${kpiRow([{ label: "Explained", value: fmtFixed(critic.explained_variance, 3), title: "Explained variance" }, { label: "Bias", value: fmtSigned(critic.bias), title: "Bias, V minus return" }, { label: "Abs error", value: fmtFixed(critic.mae), title: "Mean absolute error" }], { strip: true })}
      <div data-role="chart"></div>`;
    const points = agg.index.decisions.map((decision) => ({ x: decision.value, y: decision.return_to_go, title: `Episode ${decision.episode}, round ${decision.round + 1}, edit ${decision.edit + 1}: ${OPERATORS[decision.operator]}`, decision }));
    Scatter.render(body.querySelector("[data-role=chart]"), points, { xLabel: "V(s)", yLabel: "discounted return-to-go, points", bins: 6, height: 300, onClick: (point) => this.onNavigate(Links.decision(this.run, this.seed, point.decision)) });
  }

  confidence(agg) {
    const { body, n } = this.bodies.confidence;
    const decisions = agg.index.decisions;
    n.textContent = `n = ${plural(decisions.length, "decision")}`;
    const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
    const pending = decisions.filter((decision) => decision.unassigned > 0);
    const greedy = agg.mean("model", (agent) => agent.total_reward);
    const sampled = agg.keys.includes("model_sampled") ? agg.mean("model_sampled", (agent) => agent.total_reward) : null;
    body.innerHTML = `
      <div data-role="chart"></div>
      ${kvTable([
        ["Mean joint probability of the chosen action", `<b>${fmtFixed(mean(decisions.map((d) => d.chosen_joint_prob)), 3)}</b>`],
        ["Mean normalised entropy, operator head", fmtFixed(mean(decisions.map((d) => d.entropy.operator).filter(isNum)), 3)],
        ["Mean normalised entropy, vehicle head", fmtFixed(mean(decisions.map((d) => d.entropy.vehicle).filter(isNum)), 3)],
        ["Mean normalised entropy, job head", fmtFixed(mean(decisions.map((d) => d.entropy.job).filter(isNum)), 3)],
        ["P(DO_NOTHING) with unassigned jobs waiting", `${fmtFixed(mean(pending.map((d) => d.p_do_nothing)), 3)} <span class="faint">n = ${pending.length}</span>`],
        ["Greedy against sampled mean reward", sampled === null ? `<span class="faint">sampled twin not recorded</span>` : `${fmtSigned(greedy)} against ${fmtSigned(sampled)}`],
      ])}`;
    Histogram.render(body.querySelector("[data-role=chart]"), decisions.map((d) => d.chosen_joint_prob), { xLabel: "joint probability of the chosen action", bins: 10, domain: [0, 1], height: 170 });
  }

  jobRanking(agg) {
    const { body, n } = this.bodies["job-ranking"];
    const ranking = agg.index.aggregates.job_ranking;
    n.textContent = `n = ${ranking.n} job scores`;
    const rows = [1, 2, 3, 4, 5].map((priority) => {
      const ranks = agg.index.decisions.map((decision) => (decision.job_rank_by_priority ? decision.job_rank_by_priority[priority] : undefined)).filter(isNum);
      return { label: String(priority), value: ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null, n: ranks.length };
    });
    body.innerHTML = `
      ${kpiRow([{ label: "Spearman rho", value: fmtSigned(ranking.spearman, 3), title: "Spearman rho, priority against rank" }], { strip: true })}
      <div data-role="chart"></div>`;
    ColumnChart.render(body.querySelector("[data-role=chart]"), rows, { xLabel: "job priority (5 = most urgent)", yLabel: "mean rank", height: 190 });
  }

  agreement(agg) {
    const { body, n } = this.bodies["teacher-agreement"];
    const agreement = agg.index.aggregates.agreement;
    if (!agreement) {
      n.textContent = "";
      body.innerHTML = emptyBlock("Needs verdicts. Record traces with the teacher verdict set to insertions or full.", `<a class="btn btn--sm" href="${esc(this.recordHash)}">Record traces</a>`);
      return;
    }
    n.textContent = `n = ${plural(agreement.n, "decision")}, verdict scope ${agg.index.meta.teacher_verdict}`;
    body.innerHTML = `
      ${kpiRow([{ label: "Same operator", value: fmtPct(agreement.operator) }, { label: "Same vehicle", value: fmtPct(agreement.vehicle) }, { label: "Same job", value: fmtPct(agreement.job) }], { strip: true })}
      <h4 class="ctl__label behav__sub">Operator confusion, decisions</h4>
      <div data-role="heat"></div>
      <h4 class="ctl__label behav__sub">Profitable insertions left at round ends</h4>
      <div data-role="chart"></div>`;
    const heat = new HeatTable(body.querySelector("[data-role=heat]"), {});
    heat.render({
      rows: OPERATORS.map((name, index) => ({ key: String(index), html: `<span class="opmark opmark--op${index}"></span> model ${name}` })),
      columns: OPERATORS.map((name, index) => ({ key: String(index), html: `teacher ${name}`, title: `Teacher chose ${name}` })),
      cells: agreement.confusion.map((row, r) => row.map((value, c) => ({ value, tip: `model ${OPERATORS[r]}, teacher ${OPERATORS[c]}: ${value} of ${agreement.n}` }))),
      format: (value) => String(value),
      cornerLabel: "",
      marks: {},
    });
    const rows = agg.index.episodes.map((episode) => ({ label: String(episode.index), value: episode.agents.model ? episode.agents.model.profitable_left : null, episode }));
    ColumnChart.render(body.querySelector("[data-role=chart]"), rows, { xLabel: "episode", yLabel: "insertions left", height: 160, digits: 0, onClick: (row) => this.onNavigate(Links.episode(this.run, this.seed, row.episode.index, true)) });
  }

  execution(agg) {
    const { body, n } = this.bodies.execution;
    n.textContent = `n = ${plural(agg.index.episodes.length, "episode")}`;
    const nominal = 1 - agg.index.meta.environment.repossession_success_probability;
    const rates = agg.index.aggregates.pickup_failure_rate;
    const rows = agg.keys.map((key) => {
      const pickups = agg.sum(key, (agent) => agent.pickups.served + agent.pickups.failed);
      return `<tr><th>${this.agentLabel(key)}</th><td class="num">${agg.sum(key, (a) => a.served)}</td><td class="num">${agg.sum(key, (a) => a.failed)}</td><td class="num">${agg.sum(key, (a) => a.dropped)}</td><td class="num">${fmtPct(rates[key])} <span class="faint">of ${pickups}</span></td></tr>`;
    }).join("");
    const divergent = agg.index.episodes.filter((episode) => episode.divergent_rounds.length);
    const links = divergent.map((episode) => `<a href="${esc(Links.episode(this.run, this.seed, episode.index, true))}">${episode.index}</a>`).join(", ");
    body.innerHTML = `
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr><th>Agent</th><th class="num">Served<small>jobs</small></th><th class="num">Failed<small>jobs</small></th><th class="num">Dropped<small>bikes</small></th><th class="num" data-hint="Pickup failure rate">Failure rate</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <p class="statline"><span>Nominal pickup failure rate</span><b>${fmtPct(nominal)}</b></p>
      <p class="statline"><span>Episodes where events or pickup outcomes differ between agents</span><b>${divergent.length} of ${agg.index.episodes.length}${links ? ` (episode ${links})` : ""}</b></p>`;
  }

  render(index) {
    const agg = new Aggregate(index);
    this.layout();
    if (!agg.keys.includes("model") || !agg.agents("model").length) {
      this.needsTraces();
      return;
    }
    this.operatorMix(agg);
    this.burst(agg);
    this.rewardByTerm(agg);
    this.critic(agg);
    this.confidence(agg);
    this.jobRanking(agg);
    this.agreement(agg);
    this.execution(agg);
    bindTips(this.container);
  }
}
