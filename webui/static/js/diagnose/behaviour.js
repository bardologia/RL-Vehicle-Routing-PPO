import { el, esc, fmtFixed, fmtPct, fmtSigned, isNum, NONE, plural } from "../format.js";
import { Agents, OPERATORS, TERMS } from "../episode/encoding.js";
import { bindTips, ColumnChart, Histogram, Scatter, StackBar } from "../ui/charts.js";
import { HeatTable } from "../ui/heat_table.js";
import { kpiRow } from "../ui/kpi.js";
import { infoText, panelHead } from "../ui/panel.js";
import { emptyBlock } from "../ui/status.js";
import { Links } from "./links.js";

const PANELS = [
  { key: "operator-mix", title: "Operator mix", note: "% of decisions", wide: 4, narrow: 6, help: "Share of INSERT, REMOVE and DO_NOTHING per agent, then by edit position inside a round.", more: "The first row pools every edit position, in percent INSERT / REMOVE / DO_NOTHING. The number beside each bar is the count of decisions taken at that edit position. Filled = INSERT, hatched = REMOVE, outline = DO_NOTHING." },
  { key: "burst-profile", title: "Burst profile", note: "rounds, all episodes", wide: 4, narrow: 6, help: "How many edits the model spends per round, and how often a round ends with work left.", more: "Chart: rounds of the model by the number of edits used. Table: rounds that ended with unassigned jobs, and rounds committed with edits remaining, per agent." },
  { key: "teacher-agreement", title: "Teacher agreement", note: "model decisions", wide: 4, narrow: 6, help: "Model action against the teacher verdict on the same state.", more: "Grid: decisions by model operator (rows) and teacher operator (columns). Bars: profitable insertions left at round ends, per episode; a bar opens the episode." },
  { key: "reward-by-term", title: "Reward by term", note: "points per episode", wide: 8, narrow: 12, help: "Mean episode sum of each reward term per agent, on one scale.", more: "Bars share one scale across the board; right of the centre is gain, left is loss. Hold share = share of the holding term in the summed absolute terms." },
  { key: "execution", title: "Execution", note: "jobs, all episodes", wide: 4, narrow: 12, help: "What the vehicles did between rounds. Pickup outcomes are sampled, so this is luck, not policy." },
  { key: "critic-calibration", title: "Critic calibration", note: "points", wide: 4, narrow: 6, help: "V(s) against the realised discounted return-to-go of every model decision. Click a point to open the decision.", more: "Dashed diagonal = perfect critic. Solid line = mean return in six bins of V(s)." },
  { key: "confidence", title: "Confidence", note: "model decisions", wide: 4, narrow: 6, help: "Joint probability the policy gave to the action it took.", more: "Entropy is normalised to 0 to 1 per head. Hold with work = mean P(DO_NOTHING) on decisions with unassigned jobs waiting." },
  { key: "job-ranking", title: "Job ranking", note: "mean rank by priority", wide: 4, narrow: 6, help: "Mean rank the job head gives a job, by job priority. Rank 1 is the most probable job.", more: "A head that favours urgent jobs gives a clearly negative rho. Bars of equal height mean the job head ignores priority. n per bar is the number of decisions that offered a job of that priority." },
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
    this.agg = null;
    container.classList.add("grid-12", "behav");
    container.innerHTML = "";
    for (const panel of PANELS) {
      const node = el(`
        <section class="panel behav__panel" data-panel="${panel.key}" style="--wide:${panel.wide};--narrow:${panel.narrow}">
          ${panelHead(panel.title, { note: panel.note, info: infoText(panel.help, panel.more) })}
          <div class="panel__body" data-role="body"></div>
        </section>`);
      container.appendChild(node);
      this.bodies[panel.key] = { body: node.querySelector("[data-role=body]") };
    }
  }

  static facts(index) {
    if (!index) return "no traces recorded";
    const agg = new Aggregate(index);
    const parts = [plural(index.episodes.length, "episode"), plural(agg.rounds("model"), "round"), `${plural(index.decisions.length, "model decision")}`];
    const verdict = index.meta.teacher_verdict;
    if (verdict === "insertions") parts.push("teacher verdicts on insertions");
    if (verdict === "full") parts.push("full teacher verdicts");
    return parts.join(", ");
  }

  static missing(index) {
    if (!index) return [];
    const agg = new Aggregate(index);
    const out = [];
    if (!agg.keys.includes("model_sampled")) out.push("Confidence: greedy against sampled mean reward needs the sampled twin, which is not recorded.");
    if (!index.aggregates.agreement) out.push("Teacher agreement needs teacher verdicts.");
    return out;
  }

  needsTraces() {
    for (const { body } of Object.values(this.bodies)) body.innerHTML = `<p class="behav__none">Needs traces.</p>`;
  }

  agentLabel(key) {
    return `<span class="lanekey lanekey--${Agents.style(key)}"><i></i>${esc(Agents.label(key))}</span>`;
  }

  operatorMix(agg) {
    const { body } = this.bodies["operator-mix"];
    const byEdit = agg.keys.map((key) => ({ key, counts: agg.operatorCounts(key), edits: agg.operatorByEdit(key) }));
    const edits = agg.index.meta.caps.edits;
    const all = byEdit.map((row) => {
      const total = row.counts.reduce((a, b) => a + b, 0);
      const shares = total ? row.counts.map((value) => ((value / total) * 100).toFixed(0)).join("/") : NONE;
      const hint = `INSERT ${row.counts[0]}, REMOVE ${row.counts[1]}, DO_NOTHING ${row.counts[2]} of ${plural(total, "decision")}`;
      return `<td data-hint="${esc(hint)}" data-hint-title="${esc(Agents.label(row.key))}"><span class="mixcell">${StackBar.html(row.counts, { showWords: false })}<small class="mixcell__share">${shares}</small></span></td>`;
    }).join("");
    const rows = Array.from({ length: edits }, (_, edit) => `<tr><th class="num">${edit + 1}</th>${byEdit.map((row) => {
      const total = row.edits[edit].reduce((a, b) => a + b, 0);
      return `<td>${total ? `<span class="mixcell">${StackBar.html(row.edits[edit], { showWords: false })}<small>${total}</small></span>` : `<span class="faint">${NONE}</span>`}</td>`;
    }).join("")}</tr>`).join("");
    body.innerHTML = `
      <table class="mixgrid">
        <thead><tr><th class="num">Edit</th>${byEdit.map((row) => `<th>${this.agentLabel(row.key)}</th>`).join("")}</tr></thead>
        <tbody><tr class="mixgrid__all"><th class="num">all</th>${all}</tr>${rows}</tbody>
      </table>`;
  }

  burst(agg) {
    const { body } = this.bodies["burst-profile"];
    const rows = agg.keys.map((key) => {
      const rounds = agg.rounds(key);
      const left = agg.roundsWithUnassigned(key);
      const early = agg.sum(key, (agent) => agent.commits_with_work_left);
      return `<tr><td>${this.agentLabel(key)}</td><td class="num" data-hint="${esc(`${left} of ${rounds} rounds`)}">${fmtPct(rounds ? left / rounds : null)} <span class="faint">${left}/${rounds}</span></td><td class="num" data-hint="${esc(`${early} of ${rounds} rounds`)}">${fmtPct(rounds ? early / rounds : null)} <span class="faint">${early}/${rounds}</span></td></tr>`;
    }).join("");
    const edits = agg.agents("model").flatMap((agent) => agent.edits_per_round);
    body.innerHTML = `
      <div data-role="chart"></div>
      <div class="dtable dtable--dense dtable--static behav__table"><table>
        <thead><tr><th>Agent</th><th class="num" data-hint="Rounds ended with unassigned jobs">Ended unassigned</th><th class="num" data-hint="Rounds committed with edits remaining">Edits left</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
    Histogram.render(body.querySelector("[data-role=chart]"), edits, { xLabel: "edits used in a round, model", integer: true, domain: [1, agg.index.meta.caps.edits], height: 210 });
  }

  agreement(agg) {
    const { body } = this.bodies["teacher-agreement"];
    const agreement = agg.index.aggregates.agreement;
    if (!agreement) {
      body.innerHTML = emptyBlock("Needs verdicts. Record traces with the teacher verdict set to insertions or full.", `<a class="btn btn--sm" href="${esc(this.recordHash)}">Record traces</a>`);
      return;
    }
    const strip = kpiRow([{ label: "Same operator", value: fmtPct(agreement.operator), title: `n = ${plural(agreement.n, "decision")}, verdict scope ${agg.index.meta.teacher_verdict}` }, { label: "Same vehicle", value: fmtPct(agreement.vehicle) }, { label: "Same job", value: fmtPct(agreement.job) }], { strip: true });
    body.innerHTML = `${strip}<div class="behav__chart" data-role="heat"></div><div class="behav__chart" data-role="chart"></div>`;
    const heat = new HeatTable(body.querySelector("[data-role=heat]"), {});
    heat.render({
      rows: OPERATORS.map((name, index) => ({ key: String(index), html: `<span class="opmark opmark--op${index}"></span> model ${name}` })),
      columns: OPERATORS.map((name, index) => ({ key: String(index), html: `<small>teacher</small>${name}`, title: `Teacher chose ${name}` })),
      cells: agreement.confusion.map((row, r) => row.map((value, c) => ({ value, tip: `model ${OPERATORS[r]}, teacher ${OPERATORS[c]}: ${value} of ${agreement.n}` }))),
      format: (value) => String(value),
      cornerLabel: "",
      marks: {},
    });
    const rows = agg.index.episodes.map((episode) => ({ label: String(episode.index), value: episode.agents.model ? episode.agents.model.profitable_left : null, episode }));
    ColumnChart.render(body.querySelector("[data-role=chart]"), rows, { xLabel: "episode", yLabel: "insertions left", height: 150, digits: 0, onClick: (row) => this.onNavigate(Links.episode(this.run, this.seed, row.episode.index, true)) });
  }

  rewardByTerm(agg) {
    const { body } = this.bodies["reward-by-term"];
    const values = agg.keys.map((key) => ({ key, terms: TERMS.map((term) => agg.termMean(key, term.key)) }));
    const max = Math.max(1e-9, ...values.flatMap((row) => row.terms).map((value) => Math.abs(value)).filter(isNum));
    const cell = (value) => {
      if (!isNum(value)) return `<td class="num termcell"><b class="faint">${NONE}</b></td>`;
      const side = value > 5e-7 ? "gain" : value < -5e-7 ? "loss" : "";
      const share = (Math.min(1, Math.abs(value) / max) * 50).toFixed(2);
      const fill = side ? `<i class="termcell__fill termcell__fill--${side}" style="${side === "gain" ? "left" : "right"}:50%;width:${share}%"></i>` : "";
      return `<td class="num termcell"><b class="${side}">${fmtSigned(value)}</b><span class="termcell__track"><i class="termcell__zero"></i>${fill}</span></td>`;
    };
    const rows = values.map((row) => `<tr><td>${this.agentLabel(row.key)}</td>${row.terms.map(cell).join("")}<td class="num">${fmtPct(agg.holdingShare(row.key))}</td></tr>`).join("");
    body.innerHTML = `
      <div class="dtable dtable--static termboard"><table>
        <thead><tr><th>Agent</th>${TERMS.map((term) => `<th class="num">${esc(term.label)}</th>`).join("")}<th class="num" data-hint="Holding share of the summed absolute terms">Hold share</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
  }

  jobRanking(agg) {
    const { body } = this.bodies["job-ranking"];
    const ranking = agg.index.aggregates.job_ranking;
    const rows = [1, 2, 3, 4, 5].map((priority) => {
      const ranks = agg.index.decisions.map((decision) => (decision.job_rank_by_priority ? decision.job_rank_by_priority[priority] : undefined)).filter(isNum);
      return { label: String(priority), value: ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null, n: ranks.length };
    });
    body.innerHTML = `
      ${kpiRow([{ label: "Spearman rho", value: fmtSigned(ranking.spearman, 3), title: "Spearman rho, priority against rank" }, { label: "Job scores", value: String(ranking.n), title: "Job scores in the decisions" }], { strip: true })}
      <div class="behav__chart" data-role="chart"></div>`;
    ColumnChart.render(body.querySelector("[data-role=chart]"), rows, { xLabel: "job priority (5 = most urgent)", yLabel: "mean rank", height: 230 });
  }

  critic(agg) {
    const { body } = this.bodies["critic-calibration"];
    const critic = agg.index.aggregates.critic;
    body.innerHTML = `
      ${kpiRow([{ label: "Explained", value: fmtFixed(critic.explained_variance, 3), title: "Explained variance" }, { label: "Bias", value: fmtSigned(critic.bias), title: "Bias, V minus return" }, { label: "Abs error", value: fmtFixed(critic.mae), title: "Mean absolute error" }], { strip: true })}
      <div class="behav__chart" data-role="chart"></div>`;
    const points = agg.index.decisions.map((decision) => ({ x: decision.value, y: decision.return_to_go, title: `Episode ${decision.episode}, round ${decision.round + 1}, edit ${decision.edit + 1}: ${OPERATORS[decision.operator]}`, decision }));
    Scatter.render(body.querySelector("[data-role=chart]"), points, { xLabel: "V(s)", yLabel: "discounted return-to-go", bins: 6, height: 236, onClick: (point) => this.onNavigate(Links.decision(this.run, this.seed, point.decision)) });
  }

  confidence(agg) {
    const { body } = this.bodies.confidence;
    const decisions = agg.index.decisions;
    const mean = (values) => (values.length ? values.reduce((a, b) => a + b, 0) / values.length : null);
    const pending = decisions.filter((decision) => decision.unassigned > 0);
    const tiles = [
      { label: "Mean probability", full: "Mean joint probability of the chosen action", value: fmtFixed(mean(decisions.map((d) => d.chosen_joint_prob)), 3), title: "Mean joint probability of the chosen action" },
      { label: "Hold with work", full: "P(DO_NOTHING) with unassigned jobs waiting", value: fmtFixed(mean(pending.map((d) => d.p_do_nothing)), 3), detail: `n = ${pending.length}`, title: "Mean P(DO_NOTHING) on decisions with unassigned jobs waiting" },
    ];
    if (agg.keys.includes("model_sampled")) {
      const greedy = agg.mean("model", (agent) => agent.total_reward);
      const sampled = agg.mean("model_sampled", (agent) => agent.total_reward);
      tiles.push({ label: "Greedy / sampled", full: "Greedy against sampled mean reward", value: `${fmtSigned(greedy, 1)} / ${fmtSigned(sampled, 1)}`, title: `Greedy against sampled mean reward: ${fmtSigned(greedy)} against ${fmtSigned(sampled)} points` });
    }
    const entropy = ["operator", "vehicle", "job"].map((head) => `<span class="read" data-hint="Mean normalised entropy, ${head} head"><span class="read__name">${head}</span><b>${fmtFixed(mean(decisions.map((d) => d.entropy[head]).filter(isNum)), 3)}</b></span>`).join("");
    body.innerHTML = `
      ${kpiRow(tiles, { strip: true })}
      <div class="reads behav__reads"><span class="ctl__label">Entropy</span>${entropy}</div>
      <div class="behav__chart" data-role="chart"></div>`;
    Histogram.render(body.querySelector("[data-role=chart]"), decisions.map((d) => d.chosen_joint_prob), { xLabel: "joint probability of the chosen action", bins: 10, domain: [0, 1], height: 196 });
  }

  execution(agg) {
    const { body } = this.bodies.execution;
    const nominal = 1 - agg.index.meta.environment.repossession_success_probability;
    const rates = agg.index.aggregates.pickup_failure_rate;
    const rows = agg.keys.map((key) => {
      const pickups = agg.sum(key, (agent) => agent.pickups.served + agent.pickups.failed);
      return `<tr><td>${this.agentLabel(key)}</td><td class="num">${agg.sum(key, (a) => a.served)}</td><td class="num">${agg.sum(key, (a) => a.failed)}</td><td class="num">${agg.sum(key, (a) => a.dropped)}</td><td class="num" data-hint="${esc(`${fmtPct(rates[key])} of ${pickups} pickups`)}">${fmtPct(rates[key])}</td></tr>`;
    }).join("");
    const divergent = agg.index.episodes.filter((episode) => episode.divergent_rounds.length);
    const links = divergent.map((episode) => `<a href="${esc(Links.episode(this.run, this.seed, episode.index, true))}">${episode.index}</a>`).join(", ");
    body.innerHTML = `
      <div class="dtable dtable--dense dtable--static"><table>
        <thead><tr><th>Agent</th><th class="num">Served</th><th class="num">Failed</th><th class="num">Dropped<small>bikes</small></th><th class="num" data-hint="Pickup failure rate, of all pickups">Fail rate</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <div class="reads behav__reads behav__reads--foot">
        <span class="read" data-hint="Nominal pickup failure rate of the environment"><span class="read__name">Nominal fail rate</span><b>${fmtPct(nominal)}</b></span>
        <span class="read" data-hint="Episodes where events or pickup outcomes differ between agents"><span class="read__name">Divergent</span><b>${divergent.length} of ${agg.index.episodes.length}</b>${links ? `<small>episode ${links}</small>` : ""}</span>
      </div>`;
  }

  draw(key) {
    const agg = this.agg;
    if (key === "operator-mix") this.operatorMix(agg);
    else if (key === "burst-profile") this.burst(agg);
    else if (key === "teacher-agreement") this.agreement(agg);
    else if (key === "reward-by-term") this.rewardByTerm(agg);
    else if (key === "execution") this.execution(agg);
    else if (key === "critic-calibration") this.critic(agg);
    else if (key === "confidence") this.confidence(agg);
    else this.jobRanking(agg);
    bindTips(this.container);
  }

  render(index) {
    const agg = new Aggregate(index);
    if (!agg.keys.includes("model") || !agg.agents("model").length) {
      this.agg = null;
      this.needsTraces();
      return;
    }
    this.agg = agg;
    for (const panel of PANELS) this.draw(panel.key);
  }
}
