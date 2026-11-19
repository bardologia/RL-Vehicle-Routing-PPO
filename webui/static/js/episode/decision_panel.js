import { esc, fmtDur, fmtFixed, fmtProb, fmtSigned, icon, isNum, listIds, plural, signClass, Store } from "../format.js";
import { BarChart } from "../ui/charts.js";
import { HeatTable } from "../ui/heat_table.js";
import { infoText } from "../ui/panel.js";
import { Popover } from "../ui/popover.js";
import { statusChip } from "../ui/status.js";
import { FigureStates } from "../ui/figure_states.js";
import { kvTable } from "../ui/table.js";
import { Segmented } from "../ui/tabs.js";
import { Agents, OPERATORS, Sentences, TERMS, Vehicles } from "./encoding.js";
import { VerdictPanel } from "./verdict_panel.js";

export class DecisionPanel {
  static OPEN = { operator: true, heat: true, critic: false, reward: true, breakeven: false, teacher: false, happened: true, executed: true, plan: false };

  constructor(container, { onHighlight, onIsolate, onJump, onVerdict }) {
    this.container = container;
    this.onHighlight = onHighlight;
    this.onIsolate = onIsolate;
    this.onJump = onJump;
    this.onVerdict = onVerdict;
    this.heatOperator = null;
    this.heatMode = "policy";
    this.current = null;
    this.open = { ...DecisionPanel.OPEN, ...Store.read("episode.sections", {}) };
    this.override = {};
    container.classList.add("decision");
    container.innerHTML = `<div class="fig-band decision__band">${HairlineFigure.html("insert", 'aria-hidden="true"')}</div><div class="decision__body"></div>`;
    this.band = container.querySelector(".decision__band > hairline-figure");
    this.body = container.querySelector(".decision__body");
    container.addEventListener("click", (event) => {
      const toggle = event.target.closest("[data-section-toggle]");
      if (toggle) {
        this.toggleSection(toggle);
        return;
      }
      const jump = event.target.closest("[data-jump]");
      if (jump) this.onJump(Number(jump.dataset.jump));
      const verdict = event.target.closest("button[data-verdict]");
      if (verdict && !verdict.disabled) this.onVerdict(verdict.dataset.verdict);
    });
  }

  isOpen(key, forced) {
    if (key in this.override) return this.override[key];
    return forced ? false : this.open[key];
  }

  section(key, label, summary, body, { info, forced } = {}) {
    const open = this.isOpen(key, forced);
    return `
      <section class="dsec${open ? " is-open" : ""}" data-section="${key}"${forced ? " data-forced" : ""}>
        <header class="dsec__head">
          <button type="button" class="dsec__toggle" data-section-toggle aria-expanded="${open}" aria-controls="dsec-${key}">${icon("chevron-right")}<span class="dsec__label">${esc(label)}</span><b class="dsec__sum">${summary}</b></button>
          ${info ? `<span class="dsec__info">${Popover.infoButton(label, info)}</span>` : ""}
        </header>
        <div class="dsec__body" id="dsec-${key}"${open ? "" : " hidden"}>${body}</div>
      </section>`;
  }

  reveal(key) {
    this.override[key] = true;
  }

  toggleSection(toggle) {
    const section = toggle.closest(".dsec");
    const key = section.dataset.section;
    const open = toggle.getAttribute("aria-expanded") !== "true";
    if (section.hasAttribute("data-forced")) {
      this.override[key] = open;
    } else {
      this.open[key] = open;
      delete this.override[key];
      Store.write("episode.sections", this.open);
    }
    toggle.setAttribute("aria-expanded", String(open));
    section.classList.toggle("is-open", open);
    section.querySelector(".dsec__body").hidden = !open;
  }

  roundOpen(trace, frame) {
    const open = frame.round_open;
    const executedList = (entries, word) => (entries.length ? entries.map((entry) => `job ${entry.job_id} <span class="muted">(${entry.kind}, priority ${entry.priority}, vehicle ${entry.vehicle_id})</span>`).join("<br>") : `<span class="muted">no job ${word}</span>`);
    const event = open.event;
    const eventRows = [];
    if (event) {
      eventRows.push(["Event", `<b>${esc(Sentences.event(event))}</b>`]);
      if (event.added_job_ids.length) eventRows.push(["Added jobs", listIds(event.added_job_ids)]);
      if (event.removed_job_ids.length) eventRows.push(["Cancelled jobs", listIds(event.removed_job_ids)]);
      if (event.added_vehicle_ids.length) eventRows.push(["Added vehicles", event.added_vehicle_ids.map((id) => Vehicles.swatch(id)).join(" ")]);
      if (event.removed_vehicle_ids.length) eventRows.push(["Removed vehicles", listIds(event.removed_vehicle_ids)]);
      if (event.orphaned_job_ids.length) eventRows.push(["Orphaned jobs", listIds(event.orphaned_job_ids)]);
    } else {
      eventRows.push(["Event", `<span class="muted">no disruption</span>`]);
    }

    const charged = open.holding_for_seq !== null && open.holding_for_seq !== undefined;
    const holding = charged
      ? `<span class="${signClass(open.holding_charged)}">${fmtSigned(open.holding_charged, 2)}</span> for the plan left at <button type="button" class="linkbtn" data-jump="${open.holding_for_seq}">the commit of round ${frame.round}</button> over ${Math.round(open.gap_s / 60)} min`
      : `<span class="muted">nothing charged in the first round</span>`;

    const dropped = open.executed.dropped.length ? open.executed.dropped.map((entry) => `${plural(entry.count, "bike")} by vehicle ${entry.vehicle_id}`).join(", ") : `<span class="muted">no bikes dropped</span>`;
    const droppedCount = open.executed.dropped.reduce((sum, entry) => sum + entry.count, 0);
    const eventSummary = event ? `${Sentences.eventKind(event.type)} ${event.count}` : "no event";
    const lane = Agents.label(trace.agentKey);

    const sections = [
      this.section("happened", "What happened", esc(eventSummary), kvTable([
        ["Gap since the last commit", frame.round === 0 ? `<span class="muted">episode start</span>` : `${Math.round(open.gap_s / 60)} min`],
        ["Holding, points", holding],
        ...eventRows,
      ]), { info: infoText("The clock advances, vehicles execute the committed plan, then a disruption arrives. The agent decides on the next frames.") }),
    ];
    if (frame.round > 0) {
      sections.push(this.section("executed", "Executed", `${open.executed.served.length} / ${open.executed.failed.length} / ${droppedCount}`, kvTable([
        ["Served", executedList(open.executed.served, "served")],
        ["Failed pickups", executedList(open.executed.failed, "failed")],
        ["Dropped at the depot", dropped],
      ]), { info: infoText("Jobs served, pickups failed and bikes dropped at the depot while the committed plan ran during the gap.") }));
    }
    sections.push(this.section("plan", "Plan", `${frame.plan.unassigned.length} unassigned`, kvTable([
      ["Unassigned jobs", frame.plan.unassigned.length ? frame.plan.unassigned.map((entry) => `job ${entry.job_id} <span class="muted">(priority ${entry.priority})</span>`).join(", ") : `<span class="muted">none</span>`],
      ["Idle vehicles", frame.plan.idle_vehicle_ids.length ? frame.plan.idle_vehicle_ids.map((id) => Vehicles.swatch(id, true)).join(" ") : `<span class="muted">none</span>`],
      ["Holding rate if committed", `${fmtFixed(frame.plan.kpis.holding_rate, 1)} pts/h`],
    ])));

    return `
      <header class="decision__head">
        <h2>Round ${frame.round + 1} opens</h2>
        <p class="decision__meta">${esc(lane)}, ${frame.round === 0 ? "episode start" : `after ${Math.round(open.gap_s / 60)} min`}</p>
      </header>
      ${sections.join("")}`;
  }

  operatorBars(decision) {
    const policy = decision.policy;
    const reasons = ["nothing unassigned", "no vehicle has jobs", ""];
    return OPERATORS.map((name, index) => {
      const masked = policy.operator.masked[index];
      const probability = masked ? 0 : policy.operator.probs[index];
      const raw = policy.operator.raw_probs[index];
      const marks = [];
      if (decision.action.operator === index) marks.push("chosen");
      if (policy.greedy_action.operator === index) marks.push("greedy");
      return `
        <div class="opbar${decision.action.operator === index ? " is-chosen" : ""}">
          <div class="opbar__name"><i class="opmark opmark--op${index}"></i>${name}</div>
          <div class="opbar__track${masked ? " is-masked" : ""}" data-hint-title="${name}" data-hint="${masked ? `masked: ${reasons[index]}` : `masked probability ${fmtProb(probability)}, raw ${fmtProb(raw)}`}">
            <i class="opbar__fill" style="width:${(probability * 100).toFixed(1)}%"></i>
            <i class="opbar__raw" style="left:${(raw * 100).toFixed(1)}%"></i>
          </div>
          <div class="opbar__value">${masked ? `<span class="muted">masked</span>` : fmtProb(probability)}</div>
          <div class="opbar__note">${masked ? esc(reasons[index]) : marks.join(", ")}</div>
        </div>`;
    }).join("");
  }

  operatorInfo(policy) {
    return infoText("Bar: probability after masking. Tick: raw probability.", `Normalised entropy: operator ${fmtFixed(policy.entropy.operator, 2)}, vehicle ${fmtFixed(policy.entropy.vehicle, 2)}, job ${fmtFixed(policy.entropy.job, 2)} (1 is uniform).`);
  }

  heatData(trace, frame, operator, mode) {
    const decision = frame.decision;
    const policy = decision.policy;
    const verdict = decision.verdict;
    const jobs = trace.entities.jobs;
    const cellKey = (action) => (action && action.operator === operator && action.vehicle_id !== null ? `${action.vehicle_id}:${action.job_id}` : null);
    const marks = {};

    let vehicleIds;
    let jobIds;
    let value;
    if (mode === "teacher") {
      const table = new Map();
      const infeasible = new Set();
      if (operator === 0) {
        for (const entry of verdict.insertions) table.set(`${entry.vehicle_id}:${entry.job_id}`, entry.reward);
        for (const [jobId, vehicleId] of verdict.infeasible) infeasible.add(`${vehicleId}:${jobId}`);
      } else {
        for (const entry of verdict.removals || []) table.set(`${entry.vehicle_id}:${entry.job_id}`, entry.reward);
      }
      const keys = [...table.keys(), ...infeasible];
      vehicleIds = [...new Set(keys.map((key) => Number(key.split(":")[0])))].sort((a, b) => a - b);
      jobIds = [...new Set(keys.map((key) => Number(key.split(":")[1])))];
      value = (vehicleId, jobId) => {
        const key = `${vehicleId}:${jobId}`;
        if (table.has(key)) return { value: table.get(key), tip: `job ${jobId} ${operator === 0 ? "into" : "from"} vehicle ${vehicleId}: teacher one-step reward ${fmtSigned(table.get(key), 2)}` };
        if (infeasible.has(key)) return { state: "infeasible", tip: `job ${jobId} into vehicle ${vehicleId}: infeasible` };
        return { state: "masked", tip: "not an option" };
      };
      marks.teacher = cellKey(verdict.teacher_action);
      marks.chosen = cellKey(decision.action);
    } else {
      vehicleIds = policy.vehicle_ids;
      const column = (jobId) => policy.job_ids.indexOf(jobId);
      jobIds = policy.job_ids.filter((jobId, index) => policy.job_probs[operator].some((row) => row[index] !== null));
      value = (vehicleId, jobId) => {
        const v = policy.vehicle_ids.indexOf(vehicleId);
        const pVehicle = policy.vehicle_probs[operator][v];
        const pJob = policy.job_probs[operator][v][column(jobId)];
        if (pVehicle === null || pJob === null || pJob === undefined) return { state: "masked", tip: "masked" };
        return { value: pVehicle * pJob, tip: `job ${jobId}, vehicle ${vehicleId}: P(vehicle) ${fmtProb(pVehicle)} x P(job) ${fmtProb(pJob)} = ${fmtProb(pVehicle * pJob)}` };
      };
      marks.chosen = cellKey(decision.action);
      const argmax = cellKey(policy.joint_argmax_action);
      if (argmax && argmax !== cellKey(policy.greedy_action)) marks.argmax = argmax;
    }

    jobIds.sort((a, b) => jobs[b].priority - jobs[a].priority || a - b);
    const cells = vehicleIds.map((vehicleId) => jobIds.map((jobId) => value(vehicleId, jobId)));
    const sum = (list) => list.reduce((total, cell) => total + (cell && isNum(cell.value) ? cell.value : 0), 0);
    return {
      rows: vehicleIds.map((id) => ({ key: String(id), html: Vehicles.swatch(id) })),
      columns: jobIds.map((id) => ({ key: String(id), html: `<b>${id}</b><small>${jobs[id].kind === "repossession" ? "rep" : "sup"} p${jobs[id].priority}</small>`, title: `Job ${id}: ${jobs[id].kind}, priority ${jobs[id].priority}` })),
      cells,
      rowTotals: mode === "teacher" ? null : cells.map(sum),
      columnTotals: mode === "teacher" ? null : jobIds.map((_, c) => sum(cells.map((row) => row[c]))),
      mode: mode === "teacher" ? "diverging" : "ramp",
      format: mode === "teacher" ? (v) => fmtSigned(v, 1) : DecisionPanel.percent,
      cornerLabel: "veh / job",
      totalLabel: "sum",
      marks,
    };
  }

  static percent(value) {
    return value * 100 >= 99.5 ? "100" : (value * 100).toFixed(value < 0.0995 ? 1 : 0);
  }

  heatOperatorFor(decision) {
    return decision.action.operator === 1 ? 1 : decision.policy.operator.masked[0] && !decision.policy.operator.masked[1] ? 1 : 0;
  }

  heatSummary(decision) {
    const policy = decision.policy;
    const action = decision.action;
    if (action.operator === 2) return policy.operator.masked[this.heatOperatorFor(decision)] ? "masked" : "commit";
    const v = policy.vehicle_ids.indexOf(action.vehicle_id);
    const c = policy.job_ids.indexOf(action.job_id);
    const pVehicle = v < 0 ? null : policy.vehicle_probs[action.operator][v];
    const pJob = v < 0 || c < 0 ? null : policy.job_probs[action.operator][v][c];
    if (!isNum(pVehicle) || !isNum(pJob)) return "masked";
    return `${DecisionPanel.percent(pVehicle * pJob)}%`;
  }

  heatInfo() {
    return infoText(
      "Policy probability: each cell is P(vehicle | operator) x P(job | vehicle) in percent; the outline is the action taken, and a diamond marks the joint argmax when it differs from the sequential greedy choice. Columns by priority, highest first.",
      "Teacher reward: each cell is the teacher one-step reward in points; the dashed outline is the teacher's pick, the solid outline the action taken.",
    );
  }

  critic(decision) {
    const error = decision.policy.value - decision.return_to_go_discounted;
    return `
      <div class="decision__triple">
        <div data-hint-title="V(s)" data-hint="The critic estimate of this state."><span>V(s)</span><b>${fmtSigned(decision.policy.value, 2)}</b></div>
        <div data-hint-title="Return-to-go" data-hint="The realised discounted return from this decision on."><span>Return-to-go</span><b>${fmtSigned(decision.return_to_go_discounted, 2)}</b></div>
        <div data-hint-title="Error" data-hint="V minus the realised return-to-go."><span>Error</span><b class="${signClass(-Math.abs(error))}">${fmtSigned(error, 2)}</b></div>
      </div>`;
  }

  criticInfo(decision) {
    return infoText("V(s) is the critic estimate; return-to-go is the realised discounted return; error is V minus return-to-go.", `Discounted with gamma ${fmtFixed(this.current.trace.meta.gamma, 2)}; undiscounted return-to-go ${fmtSigned(decision.return_to_go, 2)}.`);
  }

  reward(trace, frame) {
    const decision = frame.decision;
    const open = trace.roundOpen(frame.round);
    const rows = TERMS.map((term) => {
      const row = { label: term.label, value: decision.rewards[term.key] };
      let hint = null;
      if (term.cost) hint = `cost ${fmtFixed(decision.costs[`old_${term.cost}`], 2)} to ${fmtFixed(decision.costs[`new_${term.cost}`], 2)}`;
      if (term.key === "action_reward") hint = `operator cost ${fmtFixed(decision.costs.action_cost, 2)}, ${plural(Math.round(decision.costs.disruption), "stop")} changed vehicle`;
      if (term.key === "holding_reward") {
        row.hatched = true;
        if (Math.abs(decision.rewards.holding_reward) > 1e-9 && open && open.round_open.holding_for_seq !== null) {
          row.note = `charged for the plan left at <button type="button" class="linkbtn" data-jump="${open.round_open.holding_for_seq}">round ${frame.round}</button> over ${Math.round(open.round_open.gap_s / 60)} min`;
        } else {
          hint = frame.edit === 0 && frame.round > 0 ? "the previous commit left nothing waiting" : "only charged on the first decision of a round";
        }
      }
      if (hint) row.attrs = ` data-hint-title="${esc(term.label)}" data-hint="${esc(hint)}"`;
      return row;
    });
    return `
      ${BarChart.html(rows)}
      <div class="decision__total"><span>cumulative <b class="${signClass(decision.cumulative)}">${fmtSigned(decision.cumulative, 2)}</b></span></div>`;
  }

  breakEven(decision) {
    const added = decision.added_drive_s;
    const limit = decision.break_even_s;
    const top = Math.max(added, limit, 1) * 1.1;
    return `
      <div class="breakeven">
        <div class="breakeven__track"><i class="breakeven__fill" style="width:${((Math.max(0, added) / top) * 100).toFixed(1)}%"></i><i class="breakeven__limit" style="left:${((limit / top) * 100).toFixed(1)}%"></i></div>
        <div class="breakeven__labels"><span>added drive ${fmtDur(added)}</span><span>break-even ${fmtDur(limit)}</span></div>
      </div>`;
  }

  breakEvenInfo(decision) {
    const added = decision.added_drive_s;
    const limit = decision.break_even_s;
    if (decision.wasted) return infoText("The insertion was infeasible, so the plan did not change and the edit cost was paid for nothing.");
    return infoText(`This insertion added ${fmtDur(added)} h:mm of driving. It pays off up to ${fmtDur(limit)}: ${added <= limit ? "worth it" : "it costs more than the job is worth"}.`);
  }

  empty(html) {
    this.current = null;
    this.body.innerHTML = html;
  }

  renderBand(frame) {
    const name = frame.kind === "round_open" ? "disruption" : frame.decision.policy ? "heads" : "insert";
    if (this.band.getAttribute("name") !== name) {
      delete this.band.dataset.stateKey;
      this.band.setAttribute("name", name);
    }
    const state = name === "disruption" ? FigureStates.disruption(frame) : name === "heads" ? FigureStates.decisionHeads(frame.decision) : FigureStates.insert(frame);
    FigureStates.write(this.band, state);
  }

  head(trace, frame) {
    const decision = frame.decision;
    const policy = decision.policy;
    const states = [VerdictPanel.headState(decision.verdict)];
    if (decision.wasted) states.push(statusChip("WASTED", "edit"));
    if (decision.is_commit && frame.plan.kpis.unassigned > 0 && decision.edits_remaining > 0) states.push(`<span class="state state--warn"><i></i>${plural(frame.plan.kpis.unassigned, "job")} left unassigned</span>`);
    if (policy) states.push(`<span class="decision__note" data-hint-title="Joint probability" data-hint="Rank ${policy.chosen_rank} among all actions.">${trace.meta.greedy ? "greedy" : "sampled"}, p ${fmtProb(policy.chosen_joint_prob)}</span>`);
    return `
      <header class="decision__head">
        <h2>${esc(Sentences.action(decision.action))}</h2>
        <p class="decision__meta">${esc(Agents.label(trace.agentKey))}, round ${frame.round + 1}, edit ${frame.edit + 1} of ${trace.meta.caps.edits}${decision.ends_round ? ", ends the round" : ""}</p>
        <div class="decision__states">${states.join("")}</div>
      </header>`;
  }

  render({ trace, seq, verdictState }) {
    const frame = trace.frame(seq);
    const frameChanged = !this.current || this.current.trace !== trace || this.current.seq !== seq;
    const scroll = frameChanged ? 0 : this.container.scrollTop;
    if (frameChanged) this.override = {};
    this.current = { trace, seq };
    if (!frame) {
      this.body.innerHTML = "";
      return;
    }
    this.renderBand(frame);
    if (frame.kind === "round_open") {
      this.body.innerHTML = this.roundOpen(trace, frame);
      this.container.scrollTop = scroll;
      return;
    }

    const decision = frame.decision;
    const policy = decision.policy;
    const verdict = decision.verdict;
    const parts = [this.head(trace, frame)];
    if (policy) {
      const commit = decision.action.operator === 2;
      const heatSummary = this.heatSummary(decision);
      parts.push(this.section("operator", "Operator", fmtProb(policy.operator.probs[decision.action.operator]), this.operatorBars(decision), { info: this.operatorInfo(policy) }));
      if (this.heatMode === "teacher" && !verdict) this.heatMode = "policy";
      parts.push(this.section("heat", "Vehicle by job", heatSummary, `
        <div class="decision__switches"><div data-role="heat-op"></div><div data-role="heat-mode"></div></div>
        <div data-role="heat"></div>`, { info: this.heatInfo(), forced: commit || heatSummary === "masked" }));
      parts.push(this.section("critic", "Critic", fmtSigned(decision.policy.value - decision.return_to_go_discounted, 2), this.critic(decision), { info: this.criticInfo(decision) }));
    } else {
      parts.push(`<p class="decision__rule">Rule-based agent, no policy distribution.</p>`);
    }
    parts.push(this.section("reward", "Reward", fmtSigned(decision.reward, 2), this.reward(trace, frame), { info: infoText("Points; right is gain, left is loss. The step reward is the sum of the six terms. Holding, hatched, is only charged on the first decision of a round.") }));
    if (decision.action.operator === 0 && isNum(decision.break_even_s)) parts.push(this.section("breakeven", "Break-even", decision.wasted ? "wasted" : `${fmtDur(decision.added_drive_s)} of ${fmtDur(decision.break_even_s)}`, this.breakEven(decision), { info: this.breakEvenInfo(decision) }));
    parts.push(this.section("teacher", "Teacher", esc(VerdictPanel.summary(verdict, verdictState)), `<div data-role="verdict">${VerdictPanel.html(frame, verdictState)}</div>`, { info: VerdictPanel.info(verdict) }));

    this.body.innerHTML = parts.join("");
    this.container.scrollTop = scroll;
    if (policy) this.mountHeat(trace, frame);
  }

  mountHeat(trace, frame) {
    const decision = frame.decision;
    const verdict = decision.verdict;
    let operator = this.heatOperator === null ? this.heatOperatorFor(decision) : this.heatOperator;
    const heat = new HeatTable(this.container.querySelector("[data-role=heat]"), {
      onHover: (cell) => this.onHighlight(cell ? { vehicleId: cell.row === "" ? null : Number(cell.row), jobId: cell.column === null ? null : Number(cell.column) } : null),
      onClick: (cell) => {
        if (cell.row !== "" && cell.column === null) this.onIsolate(Number(cell.row));
      },
    });

    const draw = () => {
      const masked = decision.policy.operator.masked[operator];
      if (this.heatMode === "policy" && masked) {
        heat.container.innerHTML = `<p class="muted">${OPERATORS[operator]} is masked on this state: ${operator === 0 ? "nothing is unassigned" : "no vehicle has jobs"}.</p>`;
        return;
      }
      if (this.heatMode === "teacher" && operator === 1 && !verdict.removals) {
        heat.container.innerHTML = `<p class="muted">The verdict covers insertions only. Ask with removals to fill this grid.</p>`;
        return;
      }
      const data = this.heatData(trace, frame, operator, this.heatMode);
      if (!data.columns.length) {
        heat.container.innerHTML = `<p class="muted">No ${operator === 0 ? "insertion" : "removal"} to show on this state.</p>`;
        return;
      }
      heat.render(data);
    };

    new Segmented(this.container.querySelector("[data-role=heat-op]"), [{ value: "0", label: "INSERT" }, { value: "1", label: "REMOVE" }], String(operator), (value) => {
      operator = Number(value);
      this.heatOperator = operator;
      draw();
    }, "Operator shown in the grid");
    new Segmented(this.container.querySelector("[data-role=heat-mode]"), [{ value: "policy", label: "Policy" }, { value: "teacher", label: "Teacher reward", disabled: !verdict, title: verdict ? "" : "Ask the teacher first" }], this.heatMode, (value) => {
      this.heatMode = value;
      draw();
    }, "Quantity shown in the grid");
    draw();
  }
}
