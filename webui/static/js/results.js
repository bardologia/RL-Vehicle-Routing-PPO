"use strict";

class ResultsView {
  constructor() {
    this.rail = document.getElementById("results-rail");
    this.main = document.getElementById("results-main");
    this.runs = [];
    this.activeRun = null;
    this.activeTab = "overview";
    this.detail = null;
    this.curves = {};
    this.charts = [];
    this.curveGroup = null;
  }

  async enter(param) {
    const res = await apiGet("/api/runs");
    this.runs = res.runs || [];
    this._renderRail();

    const name = param ? decodeURIComponent(param) : this.activeRun || (this.runs[0] && this.runs[0].name);
    if (name) this._select(name);
    else this.main.innerHTML = `<p class="scn-hint">No runs found under runs/.</p>`;
  }

  leave() {
    this._destroyCharts();
  }

  _destroyCharts() {
    this.charts.forEach((c) => c.destroy());
    this.charts = [];
  }

  _renderRail() {
    this.rail.innerHTML = this.runs.map((r) => `
      <button class="run-card" data-name="${escapeHtml(r.name)}">
        <div class="run-card__row">
          <span class="run-card__name">${escapeHtml(r.name)}</span>
          <span class="badge badge--${r.kind}">${r.kind}</span>
        </div>
        ${r.evaluation ? `<div class="run-card__eval">model <b>${fmtNum(r.evaluation.model)}</b> · teacher ${fmtNum(r.evaluation.teacher)}</div>` : ""}
      </button>`).join("");

    this.rail.querySelectorAll(".run-card").forEach((el) => {
      el.addEventListener("click", () => this._select(el.dataset.name));
    });
  }

  async _select(name) {
    this.activeRun = name;
    this.rail.querySelectorAll(".run-card").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.name === name);
    });

    this.main.innerHTML = `<p class="scn-hint">loading…</p>`;
    this.detail = await apiGet(`/api/runs/${encodeURIComponent(name)}`);
    if (this.detail.error) {
      this.main.innerHTML = `<p class="scn-hint">${escapeHtml(this.detail.error)}</p>`;
      return;
    }
    this._renderDetail();
  }

  _renderDetail() {
    const d = this.detail;
    const tabs = [
      ["overview", "Overview"],
      ["curves", `Curves${d.has_events ? "" : " (none)"}`],
      ["gallery", `Gallery (${d.analysis.length})`],
      ["logs", `Logs (${d.logs.length})`],
      ["files", "Files"],
    ];

    this.main.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px; margin-bottom: 10px">
        <h2 style="margin:0; font-size:20px; font-weight:800">${escapeHtml(d.name)}</h2>
        <span class="badge badge--${d.kind}">${d.kind}</span>
        ${d.has_checkpoint ? `<span class="badge">checkpoint</span>` : ""}
      </div>
      <div class="tabs">${tabs.map(([k, t]) => `<button class="tab ${k === this.activeTab ? "is-active" : ""}" data-tab="${k}">${t}</button>`).join("")}</div>
      <div id="results-tab-body"></div>`;

    this.main.querySelectorAll(".tab").forEach((el) => {
      el.addEventListener("click", () => {
        this.activeTab = el.dataset.tab;
        this.main.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === el));
        this._renderTab();
      });
    });

    this._renderTab();
  }

  _renderTab() {
    this._destroyCharts();
    const body = document.getElementById("results-tab-body");
    if (this.activeTab === "overview") this._renderOverview(body);
    else if (this.activeTab === "curves") this._renderCurves(body);
    else if (this.activeTab === "gallery") this._renderGallery(body);
    else if (this.activeTab === "logs") this._renderLogs(body);
    else this._renderFiles(body);
  }

  _renderOverview(body) {
    const d = this.detail;
    const evals = Object.entries(d.evaluations);
    const mds = Object.entries(d.markdown);

    if (!evals.length && !mds.length) {
      body.innerHTML = `<p class="scn-hint">No evaluation reports or markdown files in this run.</p>`;
      return;
    }

    body.innerHTML = `
      ${evals.map(([file, report]) => `
        <div class="card" style="margin-bottom:14px; overflow-x:auto">
          <h3 class="card__title">${escapeHtml(file)}</h3>
          ${this._evalTable(report)}
        </div>`).join("")}
      ${mds.map(([file, text]) => `
        <div class="card" style="margin-bottom:14px">
          <h3 class="card__title">${escapeHtml(file)}</h3>
          <div class="md-block">${escapeHtml(text)}</div>
        </div>`).join("")}`;
  }

  _evalTable(report) {
    const agents = Object.keys(report);
    if (!agents.length) return "";
    const opNames = { op0: "insert", op1: "remove", op2: "no-op", op3: "reopt" };

    const bestReward = Math.max(...agents.map((a) => report[a].mean_reward));

    const rows = agents.map((agent) => {
      const m = report[agent];
      const ops = Object.entries(m.operator_frequency || {}).map(([k, v]) => `${opNames[k] || k} ${(v * 100).toFixed(0)}%`).join(" · ");
      return `<tr>
        <td><b>${escapeHtml(agent)}</b></td>
        <td class="${m.mean_reward === bestReward ? "is-best" : ""}">${fmtNum(m.mean_reward)}</td>
        <td>${fmtNum(m.std_reward)}</td>
        <td>${fmtNum(m.mean_final_cost, 0)}</td>
        <td>${fmtNum(m.mean_final_unassigned, 2)}</td>
        <td>${m.episodes}</td>
        <td style="white-space:normal">${ops}</td>
      </tr>`;
    }).join("");

    return `<table class="tbl">
      <thead><tr><th>agent</th><th>mean reward</th><th>std</th><th>final cost</th><th>unassigned</th><th>episodes</th><th>operators</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  }

  async _renderCurves(body) {
    const name = this.detail.name;
    if (!this.detail.has_events) {
      body.innerHTML = `<p class="scn-hint">This run has no TensorBoard event files.</p>`;
      return;
    }

    if (!this.curves[name]) {
      body.innerHTML = `<p class="scn-hint">reading event files…</p>`;
      const res = await apiGet(`/api/runs/${encodeURIComponent(name)}/curves`);
      if (res.error) { body.innerHTML = `<p class="scn-hint">${escapeHtml(res.error)}</p>`; return; }
      this.curves[name] = res.tags || {};
    }

    const tags = this.curves[name];
    const groups = {};
    for (const tag of Object.keys(tags)) {
      const group = tag.includes("/") ? tag.split("/")[0] : "other";
      (groups[group] = groups[group] || []).push(tag);
    }

    const groupNames = Object.keys(groups).sort();
    if (!groupNames.length) { body.innerHTML = `<p class="scn-hint">No scalar tags found.</p>`; return; }
    if (!this.curveGroup || !groups[this.curveGroup]) {
      this.curveGroup = groups["episode"] ? "episode" : groupNames[0];
    }

    body.innerHTML = `
      <div class="curve-controls" id="curve-groups">
        ${groupNames.map((g) => `<button class="chip ${g === this.curveGroup ? "is-on" : ""}" data-group="${escapeHtml(g)}">${escapeHtml(g)} <span style="opacity:.55">${groups[g].length}</span></button>`).join("")}
      </div>
      <div class="chart-grid" id="curve-charts"></div>`;

    body.querySelectorAll("[data-group]").forEach((el) => {
      el.addEventListener("click", () => {
        this.curveGroup = el.dataset.group;
        this._renderTab();
      });
    });

    const host = document.getElementById("curve-charts");
    const list = groups[this.curveGroup].sort().slice(0, 24);

    for (const tag of list) {
      const card = document.createElement("div");
      card.className = "chart-card";
      card.innerHTML = `<div class="chart-card__title">${escapeHtml(tag)}</div>`;
      host.appendChild(card);

      const chart = new LineChart(card, { height: 190 });
      const t = tags[tag];
      chart.setSeries([{ name: tag.split("/").pop(), points: t.steps.map((s, i) => ({ x: s, y: t.values[i] })) }]);
      this.charts.push(chart);
    }

    if (groups[this.curveGroup].length > 24) {
      const note = document.createElement("p");
      note.className = "scn-hint";
      note.textContent = `showing 24 of ${groups[this.curveGroup].length} tags in this group`;
      host.after(note);
    }
  }

  _renderGallery(body) {
    const figures = this.detail.analysis;
    if (!figures.length) {
      body.innerHTML = `<p class="scn-hint">No analysis figures in this run.</p>`;
      return;
    }

    body.innerHTML = `<div class="gallery">${figures.map((fig) => `
      <figure data-src="${escapeHtml(fig.url)}" data-cap="${escapeHtml(fig.name)}">
        <img src="${escapeHtml(fig.url)}" loading="lazy" alt="${escapeHtml(fig.name)}">
        <figcaption>${escapeHtml(fig.name)}</figcaption>
      </figure>`).join("")}</div>`;

    body.querySelectorAll("figure").forEach((el) => {
      el.addEventListener("click", () => openLightbox(el.dataset.src, el.dataset.cap));
    });
  }

  _renderLogs(body) {
    const logs = this.detail.logs;
    if (!logs.length) {
      body.innerHTML = `<p class="scn-hint">No log files in this run.</p>`;
      return;
    }

    body.innerHTML = logs.map((log) => `
      <div class="card" style="margin-bottom:14px">
        <h3 class="card__title">${escapeHtml(log.name)}</h3>
        <pre class="job-log" style="max-height: 480px">${escapeHtml(log.text)}</pre>
      </div>`).join("");

    body.querySelectorAll(".job-log").forEach((el) => { el.scrollTop = el.scrollHeight; });
  }

  _renderFiles(body) {
    body.innerHTML = `<div class="card" style="overflow-x:auto"><table class="tbl">
      <thead><tr><th>file</th><th>size</th></tr></thead>
      <tbody>${this.detail.files.map((f) => `<tr><td>${escapeHtml(f.rel)}</td><td>${fmtBytes(f.size)}</td></tr>`).join("")}</tbody>
    </table></div>`;
  }
}

window.ResultsView = ResultsView;
