import { esc } from "../format.js";

export class ScriptCatalogView {
  static SECTIONS = [
    { category: "Training", blurb: "Clone the regret-insertion teacher into a warm start, then improve the policy with PPO." },
    { category: "Evaluation", blurb: "Score a checkpoint on held-out episodes against the baselines, or watch it act on one random scenario." },
    { category: "Analysis", blurb: "Record the traces of every agent that the Diagnose tab and the episode replay read." },
  ];

  constructor(view, host) {
    this.view = view;
    this.host = host;
    this.filter = "All";
    this.query = "";
  }

  render(scripts) {
    this.scripts = scripts;
    this.host.innerHTML = `
      <div class="catalog">
        ${this.view.head({ title: "Entry points", lead: "Orchestrated entry scripts. Review the configuration defaults, override any field, and launch with live output.", figure: true, figureName: "cabinet" })}
        <div class="catalog__filters ctls" data-role="filters"></div>
        <div class="catalog__sections" data-role="sections"></div>
      </div>`;
    this.renderFilters();
    this.renderSections();
  }

  focusSearch() {
    const input = this.host.querySelector(".catalog__search");
    if (input) input.focus();
  }

  renderFilters() {
    const host = this.host.querySelector("[data-role=filters]");
    const counts = new Map();
    this.scripts.forEach((script) => counts.set(script.category, (counts.get(script.category) || 0) + 1));
    const categories = ["All", ...ScriptCatalogView.SECTIONS.map((section) => section.category)];

    host.innerHTML = `
      <input class="input catalog__search" type="search" placeholder="Search scripts" aria-label="Search scripts" spellcheck="false">
      <div class="chips" role="group" aria-label="Category">${categories.map((category) => `<button type="button" class="chip" data-category="${esc(category)}" aria-pressed="${category === this.filter}">${esc(category)} <b>${category === "All" ? this.scripts.length : counts.get(category) || 0}</b></button>`).join("")}</div>`;

    host.querySelector("input").addEventListener("input", (event) => {
      this.query = event.target.value.trim().toLowerCase();
      this.renderSections();
    });
    host.querySelectorAll(".chip").forEach((chip) => chip.addEventListener("click", () => {
      this.filter = chip.dataset.category;
      host.querySelectorAll(".chip").forEach((other) => other.setAttribute("aria-pressed", String(other === chip)));
      this.renderSections();
    }));
  }

  matches(script) {
    if (this.filter !== "All" && script.category !== this.filter) return false;
    if (!this.query) return true;
    const hay = [script.title, script.purpose, script.file, script.key].join(" ").toLowerCase();
    return this.query.split(/\s+/).every((term) => hay.includes(term));
  }

  renderSections() {
    const host = this.host.querySelector("[data-role=sections]");
    const html = ScriptCatalogView.SECTIONS.map((section) => {
      const items = this.scripts.filter((script) => script.category === section.category && this.matches(script));
      if (!items.length) return "";
      return `
        <section class="catalog__section">
          <div class="catalog__sectionhead"><h2 class="catalog__sectiontitle">${esc(section.category)}</h2><span class="catalog__count">${items.length} ${items.length === 1 ? "entry point" : "entry points"}</span></div>
          <p class="catalog__blurb">${esc(section.blurb)}</p>
          <div class="catalog__grid">${items.map((script) => this.card(script)).join("")}</div>
        </section>`;
    }).join("");

    host.innerHTML = html || `<p class="catalog__none">No entry points match this search.</p>`;
    host.querySelectorAll("[data-role=sequence]").forEach((button) => button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.view.sequences.quickAdd(button.dataset.key, button.dataset.title);
    }));
  }

  card(script) {
    return `
      <article class="script-card">
        <div class="script-card__top"><span class="script-card__cat">${esc(script.category)}</span><span class="script-card__file">${esc(script.file)}</span></div>
        <h3 class="script-card__title"><a class="script-card__link" href="#/launch/${esc(script.key)}">${esc(script.title)}</a></h3>
        <p class="script-card__purpose" title="${esc(script.purpose)}">${esc(script.purpose)}</p>
        <div class="script-card__foot">
          <span class="script-card__class">${esc(script.config_class)}</span>
          <button type="button" class="btn btn--sm script-card__seq" data-role="sequence" data-key="${esc(script.key)}" data-title="${esc(script.title)}" title="Add this entry as a block to the active sequence">Add to sequence</button>
          <span class="script-card__go" aria-hidden="true">Configure</span>
        </div>
      </article>`;
  }
}
