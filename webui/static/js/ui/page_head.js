import { esc } from "../format.js";

export class PageHead {
  static FORMS = { head: "", hero: " view-head--hero", narrow: " view-head--narrow", duo: " view-head--duo", plain: " view-head--plain" };

  static control(label, html) {
    return `<label class="ctl"><span class="ctl__label">${esc(label)}</span>${html}</label>`;
  }

  static group(label, html) {
    return `<span class="ctl" role="group" aria-label="${esc(label)}"><span class="ctl__label">${esc(label)}</span>${html}</span>`;
  }

  static html({ title = "", lead = "", controls = "", dateline = "", datelineExtra = "", kpis = "", extra = "", after = "", figure = "", form = "head", className = "" } = {}) {
    return `
      <header class="view-head${PageHead.FORMS[form]}${className ? ` ${className}` : ""}">
        <div class="view-head__main">
          <h1 data-role="title">${esc(title)}</h1>
          <p class="view-head__lead" data-role="lead">${esc(lead)}</p>
          <div class="view-head__controls ctls" data-role="controls">${controls}</div>
          <p class="view-head__date"><span data-role="dateline">${esc(dateline)}</span>${datelineExtra}</p>
          ${extra}
          <div class="view-head__kpis" data-role="kpis">${kpis}</div>
          ${after}
        </div>
        <div class="view-head__figure" data-role="figure">${figure}</div>
      </header>`;
  }
}
