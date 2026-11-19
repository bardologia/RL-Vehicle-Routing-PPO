import { el } from "../format.js";
import { FigureStates } from "../ui/figure_states.js";
import { RunConsole } from "../console/run_console.js";

export class ConsoleView {
  constructor(app, root) {
    this.app = app;
    this.root = root;

    root.appendChild(el(`
      <div class="console">
        <aside class="console__jobs" aria-label="Jobs">
          <header class="console__head"><h2 class="console__label">Jobs</h2><span class="console__count" data-role="count"></span></header>
          <ul class="job-list" data-role="list"></ul>
          <div class="console__foot">${HairlineFigure.html("terminal", 'aria-hidden="true"')}</div>
        </aside>
        <div class="console__stage">
          <div class="console__hint" data-role="hint">
            ${HairlineFigure.html("queue")}
            <p class="console__hinttext">Launch a script to see live output.</p>
            <a class="btn btn--sm" href="#/launch">Open Launch</a>
          </div>
          <div class="console__tiles" data-role="tiles" hidden></div>
        </div>
      </div>`));

    this.figure = root.querySelector("hairline-figure[name=queue]");
    this.console = new RunConsole(app, {
      list: root.querySelector("[data-role=list]"),
      tiles: root.querySelector("[data-role=tiles]"),
      hint: root.querySelector("[data-role=hint]"),
      count: root.querySelector("[data-role=count]"),
      onJobs: (jobs) => FigureStates.write(this.figure, FigureStates.queue(jobs)),
    });
  }

  focusSearch() {
    const input = this.root.querySelector(".console-tile__log:not([hidden]) input[type=search]");
    if (input) input.focus();
  }

  async enter(parts) {
    this.console.show();
    await this.console.refresh();
    if (parts[0]) this.console.open(parts[0]);
    this.console.queueFit();
  }

  leave() {
    this.console.hide();
  }
}
