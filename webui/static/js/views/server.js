import { el } from "../format.js";
import { Orb } from "../fx/orb.js";
import { StatusBoard } from "../server/status_board.js";

export class ServerView {
  constructor(app, root) {
    this.app = app;
    this.root = root;

    root.appendChild(el(`
      <div class="view__pad server">
        <header class="mast server-mast">
          <div class="mast__flank server-mast__flank"><div class="fig-fill">${HairlineFigure.html("fleet", 'data-role="mast-fleet" autopilot="off" aria-hidden="true"')}</div></div>
          <div class="mast__center server-mast__center">
            <p class="server-mast__kicker">Server observatory · vehicle routing</p>
            <h1 class="server-mast__host" data-role="title">Server</h1>
            <p class="server-mast__summary" data-role="dateline">connecting to the backend</p>
          </div>
          <div class="mast__flank server-mast__flank"><div class="fig-fill">${HairlineFigure.html("training", 'data-role="mast-load" autopilot="off" aria-hidden="true"')}</div></div>
        </header>
        <div class="server__board" data-role="board">${Orb.panel("Reading the machine", "working")}</div>
      </div>`));

    const mast = root.querySelector(".server-mast");
    this.mastFigures = new MastFigures(mast);
    this.board = new StatusBoard(app, root.querySelector("[data-role=board]"), {
      title: root.querySelector("[data-role=title]"),
      dateline: root.querySelector("[data-role=dateline]"),
      fleet: root.querySelector("[data-role=mast-fleet]"),
      load: root.querySelector("[data-role=mast-load]"),
    });
  }

  enter() {
    this.board.start();
  }

  leave() {
    this.board.stop();
  }
}
