import { el } from "../format.js";
import { Orb } from "../fx/orb.js";
import { StatusBoard } from "../server/status_board.js";
import { PageHead } from "../ui/page_head.js";

export class ServerView {
  constructor(app, root) {
    this.app = app;
    this.root = root;

    root.appendChild(el(`
      <div class="view__pad server">
        ${PageHead.html({
          title: "Server",
          lead: "Live readings of the machine the console runs on, the jobs it started, and the watchdog and kill switches that guard them.",
          dateline: "connecting to the backend",
          form: "duo",
          className: "server-head",
          figure: `${HairlineFigure.html("fleet", 'aria-hidden="true"')}${HairlineFigure.html("training", 'aria-hidden="true"')}`,
        })}
        <div class="server__board" data-role="board">${Orb.panel("Reading the machine", "working")}</div>
      </div>`));

    this.board = new StatusBoard(app, root.querySelector("[data-role=board]"), {
      title: root.querySelector("[data-role=title]"),
      dateline: root.querySelector("[data-role=dateline]"),
    });
  }

  enter() {
    this.board.start();
  }

  leave() {
    this.board.stop();
  }
}
