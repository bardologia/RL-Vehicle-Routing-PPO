# Product

## Register

product

## Users

One researcher, the author of the project, working alone at a desk on a 1920x1080 monitor or a 1366x768 laptop panel, in a dark room or a dim office. He trains a PPO policy that edits vehicle routing plans under disruptions and compares it with a regret-insertion teacher and two baselines. He opens the console many times a day, between training runs, to answer one of seven questions:

- Overview: is the selected checkpoint usable, and where should I look next?
- Diagnose: why is the model behind or ahead of the teacher, in which episodes and through which behaviour?
- Episode: what did each agent do, decision by decision, on one episode, and was each decision right?
- Lab: what happens on a scenario I build by hand?
- Runs: is this training run healthy and what did it write?
- Launch: configure a script and start it.
- Console: is the job running, and what does its log say?

He knows the domain and the code. He does not need the interface to explain reinforcement learning; he needs it to put the one number, map or chart that answers the question in front of him without making him read.

## Product Purpose

A local web console over the training, evaluation and trace files of the routing project. It exists so that judging a policy takes a glance and investigating it takes a few clicks, instead of reading logs and notebooks. Success: on every tab the answer to that tab's question is the largest object on screen, and everything else waits one interaction away. No number or capability is ever removed to get there; detail moves behind disclosure.

## Brand Personality

Quiet, exact, instrument-like. The console belongs to a family of the author's research consoles and takes its standard from the Mercado market console: a dark ground, heavy grotesque titles, monospaced numbers, two-pixel hairlines, and isometric hairline figures that are always moving and are driven by the page's data. The voice is that of a lab instrument: short labels, units on every number, sentences only where a finding has to be stated. Academic and professional. No emojis, no decorative symbols, no exclamation.

## Anti-references

- The console's own state before this redesign: status chips in every row, a help sentence beside every panel title, the same list of findings rendered on three tabs, twelve boxed KPI tiles in a strip, an open legend over the map, condensed uppercase labels, sentences set in monospace.
- Generic admin dashboards: grids of identical bordered cards, coloured PASS and FAIL badges as the main visual element, icons beside every label.
- Monitoring products that show everything at once and leave ranking to the reader.
- `Mercado/webui_agent` is not a reference. Only `Mercado/webui_market` is.

## Design Principles

1. One answer per screen. Each tab has one question; the object that answers it leads, at a size nothing else on the page matches.
2. Disclose, never delete. Secondary numbers, definitions, column sets and help move behind one interaction (popover, row expand, column switch, tooltip). Nothing reachable before becomes unreachable.
3. Numbers are read, sentences are rare. Values are monospaced, sized and aligned to be compared. Prose appears only to state a finding or an expectation, never to restate a heading or explain a control.
4. The standard is ported, not reinterpreted. Where the Mercado market console has a rule for a role, this console uses that rule and that value.
5. Motion belongs to the figures. Interface chrome is still; every page carries a hairline figure about routing that moves all the time and shows the page's own data.

## Accessibility & Inclusion

- Dark only. Text contrast at least 4.5:1 against its surface; the dimmest text token (`--ink-hint` #8a98a3 on #0b0f13) passes at 6.4:1 and is used only for labels and ticks, never for values.
- Status is never carried by colour alone: every state has a word, and operators and lanes have a shape or a line style as well as a colour.
- Every interactive element is reachable by keyboard and has a visible focus ring; the existing keyboard shortcuts are kept unchanged.
- `prefers-reduced-motion: reduce` stops page-entry animation, autopilot and figure loops; figures draw their settled data state and still update when data changes.
- Minimum text size 11px.
