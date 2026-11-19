---
name: Routing console
description: Dark research console for a routing policy, built to the standard of the Mercado market console.
colors:
  bg: "#0b0f13"
  well: "#090d11"
  panel-solid: "#10161c"
  line: "#4f5d6a"
  line-strong: "#8b9ba8"
  line-card: "#3d4a56"
  line-soft: "#2c353e"
  ink: "#f5f8fa"
  ink-dim: "#d6dfe4"
  ink-faint: "#b4c1c9"
  ink-hint: "#8a98a3"
  accent: "#a3c0ff"
  accent-deep: "#c9d9ff"
  on-accent: "#0b0f13"
  live: "#5eead4"
  pass: "#4ade80"
  warn: "#fcd34d"
  fail: "#f87171"
  gain: "#7c8cff"
  loss: "#fb923c"
  lane-model: "#5eead4"
  lane-teacher: "#f0abfc"
  lane-insertion: "#90a9c8"
  lane-nothing: "#8a98a3"
  veh-1: "#38bdf8"
  veh-2: "#d946ef"
  veh-3: "#bef264"
  veh-4: "#d0a058"
typography:
  page-title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "clamp(1.8rem, 3.2vw, 2.7rem)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  section-title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "clamp(20px, 1.7vw, 27px)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.015em"
  card-title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.01em"
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "normal"
  card-prose:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "0.08em"
  kpi-label:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1.25
    letterSpacing: "0.08em"
  kpi-value:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "clamp(26px, 2.3vw, 42px)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.02em"
  kpi-value-small:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "clamp(20px, 1.5vw, 26px)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.02em"
  data:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "normal"
  note:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "11.5px"
    fontWeight: 600
    lineHeight: 1.45
    letterSpacing: "normal"
  tick:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "11px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "normal"
rounded:
  r: "2px"
  r-in: "1px"
spacing:
  pad: "32px"
  gap: "16px"
  column-max: "1640px"
  bar-h: "56px"
components:
  button:
    backgroundColor: "rgba(16, 22, 28, 0.74)"
    textColor: "{colors.ink}"
    rounded: "{rounded.r}"
    padding: "9px 18px"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.r}"
    padding: "9px 18px"
  button-small:
    backgroundColor: "rgba(16, 22, 28, 0.74)"
    textColor: "{colors.ink}"
    rounded: "{rounded.r}"
    padding: "5px 11px"
  icon-button:
    backgroundColor: "rgba(16, 22, 28, 0.74)"
    textColor: "{colors.ink-dim}"
    rounded: "{rounded.r}"
    size: "34px"
  input:
    backgroundColor: "rgba(16, 22, 28, 0.74)"
    textColor: "{colors.ink}"
    rounded: "{rounded.r}"
    height: "34px"
    padding: "0 10px"
  segmented-item:
    textColor: "{colors.ink-faint}"
    typography: "{typography.data}"
    padding: "4px 10px"
  chip:
    textColor: "{colors.ink-dim}"
    typography: "{typography.data}"
    rounded: "{rounded.r}"
    padding: "5px 9px"
  card:
    backgroundColor: "rgba(16, 22, 28, 0.70)"
    rounded: "{rounded.r}"
    padding: "16px 18px 18px"
---

# Routing console design system

## Overview

The console is a dark instrument for one researcher. Its standard is the Mercado market console (`Mercado/webui_market`); rules are ported from it and deviations need a stated reason. A page is one centred column, at most 1640px wide with 32px side padding, and the top bar shares the same inset. Each page opens with a mast in a fixed order: title, at most one lead sentence, controls, one monospaced dateline, three to six unboxed KPIs, with one or two hairline figures on the right taking 20 to 45 percent of the mast width. Below the mast, flat cards sit on a 12-column grid with a 16px gap. Episode (record view) and Lab are the two full-bleed workspaces because their subject is a map; they keep the same components and type.

Density rule: a 1920x1080 viewport holds about 100 to 230 words, of which fewer than 55 are in sentences, and fewer than 40 bordered boxes. Explanations are delivered through an info popover per card, tooltips, row expansion and the key sheet, never as a sentence beside a title.

How the density rule is measured, re-baselined against the reference consoles. The share of characters set at 13px or above stays inside the range Mercado's own views measure (16 to 64 percent); on data views monospace carries at least 40 percent of the characters (Mercado measures 44 to 79). A table row rule is drawn once per row, on the row, and counts once; map data glyphs (stop, depot and label markers) are content, not boxes. The Launch catalogue and the Launch form follow the DLR consoles, where each entry point's purpose and each field's help are the content, so their prose and monospace shares are those of the DLR pages rather than of the data views.

Layout is structural, not fluid: the two reference viewports are 1920x1080 and 1366x768; below 1100px wide the mast stacks and figures move under the text; nothing is hidden by a width or height breakpoint.

## Colors

Restrained strategy: cool tinted neutrals and one accent.

- Ground `bg` #0b0f13 with a 28px grid at 4.5 percent alpha, a backdrop orb at opacity 0.14 and grain at 0.03. Wells `well` #090d11 for map side panels and logs.
- Lines: cards `line-card` 2px; controls `line` 2px; hover and figure edges `line-strong`; row rules and chart grid `line-soft` 1px.
- Ink ramp, four steps: `ink` for values, titles and body; `ink-dim` for lead sentences and text cells; `ink-faint` for KPI labels, notes and card prose; `ink-hint` for control labels, axis ticks and right-aligned card notes. A view uses at most eight text colours including series colours.
- `accent` #a3c0ff marks the current selection, the primary button, links and the focus ring. It is never decoration.
- State colours `pass`, `warn`, `fail` appear as a 7px dot beside a word, and as a 2px chip border only for WARN and FAIL findings. `live` marks a running job.
- Series colours: lanes (model, teacher, insertion only, do nothing) and vehicles 1 to 4, each paired with a line style or a shape. `gain` and `loss` colour signed reward terms.
- Hairline figures are monochrome: `ink` for the one lit silhouette, `line-strong` for silhouettes, `line` for strokes, `line-soft` for creases. No series or state colour enters a figure.

## Typography

Two families, each one variable file, the same builds the Mercado console renders: Archivo (latin, weight axis 600 to 900, no width axis) and JetBrains Mono (latin, weight axis 400 to 800, unhinted). A third face name, `JetBrains Mono Log`, points at a full-coverage static cut and is used only for job logs, which contain box-drawing characters.

Family rule: monospace carries everything that is read as a value (numbers, ids, paths, commands, times, key hints, axis ticks, direct labels), segmented-control items, chips, and one-line fact notes without a verb. Archivo carries titles, labels, buttons, navigation, names in tables and every sentence. A string with a verb is never set in monospace.

Scale (all sizes in px; nothing below 11): page title clamp(28.8, 3.2vw, 43.2)/700; section title clamp(20, 1.7vw, 27)/700; card title 18/700; body and lead 15/600/1.55; card prose 13/600/1.5; nav and buttons 14/700; small buttons 12.5/700; KPI label 12/700 uppercase 0.08em; control and table-head label 11/700 uppercase (0.08em and 0.06em); KPI value mono clamp(26, 2.3vw, 42)/700/1.05 in masts and clamp(20, 1.5vw, 26) in side panels and strips, unit as `small` at half size; data cells mono 12/600/1.55; inputs mono 13/600; notes mono 11.5/600/1.45; ticks mono 11/600.

Fixed rules: `font-stretch` is always 100 percent; no `word-spacing`; negative tracking only on titles (to -0.02em) and KPI values (-0.02em); uppercase only on labels of four words or fewer; no `font:` shorthand that resets line height; lead and prose capped at 68ch and 80ch; `text-wrap: balance` on titles; every changing number reserves its width in `ch` and every changing line has a fixed height.

## Elevation

Flat. There are no shadows. Depth is carried by three surfaces: the page ground, flat cards (2px `line-card` border, 2px radius, faint top gradient) and glass chrome (blur 14px, saturate 1.25) used only for the top bar, the episode transport bar, popovers, tooltips and controls that float over a map. Cards never nest; inside a card the only dividers are 1px `line-soft` row rules and the 2px KPI top rule. Z order, low to high: backdrop, content, sticky heads, top bar, popover, key sheet, toast, tooltip.

Motion: easing `cubic-bezier(.16, 1, .3, 1)` everywhere. Press 90ms scale 0.98; tooltip 120ms; hover colour and border 200ms; bar fills 220ms `transform: scaleX`; navigation underline and fades 240ms; drawer 320ms; page entry 450ms opacity and 8px rise, only under `prefers-reduced-motion: no-preference`. Interface chrome does not move on its own. Ambient motion comes only from hairline figures and the backdrop orb.

## Components

- Mast (`.view-head`, `.mast`): title, lead, controls, dateline, KPIs on the left; figures right. Actions sit in the controls row, never on the title row.
- KPI (`.kpi`): 2px top rule, uppercase label, mono value with half-size unit, one or two fixed-height mono comparison lines coloured by series. No box, icon or chip.
- Card (`.panel`): title left; one mono note of at most eight words on the right stating unit and scope; an optional info button that opens a popover with definitions. No help sentence beside the title.
- Reads row (`.reads`): above a chart, a 14x2px colour key, an uppercase series name and a mono 16/700 value per series. Legends exist only for categorical bands, in one mono row.
- Controls, a vocabulary of five: filter chip (toggle), segmented control (exclusive choice, preceded by one uppercase label), button (one primary per view at most), icon button with a tooltip, text field. Native selects are kept for run and file choice and share the field metrics. Booleans are two-item segmented controls.
- State (`.state`): 7px dot and a lower-case mono word. A bordered chip is used only for WARN and FAIL findings.
- Table (`.dtable`): uppercase 11px heads on one line, 29px rows, names in Archivo 12, numbers in mono 12 right-aligned, 1px row rules, a column-set switch or the Columns menu for secondary columns, the row itself as the link.
- Chart: 56/14/22/24 padding, four nice ticks with 1px grid lines, mono 11 ticks in `ink-hint`, no axis lines or axis titles, the unit as a short mono caption at the top left, 1.6px lines, hover rule and dots with a tooltip that lists swatch, label and value.
- Disclosure: info popover, row expansion, collapsible section with its summary value in the head, a "More" popover for secondary KPIs, tooltips for per-item numbers.
- Hairline figure (`<hairline-figure>`): 400x320 view box, no caption, frame or background, geometry filling 0.30 to 0.55 of the box, exactly one lit silhouette, state assigned from the page's payload, autopilot when idle. Sizes: hero clamp(360px, 34vw, 560px), flank clamp(200px, 16vw, 320px), in-card and band figures sized by container queries. Every page shows at least one at both reference viewports.
- Loading and empty: a 64px thinking orb over a 72 percent veil for a board, a 20px orb inline; an empty state is the section's own figure in its zero state above one sentence.

## Do's and Don'ts

Do

- Port the Mercado rule and value for a role before writing a new one.
- Lead each page with the object that answers its question, at mast KPI size or as the largest board.
- Set values in monospace with units, reserve their width, and align them for comparison.
- Move secondary detail behind exactly one interaction and keep it reachable.
- Bind every figure to the payload the page already holds.

Don't

- Put a help sentence beside a card title, a footnote under a table, or an axis title under a chart.
- Repeat a number that a KPI on the same screen already states, or render the same list on more than one tab.
- Use status chips, PASS badges, rule ids or icons beside labels in content.
- Condense Archivo, set sentences in monospace, or go below 11px.
- Use side-stripe borders, shadows, nested cards, gradient text or an open legend over a visual.
- Hide a figure at any viewport size, add a caption to it, or colour it.
- Add a framework, a bundler, an npm dependency or a CDN link; write comments in code; use emojis or decorative symbols.
