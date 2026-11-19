import { esc, fmtNum, icon, isNum, plural } from "../format.js";
import { Popover } from "../ui/popover.js";
import { statusChip } from "../ui/status.js";
import { Links } from "./links.js";

export class FindingsList {
  static ORDER = { FAIL: 0, WARN: 1, PASS: 2, "NO DATA": 3 };

  static NEEDS = { report: "an evaluation report", traces: "recorded traces", verdicts: "teacher verdicts", curves: "TensorBoard events" };

  static FRAMES = 6;

  static INLINE = 3;

  static bound = false;

  static bind() {
    if (FindingsList.bound) return;
    FindingsList.bound = true;
    document.addEventListener("click", (event) => {
      if (!(event.target instanceof Element)) return;
      const toggle = event.target.closest("[data-finding-toggle]");
      if (toggle) {
        const row = toggle.closest(".finding");
        const open = row.classList.toggle("is-open");
        toggle.setAttribute("aria-expanded", String(open));
        Popover.reflow();
        return;
      }
      const more = event.target.closest("[data-finding-frames]");
      if (more) {
        const holder = more.closest(".finding__jumps");
        for (const node of holder.querySelectorAll("[hidden]")) node.hidden = false;
        more.remove();
        Popover.reflow();
        return;
      }
      const all = event.target.closest("[data-findings-all]");
      if (all) {
        const source = all.parentElement.querySelector("template[data-findings-body]");
        Popover.toggle(all, source.innerHTML, { label: "Findings", className: "popover--wide" });
      }
    });
  }

  static sorted(findings) {
    return [...findings].sort((a, b) => FindingsList.ORDER[a.status] - FindingsList.ORDER[b.status]);
  }

  static flagged(findings) {
    return FindingsList.sorted(findings).filter((finding) => finding.status === "FAIL" || finding.status === "WARN");
  }

  static counts(findings) {
    const counts = { FAIL: 0, WARN: 0, PASS: 0, "NO DATA": 0 };
    for (const finding of findings) counts[finding.status] += 1;
    return counts;
  }

  static numbers(finding) {
    const parts = [];
    if (finding.metric && isNum(finding.value)) parts.push(`${esc(finding.metric)} ${fmtNum(finding.value, 3)}`);
    if (isNum(finding.threshold)) parts.push(`threshold ${fmtNum(finding.threshold, 3)}`);
    if (isNum(finding.n)) parts.push(`n = ${finding.n}`);
    return parts.join(", ");
  }

  static target(finding) {
    const evidence = finding.evidence || [];
    return evidence.length ? Links.evidence(evidence[0]) : null;
  }

  static frames(finding) {
    const seqs = finding.seqs || [];
    if (!seqs.length) return "";
    const buttons = seqs.map((seq, index) => `<button type="button" class="btn btn--sm" data-jump="${seq}"${index >= FindingsList.FRAMES ? " hidden" : ""}>${seq + 1}</button>`).join("");
    const rest = seqs.length - FindingsList.FRAMES;
    return `<span class="finding__numbers">frames</span>${buttons}${rest > 0 ? `<button type="button" class="btn btn--sm btn--quiet" data-finding-frames>${rest} more</button>` : ""}`;
  }

  static row(finding, actions) {
    const fail = finding.status === "FAIL";
    const href = FindingsList.target(finding);
    const title = href ? `<a class="finding__title" href="${esc(href)}">${esc(finding.title)}</a>` : `<span class="finding__title">${esc(finding.title)}</span>`;
    const links = (finding.evidence || []).slice(1).map((entry) => `<a class="btn btn--sm" href="${esc(Links.evidence(entry))}">${esc(entry.label)}</a>`);
    if (finding.needs && actions && actions[finding.needs]) links.push(actions[finding.needs]);
    const numbers = FindingsList.numbers(finding);
    const frames = FindingsList.frames(finding);
    const text = `<p class="finding__text">${esc(finding.text)}</p>`;
    return `
      <li class="finding finding--${FindingsList.tone(finding)}" data-finding="${esc(finding.id)}">
        <div class="finding__head">${statusChip(finding.status)}${title}<button type="button" class="tbtn tbtn--sm finding__more" data-finding-toggle aria-expanded="false" aria-label="Details of ${esc(finding.title)}">${icon("chevron-right")}</button></div>
        ${fail ? text : ""}
        <div class="finding__body">
          ${fail ? "" : text}
          <div class="finding__jumps"><span class="finding__id">${esc(finding.id)}</span>${numbers ? `<span class="finding__numbers">${numbers}</span>` : ""}${links.join("")}</div>
          ${frames ? `<div class="finding__jumps">${frames}</div>` : ""}
        </div>
      </li>`;
  }

  static tone(finding) {
    return { FAIL: "fail", WARN: "warn", PASS: "pass", "NO DATA": "none" }[finding.status];
  }

  static restLabel(counts) {
    return [counts.PASS ? `${counts.PASS} pass` : null, counts["NO DATA"] ? `${counts["NO DATA"]} without data` : null].filter((part) => part).join(", ");
  }

  static list(findings, { actions, columns, expanded, emptyText } = {}) {
    FindingsList.bind();
    const sorted = FindingsList.sorted(findings);
    const flagged = sorted.filter((finding) => FindingsList.ORDER[finding.status] < 2);
    const rest = sorted.filter((finding) => FindingsList.ORDER[finding.status] >= 2);
    const cols = columns ? " findings--cols" : "";
    const head = flagged.length ? `<ul class="findings${cols}">${flagged.map((finding) => FindingsList.row(finding, actions)).join("")}</ul>` : `<p class="muted">${esc(emptyText || "No rule is flagged.")}</p>`;
    if (!rest.length) return head;
    return `${head}<details class="disclosure findings__rest"${expanded ? " open" : ""}><summary>${FindingsList.restLabel(FindingsList.counts(findings))}</summary><ul class="findings${cols}">${rest.map((finding) => FindingsList.row(finding, actions)).join("")}</ul></details>`;
  }

  static countLabel(counts, total) {
    const flagged = [counts.FAIL ? `${counts.FAIL} fail` : null, counts.WARN ? `${counts.WARN} warn` : null].filter((part) => part).join(", ");
    return flagged || `${plural(total, "rule")}, none flagged`;
  }

  static strip(findings, { actions, link, tail, inline = true, scope = "" } = {}) {
    FindingsList.bind();
    const counts = FindingsList.counts(findings);
    const tone = counts.FAIL ? "fail" : counts.WARN ? "warn" : "pass";
    const fails = inline ? FindingsList.flagged(findings).filter((finding) => finding.status === "FAIL").slice(0, FindingsList.INLINE) : [];
    const rules = fails.map((finding) => {
      const href = FindingsList.target(finding) || link;
      const attrs = `class="findstrip__rule" data-hint="${esc(finding.text)}" data-hint-title="${esc(`fail: ${finding.title}`)}" data-finding="${esc(finding.id)}"`;
      return href ? `<a ${attrs} href="${esc(href)}"><i></i>${esc(finding.title)}</a>` : `<span ${attrs}><i></i>${esc(finding.title)}</span>`;
    }).join("");
    const rest = FindingsList.restLabel(counts);
    const head = `<header class="popover__head"><h3>Findings</h3></header>${scope ? `<p class="popover__lede">${esc(scope)}</p>` : ""}`;
    const button = `<button type="button" class="btn btn--sm btn--quiet findstrip__count" data-findings-all aria-haspopup="dialog" aria-expanded="false" aria-label="Findings: ${esc(FindingsList.countLabel(counts, findings.length))}" data-hint-title="Findings" data-hint="${esc([FindingsList.countLabel(counts, findings.length), rest].filter((part) => part).join(", "))}"><span class="state state--${tone}"><i></i></span>${esc(FindingsList.countLabel(counts, findings.length))}</button>`;
    return `<div class="findstrip"><span class="findstrip__more">${button}<template data-findings-body>${head}${FindingsList.list(findings, { actions })}</template></span>${tail || ""}${rules}</div>`;
  }

  static html(findings, options = {}) {
    return FindingsList.list(findings, options);
  }
}
