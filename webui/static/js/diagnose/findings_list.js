import { esc, fmtNum, isNum, plural } from "../format.js";
import { plainChip, statusChip } from "../ui/status.js";
import { Links } from "./links.js";

export class FindingsList {
  static ORDER = { FAIL: 0, WARN: 1, PASS: 2, "NO DATA": 3 };

  static NEEDS = { report: "an evaluation report", traces: "recorded traces", verdicts: "teacher verdicts", curves: "TensorBoard events" };

  static flagged(findings) {
    return findings.filter((finding) => finding.status === "FAIL" || finding.status === "WARN");
  }

  static counts(findings) {
    const counts = { FAIL: 0, WARN: 0, PASS: 0, "NO DATA": 0 };
    for (const finding of findings) counts[finding.status] += 1;
    return counts;
  }

  static summaryChips(findings) {
    const counts = FindingsList.counts(findings);
    const chips = [];
    if (counts.FAIL) chips.push(statusChip("FAIL", String(counts.FAIL)));
    if (counts.WARN) chips.push(statusChip("WARN", String(counts.WARN)));
    if (!counts.FAIL && !counts.WARN && counts.PASS) chips.push(statusChip("PASS", String(counts.PASS)));
    if (counts["NO DATA"]) chips.push(plainChip(`${counts["NO DATA"]} without data`));
    return chips.join(" ");
  }

  static numbers(finding) {
    const parts = [];
    if (finding.metric && isNum(finding.value)) parts.push(`${esc(finding.metric)} ${fmtNum(finding.value, 3)}`);
    if (isNum(finding.threshold)) parts.push(`threshold ${fmtNum(finding.threshold, 3)}`);
    if (isNum(finding.n)) parts.push(`n = ${finding.n}`);
    return parts.join(", ");
  }

  static row(finding, actions) {
    const chip = finding.status === "NO DATA" ? plainChip("NO DATA") : statusChip(finding.status);
    const links = finding.evidence.map((entry) => `<a class="btn btn--sm" href="${esc(Links.evidence(entry))}">${esc(entry.label)}</a>`);
    if (finding.needs && actions && actions[finding.needs]) links.push(actions[finding.needs]);
    const numbers = FindingsList.numbers(finding);
    return `
      <li class="finding" data-finding="${esc(finding.id)}">
        <div class="finding__head"><span class="finding__id">${esc(finding.id)}</span>${chip}<strong>${esc(finding.title)}</strong></div>
        <p>${esc(finding.text)}</p>
        <div class="finding__jumps">${numbers ? `<span class="faint finding__numbers">${numbers}</span>` : ""}${links.join("")}</div>
      </li>`;
  }

  static html(findings, { actions, expanded, columns } = {}) {
    const sorted = [...findings].sort((a, b) => FindingsList.ORDER[a.status] - FindingsList.ORDER[b.status]);
    const flagged = sorted.filter((finding) => FindingsList.ORDER[finding.status] < 2);
    const rest = sorted.filter((finding) => FindingsList.ORDER[finding.status] >= 2);
    const counts = FindingsList.counts(findings);
    const head = flagged.length ? `<ul class="findings${columns ? " findings--cols" : ""}">${flagged.map((finding) => FindingsList.row(finding, actions)).join("")}</ul>` : `<p class="muted">No rule is flagged WARN or FAIL.</p>`;
    if (!rest.length) return head;
    const label = [counts.PASS ? `${plural(counts.PASS, "rule")} at PASS` : null, counts["NO DATA"] ? `${plural(counts["NO DATA"], "rule")} without data` : null].filter((part) => part).join(", ");
    return `${head}<details class="disclosure"${expanded || !flagged.length ? " open" : ""}><summary>${label}</summary><ul class="findings${columns ? " findings--cols" : ""}">${rest.map((finding) => FindingsList.row(finding, actions)).join("")}</ul></details>`;
  }
}
