import { esc } from "../format.js";

export class MarkdownTable {
  static cells(line) {
    return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  }

  static isRule(line) {
    return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
  }

  static inline(text) {
    return esc(text).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  }

  static table(header, rows) {
    const numeric = header.map((_, column) => rows.length > 0 && rows.every((row) => /^-?[\d,.]+%?$/.test(row[column] || "")));
    const head = header.map((cell, column) => `<th class="${numeric[column] ? "num" : ""}">${MarkdownTable.inline(cell)}</th>`).join("");
    const body = rows.map((row) => `<tr>${header.map((_, column) => `<td class="${numeric[column] ? "num" : ""}">${MarkdownTable.inline(row[column] || "")}</td>`).join("")}</tr>`).join("");
    return `<div class="dtable dtable--dense dtable--static"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  static render(text) {
    const lines = text.split("\n");
    const out = [];
    let paragraph = [];
    const flush = () => {
      if (paragraph.length) out.push(`<p>${paragraph.map(MarkdownTable.inline).join(" ")}</p>`);
      paragraph = [];
    };

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      if (line.includes("|") && index + 1 < lines.length && MarkdownTable.isRule(lines[index + 1])) {
        flush();
        const header = MarkdownTable.cells(line);
        const rows = [];
        index += 2;
        while (index < lines.length && lines[index].includes("|")) {
          rows.push(MarkdownTable.cells(lines[index]));
          index += 1;
        }
        index -= 1;
        out.push(MarkdownTable.table(header, rows));
      } else if (/^#{1,6}\s/.test(line)) {
        flush();
        out.push(`<h4>${MarkdownTable.inline(line.replace(/^#{1,6}\s+/, ""))}</h4>`);
      } else if (!line.trim()) {
        flush();
      } else {
        paragraph.push(line.trim());
      }
    }
    flush();
    return `<div class="markdown">${out.join("")}</div>`;
  }
}
