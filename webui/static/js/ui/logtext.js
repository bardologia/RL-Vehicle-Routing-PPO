import { esc } from "../format.js";

export class LogText {
  static html(text, query) {
    if (!query) return { html: esc(text), count: 0 };
    const needle = query.toLowerCase();
    const lower = text.toLowerCase();
    const parts = [];
    let from = 0;
    let count = 0;
    let at = lower.indexOf(needle, from);
    while (at !== -1 && count < 5000) {
      parts.push(esc(text.slice(from, at)), `<mark>${esc(text.slice(at, at + needle.length))}</mark>`);
      from = at + needle.length;
      count += 1;
      at = lower.indexOf(needle, from);
    }
    parts.push(esc(text.slice(from)));
    return { html: parts.join(""), count };
  }

  static fill(node, text, query) {
    if (!query) {
      node.textContent = text;
      return 0;
    }
    const { html, count } = LogText.html(text, query);
    node.innerHTML = html;
    return count;
  }
}
