export class PythonLiteral {
  static OPENERS = "[{(";
  static CLOSERS = "]})";

  static split(text) {
    const raw   = String(text);
    const parts = [];

    let depth = 0;
    let quote = "";
    let start = 0;

    for (let pos = 0; pos < raw.length; pos += 1) {
      const ch = raw[pos];

      if (quote) {
        if (ch === "\\") pos += 1;
        else if (ch === quote) quote = "";
        continue;
      }

      if (ch === "'" || ch === '"') quote = ch;
      else if (PythonLiteral.OPENERS.includes(ch)) depth += 1;
      else if (PythonLiteral.CLOSERS.includes(ch)) depth = Math.max(0, depth - 1);
      else if (ch === "," && depth === 0) {
        parts.push(raw.slice(start, pos));
        start = pos + 1;
      }
    }

    parts.push(raw.slice(start));

    return parts.map((part) => part.trim()).filter(Boolean);
  }
}
