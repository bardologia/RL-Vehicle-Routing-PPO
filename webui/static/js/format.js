const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export const NONE = "-";

export function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

export function icon(name, extra) {
  return `<svg class="icon${extra ? " " + extra : ""}" aria-hidden="true"><use href="#i-${name}"></use></svg>`;
}

export function el(html) {
  const holder = document.createElement("template");
  holder.innerHTML = html.trim();
  return holder.content.firstElementChild;
}

export function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function fmtNum(value, digits = 2) {
  if (!isNum(value)) return NONE;
  if (Number.isInteger(value) && digits !== 0 && Math.abs(value) < 1e7) return String(value);
  return value.toFixed(digits);
}

export function fmtFixed(value, digits = 2) {
  return isNum(value) ? value.toFixed(digits) : NONE;
}

export function fmtSigned(value, digits = 2) {
  if (!isNum(value)) return NONE;
  const text = Math.abs(value).toFixed(digits);
  if (Number(text) === 0) return text;
  return (value > 0 ? "+" : "-") + text;
}

export function fmtInt(value) {
  return isNum(value) ? String(Math.round(value)) : NONE;
}

export function fmtSignedInt(value) {
  if (!isNum(value)) return NONE;
  const rounded = Math.round(value);
  return rounded > 0 ? `+${rounded}` : String(rounded);
}

export function fmtPct(value, digits = 0) {
  return isNum(value) ? `${(value * 100).toFixed(digits)}%` : NONE;
}

export function fmtProb(value) {
  if (!isNum(value)) return NONE;
  if (value > 0 && value < 0.001) return "<0.001";
  return value.toFixed(3);
}

export function fmtDur(seconds) {
  if (!isNum(seconds)) return NONE;
  const sign = seconds < 0 ? "-" : "";
  const total = Math.round(Math.abs(seconds) / 60);
  return `${sign}${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function fmtSignedDur(seconds) {
  if (!isNum(seconds)) return NONE;
  const text = fmtDur(Math.abs(seconds));
  if (text === "0:00") return text;
  return (seconds > 0 ? "+" : "-") + text;
}

export function fmtMin(seconds, digits = 0) {
  return isNum(seconds) ? (seconds / 60).toFixed(digits) : NONE;
}

export function fmtClock(seconds) {
  if (!isNum(seconds)) return NONE;
  const total = Math.round(seconds / 60);
  return `${String(Math.floor(total / 60) % 24).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

export function fmtKm(metres, digits = 1) {
  return isNum(metres) ? (metres / 1000).toFixed(digits) : NONE;
}

export function fmtAge(epoch) {
  if (!isNum(epoch)) return NONE;
  const delta = Date.now() / 1000 - epoch;
  if (delta < 90) return "just now";
  if (delta < 3600) return `${Math.round(delta / 60)} min ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)} h ago`;
  return `${Math.round(delta / 86400)} d ago`;
}

export function fmtBytes(size) {
  if (!isNum(size)) return NONE;
  if (size > 1e9) return `${(size / 1e9).toFixed(2)} GB`;
  if (size > 1e6) return `${(size / 1e6).toFixed(1)} MB`;
  if (size > 1e3) return `${(size / 1e3).toFixed(1)} KB`;
  return `${size} B`;
}

export function signClass(value) {
  if (!isNum(value) || Math.abs(value) < 5e-7) return "";
  return value > 0 ? "gain" : "loss";
}

export function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

export function listIds(ids) {
  return ids.join(", ");
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class Store {
  static read(key, fallback) {
    const raw = localStorage.getItem(`routing-console.${key}`);
    return raw === null ? fallback : JSON.parse(raw);
  }

  static write(key, value) {
    localStorage.setItem(`routing-console.${key}`, JSON.stringify(value));
  }

  static remove(key) {
    localStorage.removeItem(`routing-console.${key}`);
  }
}
