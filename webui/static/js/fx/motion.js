"use strict";

class FxMotion {
  static QUERY     = "(prefers-reduced-motion: reduce)";
  static listeners = new Set();
  static media     = null;

  static start() {
    if (FxMotion.media) throw new Error("FxMotion: already started");
    FxMotion.media        = window.matchMedia(FxMotion.QUERY);
    window.REDUCED_MOTION = FxMotion.media.matches;
    FxMotion.media.addEventListener("change", (event) => FxMotion._change(event.matches));
  }

  static _change(reduced) {
    window.REDUCED_MOTION = reduced;
    for (const fn of [...FxMotion.listeners]) fn(reduced);
  }

  static calm() {
    return window.REDUCED_MOTION || document.documentElement.dataset.governor === "2";
  }

  static subscribe(fn) {
    if (typeof fn !== "function") throw new Error("FxMotion: subscribe takes a function");
    FxMotion.listeners.add(fn);
    return () => FxMotion.listeners.delete(fn);
  }
}

FxMotion.start();
window.FxMotion = FxMotion;
