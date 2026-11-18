import { Router } from "../router.js";

export class Links {
  static diagnose(run, seed, panel) {
    return Router.build(["diagnose", run, seed], panel ? { panel } : {});
  }

  static episode(run, seed, index, hasTrace, position) {
    if (hasTrace) return Router.build(["episode", "diag", run, seed, index, "model", position || "r0"]);
    return Router.build(["episode", "seed", run, seed, index]);
  }

  static decision(run, seed, decision) {
    return Router.build(["episode", "diag", run, seed, decision.episode, "model", `r${decision.round}e${decision.edit}`]);
  }

  static runs(run, section) {
    return Router.build(["runs", run, section || "summary"]);
  }

  static evaluate(run) {
    return Router.build(["launch", "evaluate"], { "io.run_name": run });
  }

  static recordTraces(run, seed, episodes) {
    const query = { "io.run_name": run };
    if (seed !== null && seed !== undefined) query["diagnostics.seed"] = seed;
    if (episodes) query["diagnostics.episodes"] = episodes;
    return Router.build(["launch", "diagnose"], query);
  }

  static slug(label) {
    return label.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  }

  static evidence(entry) {
    if (!entry.hash.startsWith("#/diagnose/") || entry.hash.includes("?")) return entry.hash;
    return `${entry.hash}?panel=${Links.slug(entry.label)}`;
  }
}
