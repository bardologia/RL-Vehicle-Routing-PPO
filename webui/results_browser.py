"""Read side of the runs directory: run listings, evaluation reports with paired statistics, training curves and diagnostics indexes."""

import json
import math
import re
from pathlib import Path

from web_errors import ApiError


class RunBrowser:
    """Serves what training, evaluation and diagnostics wrote into each run directory.

    Attributes:
        paths   : ProjectPaths of the console.
        layout  : RunLayout naming the files of a run.
        catalog : CheckpointCatalog used for the loadability verdict.
    """

    ANSI           = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")
    LOG_TAIL_BYTES = 262144
    MAX_POINTS     = 1000
    TIE            = 1e-6
    BASELINES      = ("teacher", "insertion_only", "do_nothing")
    AGENT_KEYS     = ("model", "model_sampled", "teacher", "insertion_only", "do_nothing")

    HEALTH = {
        "ppo" : [
            {"key": "episode_reward",     "title": "Episode reward",            "unit": "reward points", "tags": [("episode return", "episode/total_reward")]},
            {"key": "operator_frequency", "title": "Operator frequency",        "unit": "share",         "tags": [("INSERT", "episode/operator_frequency/op0"), ("REMOVE", "episode/operator_frequency/op1"), ("DO_NOTHING", "episode/operator_frequency/op2")]},
            {"key": "entropy",            "title": "Entropy per head",          "unit": "normalised",    "tags": [("operator", "sample/entropy/operator_entropy"), ("vehicle", "sample/entropy/vehicle_entropy_exp"), ("job", "sample/entropy/job_entropy_exp")]},
            {"key": "mean_kl",            "title": "Mean KL per batch",         "unit": "nats",          "tags": [("mean KL", "batch/mean_kl")]},
            {"key": "clip_fraction",      "title": "Clip fraction per head",    "unit": "share",         "tags": [("operator", "sample/clip_fraction/operator"), ("vehicle", "sample/clip_fraction/vehicle"), ("job", "sample/clip_fraction/job")]},
            {"key": "explained_variance", "title": "Explained variance",        "unit": "fraction",      "tags": [("explained variance", "batch/baseline_stats/explained_variance")]},
            {"key": "value_loss",         "title": "Value loss",                "unit": "squared points", "tags": [("value loss", "sample/value_loss/value_loss")]},
            {"key": "grad_norm",          "title": "Gradient norm before clip", "unit": "L2 norm",       "tags": [("gradient norm", "batch/grad_norm_before_clip")]},
            {"key": "anchor_coefficient", "title": "Anchor coefficient",        "unit": "weight",        "tags": [("anchor coefficient", "batch/anchor_coefficient")]},
            {"key": "learning_rate",      "title": "Learning rate",             "unit": "step size",     "tags": [("operator actor", "batch/learning_rate/operator_actor"), ("vehicle actor", "batch/learning_rate/vehicle_actor"), ("job actor", "batch/learning_rate/job_actor"), ("critic", "batch/learning_rate/critic"), ("graph embedding", "batch/learning_rate/graph_embedding"), ("operator embedding", "batch/learning_rate/operator_embedding")]},
        ],
        "pretrain" : [
            {"key": "rollout_reward", "title": "Teacher rollout reward", "unit": "reward points", "tags": [("rollout reward", "pretrain/rollout_reward")]},
            {"key": "loss",           "title": "Loss per head",          "unit": "nats",          "tags": [("operator", "pretrain/loss/operator"), ("vehicle", "pretrain/loss/vehicle"), ("job", "pretrain/loss/job"), ("value", "pretrain/loss/value")]},
            {"key": "accuracy",       "title": "Accuracy per head",      "unit": "share",         "tags": [("operator", "pretrain/accuracy/operator"), ("vehicle", "pretrain/accuracy/vehicle"), ("job", "pretrain/accuracy/job")]},
        ],
        "checkpoint" : [],
    }

    def __init__(self, paths, layout, catalog):
        self.paths         = paths
        self.layout        = layout
        self.catalog       = catalog
        self._curves_cache = {}

    def _require_run(self, name):
        run_dir = self.layout.run_dir(name)
        if run_dir is None:
            raise ApiError(404, f"unknown run '{name}'")

        return run_dir

    def read_report(self, name, seed):
        """Return the raw evaluation report of a run and seed as written by the evaluation pipeline.

        Raises:
            ApiError : The run or the report does not exist (404).
        """
        path = self.layout.report_path(self._require_run(name), seed)
        if not path.exists():
            raise ApiError(404, f"run '{name}' has no evaluation report for seed {seed}")

        return json.loads(path.read_text())

    def _report_brief(self, run_dir, seed):
        report = json.loads(self.layout.report_path(run_dir, seed).read_text())
        meta   = report["meta"]

        return {
            "seed"     : meta["seed"],
            "episodes" : meta["episodes"],
            "greedy"   : meta["greedy"],
            "agents"   : {name: {"mean_reward": metrics["mean_reward"], "sem_reward": metrics["sem_reward"]} for name, metrics in report["agents"].items()},
        }

    def list_runs(self):
        """Return every run, newest first.

        Returns:
            List of {"name", "kind", "mtime", "has_checkpoint", "loadable", "reason", "has_events", "analysis_count", "reports": [{"seed", "episodes", "greedy", "agents"}], "diagnostics": [seed]}.
        """
        runs = []
        for run_dir in self.layout.run_dirs():
            status = self.catalog.status(run_dir)

            runs.append({
                "name"           : run_dir.name,
                "kind"           : self.layout.kind(run_dir),
                "mtime"          : max((child.stat().st_mtime for child in run_dir.iterdir()), default=run_dir.stat().st_mtime),
                "has_checkpoint" : status["has_checkpoint"],
                "loadable"       : status["loadable"],
                "reason"         : status["reason"],
                "has_events"     : bool(list(run_dir.glob("events.out.tfevents.*"))),
                "analysis_count" : len(list((run_dir / "analysis").glob("*.png"))) if (run_dir / "analysis").is_dir() else 0,
                "reports"        : [self._report_brief(run_dir, seed) for seed in self.layout.report_seeds(run_dir)],
                "diagnostics"    : self.layout.diagnostics_seeds(run_dir),
            })

        runs.sort(key=lambda run: run["mtime"], reverse=True)
        return runs

    def _read_log(self, path):
        raw = path.read_bytes()
        if len(raw) > self.LOG_TAIL_BYTES:
            raw = raw[-self.LOG_TAIL_BYTES:]

        return self.ANSI.sub("", raw.decode("utf-8", "replace"))

    def run_detail(self, name):
        """Return the files and status of one run.

        Returns:
            {"name", "kind", "has_checkpoint", "loadable", "reason", "has_events", "reports": [seed], "diagnostics": [seed], "markdown": {file: text}, "logs": [{"name", "text"}], "analysis": [{"name", "url"}], "files": [{"rel", "size"}]}.

        Raises:
            ApiError : The run does not exist (404).
        """
        run_dir = self._require_run(name)
        status  = self.catalog.status(run_dir)

        analysis_dir = run_dir / "analysis"
        analysis     = [{"name": path.stem, "url": f"/runsmedia?path={path}"} for path in sorted(analysis_dir.glob("*.png"))] if analysis_dir.is_dir() else []

        return {
            "name"           : name,
            "kind"           : self.layout.kind(run_dir),
            "has_checkpoint" : status["has_checkpoint"],
            "loadable"       : status["loadable"],
            "reason"         : status["reason"],
            "has_events"     : bool(list(run_dir.glob("events.out.tfevents.*"))),
            "reports"        : self.layout.report_seeds(run_dir),
            "diagnostics"    : self.layout.diagnostics_seeds(run_dir),
            "markdown"       : {path.name: path.read_text() for path in sorted(run_dir.glob("*.md"))},
            "logs"           : [{"name": path.name, "text": self._read_log(path)} for path in sorted(run_dir.glob("*.log"))],
            "analysis"       : analysis,
            "files"          : [{"rel": str(path.relative_to(run_dir)), "size": path.stat().st_size} for path in sorted(run_dir.rglob("*")) if path.is_file()],
        }

    def _interval(self, mean, sem):
        return [mean - 1.96 * sem, mean + 1.96 * sem]

    def _identical_to(self, name, agents):
        rewards    = agents[name]["per_episode"]["reward"]
        candidates = [other for other in ("do_nothing", "insertion_only", "teacher") if other != name and other in agents]

        if name != "model":
            candidates = [other for other in candidates if other == "do_nothing"]

        for other in candidates:
            theirs = agents[other]["per_episode"]["reward"]
            if len(theirs) == len(rewards) and all(abs(mine - other_value) < self.TIE for mine, other_value in zip(rewards, theirs)):
                return other

        return None

    def _paired(self, model, other):
        differences = [mine - theirs for mine, theirs in zip(model["per_episode"]["reward"], other["per_episode"]["reward"])]
        count       = len(differences)
        mean        = sum(differences) / count
        sem         = math.sqrt(sum((value - mean) ** 2 for value in differences) / count) / math.sqrt(count)

        return {
            "mean_diff" : mean,
            "sem_diff"  : sem,
            "ci95"      : self._interval(mean, sem),
            "wins"      : sum(1 for value in differences if value > self.TIE),
            "ties"      : sum(1 for value in differences if abs(value) <= self.TIE),
            "losses"    : sum(1 for value in differences if value < -self.TIE),
        }

    def _episode_rows(self, run_dir, meta, agents):
        if "model" not in agents:
            return []

        rows   = []
        model  = agents["model"]["per_episode"]
        seed   = meta["seed"]
        traces = self.layout.diagnostics_dir(run_dir, seed)

        for index in range(len(model["reward"])):
            present = {name: metrics["per_episode"] for name, metrics in agents.items() if index < len(metrics["per_episode"]["reward"])}
            others  = [present[name]["reward"][index] for name in self.BASELINES if name in present]
            teacher = present.get("teacher")

            rows.append({
                "index"                : index,
                "seed"                 : seed + index,
                "reward"               : {name: values["reward"][index] for name, values in present.items()},
                "final_cost"           : {name: values["final_cost"][index] for name, values in present.items()},
                "unassigned"           : {name: values["unassigned"][index] for name, values in present.items()},
                "served"               : {name: values["served"][index] for name, values in present.items()},
                "regret_teacher"       : None if teacher is None else teacher["reward"][index] - model["reward"][index],
                "regret_best_baseline" : max(others) - model["reward"][index] if others else None,
                "unassigned_delta"     : None if teacher is None else model["unassigned"][index] - teacher["unassigned"][index],
                "served_delta"         : None if teacher is None else model["served"][index] - teacher["served"][index],
                "has_trace"            : {key: self.layout.trace_path(run_dir, seed, index, key).exists() for key in self.AGENT_KEYS} if traces.is_dir() else {key: False for key in self.AGENT_KEYS},
            })

        return rows

    def evaluation(self, name, seed):
        """Return one evaluation report with intervals, paired differences and per-episode regret.

        Args:
            name : Run name.
            seed : Report seed.

        Returns:
            {"meta", "agents": {agent: aggregate metrics plus "ci95" and "identical_to"}, "paired": {agent: model minus agent statistics}, "missing_agents", "episodes": [per-episode row]}. Final cost is drive time in seconds.

        Raises:
            ApiError : The run or the report does not exist (404).
        """
        run_dir = self._require_run(name)
        report  = self.read_report(name, seed)
        agents  = report["agents"]

        rendered = {}
        for agent, metrics in agents.items():
            entry                 = {key: value for key, value in metrics.items() if key != "per_episode"}
            entry["ci95"]         = self._interval(metrics["mean_reward"], metrics["sem_reward"])
            entry["identical_to"] = self._identical_to(agent, agents)
            rendered[agent]       = entry

        paired = {}
        if "model" in agents:
            paired = {agent: self._paired(agents["model"], metrics) for agent, metrics in agents.items() if agent != "model" and len(metrics["per_episode"]["reward"]) == len(agents["model"]["per_episode"]["reward"])}

        return {
            "meta"           : report["meta"],
            "agents"         : rendered,
            "paired"         : paired,
            "missing_agents" : [agent for agent in ("model",) + self.BASELINES if agent not in agents],
            "episodes"       : self._episode_rows(run_dir, report["meta"], agents),
        }

    def diagnostics(self, name, seed):
        """Return the diagnostics index of a run and seed as written by the diagnose script.

        Raises:
            ApiError : The run or the index does not exist (404).
        """
        path = self.layout.diagnostics_dir(self._require_run(name), seed) / "index.json"
        if not path.exists():
            raise ApiError(404, f"run '{name}' has no diagnostics for seed {seed}")

        return json.loads(path.read_text())

    def _events_signature(self, run_dir):
        signature = []
        for path in sorted(run_dir.glob("events.out.tfevents.*")):
            stat = path.stat()
            signature.append((path.name, stat.st_mtime, stat.st_size))
        return tuple(signature)

    def _downsample(self, points):
        if len(points) <= self.MAX_POINTS:
            return points
        stride = len(points) // self.MAX_POINTS + 1
        return points[::stride]

    def run_curves(self, name):
        """Return every TensorBoard scalar of a run, at most 1000 points per tag.

        Returns:
            {"tags": {tag: {"steps": [int], "values": [float]}}}.

        Raises:
            ApiError : The run does not exist (404).
        """
        run_dir   = self._require_run(name)
        signature = self._events_signature(run_dir)
        if not signature:
            return {"tags": {}}

        cached = self._curves_cache.get(name)
        if cached is not None and cached[0] == signature:
            return cached[1]

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        accumulator.Reload()

        tags = {}
        for tag in accumulator.Tags()["scalars"]:
            points    = self._downsample(accumulator.Scalars(tag))
            tags[tag] = {"steps": [point.step for point in points], "values": [float(point.value) for point in points]}

        payload                  = {"tags": tags}
        self._curves_cache[name] = (signature, payload)
        return payload

    def health(self, name):
        """Return the fixed training-health charts of a run.

        Returns:
            {"kind", "charts": [{"key", "title", "unit", "series": [{"label", "tag", "steps", "values"}]}]}; a chart whose tags were not logged has an empty series list.

        Raises:
            ApiError : The run does not exist (404).
        """
        run_dir = self._require_run(name)
        kind    = self.layout.kind(run_dir)
        tags    = self.run_curves(name)["tags"]

        charts = []
        for chart in self.HEALTH[kind]:
            series = [{"label": label, "tag": tag, "steps": tags[tag]["steps"], "values": tags[tag]["values"]} for label, tag in chart["tags"] if tag in tags]
            charts.append({"key": chart["key"], "title": chart["title"], "unit": chart["unit"], "series": series})

        return {"kind": kind, "charts": charts}

    def media_path(self, raw):
        """Return the resolved path of a media file under runs/, or None when it is outside or missing."""
        target = Path(raw).resolve()

        if target.is_relative_to(self.paths.runs_dir.resolve()) and target.is_file():
            return target

        return None
