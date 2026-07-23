import json
import re
from pathlib import Path


class RunBrowser:
    ANSI           = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")
    LOG_TAIL_BYTES = 262144
    MAX_POINTS     = 1000

    def __init__(self, paths):
        self.paths         = paths
        self._curves_cache = {}

    def _run_kind(self, run_dir):
        if (run_dir / "training.log").exists():
            return "ppo"
        if (run_dir / "pretraining.log").exists():
            return "pretrain"
        return "checkpoint"

    def _eval_brief(self, run_dir):
        path = run_dir / "evaluation.json"
        if not path.exists():
            return None

        report = json.loads(path.read_text())
        return {name: round(metrics["mean_reward"], 3) for name, metrics in report.items() if "mean_reward" in metrics}

    def list_runs(self):
        if not self.paths.runs_dir.is_dir():
            return []

        runs = []
        for run_dir in sorted(self.paths.runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            events = list(run_dir.glob("events.out.tfevents.*"))
            runs.append({
                "name"           : run_dir.name,
                "kind"           : self._run_kind(run_dir),
                "mtime"          : max((child.stat().st_mtime for child in run_dir.iterdir()), default=run_dir.stat().st_mtime),
                "has_checkpoint" : (run_dir / "graph_ppo_policy.pt").exists(),
                "has_events"     : bool(events),
                "analysis_count" : len(list((run_dir / "analysis").glob("*.png"))) if (run_dir / "analysis").is_dir() else 0,
                "evaluation"     : self._eval_brief(run_dir),
            })

        runs.sort(key=lambda run: run["mtime"], reverse=True)
        return runs

    def _read_log(self, path):
        raw = path.read_bytes()
        if len(raw) > self.LOG_TAIL_BYTES:
            raw = raw[-self.LOG_TAIL_BYTES:]

        text = raw.decode("utf-8", "replace")
        return self.ANSI.sub("", text)

    def run_detail(self, name):
        run_dir = self.paths.run_dir(name)
        if not run_dir.is_dir():
            return {"error": f"unknown run '{name}'"}

        evaluations = {}
        for path in sorted(run_dir.glob("evaluation*.json")):
            evaluations[path.name] = json.loads(path.read_text())

        markdown = {}
        for path in sorted(run_dir.glob("*.md")):
            markdown[path.name] = path.read_text()

        logs = []
        for path in sorted(run_dir.glob("*.log")):
            logs.append({"name": path.name, "text": self._read_log(path)})

        analysis_dir = run_dir / "analysis"
        analysis     = []
        if analysis_dir.is_dir():
            for path in sorted(analysis_dir.glob("*.png")):
                analysis.append({"name": path.stem, "url": f"/runsmedia?path={path}"})

        files = []
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                files.append({"rel": str(path.relative_to(run_dir)), "size": path.stat().st_size})

        return {
            "name"           : name,
            "kind"           : self._run_kind(run_dir),
            "has_checkpoint" : (run_dir / "graph_ppo_policy.pt").exists(),
            "has_events"     : bool(list(run_dir.glob("events.out.tfevents.*"))),
            "evaluations"    : evaluations,
            "markdown"       : markdown,
            "logs"           : logs,
            "analysis"       : analysis,
            "files"          : files,
        }

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
        run_dir = self.paths.run_dir(name)
        if not run_dir.is_dir():
            return {"error": f"unknown run '{name}'"}

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
            points = self._downsample(accumulator.Scalars(tag))
            tags[tag] = {
                "steps"  : [point.step for point in points],
                "values" : [float(point.value) for point in points],
            }

        payload = {"tags": tags}
        self._curves_cache[name] = (signature, payload)
        return payload

    def media_path(self, raw):
        target = Path(raw).resolve()

        if target.is_relative_to(self.paths.runs_dir.resolve()) and target.is_file():
            return target

        return None
