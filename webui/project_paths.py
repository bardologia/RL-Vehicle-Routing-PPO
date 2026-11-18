"""Filesystem layout of the console: repository paths and the files that make up a run directory."""

import re
import sys
from pathlib import Path


class ProjectPaths:
    """Paths the console reads from and writes to.

    Attributes:
        repo_root     : Repository root.
        static_dir    : Frontend files served under /static.
        runs_dir      : Training run directories.
        episodes_dir  : Interactive episode recordings, one folder per episode.
        scenarios_dir : Saved Lab scenarios, one JSON file each.
        buildings_dir : Cached building tiles.
        interpreter   : Python interpreter used to launch scripts.
    """

    def __init__(self):
        self.webui_dir     = Path(__file__).resolve().parent
        self.repo_root     = self.webui_dir.parent
        self.static_dir    = self.webui_dir / "static"
        self.main_dir      = self.repo_root / "main"
        self.runs_dir      = self.repo_root / "runs"
        self.state_dir     = self.repo_root / "logs" / "webui"
        self.episodes_dir  = self.state_dir / "episodes"
        self.scenarios_dir = self.state_dir / "scenarios"
        self.buildings_dir = self.state_dir / "buildings"
        self.interpreter   = sys.executable

    def script_path(self, key):
        """Return the path of the entry script `main/<key>.py`."""
        return self.main_dir / f"{key}.py"

    def run_dir(self, name):
        """Return the directory of the run called `name`."""
        return self.runs_dir / name


class RunLayout:
    """Names the files of a run directory: checkpoint, evaluation reports and diagnostics traces."""

    CHECKPOINT  = "graph_ppo_policy.pt"
    REPORT      = re.compile(r"^evaluation_seed(\d+)\.json$")
    DIAGNOSTICS = re.compile(r"^seed(\d+)$")
    RUN_NAME    = re.compile(r"^[A-Za-z0-9_.\-]{1,96}$")

    def __init__(self, paths):
        self.paths = paths

    def run_dir(self, name):
        """Return the directory of a run, or None when the name is malformed or no such directory exists."""
        if not self.RUN_NAME.match(name) or name in (".", ".."):
            return None

        run_dir = self.paths.run_dir(name)
        return run_dir if run_dir.is_dir() else None

    def run_dirs(self):
        """Return every run directory, sorted by name."""
        if not self.paths.runs_dir.is_dir():
            return []

        return [run_dir for run_dir in sorted(self.paths.runs_dir.iterdir()) if run_dir.is_dir()]

    def kind(self, run_dir):
        """Return "ppo", "pretrain" or "checkpoint" from the log file a run holds."""
        if (run_dir / "training.log").exists():
            return "ppo"
        if (run_dir / "pretraining.log").exists():
            return "pretrain"
        return "checkpoint"

    def checkpoint(self, run_dir):
        """Return the checkpoint path of a run; the file may not exist."""
        return run_dir / self.CHECKPOINT

    def report_path(self, run_dir, seed):
        """Return the evaluation report path of a run for one report seed."""
        return run_dir / f"evaluation_seed{int(seed)}.json"

    def report_seeds(self, run_dir):
        """Return the sorted seeds of the evaluation reports a run holds."""
        seeds = [int(match.group(1)) for match in (self.REPORT.match(path.name) for path in run_dir.iterdir()) if match]
        return sorted(seeds)

    def diagnostics_dir(self, run_dir, seed):
        """Return the diagnostics directory of a run for one seed."""
        return run_dir / "diagnostics" / f"seed{int(seed)}"

    def diagnostics_seeds(self, run_dir):
        """Return the sorted seeds that have a diagnostics index."""
        root = run_dir / "diagnostics"
        if not root.is_dir():
            return []

        seeds = [int(match.group(1)) for match, path in ((self.DIAGNOSTICS.match(path.name), path) for path in root.iterdir()) if match and (path / "index.json").exists()]
        return sorted(seeds)

    def trace_path(self, run_dir, seed, index, agent_key):
        """Return the path of one diagnostics trace."""
        return self.diagnostics_dir(run_dir, seed) / f"ep{int(index)}_{agent_key}.json.gz"
