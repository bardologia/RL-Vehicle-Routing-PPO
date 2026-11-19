"""Filesystem layout of the console: repository paths, entry-point registry, interpreter discovery and the files of a run directory.

ProjectPaths resolves the directories the console reads from and writes to,
maps every launchable script key to its file under main/, and finds the conda
interpreters a job can be launched with. RunLayout names the files a run
directory holds: checkpoint, evaluation reports and diagnostics traces.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


class ProjectPaths:
    """Resolved paths of a routing checkout plus its script and interpreter registry.

    Attributes:
        webui_dir      : Directory holding the console server and its static assets.
        repo_root      : Root the console operates on: jobs start there, runs and state live under it.
        static_dir     : Frontend files served under /static.
        main_dir       : Directory holding the entry-point scripts.
        config_dir     : Root of the configuration package.
        runs_dir       : Training run directories.
        logs_dir       : Directory the console writes its state, logs and telemetry into.
        state_dir      : Same directory as logs_dir, under the name the analysis services use.
        episodes_dir   : Interactive episode recordings, one folder per episode.
        scenarios_dir  : Saved Lab scenarios, one JSON file each.
        buildings_dir  : Cached building tiles.
        saved_runs_dir : Saved launch definitions, one JSON file each.
        templates_dir  : Config templates, one JSON file each.
        sequences_dir  : Stored experiment sequences, one JSON file each.
        jobs_dir       : Output transcripts of jobs that stream to the console, one .out file per job.
        telemetry_dir  : Persisted host and GPU activity histories.
        interpreter    : Interpreter the console itself runs under.
    """

    REQUIRED_DIRS = ("main", "configuration", "core", "webui")
    SCRIPT_KEYS   = ("pretrain", "train", "evaluate", "infer", "diagnose")
    CONDA_BASES   = ("miniconda3", "anaconda3", ".conda")
    PRIORITY      = ("conda:routing-ppo",)
    CONFIG_MODULE = "configuration"
    CONFIG_CLASS  = "Config"

    def __init__(self, root: Path | None = None) -> None:
        """Resolves every directory and verifies the root is a routing checkout.

        Args:
            root : Root the console operates on; defaults to the parent of the webui directory.

        Raises:
            ValueError : If the resolved root lacks any of REQUIRED_DIRS.
        """
        self.webui_dir      = Path(__file__).resolve().parent
        self.repo_root      = self.webui_dir.parent if root is None else Path(root).resolve()
        self.static_dir     = self.webui_dir / "static"
        self.main_dir       = self.repo_root / "main"
        self.config_dir     = self.repo_root / "configuration"
        self.runs_dir       = self.repo_root / "runs"
        self.logs_dir       = self.repo_root / "logs" / "webui"
        self.state_dir      = self.logs_dir
        self.episodes_dir   = self.logs_dir / "episodes"
        self.scenarios_dir  = self.logs_dir / "scenarios"
        self.buildings_dir  = self.logs_dir / "buildings"
        self.saved_runs_dir = self.logs_dir / "saved_runs"
        self.templates_dir  = self.logs_dir / "config_templates"
        self.sequences_dir  = self.logs_dir / "sequences"
        self.jobs_dir       = self.logs_dir / "jobs"
        self.telemetry_dir  = self.logs_dir / "telemetry"
        self.interpreter    = sys.executable

        self._require_repository()

    def _require_repository(self) -> None:
        """Raises ValueError when the root is missing a required directory."""
        missing = [name for name in self.REQUIRED_DIRS if not (self.repo_root / name).is_dir()]

        if missing:
            raise ValueError(f"project root {self.repo_root} is not a routing repository: missing {', '.join(missing)}")

    def run_dir(self, name: str) -> Path:
        """Returns the directory of the run called `name`."""
        return self.runs_dir / name

    def tree_signature(self, paths) -> tuple:
        """Returns a cache key of (name, mtime_ns) pairs for the given paths, skipping unreadable ones."""
        stamps = []
        for path in paths:
            try:
                stamps.append((path.name, path.stat().st_mtime_ns))
            except OSError:
                continue
        return tuple(stamps)

    def config_signature(self) -> tuple:
        """Returns the mtime signature of every Python file under the configuration package."""
        return self.tree_signature(sorted(self.config_dir.rglob("*.py")))

    def has_script(self, key: str) -> bool:
        """Returns whether the key names a registered entry-point script."""
        return key in self.SCRIPT_KEYS

    def script_entry(self, key: str) -> dict:
        """Returns the absolute path, repo-relative path and entry config class of a script.

        Args:
            key : Registered script key, such as "train".

        Returns:
            Mapping with "path", "rel", "config_module" and "config_class"; every routing entry reads the one shared Config.

        Raises:
            KeyError : If the key is not one of SCRIPT_KEYS.
        """
        if key not in self.SCRIPT_KEYS:
            raise KeyError(key)

        return {
            "path"          : self.main_dir / f"{key}.py",
            "rel"           : f"main/{key}.py",
            "config_module" : self.CONFIG_MODULE,
            "config_class"  : self.CONFIG_CLASS,
        }

    def discover_interpreters(self) -> list[dict]:
        """Returns the conda interpreters available on this machine.

        Scans the miniconda, anaconda and .conda installs under the home
        directory for base and per-environment interpreters, and prepends the
        running interpreter when it is not one of them.

        Returns:
            List of entries with "label", "path" and a "current" flag marking the interpreter this server runs under.
        """
        found   = []
        seen    = set()
        current = str(Path(sys.executable))

        for base in [Path.home() / name for name in self.CONDA_BASES]:
            base_py = base / "bin" / "python"
            if base_py.exists() and str(base_py) not in seen:
                found.append({"label": "conda:base", "path": str(base_py), "current": str(base_py) == current})
                seen.add(str(base_py))

            envs_dir = base / "envs"
            if envs_dir.is_dir():
                for env in sorted(envs_dir.iterdir()):
                    env_py = env / "bin" / "python"
                    if env_py.exists() and str(env_py) not in seen:
                        found.append({"label": f"conda:{env.name}", "path": str(env_py), "current": str(env_py) == current})
                        seen.add(str(env_py))

        if current not in seen and Path(current).exists():
            found.insert(0, {"label": "current", "path": current, "current": True})

        return found

    def preferred_interpreter(self, interpreters: list[dict], script_key: str = "") -> str:
        """Returns the interpreter path a script defaults to.

        Args:
            interpreters : Discovered interpreter entries as returned by discover_interpreters.
            script_key   : Script key; every routing entry prefers the same environment.

        Returns:
            Path of the routing-ppo environment when present, the first discovered interpreter otherwise, or the running interpreter when the list is empty.
        """
        for wanted in self.PRIORITY:
            for item in interpreters:
                if item["label"] == wanted:
                    return item["path"]
        return interpreters[0]["path"] if interpreters else sys.executable


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
