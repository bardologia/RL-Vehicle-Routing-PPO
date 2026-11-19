"""Reading of the settings a finished run recorded, for the launch form's copy-from-run control.

Routing runs keep no resolved configuration file; what they do record is the
``meta`` block of each evaluation report: the environment, reward and episode
caps the run was evaluated under, plus the evaluation seed, episode count and
action selection. RunConfigReader lists the runs holding such a report and maps
the newest report's meta back onto dotted config paths with the same rendering
the resolver uses, so a copied value compares string-equal to a default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config_value_render import ConfigValueRenderer
from web_logger          import WebLogger


class RunConfigReader:
    """Finds and reads the evaluation settings recorded in run directories.

    Attributes:
        REPORT          : File name pattern of an evaluation report.
        SECTIONS        : Meta blocks and the config section their keys map onto.
        SCALARS         : Top-level meta keys and the dotted config path each maps onto.
        IDENTITY_LEAVES : Dotted paths naming the source run itself; withheld from every read.
        logger          : Web logger used to record what each listing and read found.
    """

    REPORT          = re.compile(r"^evaluation_seed(\d+)\.json$")
    SECTIONS        = {"environment": "env", "reward": "reward", "caps": "training"}
    SCALARS         = {"run_name": "io.run_name", "checkpoint_filename": "io.checkpoint_filename", "seed": "evaluation.seed", "episodes": "evaluation.episodes", "greedy": "evaluation.greedy"}
    IDENTITY_LEAVES = ("io.run_name",)

    def __init__(self, logger: WebLogger) -> None:
        """Stores the logger used to report listing and read results."""
        self.logger = logger

    def _report(self, run_dir: Path) -> Path | None:
        """Returns the most recently written evaluation report of a run directory, or None."""
        reports = [path for path in run_dir.iterdir() if path.is_file() and self.REPORT.match(path.name)]
        return max(reports, key=lambda path: path.stat().st_mtime) if reports else None

    def list_runs(self, raw_base: str) -> dict:
        """Lists every run directory directly below a root that holds an evaluation report.

        Args:
            raw_base : Absolute path of the runs root to search.

        Returns:
            Dict with ``ok``, the resolved ``base`` and a ``runs`` list of entries carrying the run name, its absolute path and the report file it would be copied from; ``ok`` False with an error when the base is not a directory.
        """
        base = Path(raw_base).expanduser() if raw_base else None
        if base is None or not base.is_dir():
            return {"ok": False, "error": f"not a directory: {raw_base}"}

        entries = []
        for run_dir in sorted(base.iterdir()):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            found = self._report(run_dir)
            if found is not None:
                entries.append({"name": run_dir.name, "path": str(run_dir), "source": found.name})

        self.logger.info(f"run configs: listed {len(entries)} under {base}")
        return {"ok": True, "base": str(base), "runs": entries}

    def _flatten(self, meta: dict) -> dict:
        """Maps an evaluation report's meta block onto dotted config paths with rendered values."""
        mapping = {}

        for block, section in self.SECTIONS.items():
            for name, value in meta[block].items():
                mapping[f"{section}.{name}"] = ConfigValueRenderer.render(value)

        for name, path in self.SCALARS.items():
            mapping[path] = ConfigValueRenderer.render(meta[name])

        return mapping

    def read(self, raw_path: str) -> dict:
        """Reads the recorded settings of one run as dotted paths with rendered values.

        The run's own name is split out into ``identity`` instead of being
        served as a copyable setting, so a copy never points a new launch at
        the run it was copied from.

        Args:
            raw_path : Absolute path of the run directory.

        Returns:
            Dict with ``ok``, the ``run`` name, the ``source`` report, the ``config`` mapping of copyable dotted config paths to rendered string values and the ``identity`` list of the withheld paths; ``ok`` False with an error when the directory holds no evaluation report or the report is unreadable.
        """
        run_dir = Path(raw_path).expanduser() if raw_path else None
        if run_dir is None or not run_dir.is_dir():
            return {"ok": False, "error": f"not a run directory: {raw_path}"}

        found = self._report(run_dir)
        if found is None:
            return {"ok": False, "error": f"{run_dir.name} holds no evaluation report; only evaluated runs record their settings"}

        try:
            meta = json.loads(found.read_text(encoding="utf-8"))["meta"]
        except (OSError, ValueError, KeyError) as exc:
            return {"ok": False, "error": f"could not read the meta block of {found}: {exc}"}

        mapping  = self._flatten(meta)
        rendered = {path: value for path, value in mapping.items() if path not in self.IDENTITY_LEAVES}
        identity = sorted(path for path in mapping if path in self.IDENTITY_LEAVES)

        self.logger.info(f"run configs: read {len(rendered)} fields from {found}, {len(identity)} identity fields withheld")
        return {"ok": True, "run": run_dir.name, "source": found.name, "config": rendered, "identity": identity}
