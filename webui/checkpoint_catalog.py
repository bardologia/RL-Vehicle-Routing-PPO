"""Catalog of the checkpoints under runs/ and whether each still fits the current policy architecture."""

import torch

from model.policy_model import Policy


class CheckpointCatalog:
    """Lists checkpoints with a loadability verdict.

    Attributes:
        runtime : ConsoleRuntime providing the base Config and the run layout.
    """

    def __init__(self, runtime):
        self.runtime    = runtime
        self.layout     = runtime.layout
        self._reference = None
        self._cache     = {}

    def _reference_shapes(self):
        if self._reference is None:
            self._reference = {key: tuple(value.shape) for key, value in Policy(self.runtime.config).state_dict().items()}

        return self._reference

    def _compare(self, path):
        stored    = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        reference = self._reference_shapes()

        missing    = [key for key in reference if key not in stored]
        unexpected = [key for key in stored if key not in reference]
        mismatched = [key for key in reference if key in stored and tuple(stored[key].shape) != reference[key]]

        if not missing and not unexpected and not mismatched:
            return None

        parts = []
        if missing:
            parts.append(f"{len(missing)} missing keys")
        if unexpected:
            parts.append(f"{len(unexpected)} unexpected keys")
        if mismatched:
            parts.append(f"{len(mismatched)} shape mismatches")

        return "architecture mismatch: " + ", ".join(parts)

    def status(self, run_dir):
        """Return the checkpoint status of a run directory.

        Args:
            run_dir : Path of the run directory.

        Returns:
            {"has_checkpoint": bool, "loadable": bool or None when there is no checkpoint, "reason": str or None, "mtime": checkpoint mtime in epoch seconds or None}.
        """
        path = self.layout.checkpoint(run_dir)
        if not path.exists():
            return {"has_checkpoint": False, "loadable": None, "reason": None, "mtime": None}

        mtime  = path.stat().st_mtime
        cached = self._cache.get(str(path))

        if cached is None or cached[0] != mtime:
            cached                 = (mtime, self._compare(path))
            self._cache[str(path)] = cached

        return {"has_checkpoint": True, "loadable": cached[1] is None, "reason": cached[1], "mtime": mtime}

    def list(self):
        """Return every run that holds a checkpoint, newest first.

        Returns:
            List of {"run", "kind", "mtime", "loadable", "reason", "reports": [seed], "diagnostics": [seed]}.
        """
        entries = []
        for run_dir in self.layout.run_dirs():
            status = self.status(run_dir)
            if not status["has_checkpoint"]:
                continue

            entries.append({
                "run"         : run_dir.name,
                "kind"        : self.layout.kind(run_dir),
                "mtime"       : status["mtime"],
                "loadable"    : status["loadable"],
                "reason"      : status["reason"],
                "reports"     : self.layout.report_seeds(run_dir),
                "diagnostics" : self.layout.diagnostics_seeds(run_dir),
            })

        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        return entries
