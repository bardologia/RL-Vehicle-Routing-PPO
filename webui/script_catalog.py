"""Curated metadata of every console entry point.

Holds the title, category and purpose text shown for each routing entry
script, lists the entries that exist on disk, and returns one entry's detail
with its source for the launch page.
"""

from __future__ import annotations

from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


class ScriptCatalog:
    """Lists the entry points that exist on disk, with their metadata.

    Attributes:
        paths    : Project paths resolving each entry key to a file.
        resolver : Config resolver providing each entry's configuration class name.
    """

    META = {
        "pretrain": {
            "title"    : "Pretrain",
            "category" : "Training",
            "purpose"  : "Clone the regret-insertion teacher into the policy by behaviour cloning, as the warm start for PPO.",
        },
        "train": {
            "title"    : "Train",
            "category" : "Training",
            "purpose"  : "PPO over inline-sampled episodes with per-head learning rates, entropy annealing and a KL anchor to the initial policy.",
        },
        "evaluate": {
            "title"    : "Evaluate",
            "category" : "Evaluation",
            "purpose"  : "Score a checkpoint on held-out episodes against the teacher, insertion-only and do-nothing baselines, and write the evaluation report.",
        },
        "infer": {
            "title"    : "Infer",
            "category" : "Evaluation",
            "purpose"  : "Run a checkpoint once on a random scenario and log the decision table of the episode.",
        },
        "diagnose": {
            "title"    : "Diagnose",
            "category" : "Analysis",
            "purpose"  : "Record the full trace of every agent on held-out episodes, with policy distributions and teacher verdicts, and index them for the Diagnose tab.",
        },
    }

    def __init__(self, paths: ProjectPaths, resolver: ScriptConfigResolver) -> None:
        """Stores the project paths and the config resolver."""
        self.paths    = paths
        self.resolver = resolver

    def titles(self) -> dict:
        """Returns the catalog title of every entry point, keyed by script key."""
        return {key: meta["title"] for key, meta in self.META.items()}

    def list_scripts(self) -> list[dict]:
        """Returns one catalog record per entry point present on disk.

        Each record carries the entry key and relative file, its title, category and purpose, and its configuration class.
        """
        entries = []
        for key, meta in self.META.items():
            spec = self.paths.script_entry(key)
            if not spec["path"].exists():
                continue

            entries.append({
                "key"          : key,
                "file"         : spec["rel"],
                "title"        : meta["title"],
                "category"     : meta["category"],
                "purpose"      : meta["purpose"],
                "config_class" : self.resolver.entry_config(key)["class"],
            })
        return entries

    def get_script(self, key: str) -> dict | None:
        """Returns one entry point's full detail, source text included.

        Args:
            key : Entry-point key.

        Returns:
            Record with the catalog metadata, the entry source, the shell command and the preferred interpreter, or None when the entry file does not exist.
        """
        spec = self.paths.script_entry(key)
        if not spec["path"].exists():
            return None

        meta = self.META[key]

        return {
            "key"          : key,
            "file"         : spec["rel"],
            "title"        : meta["title"],
            "category"     : meta["category"],
            "purpose"      : meta["purpose"],
            "source"       : spec["path"].read_text(encoding="utf-8"),
            "language"     : "python",
            "config_class" : self.resolver.entry_config(key)["class"],
            "command"      : f"python -u {spec['rel']}",
            "preferred"    : self.paths.preferred_interpreter(self.paths.discover_interpreters(), key),
        }
