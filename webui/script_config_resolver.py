"""Resolution of the routing configuration into editable leaves.

Every routing entry point reads the one shared ``Config``. A bootstrap snippet
runs under the target interpreter, instantiates the configuration and prints
every leaf with its rendered value, form type, editability, section and
whether it accepts None. Leaf values go through the shared
``ConfigValueRenderer``, the same renderer the run-config reader uses, so a
copied run compares string-equal to the resolved defaults. Results are cached
against a signature of the configuration tree, the entry script and the
renderer, and concurrent requests for the same entry share a single subprocess.
"""

from __future__ import annotations

import json
import subprocess
import threading

from project_paths import ProjectPaths


class ScriptConfigResolver:
    """Resolves and caches the config leaves of each entry point.

    Attributes:
        paths    : Project paths providing entry points and the configuration tree.
        cache    : Resolved results keyed by entry point and interpreter, with the signature they were produced under.
        inflight : Pending resolutions shared by concurrent callers.
        lock     : Guards the cache and the in-flight registry.
    """

    BOOTSTRAP = (
        "import json, sys\n"
        "from dataclasses import fields\n"
        "repo = sys.argv[1]\n"
        "sys.path.insert(0, repo)\n"
        "sys.path.insert(0, sys.argv[4])\n"
        "from config_value_render import ConfigValueRenderer\n"
        "module = __import__(sys.argv[2], fromlist=[sys.argv[3]])\n"
        "config = getattr(module, sys.argv[3])()\n"
        "leaves = []\n"
        "for section in fields(config):\n"
        "    block = getattr(config, section.name)\n"
        "    for spec in fields(block):\n"
        "        if not spec.init:\n"
        "            continue\n"
        "        value = getattr(block, spec.name)\n"
        "        leaves.append({'path': section.name + '.' + spec.name, 'value': ConfigValueRenderer.render(value), 'type': ConfigValueRenderer.kind(spec.type), 'editable': ConfigValueRenderer.editable(spec.type), 'nullable': ConfigValueRenderer.nullable(spec.type), 'choices': None, 'section': section.name, 'section_class': type(block).__name__, 'source': 'default'})\n"
        "print(json.dumps(leaves))\n"
    )

    def __init__(self, paths: ProjectPaths) -> None:
        """Stores the project paths and creates the empty cache and lock."""
        self.paths    = paths
        self.cache    = {}
        self.inflight = {}
        self.lock     = threading.Lock()

    def entry_config(self, key: str) -> dict:
        """Returns the configuration module and class an entry point reads.

        Args:
            key : Entry-point key.

        Returns:
            Mapping with `module` and `class`.
        """
        entry = self.paths.script_entry(key)
        return {"module": entry["config_module"], "class": entry["config_class"]}

    def _signature(self, key: str) -> tuple:
        """Returns the change signature of the configuration tree, the entry script and the renderer."""
        watched = sorted(self.paths.config_dir.rglob("*.py"))
        watched.append(self.paths.script_entry(key)["path"])
        watched.append(self.paths.webui_dir / "config_value_render.py")

        return self.paths.tree_signature(watched)

    def _run_bootstrap(self, entry: dict, interpreter: str) -> dict:
        """Instantiates the configuration under an interpreter and returns its leaves.

        Args:
            entry       : Mapping with the configuration `module` and `class`.
            interpreter : Python interpreter to run the bootstrap snippet under.

        Returns:
            Result with `module`, `config_class` and the `leaves` list, or `ok` False with the tail of the subprocess stderr when the import or run fails.
        """
        argv = [interpreter, "-c", self.BOOTSTRAP, str(self.paths.repo_root), entry["module"], entry["class"], str(self.paths.webui_dir)]

        try:
            proc = subprocess.run(argv, cwd=str(self.paths.repo_root), capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": f"config resolution failed under {interpreter}: {exc}"}

        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-4:])
            return {"ok": False, "error": f"config resolution failed under {interpreter}:\n{tail}"}

        try:
            leaves = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return {"ok": False, "error": "config resolution produced no output"}

        return {"ok": True, "module": entry["module"], "config_class": entry["class"], "leaves": leaves}

    def resolve(self, key: str, interpreter: str) -> dict:
        """Returns an entry point's config leaves, cached against the configuration tree.

        Serves the cached result while the signature is unchanged. A first
        caller runs the bootstrap subprocess and later callers wait on the same
        result; only successful results enter the cache.

        Args:
            key         : Entry-point key.
            interpreter : Python interpreter to resolve under.

        Returns:
            The resolution result, or `ok` False with an error when the bootstrap fails.

        Raises:
            BaseException : Re-raises whatever the leading resolution raised, in the waiting callers as well.
        """
        entry        = self.entry_config(key)
        signature    = self._signature(key)
        cache_key    = (key, interpreter)
        inflight_key = (key, interpreter, signature)

        with self.lock:
            hit = self.cache.get(cache_key)
            if hit is not None and hit[0] == signature:
                return hit[1]

            pending = self.inflight.get(inflight_key)
            leader  = pending is None
            if leader:
                pending                     = {"event": threading.Event(), "result": None, "failure": None}
                self.inflight[inflight_key] = pending

        if not leader:
            pending["event"].wait()
            if pending["failure"] is not None:
                raise pending["failure"]
            return pending["result"]

        try:
            result            = self._run_bootstrap(entry, interpreter)
            pending["result"] = result
        except BaseException as exc:
            pending["failure"] = exc
            raise
        finally:
            with self.lock:
                self.inflight.pop(inflight_key, None)
            pending["event"].set()

        if result.get("ok"):
            with self.lock:
                self.cache[cache_key] = (signature, result)

        return result
