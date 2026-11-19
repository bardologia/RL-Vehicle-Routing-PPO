"""Tests for the console's project path registry.

Covers the directories derived from the root, the root override that keeps a
console off the live repository, the refusal of a root that is not a routing
checkout, the entry-point registry and the interpreter preference.
"""

from __future__ import annotations

import sys

import pytest

from project_paths import ProjectPaths


def _root(path):
    """Creates the directories a routing checkout must hold under a temporary root."""
    for name in ProjectPaths.REQUIRED_DIRS:
        (path / name).mkdir(parents=True)
    return path


def test_a_root_override_moves_every_state_directory(tmp_path):
    """Jobs, saved runs, templates, sequences and telemetry all live under the given root's logs/webui."""
    paths = ProjectPaths(_root(tmp_path))

    assert paths.repo_root == tmp_path
    assert paths.runs_dir == tmp_path / "runs"
    for directory in (paths.jobs_dir, paths.saved_runs_dir, paths.templates_dir, paths.sequences_dir, paths.telemetry_dir, paths.episodes_dir):
        assert directory.parent == tmp_path / "logs" / "webui"
    assert paths.static_dir == ProjectPaths().static_dir


def test_a_root_that_is_not_a_checkout_is_refused(tmp_path):
    """A root missing a required directory is refused, naming what is missing."""
    with pytest.raises(ValueError, match="missing main, configuration, core, webui"):
        ProjectPaths(tmp_path)


def test_the_entry_registry_names_the_five_scripts(tmp_path):
    """The five routing scripts map to main/<key>.py and the shared Config; anything else is unknown."""
    paths = ProjectPaths(_root(tmp_path))
    entry = paths.script_entry("diagnose")

    assert [key for key in ("pretrain", "train", "evaluate", "infer", "diagnose", "tune") if paths.has_script(key)] == ["pretrain", "train", "evaluate", "infer", "diagnose"]
    assert (entry["path"], entry["rel"], entry["config_module"], entry["config_class"]) == (tmp_path / "main" / "diagnose.py", "main/diagnose.py", "configuration", "Config")
    with pytest.raises(KeyError):
        paths.script_entry("tune")


def test_the_routing_environment_is_preferred(tmp_path):
    """The routing-ppo environment wins over others, the first one is the fallback, the running one the last resort."""
    paths = ProjectPaths(_root(tmp_path))
    found = [{"label": "conda:base", "path": "/base/python"}, {"label": "conda:routing-ppo", "path": "/env/python"}]

    assert paths.preferred_interpreter(found) == "/env/python"
    assert paths.preferred_interpreter(found[:1]) == "/base/python"
    assert paths.preferred_interpreter([]) == sys.executable
