"""Tests for the resolution of the routing configuration into launch form leaves.

The bootstrap runs under a real interpreter and lists every init field of the
shared Config with the value rendered the way ConfigCli parses it back, the
form type and editability of its annotation, and its section; results are
cached against the configuration signature and a failing bootstrap is
reported with the tail of its stderr.
"""

from __future__ import annotations

import sys
from dataclasses import fields

import pytest

from config_value_render    import ConfigValueRenderer
from configuration          import Config
from configuration.cli      import ConfigCli
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


@pytest.fixture(scope="module")
def resolved():
    """Returns one resolution of the train entry under the running interpreter."""
    result = ScriptConfigResolver(ProjectPaths()).resolve("train", sys.executable)
    assert result["ok"], result.get("error")
    return result


def _leaf(resolved: dict, path: str) -> dict:
    """Returns the resolved leaf of one dotted path."""
    return next(leaf for leaf in resolved["leaves"] if leaf["path"] == path)


def test_every_init_field_of_the_config_is_a_leaf(resolved):
    """The leaves are exactly the init fields of every configuration section."""
    config   = Config()
    expected = {f"{section.name}.{spec.name}" for section in fields(config) for spec in fields(getattr(config, section.name)) if spec.init}

    assert {leaf["path"] for leaf in resolved["leaves"]} == expected
    assert resolved["module"] == "configuration" and resolved["config_class"] == "Config"


def test_values_render_in_the_form_config_cli_parses(resolved):
    """Tuples render comma-separated, booleans as True or False and an unset optional as empty, each round-tripping through ConfigCli."""
    assert _leaf(resolved, "env.center")["value"] == "-46.63, -23.55"
    assert _leaf(resolved, "diagnostics.agents")["value"] == "model, insertion_only, do_nothing"
    assert _leaf(resolved, "training.use_mixed_precision")["value"] == "False"
    assert _leaf(resolved, "io.run_name")["value"] == ""

    config = ConfigCli(Config(), ["--env.center", _leaf(resolved, "env.center")["value"], "--diagnostics.agents", _leaf(resolved, "diagnostics.agents")["value"], "--training.use_mixed_precision", "False"]).apply()

    assert config.env.center == Config().env.center
    assert config.diagnostics.agents == Config().diagnostics.agents
    assert config.training.use_mixed_precision is False


def test_types_and_editability_follow_the_annotations(resolved):
    """Scalars and tuples are editable with their form type; the solver options dictionary is not."""
    assert (_leaf(resolved, "training.num_updates")["type"], _leaf(resolved, "training.num_updates")["editable"]) == ("int", True)
    assert (_leaf(resolved, "lr.lr_min")["type"], _leaf(resolved, "lr.lr_min")["editable"]) == ("float", True)
    assert (_leaf(resolved, "io.run_name")["type"], _leaf(resolved, "io.run_name")["nullable"]) == ("str", True)
    assert _leaf(resolved, "env.center")["type"] == "tuple"
    assert (_leaf(resolved, "service.options")["type"], _leaf(resolved, "service.options")["editable"]) == ("dict", False)
    assert _leaf(resolved, "reward.distance_weight")["section"] == "reward"
    assert _leaf(resolved, "reward.distance_weight")["section_class"] == "RewardConfig"


def test_a_second_resolution_is_served_from_the_cache(monkeypatch):
    """An unchanged configuration is resolved once; later calls never start the bootstrap."""
    resolver = ScriptConfigResolver(ProjectPaths())
    first    = resolver.resolve("infer", sys.executable)

    monkeypatch.setattr(resolver, "_run_bootstrap", lambda entry, interpreter: pytest.fail("the cache was bypassed"))

    assert resolver.resolve("infer", sys.executable) is first


def test_a_failing_interpreter_is_reported_with_its_stderr(tmp_path):
    """An interpreter that cannot import the configuration yields a failed result naming it."""
    broken = tmp_path / "python"
    broken.write_text("#!/bin/sh\necho 'ModuleNotFoundError: no configuration here' >&2\nexit 1\n")
    broken.chmod(0o755)

    result = ScriptConfigResolver(ProjectPaths()).resolve("train", str(broken))

    assert result["ok"] is False
    assert "no configuration here" in result["error"]


def test_the_renderer_classifies_optional_and_tuple_annotations():
    """The renderer reads through Optional, names tuples and refuses unsupported containers."""
    annotations = {spec.name: spec.type for spec in fields(Config().io)}

    assert ConfigValueRenderer.kind(annotations["run_name"]) == "str"
    assert ConfigValueRenderer.nullable(annotations["run_name"]) is True
    assert ConfigValueRenderer.render(None) == ""
    assert ConfigValueRenderer.render((1.5, 2)) == "1.5, 2"
