"""Tests for the launch form layouts over the live routing configuration.

Every entry point's layout builds against the resolved leaves of the shared
Config and claims each leaf of its shown sections exactly once, a field added
to a shown section or a stray claim fails the build loudly, a hidden section
may not be claimed, number widgets carry bounds, and the catalog route answers
the layout error instead of a form.
"""

from __future__ import annotations

import copy
import json
import sys

import pytest

from config_registry        import ConfigRegistry
from launch_layout          import LaunchLayout, LayoutError
from project_paths          import ProjectPaths
from routers.dispatch       import HttpExchange
from routers.launch_routers import CatalogRouter
from script_config_resolver import ScriptConfigResolver

from tests.webui.conftest import FakeHandler

SCRIPTS = ("pretrain", "train", "evaluate", "infer", "diagnose")


@pytest.fixture(scope="module")
def paths():
    """Returns the project paths of the live repository."""
    return ProjectPaths()


@pytest.fixture(scope="module")
def leaves(paths):
    """Returns the resolved leaves of the shared configuration, resolved once."""
    resolved = ScriptConfigResolver(paths).resolve("train", sys.executable)
    assert resolved["ok"], resolved.get("error")
    return resolved["leaves"]


def _with_extra_field(leaves: list, section: str, name: str) -> list:
    """Returns a copy of the leaves with one more field in the given section."""
    extra = {**next(leaf for leaf in leaves if leaf["section"] == section), "path": f"{section}.{name}", "value": "1"}
    return copy.deepcopy(leaves) + [extra]


@pytest.mark.parametrize("key", SCRIPTS)
def test_every_entry_layout_builds_against_the_live_config(key, leaves):
    """Each entry's layout validates and runs in sections mode with its essentials first."""
    layout = LaunchLayout().build(key, leaves)

    assert layout["mode"] == "sections"
    assert layout["essentials"][0]["path"] == "io.run_name"
    assert all(section["panels"] for section in layout["sections"])


@pytest.mark.parametrize("key", SCRIPTS)
def test_a_new_field_in_a_shown_section_fails_until_a_layout_claims_it(key, leaves):
    """A field added to a section the entry shows makes the layout refuse to build, naming the field."""
    with pytest.raises(LayoutError, match="leaves fields unclaimed: reward.brand_new_weight"):
        LaunchLayout().build(key, _with_extra_field(leaves, "reward", "brand_new_weight"))


def test_a_new_field_in_a_hidden_section_is_accepted(leaves):
    """A field added to a section the entry hides needs no claim."""
    layout = LaunchLayout().build("evaluate", _with_extra_field(leaves, "ppo", "brand_new_coef"))

    assert "ppo" in layout["hidden_sections"]


def test_claiming_a_hidden_section_field_is_refused(leaves, monkeypatch):
    """A layout that shows a field of a section it also hides is refused."""
    spec = copy.deepcopy(LaunchLayout.LAYOUTS["evaluate"])
    spec["sections"][0]["panels"][0]["groups"].append({"title": "Stray", "fields": ["ppo.gamma"]})
    monkeypatch.setitem(LaunchLayout.LAYOUTS, "evaluate", spec)

    with pytest.raises(LayoutError, match="claims fields of hidden sections: ppo.gamma"):
        LaunchLayout().build("evaluate", leaves)


def test_a_field_claimed_twice_and_an_unknown_field_are_both_reported(leaves, monkeypatch):
    """A duplicate claim and a claim of a field that does not exist are reported together."""
    spec = copy.deepcopy(LaunchLayout.LAYOUTS["infer"])
    spec["sections"][0]["panels"][0]["groups"].append({"title": "Stray", "fields": ["inference.greedy", "inference.nope"]})
    monkeypatch.setitem(LaunchLayout.LAYOUTS, "infer", spec)

    with pytest.raises(LayoutError) as caught:
        LaunchLayout().build("infer", leaves)

    assert "names unknown fields: inference.nope" in str(caught.value)
    assert "claims fields twice: inference.greedy" in str(caught.value)


def test_a_number_widget_without_bounds_is_refused(leaves, monkeypatch):
    """A number widget lacking min or max is refused."""
    spec = copy.deepcopy(LaunchLayout.LAYOUTS["infer"])
    spec["essentials"][1] = {"path": "inference.seed", "widget": {"kind": "number", "presets": [1, 2]}}
    monkeypatch.setitem(LaunchLayout.LAYOUTS, "infer", spec)

    with pytest.raises(LayoutError, match="number widget for inference.seed lacks min/max bounds"):
        LaunchLayout().build("infer", leaves)


@pytest.mark.parametrize("key", SCRIPTS)
def test_entries_that_load_a_checkpoint_require_the_run_name(key, leaves):
    """Evaluate, infer and diagnose mark io.run_name as required; the training entries leave it optional."""
    widgets  = LaunchLayout().build(key, leaves)["widgets"]
    required = "io.run_name" in widgets and widgets["io.run_name"].get("required") is True

    assert required is (key in ("evaluate", "infer", "diagnose"))


def test_an_unknown_script_has_no_layout(leaves):
    """A script without a declared layout is refused by name."""
    with pytest.raises(LayoutError, match="no launch layout declared for nope"):
        LaunchLayout().build("nope", leaves)


class FixedLauncher:
    """Launcher stand-in naming the running interpreter as the preferred one."""

    def preferred_interpreter(self, key: str = "") -> str:
        """Returns the interpreter running the tests."""
        return sys.executable


def test_the_config_route_answers_the_layout_and_the_field_descriptions(paths):
    """The catalog config route answers the leaves, the layout and one description per leaf."""
    router   = CatalogRouter(paths, None, ScriptConfigResolver(paths), LaunchLayout(), ConfigRegistry(paths), FixedLauncher())
    handler  = FakeHandler("GET", "/api/scripts/diagnose/config")
    exchange = HttpExchange.of(handler)

    router.script_config(exchange, "diagnose")
    payload = handler.payload()

    assert handler.status == 200 and payload["ok"] is True
    assert payload["layout"]["widgets"]["diagnostics.agents"]["kind"] == "multi"
    assert set(payload["descriptions"]) == {leaf["path"] for leaf in payload["leaves"]}


def test_the_config_route_answers_a_layout_error_as_a_failed_result(paths, monkeypatch):
    """A layout that no longer matches the configuration is answered as a 400 carrying the error."""
    spec = copy.deepcopy(LaunchLayout.LAYOUTS["infer"])
    spec["essentials"].append("inference.nope")
    monkeypatch.setitem(LaunchLayout.LAYOUTS, "infer", spec)

    router   = CatalogRouter(paths, None, ScriptConfigResolver(paths), LaunchLayout(), ConfigRegistry(paths), FixedLauncher())
    handler  = FakeHandler("GET", "/api/scripts/infer/config")
    router.script_config(HttpExchange.of(handler), "infer")

    assert handler.status == 400
    assert "inference.nope" in json.loads(handler.wfile.getvalue())["error"]
