"""Tests that every configuration dataclass exposed in the webui is documented.

Walks the parsed configuration package and asserts each section, class and
field carries a description, that no description survives for a class or field
that was removed, that an undescribed field raises KeyError, that every
resolved launch leaf gets its description, and that the text stays plain ASCII
without dashes the console's style forbids.
"""

from __future__ import annotations

import sys

import pytest

from config_registry        import ConfigRegistry
from project_paths          import ProjectPaths
from script_config_resolver import ScriptConfigResolver


@pytest.fixture(scope="module")
def registry():
    """Returns a module-scoped config registry over the real project paths."""
    return ConfigRegistry(ProjectPaths())


@pytest.fixture(scope="module")
def groups(registry):
    """Returns the registry's collected section groups, computed once per module."""
    return registry.collect()


def _classes(groups):
    """Yields ``(group, class)`` pairs for every documented config class."""
    for group in groups:
        for cls in group["classes"]:
            yield group, cls


def test_collect_covers_every_configuration_module(groups):
    """Checks every module of the configuration package with a dataclass becomes a section."""
    assert [group["module"] for group in groups] == ConfigRegistry.SECTION_ORDER


def test_every_section_class_and_field_is_described(groups):
    """Checks each section, class and field carries a non-empty description."""
    for group, cls in _classes(groups):
        assert group["desc"], f"section {group['module']} is missing a description"
        assert cls["desc"], f"{cls['module']}::{cls['name']} is missing a summary"
        for field in cls["fields"]:
            assert field["desc"], f"{cls['module']}::{cls['name']}.{field['name']} has no description"


def test_no_description_outlives_its_class_or_field(registry, groups):
    """Checks the descriptions file holds nothing for a class or field that no longer exists."""
    parsed = {f"{cls['module']}::{cls['name']}": {field["name"] for field in cls["fields"]} for _, cls in _classes(groups)}

    assert set(registry.descriptions["configs"]) == set(parsed)
    for key, entry in registry.descriptions["configs"].items():
        assert set(entry["fields"]) == parsed[key], key
    assert set(registry.descriptions["sections"]) == {group["module"] for group in groups}


def test_an_undescribed_field_raises(registry):
    """Checks a class entry missing one field description raises KeyError, which is how new settings are caught."""
    cls = {"module": "reward", "name": "RewardConfig", "fields": [{"name": "brand_new_weight"}]}

    with pytest.raises(KeyError):
        registry._attach_descriptions(cls)


def test_every_resolved_leaf_gets_its_description(registry):
    """Checks the launch form receives one description per resolved leaf."""
    leaves       = ScriptConfigResolver(ProjectPaths()).resolve("train", sys.executable)["leaves"]
    descriptions = registry.leaf_descriptions(leaves)

    assert set(descriptions) == {leaf["path"] for leaf in leaves}
    assert descriptions["training.max_edits_per_round"].startswith("Most edits the agent may make")


def test_descriptions_are_plain_ascii_without_dashes(registry):
    """Checks the descriptions are ASCII and carry no em dash or double hyphen."""
    texts = list(registry.descriptions["sections"].values())
    for entry in registry.descriptions["configs"].values():
        texts.append(entry["summary"])
        texts.extend(entry["fields"].values())

    for text in texts:
        assert text.isascii(), text
        assert "--" not in text, text
