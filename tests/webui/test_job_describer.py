"""Tests for the one-line job descriptions of the routing entry points.

Covers the per-script detail fields over resolved defaults and overrides, the
start lead of a PPO run, the silent unset fields, the trailing extras with
their cap, and the description of an unknown script from its overrides alone.
"""

from __future__ import annotations

import pytest

from job_describer import JobDescriber


class StubResolver:
    """Resolver stand-in serving fixed leaves.

    Attributes:
        leaves : Dotted path to rendered default served for every script.
    """

    def __init__(self, leaves: dict) -> None:
        """Stores the leaves served as resolved defaults."""
        self.leaves = leaves

    def resolve(self, key: str, interpreter: str) -> dict:
        """Returns the leaves as a successful resolution."""
        return {"ok": True, "leaves": [{"path": path, "value": value} for path, value in self.leaves.items()]}


class StubPaths:
    """Path registry stand-in knowing the five routing scripts."""

    def has_script(self, key: str) -> bool:
        """Returns whether the key is a routing entry point."""
        return key in ("pretrain", "train", "evaluate", "infer", "diagnose")


DEFAULTS = {
    "io.run_name"                  : "",
    "io.init_from_run"             : "",
    "io.resume_from_run"           : "",
    "pretrain.episodes"            : "2000",
    "pretrain.bc_epochs"           : "4",
    "training.num_updates"         : "16",
    "training.episodes_per_update" : "1024",
    "evaluation.episodes"          : "200",
    "evaluation.seed"              : "1000000",
    "evaluation.greedy"            : "True",
    "inference.seed"               : "2000000",
    "inference.greedy"             : "True",
    "diagnostics.seed"             : "1000000",
    "diagnostics.episodes"         : "10",
    "diagnostics.agents"           : "model, insertion_only, do_nothing",
    "diagnostics.teacher_verdict"  : "insertions",
}


@pytest.fixture
def describer():
    """Returns a describer over the routing defaults."""
    return JobDescriber(StubPaths(), StubResolver(DEFAULTS))


def test_a_train_launch_names_where_it_starts(describer):
    """A PPO launch leads with its init or resume run, or from scratch."""
    assert describer.describe("train", "py", {"io.run_name": "remake_ppo", "io.init_from_run": "remake_pretrain"}) == "init remake_pretrain · remake_ppo · updates 16 · episodes/update 1024"
    assert describer.describe("train", "py", {"io.resume_from_run": "remake_ppo"}).startswith("resume remake_ppo · updates 16")
    assert describer.describe("train", "py", {}).startswith("from scratch · updates 16")


def test_each_script_surfaces_its_own_fields(describer):
    """Every entry point names the fields that tell its launches apart, silent where unset."""
    assert describer.describe("pretrain", "py", {"io.run_name": "bc"}) == "bc · episodes 2000 · BC epochs 4"
    assert describer.describe("evaluate", "py", {"io.run_name": "remake_ppo"}) == "run remake_ppo · episodes 200 · seed 1000000"
    assert describer.describe("infer", "py", {"io.run_name": "remake_ppo", "inference.greedy": "False"}) == "run remake_ppo · seed 2000000 · sampled"
    assert describer.describe("diagnose", "py", {"io.run_name": "r"}) == "run r · seed 1000000 · episodes 10 · agents model, insertion_only, do_nothing · verdict insertions"


def test_overrides_no_spec_covers_trail_as_capped_extras(describer):
    """Overrides outside the spec trail as path=value, at most three plus a count."""
    overrides = {"training.device": "cpu", "env.radius": "10.0", "reward.distance_weight": "2.0", "ppo.gamma": "0.9", "lr.lr_min": "1e-6"}

    text = describer.describe("infer", "py", overrides)

    assert text.endswith("training.device=cpu · env.radius=10.0 · reward.distance_weight=2.0 · +2 more overrides")


def test_an_unknown_script_is_described_from_its_overrides(describer):
    """A script the registry does not know is described by its overrides alone."""
    assert describer.describe("adopted_tool", "py", {"x.y": "1"}) == "x.y=1"
