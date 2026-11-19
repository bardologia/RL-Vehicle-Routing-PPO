"""Tests for the copy-from-run reader over evaluation report meta blocks.

Covers the listing of runs that hold an evaluation report, the mapping of the
newest report's environment, reward, caps and evaluation settings onto dotted
config paths rendered as the resolver renders defaults, the withheld run name,
and the failures for a run without a report or with an unreadable one.
"""

from __future__ import annotations

import json
import os

import pytest

from config_value_render import ConfigValueRenderer
from configuration       import Config
from run_config_reader   import RunConfigReader
from web_logger          import WebLogger


def _meta(run_name: str, seed: int, mean_jobs: int = 16) -> dict:
    """Returns an evaluation meta block shaped like the ones evaluate.py writes."""
    config = Config()
    env    = {name: getattr(config.env, name) for name in vars(config.env)}
    reward = {name: getattr(config.reward, name) for name in vars(config.reward)}
    env["center"]    = list(env["center"])
    env["mean_jobs"] = mean_jobs

    return {"run_name": run_name, "checkpoint_filename": "graph_ppo_policy.pt", "seed": seed, "episodes": 10, "greedy": True, "scenario_seed": 42, "reward": reward, "environment": env, "caps": {"max_disruptions_per_episode": 3, "max_edits_per_round": 8}}


def _report(run_dir, seed: int, meta: dict, mtime: float) -> None:
    """Writes one evaluation report with a given modification time."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"evaluation_seed{seed}.json"
    path.write_text(json.dumps({"meta": meta, "agents": {}}))
    os.utime(path, (mtime, mtime))


@pytest.fixture
def reader():
    """Returns a reader with a real logger."""
    return RunConfigReader(WebLogger())


def test_runs_with_a_report_are_listed_with_their_newest_report(reader, tmp_path):
    """Only run directories holding a report are listed, each naming its newest report."""
    _report(tmp_path / "alpha", 1000, _meta("alpha", 1000), 100.0)
    _report(tmp_path / "alpha", 2000, _meta("alpha", 2000), 200.0)
    (tmp_path / "beta").mkdir()

    listed = reader.list_runs(str(tmp_path))

    assert listed["ok"] is True
    assert listed["runs"] == [{"name": "alpha", "path": str(tmp_path / "alpha"), "source": "evaluation_seed2000.json"}]


def test_a_read_maps_the_meta_onto_config_paths_rendered_like_defaults(reader, tmp_path):
    """Environment, reward, caps and evaluation settings land on their dotted paths, rendered like resolved defaults."""
    _report(tmp_path / "alpha", 1000, _meta("alpha", 1000, mean_jobs=20), 100.0)

    result = reader.read(str(tmp_path / "alpha"))
    config = result["config"]

    assert result["ok"] is True and result["source"] == "evaluation_seed1000.json"
    assert config["env.mean_jobs"] == "20"
    assert config["env.center"] == ConfigValueRenderer.render(Config().env.center)
    assert config["reward.distance_weight"] == ConfigValueRenderer.render(Config().reward.distance_weight)
    assert config["training.max_disruptions_per_episode"] == "3"
    assert (config["evaluation.seed"], config["evaluation.episodes"], config["evaluation.greedy"]) == ("1000", "10", "True")
    assert config["io.checkpoint_filename"] == "graph_ppo_policy.pt"


def test_the_run_name_is_withheld_as_identity(reader, tmp_path):
    """The source run's own name is never served as a copyable setting."""
    _report(tmp_path / "alpha", 1000, _meta("alpha", 1000), 100.0)

    result = reader.read(str(tmp_path / "alpha"))

    assert "io.run_name" not in result["config"]
    assert result["identity"] == ["io.run_name"]


def test_a_run_without_a_report_or_with_a_broken_one_fails(reader, tmp_path):
    """A directory without a report, a missing directory and an unreadable report each fail with a reason."""
    (tmp_path / "empty").mkdir()
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "evaluation_seed1.json").write_text("{not json")

    assert "holds no evaluation report" in reader.read(str(tmp_path / "empty"))["error"]
    assert reader.read(str(tmp_path / "missing"))["ok"] is False
    assert "could not read the meta block" in reader.read(str(tmp_path / "broken"))["error"]
    assert reader.list_runs(str(tmp_path / "missing"))["ok"] is False
