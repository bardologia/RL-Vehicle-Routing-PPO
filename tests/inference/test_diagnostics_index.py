import json

import pytest

import core.shared.services as services_module
from core.inference import DiagnosticsPipeline, TraceFile
from model.policy_model import Policy, PolicyCheckpoint
from tools.logger import NullLogger


def prepare(cpu_config, tmp_path):
    run_dir = tmp_path / "runs" / "diag_run"
    PolicyCheckpoint().save(Policy(cpu_config), cpu_config.io.checkpoint_filename, str(run_dir), training_state=None)

    cpu_config.io.runs_dir                          = str(tmp_path / "runs")
    cpu_config.io.run_name                          = "diag_run"
    cpu_config.diagnostics.episodes                 = 2
    cpu_config.diagnostics.sampled_twin             = True
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 2

    return run_dir


def test_pipeline_writes_traces_and_an_index(cpu_config, seeded, fake_vroom, monkeypatch, tmp_path):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)
    run_dir = prepare(cpu_config, tmp_path)

    index   = DiagnosticsPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger()).run()
    out_dir = run_dir / "diagnostics" / f"seed{cpu_config.diagnostics.seed}"

    assert json.loads((out_dir / "index.json").read_text()) == json.loads(json.dumps(index))
    assert index["coverage"] == {key: {"done": 2, "total": 2} for key in ("model", "insertion_only", "do_nothing", "model_sampled")}
    assert set(index["meta"]) == {"run_name", "seed", "episodes", "greedy", "teacher_verdict", "reward", "environment", "caps"}

    for episode in index["episodes"]:
        assert episode["seed"] == cpu_config.diagnostics.seed + episode["index"]
        assert episode["jobs"] > 0 and episode["vehicles"] > 0
        assert len(episode["events"]) == 2

        for key, digest in episode["agents"].items():
            trace = TraceFile().read(out_dir / f"ep{episode['index']}_{key}.json.gz")

            assert trace["summary"]["total_reward"] == pytest.approx(digest["total_reward"])
            assert len(digest["operator_by_edit"]) == 2
            assert sum(sum(row) for row in digest["operator_by_edit"]) == digest["decisions"]
            assert (digest["profitable_left"] is not None) == (key == "model")

    assert index["decisions"]
    assert all(row["match"] is not None for row in index["decisions"])
    assert index["aggregates"]["critic"]["n"] == len(index["decisions"])
    assert index["aggregates"]["agreement"]["n"] == len(index["decisions"])
    assert sum(sum(row) for row in index["aggregates"]["agreement"]["confusion"]) == len(index["decisions"])
    assert set(index["aggregates"]["pickup_failure_rate"]) == set(index["coverage"])


def test_pipeline_takes_the_episode_settings_from_the_report(cpu_config, seeded, fake_vroom, monkeypatch, tmp_path):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)
    run_dir = prepare(cpu_config, tmp_path)

    cpu_config.diagnostics.sampled_twin = False
    cpu_config.diagnostics.episodes     = 1
    cpu_config.training.max_disruptions_per_episode = 5

    meta = {"environment": {"min_jobs": 6}, "reward": {"distance_weight": 1.5}, "caps": {"max_disruptions_per_episode": 3, "max_edits_per_round": 2}}
    (run_dir / f"evaluation_seed{cpu_config.diagnostics.seed}.json").write_text(json.dumps({"meta": meta, "agents": {}}))

    index = DiagnosticsPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger()).run()

    assert index["meta"]["caps"] == {"rounds": 3, "edits": 2}
    assert index["meta"]["environment"]["min_jobs"] == 6


def test_pipeline_rejects_overrides_that_contradict_the_report(cpu_config, seeded, fake_vroom, monkeypatch, tmp_path):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)
    run_dir = prepare(cpu_config, tmp_path)

    meta = {"environment": {"min_jobs": 6}, "reward": {}, "caps": {"max_disruptions_per_episode": 3, "max_edits_per_round": 8}}
    (run_dir / f"evaluation_seed{cpu_config.diagnostics.seed}.json").write_text(json.dumps({"meta": meta, "agents": {}}))

    with pytest.raises(ValueError, match="contradicts"):
        DiagnosticsPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger()).run()


def test_pipeline_rejects_unknown_agents_and_verdict_scopes(cpu_config, tmp_path):
    prepare(cpu_config, tmp_path)

    cpu_config.diagnostics.agents = ("model", "oracle")
    with pytest.raises(ValueError, match="unknown agents"):
        DiagnosticsPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger()).run()

    cpu_config.diagnostics.agents          = ("model",)
    cpu_config.diagnostics.teacher_verdict = "everything"
    with pytest.raises(ValueError, match="teacher_verdict"):
        DiagnosticsPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger()).run()
