import json
import os

from core.inference import EpisodeEvaluator, EvaluationPipeline, FixedOperatorAgent
from core.shared import Environment
from model.policy_model import Policy, PolicyCheckpoint
from tools.logger import NullLogger


def test_fixed_operator_agent_emits_constant_action():
    agent  = FixedOperatorAgent(2)
    action = agent.act(None, None, None, 1)

    assert action.operator == 2
    assert action.vehicle_index == 0
    assert action.job_index == 0


def test_episode_evaluator_is_deterministic_per_seed(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 3
    cpu_config.env.step_event_probability     = 1.0

    environment = Environment(cpu_config)
    evaluator   = EpisodeEvaluator(environment, cpu_config)
    agent       = FixedOperatorAgent(2)

    first  = evaluator.run(agent, episode_seed=99)
    second = evaluator.run(agent, episode_seed=99)

    assert first == second


def build_evaluation_setup(cpu_config, tmp_path):
    run_dir = tmp_path / "runs" / "eval_run"
    PolicyCheckpoint().save(Policy(cpu_config), cpu_config.io.checkpoint_filename, str(run_dir), training_state=None)

    cpu_config.io.runs_dir                    = str(tmp_path / "runs")
    cpu_config.io.run_name                    = "eval_run"
    cpu_config.evaluation.episodes            = 2
    cpu_config.training.max_steps_per_episode = 2
    cpu_config.env.step_event_probability     = 0.0


def test_evaluation_pipeline_reports_all_baselines(cpu_config, seeded, fake_vroom, tmp_path):
    build_evaluation_setup(cpu_config, tmp_path)

    pipeline = EvaluationPipeline(cpu_config, repo_root=str(tmp_path), logger=NullLogger())
    results  = pipeline.run()

    assert set(results.keys()) == {"model", "teacher", "insertion_only", "do_nothing"}

    for metrics in results.values():
        assert metrics["episodes"] == 2
        assert set(metrics["operator_frequency"].keys()) == {"op0", "op1", "op2"}

    assert results["insertion_only"]["operator_frequency"]["op1"] == 0.0
    assert results["do_nothing"]["operator_frequency"]["op2"] == 1.0

    report_path = os.path.join(str(tmp_path / "runs" / "eval_run"), "evaluation.json")
    with open(report_path) as handle:
        saved = json.load(handle)

    assert set(saved.keys()) == set(results.keys())
