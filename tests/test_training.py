import os

import pytest

from core.dataset import Dataset, generate_events
from core.shared import Environment
from core.training import EpisodeRunner, RunDirectory, Trainer
from model.policy_model import Policy
from tools.inspection import TensorLogger
from tools.logger import NullLogger
from tools.telemetry import PPOTelemetry
from tools.tracker import NullTracker


EXPERIENCE_KEYS = {
    "graph",
    "mask_info",
    "reward",
    "action",
    "log_prob_operator",
    "log_prob_vehicle",
    "log_prob_job",
    "state_value",
    "old_operator_logits",
    "old_vehicle_logits",
    "old_job_logits",
    "bootstrap_value",
    "done",
}


def build_runner(cpu_config, tmp_path):
    environment   = Environment(cpu_config)
    policy        = Policy(cpu_config)
    telemetry     = PPOTelemetry(NullTracker(), cpu_config)
    tensor_logger = TensorLogger(policy).attach()

    runner = EpisodeRunner(
        environment   = environment,
        policy        = policy,
        telemetry     = telemetry,
        tensor_logger = tensor_logger,
        logger        = NullLogger(),
        config        = cpu_config,
    )

    return runner, tensor_logger


def test_episode_runner_produces_experiences_and_operator_stats(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 3

    items, _     = generate_events(batch_size=1, seed=7, config=cpu_config)
    dataset_item = items[0]

    runner, tensor_logger  = build_runner(cpu_config, tmp_path)
    experiences, op_stats  = runner.run(dataset_item, global_step_counter=0)

    assert len(experiences) == 3
    assert set(experiences[0].keys()) == EXPERIENCE_KEYS
    assert experiences[-1]["done"] is True
    assert all(experience["done"] is False for experience in experiences[:-1])
    assert sum(op_stats["count"].values()) == 3


def test_episode_runner_detaches_tensor_logger_after_first_step(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 2

    items, _     = generate_events(batch_size=1, seed=3, config=cpu_config)
    dataset_item = items[0]

    runner, tensor_logger = build_runner(cpu_config, tmp_path)
    runner.run(dataset_item, global_step_counter=0)

    assert runner.attached is False
    assert tensor_logger.hooks == []
    assert (tmp_path / "tensor_shape.md").exists()


def test_run_directory_creates_fresh_run_with_active_tracker(cpu_config, tmp_path):
    session = RunDirectory(cpu_config, str(tmp_path)).prepare()

    assert os.path.isdir(session.path)
    assert cpu_config.io.logdir == session.path
    assert session.tracker.active is True

    session.writer.close()


def test_run_directory_resume_missing_run_raises(cpu_config, tmp_path):
    cpu_config.io.resume_from_run = "ghost_run"

    with pytest.raises(FileNotFoundError):
        RunDirectory(cpu_config, str(tmp_path)).prepare()


def test_run_directory_resume_reuses_existing_run(cpu_config, tmp_path):
    prior = tmp_path / "run_prior"
    prior.mkdir()

    cpu_config.io.resume_from_run = "run_prior"
    session = RunDirectory(cpu_config, str(tmp_path)).prepare()

    assert session.path == str(prior)

    session.writer.close()


def test_trainer_resume_restores_counters_from_checkpoint(cpu_config, seeded, fake_vroom, tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    cpu_config.io.logdir = str(run_dir)

    dataset = Dataset(dataset_dir=str(tmp_path / "data"), config=cpu_config)

    first = Trainer(dataset=dataset, config=cpu_config, logger=NullLogger(), tracker=NullTracker())
    first.global_step_counter = 123
    first.episode_index       = 45
    first.ppo_update_index    = 7
    first.ppo.lr_scheduler.set_step(11)
    first.ppo.entropy_scheduler.set_step(13)
    first.checkpoint.save(first.ppo, first)

    cpu_config.io.resume_from_run = "run_x"
    second = Trainer(dataset=dataset, config=cpu_config, logger=NullLogger(), tracker=NullTracker())

    assert second.resume is True
    assert second.global_step_counter == 123
    assert second.episode_index == 45
    assert second.ppo_update_index == 7
    assert second.ppo.lr_scheduler.current_step == 11
    assert second.ppo.entropy_scheduler.current_step == 13
