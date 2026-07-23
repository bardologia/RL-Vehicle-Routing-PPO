import os

import pytest

from core.training import RunDirectory, Trainer
from tools.logger import NullLogger
from tools.tracker import NullTracker


def test_run_directory_creates_fresh_run_with_active_tracker(cpu_config, tmp_path):
    session = RunDirectory(cpu_config, str(tmp_path)).prepare()

    assert os.path.isdir(session.path)
    assert cpu_config.io.logdir == session.path
    assert session.tracker.active is True

    session.writer.close()


def test_run_directory_uses_configured_run_name(cpu_config, tmp_path):
    cpu_config.io.run_name = "named_run"

    session = RunDirectory(cpu_config, str(tmp_path)).prepare()

    assert session.path == os.path.join(str(tmp_path), "named_run")

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


def test_trainer_state_snapshot_carries_all_counters(cpu_config, seeded, fake_vroom, tmp_path):
    run_dir = tmp_path / "run_s"
    run_dir.mkdir()
    cpu_config.io.logdir = str(run_dir)

    trainer = Trainer(config=cpu_config, logger=NullLogger(), tracker=NullTracker())

    trainer.global_step_counter = 12
    trainer.episode_index       = 3
    trainer.ppo_update_index    = 1

    state = trainer.state()

    assert state["global_step_counter"] == 12
    assert state["episode_index"] == 3
    assert state["ppo_update_index"] == 1
    assert "lr_scheduler_step" in state
    assert "entropy_scheduler_step" in state


def test_trainer_resume_restores_counters_from_checkpoint(cpu_config, seeded, fake_vroom, tmp_path):
    run_dir = tmp_path / "run_x"
    run_dir.mkdir()
    cpu_config.io.logdir = str(run_dir)

    first = Trainer(config=cpu_config, logger=NullLogger(), tracker=NullTracker())
    first.global_step_counter = 123
    first.episode_index       = 45
    first.ppo_update_index    = 7
    first.ppo.lr_scheduler.set_step(11)
    first.ppo.entropy_scheduler.set_step(13)
    first.checkpoint.save(first.ppo, first)

    cpu_config.io.resume_from_run = "run_x"
    second = Trainer(config=cpu_config, logger=NullLogger(), tracker=NullTracker())

    assert second.resume is True
    assert second.global_step_counter == 123
    assert second.episode_index == 45
    assert second.ppo_update_index == 7
    assert second.ppo.lr_scheduler.current_step == 11
    assert second.ppo.entropy_scheduler.current_step == 13
