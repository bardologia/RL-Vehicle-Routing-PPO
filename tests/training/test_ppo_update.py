import math
import os

import pytest
import torch

from core.training import Trainer
from tools.logger import NullLogger
from tools.tracker import Tracker
from tests.conftest import FakeWriter


def build_trainer(cpu_config, tmp_path, tracker):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cpu_config.io.logdir                      = str(run_dir)
    cpu_config.training.max_steps_per_episode = 2
    cpu_config.training.minibatch_size        = 4
    cpu_config.training.num_epochs            = 2
    cpu_config.ppo.kl_divergence_threshold    = 100.0

    return Trainer(config=cpu_config, logger=NullLogger(), tracker=tracker)


def populate_memory(trainer):
    trainer._run_update([0, 1, 2])


def test_real_update_fills_then_clears_memory(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))

    populate_memory(trainer)

    assert len(trainer.ppo.memory.rewards) == 6

    trainer.ppo.update()

    assert len(trainer.ppo.memory.rewards) == 0
    assert len(trainer.ppo.memory.graphs) == 0


def test_real_update_changes_parameters_and_stays_finite(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))
    populate_memory(trainer)

    before = [parameter.detach().clone() for parameter in trainer.ppo.policy.parameters()]

    trainer.ppo.update()

    after   = list(trainer.ppo.policy.parameters())
    changed = any(not torch.equal(b, a) for b, a in zip(before, after))
    finite  = all(torch.isfinite(a).all() for a in after)

    assert changed is True
    assert finite is True


def test_real_update_reports_finite_loss(cpu_config, seeded, fake_vroom, tmp_path):
    writer  = FakeWriter()
    trainer = build_trainer(cpu_config, tmp_path, Tracker(writer=writer))
    populate_memory(trainer)

    trainer.ppo.update()

    losses = [value for tag, value, _ in writer.scalars if tag == "batch/mean_loss"]

    assert len(losses) > 0
    assert all(math.isfinite(value) for value in losses)


def test_real_update_advances_global_steps(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))
    populate_memory(trainer)

    before_update = trainer.ppo.global_update_step
    trainer.ppo.update()

    assert trainer.ppo.global_update_step == before_update + 1
    assert trainer.ppo.global_epoch_step >= 1


def test_checkpoint_written_then_resumed_restores_counters(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))
    populate_memory(trainer)
    trainer.ppo.update()

    trainer.global_step_counter = 99
    trainer.episode_index       = 8
    trainer.ppo_update_index    = 2
    trainer.checkpoint.save(trainer.ppo, trainer)

    checkpoint_path = os.path.join(cpu_config.io.logdir, cpu_config.io.checkpoint_filename)
    assert os.path.exists(checkpoint_path)

    cpu_config.io.resume_from_run = os.path.basename(cpu_config.io.logdir)
    resumed = Trainer(config=cpu_config, logger=NullLogger(), tracker=Tracker(writer=FakeWriter()))

    assert resumed.global_step_counter == 99
    assert resumed.episode_index == 8
    assert resumed.ppo_update_index == 2
