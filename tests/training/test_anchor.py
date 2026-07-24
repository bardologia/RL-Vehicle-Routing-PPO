import math

import pytest
import torch

from core.shared import ActionMasker
from core.training import ActionDistribution, Trainer
from model.policy_model import Policy, PolicyCheckpoint
from tools.logger import NullLogger
from tools.tracker import Tracker
from tests.conftest import FakeWriter


MASK_INFO = {
    "unassigned_job_indices"         : [2, 5, 11],
    "vehicles_with_jobs_indices"     : [0, 2, 4],
    "vehicle_to_job_indices"         : {0: [1, 3], 1: [], 2: [7, 8, 9], 3: [], 4: [0], 5: []},
    "vehicles_with_capacity_indices" : [1, 3, 5],
}


def save_perturbed_policy(cpu_config, runs_dir, run_name):
    source = Policy(cpu_config)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(0.5)

    PolicyCheckpoint().save(source, cpu_config.io.checkpoint_filename, str(runs_dir / run_name), training_state={"phase": "pretraining"})
    return source


def build_anchored_trainer(cpu_config, tmp_path, tracker):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    save_perturbed_policy(cpu_config, tmp_path / "runs", "pre")

    cpu_config.io.logdir                      = str(run_dir)
    cpu_config.io.runs_dir                    = str(tmp_path / "runs")
    cpu_config.io.init_from_run               = "pre"
    cpu_config.training.max_steps_per_episode = 2
    cpu_config.training.minibatch_size        = 4
    cpu_config.training.num_epochs            = 2
    cpu_config.ppo.kl_divergence_threshold    = 100.0

    return Trainer(config=cpu_config, logger=NullLogger(), tracker=tracker)


def test_kl_terms_matches_compute_totals(cpu_config, seeded):
    distribution = ActionDistribution(cpu_config, ActionMasker(cpu_config))

    old = (torch.randn(3), torch.randn(3, 6), torch.randn(3, 6, 20))
    new = (torch.randn(3, requires_grad=True), torch.randn(3, 6, requires_grad=True), torch.randn(3, 6, 20, requires_grad=True))

    tensors  = distribution.kl_terms(*old, *new, MASK_INFO)
    _, kl    = distribution.compute(*old, *new, MASK_INFO)

    assert float(tensors["total_kl"].item()) == pytest.approx(kl["total_kl"], abs=1e-6)
    assert tensors["total_kl"].requires_grad is True


def test_init_from_run_builds_frozen_reference(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_anchored_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))

    reference = trainer.ppo.reference_policy

    assert reference is not None
    assert all(parameter.requires_grad is False for parameter in reference.parameters())
    for policy_parameter, reference_parameter in zip(trainer.ppo.policy.parameters(), reference.parameters()):
        assert torch.equal(policy_parameter, reference_parameter)

    assert trainer.ppo.current_anchor_coef == cpu_config.ppo.anchor_kl_start


def test_no_reference_without_init_from_run(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir = str(tmp_path)

    trainer = Trainer(config=cpu_config, logger=NullLogger(), tracker=Tracker(writer=FakeWriter()))

    assert trainer.ppo.reference_policy is None


def test_anchored_update_logs_kl_and_stays_finite(cpu_config, seeded, fake_vroom, tmp_path):
    writer  = FakeWriter()
    trainer = build_anchored_trainer(cpu_config, tmp_path, Tracker(writer=writer))

    trainer._run_update([0, 1, 2])
    trainer.ppo.update()

    anchor_kls = [value for tag, value, _ in writer.scalars if tag == "batch/anchor_kl"]

    assert len(anchor_kls) > 0
    assert all(math.isfinite(value) for value in anchor_kls)
    assert all(torch.isfinite(parameter).all() for parameter in trainer.ppo.policy.parameters())


def test_anchor_survives_checkpoint_roundtrip(cpu_config, seeded, fake_vroom, tmp_path):
    trainer = build_anchored_trainer(cpu_config, tmp_path, Tracker(writer=FakeWriter()))

    trainer.ppo.anchor_scheduler.set_step(17)
    trainer.checkpoint.save(trainer.ppo, trainer)

    cpu_config.io.init_from_run   = None
    cpu_config.io.resume_from_run = "run"

    resumed = Trainer(config=cpu_config, logger=NullLogger(), tracker=Tracker(writer=FakeWriter()))

    assert resumed.ppo.reference_policy is not None
    assert resumed.ppo.anchor_scheduler.current_step == 17
    for original, restored in zip(trainer.ppo.reference_policy.parameters(), resumed.ppo.reference_policy.parameters()):
        assert torch.equal(original, restored)
