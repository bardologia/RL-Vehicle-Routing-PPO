from types import SimpleNamespace

import torch

from tools.telemetry import PPOTelemetry
from tools.tracker import NullTracker
from tests.conftest import RecordingTracker


REWARDS = {"distance_reward": 0.5, "action_reward": 0.5}
COSTS   = {"old_distance_cost": 1.0, "new_distance_cost": 0.5}

OPERATOR_STATS = {
    "count"   : {0: 2, 1: 1, 2: 0, 3: 1},
    "rewards" : {0: [1.0, 2.0], 1: [0.5], 2: [], 3: [-1.0]},
}


def build(config):
    tracker = RecordingTracker()
    return tracker, PPOTelemetry(tracker, config)


def test_step_emits_on_multiple_of_step_every(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.step(REWARDS, COSTS, state_value=0.3, step=cpu_config.telemetry.step_every)

    assert "step/reward" in tracker.metric_prefixes()
    assert "step/cost" in tracker.metric_prefixes()
    assert "step/state_value" in tracker.scalar_tags()


def test_step_silent_off_cadence(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.step(REWARDS, COSTS, state_value=0.3, step=cpu_config.telemetry.step_every + 1)

    assert tracker.metrics == []
    assert tracker.scalars == []


def test_episode_emits_on_multiple_of_episode_every(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.episode(reward=5.0, length=3, operator_stats=OPERATOR_STATS, episode_index=cpu_config.telemetry.episode_every)

    assert "episode/total_reward" in tracker.scalar_tags()
    assert "episode/operator_frequency" in tracker.metric_prefixes()


def test_episode_silent_off_cadence(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.episode(reward=5.0, length=3, operator_stats=OPERATOR_STATS, episode_index=cpu_config.telemetry.episode_every + 1)

    assert tracker.scalars == []
    assert tracker.metrics == []


def test_sample_silent_off_cadence(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.sample(
        sample_step      = cpu_config.telemetry.sample_every + 1,
        advantage        = None,
        target_return    = None,
        old_log_probs    = None,
        new_log_probs    = None,
        distributions    = None,
        policy_loss_dict = None,
        value_loss_dict  = None,
        entropy_dict     = None,
        kl_dict          = None,
        entropy_loss     = None,
        total_loss       = None,
    )

    assert tracker.scalars == []
    assert tracker.metrics == []


def test_null_tracker_suppresses_gated_emitters(cpu_config):
    telemetry = PPOTelemetry(NullTracker(), cpu_config)

    telemetry.step(REWARDS, COSTS, state_value=0.3, step=cpu_config.telemetry.step_every)
    telemetry.episode(reward=1.0, length=1, operator_stats=OPERATOR_STATS, episode_index=cpu_config.telemetry.episode_every)


def test_ungated_emitters_always_record(cpu_config):
    tracker, telemetry = build(cpu_config)

    telemetry.batch(mean_loss=0.5, mean_kl=0.01, batch_step=3)
    telemetry.epoch(mean_loss=0.4, mean_kl=0.02, epoch_step=1)
    telemetry.entropy_coefficient(0.01, step=3)
    telemetry.buffer_size(64, update_step=1)

    tags = tracker.scalar_tags()
    assert "batch/mean_loss" in tags
    assert "batch/mean_kl" in tags
    assert "epoch/mean_loss" in tags
    assert "batch/entropy_coefficient" in tags
    assert "batch/buffer_size" in tags


def test_learning_rates_guarded_by_active_flag(cpu_config):
    tracker, telemetry = build(cpu_config)
    optimizer          = SimpleNamespace(param_groups=[{"name": "critic", "lr": 0.001}])

    telemetry.learning_rates(optimizer, step=1)

    assert "batch/learning_rate" in tracker.metric_prefixes()


def test_baseline_reports_explained_variance(cpu_config):
    tracker, telemetry = build(cpu_config)

    batch_data = {
        "values"     : torch.tensor([0.0, 1.0, 2.0]),
        "returns"    : torch.tensor([0.1, 1.1, 1.9]),
        "advantages" : torch.tensor([0.1, 0.1, -0.1]),
        "rewards"    : torch.tensor([1.0, 1.0, 1.0]),
    }

    telemetry.baseline(batch_data, update_step=0)

    assert "batch/baseline_stats" in tracker.metric_prefixes()
