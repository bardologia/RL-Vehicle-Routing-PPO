from core.shared import Environment
from core.training import EpisodeRollout
from model.policy_model import Policy


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


def build_rollout(cpu_config):
    environment = Environment(cpu_config)
    policy      = Policy(cpu_config)
    return EpisodeRollout(environment, policy, cpu_config)


def test_rollout_produces_experiences_stats_and_payloads(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 3

    rollout = build_rollout(cpu_config)

    experiences, op_stats, step_payloads = rollout.rollout(0)

    assert len(experiences) == 3
    assert len(step_payloads) == 3
    assert set(experiences[0].keys()) == EXPERIENCE_KEYS
    assert experiences[-1]["done"] is True
    assert all(experience["done"] is False for experience in experiences[:-1])
    assert sum(op_stats["count"].values()) == 3

    rewards, costs, value = step_payloads[0]
    assert isinstance(value, float)
    assert "distance_reward" in rewards


def test_rollout_bootstrap_sets_tail_value_on_last(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 2

    rollout = build_rollout(cpu_config)

    experiences, _, _ = rollout.rollout(0)

    assert isinstance(experiences[-1]["bootstrap_value"], float)
    assert experiences[0]["bootstrap_value"] == 0.0


def test_rollout_operator_stats_track_rewards_per_operator(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 4

    rollout = build_rollout(cpu_config)

    experiences, op_stats, _ = rollout.rollout(0)

    recorded_rewards = sum(len(values) for values in op_stats["rewards"].values())

    assert recorded_rewards == len(experiences)
    assert set(op_stats["count"].keys()) == {0, 1, 2}


def test_rollout_applies_events_between_steps_when_forced(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 3
    cpu_config.env.step_event_probability     = 1.0
    cpu_config.env.tick_seconds               = 0

    rollout = build_rollout(cpu_config)

    calls = []
    rollout.environment.generate_event = lambda: calls.append(1) or ("new_job", 1)

    rollout.rollout(0)

    assert len(calls) == cpu_config.training.max_steps_per_episode


def test_rollout_applies_no_events_at_zero_probability(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 3
    cpu_config.env.step_event_probability     = 0.0

    rollout = build_rollout(cpu_config)

    setup_generate = rollout.environment.generate_event
    calls          = []
    rollout.environment.generate_event = lambda: calls.append(1) or setup_generate()

    rollout.rollout(0)

    assert len(calls) == 1
