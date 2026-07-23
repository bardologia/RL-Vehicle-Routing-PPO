from core.dataset import generate_events
from core.shared import Environment
from core.training import EpisodeRunner
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


def build_runner(cpu_config):
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

    runner, _             = build_runner(cpu_config)
    experiences, op_stats = runner.run(dataset_item, global_step_counter=0)

    assert len(experiences) == 3
    assert set(experiences[0].keys()) == EXPERIENCE_KEYS
    assert experiences[-1]["done"] is True
    assert all(experience["done"] is False for experience in experiences[:-1])
    assert sum(op_stats["count"].values()) == 3


def test_episode_runner_bootstrap_sets_tail_value_on_last(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 2

    items, _     = generate_events(batch_size=1, seed=4, config=cpu_config)
    runner, _    = build_runner(cpu_config)

    experiences, _ = runner.run(items[0], global_step_counter=0)

    assert isinstance(experiences[-1]["bootstrap_value"], float)
    assert experiences[0]["bootstrap_value"] == 0.0


def test_episode_runner_operator_stats_track_rewards_per_operator(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 4

    items, _  = generate_events(batch_size=1, seed=1, config=cpu_config)
    runner, _ = build_runner(cpu_config)

    experiences, op_stats = runner.run(items[0], global_step_counter=0)

    recorded_rewards = sum(len(values) for values in op_stats["rewards"].values())

    assert recorded_rewards == len(experiences)
    assert set(op_stats["count"].keys()) == {0, 1, 2, 3}


def test_episode_runner_detaches_tensor_logger_after_first_step(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir                      = str(tmp_path)
    cpu_config.training.max_steps_per_episode = 2

    items, _     = generate_events(batch_size=1, seed=3, config=cpu_config)
    dataset_item = items[0]

    runner, tensor_logger = build_runner(cpu_config)
    runner.run(dataset_item, global_step_counter=0)

    assert runner.attached is False
    assert tensor_logger.hooks == []
    assert (tmp_path / "tensor_shape.md").exists()
