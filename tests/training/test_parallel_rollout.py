import torch

from core.dataset import ChunkStore, generate_events
from core.training import ParallelRolloutCollector
from core.training.pretraining import _teacher_worker_init, _teacher_worker_task
from tools.parallel import Batcher, ForkPool


def test_batcher_splits_into_bounded_chunks():
    batches = Batcher(3).split(list(range(7)))

    assert batches == [[0, 1, 2], [3, 4, 5], [6]]


def test_forkpool_serial_runs_initializer_then_worker():
    state = {}

    def initializer(offset):
        state["offset"] = offset

    def worker(value):
        return value + state["offset"]

    results = ForkPool(1).map(worker, [1, 2, 3], initializer=initializer, initargs=(10,))

    assert results == [11, 12, 13]


def test_parallel_collector_serial_matches_episode_shapes(cpu_config, seeded, fake_vroom):
    from model.policy_model import Policy

    cpu_config.training.max_steps_per_episode = 3
    cpu_config.training.rollout_batch_size    = 2

    items, _  = generate_events(batch_size=4, seed=5, config=cpu_config)
    policy    = Policy(cpu_config)
    collector = ParallelRolloutCollector(cpu_config)

    episodes = collector.collect(items, policy)

    assert len(episodes) == 4
    for experiences, operator_stats, step_payloads in episodes:
        assert len(experiences) == 3
        assert len(step_payloads) == 3
        assert experiences[-1]["done"] is True
        assert sum(operator_stats["count"].values()) == 3


def test_teacher_worker_task_returns_episode_records(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 2

    items, _ = generate_events(batch_size=2, seed=6, config=cpu_config)

    _teacher_worker_init(cpu_config)
    episodes = _teacher_worker_task((0, items))

    assert len(episodes) == 2
    for episode in episodes:
        assert len(episode["records"]) == 2
        assert set(episode["operator_counts"].keys()) == {0, 1, 2}
        assert isinstance(episode["episode_reward"], float)
        assert all("return" in record for record in episode["records"])
