import pytest

from core.training.pretraining import _teacher_worker_init, _teacher_worker_task
from tools.parallel import Batcher, ForkPool


def _double(value):
    return value * 2


def _explode(value):
    raise ValueError("worker blew up")


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


def test_forkpool_matches_the_serial_run_element_for_element():
    results = ForkPool(2).map(_double, [1, 2, 3, 4, 5])

    assert results == ForkPool(1).map(_double, [1, 2, 3, 4, 5])
    assert results == [2, 4, 6, 8, 10]


def test_forkpool_propagates_a_worker_failure():
    with pytest.raises(ValueError, match="worker blew up"):
        ForkPool(2).map(_explode, [1, 2, 3])


def test_teacher_worker_task_returns_episode_records(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 1

    _teacher_worker_init(cpu_config)
    episodes = _teacher_worker_task((0, [0, 1]))

    assert len(episodes) == 2
    for episode in episodes:
        assert len(episode["records"]) == 2
        assert set(episode["operator_counts"].keys()) == {0, 1, 2}
        assert isinstance(episode["episode_reward"], float)
        assert all("reward" in record for record in episode["records"])
        assert episode["records"][-1]["tail_graph"] is not None
