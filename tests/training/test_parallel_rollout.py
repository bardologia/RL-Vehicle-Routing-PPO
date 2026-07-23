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


def test_teacher_worker_task_returns_episode_records(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 2

    _teacher_worker_init(cpu_config)
    episodes = _teacher_worker_task((0, [0, 1]))

    assert len(episodes) == 2
    for episode in episodes:
        assert len(episode["records"]) == 2
        assert set(episode["operator_counts"].keys()) == {0, 1, 2}
        assert isinstance(episode["episode_reward"], float)
        assert all("return" in record for record in episode["records"])
