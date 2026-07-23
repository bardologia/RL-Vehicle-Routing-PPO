import pstats

import pytest
import torch

from core.dataset import generate_events, generate_events_task
from core.shared import Environment


@pytest.fixture
def batch(cpu_config, seeded, fake_vroom):
    items, _ = generate_events(batch_size=3, seed=11, config=cpu_config)
    return items


def test_generate_events_item_schema(batch):
    for item in batch:
        assert set(item.keys()) == {"state", "graph", "mask_info", "jobs", "vehicles"}
        assert item["state"]["schema"] == "routing-state-v1"
        assert len(item["jobs"]) > 0
        assert len(item["vehicles"]) > 0
        assert set(item["mask_info"].keys()) == {"unassigned_job_indices", "vehicles_with_jobs_indices", "vehicle_to_job_indices", "vehicles_with_capacity_indices"}


def test_generate_events_graphs_are_cpu_and_complete(batch):
    for item in batch:
        graph = item["graph"]
        assert graph["job"].x.shape[1] == 7
        assert graph["vehicle"].x.shape[1] == 7
        assert graph["job"].x.device.type == "cpu"


def test_generate_events_batch_size_matches_request(batch):
    assert len(batch) == 3


def test_generate_events_is_deterministic_per_seed(cpu_config, seeded, fake_vroom):
    first, _  = generate_events(batch_size=2, seed=5, config=cpu_config)
    second, _ = generate_events(batch_size=2, seed=5, config=cpu_config)

    for a, b in zip(first, second):
        assert a["state"] == b["state"]
        assert a["jobs"] == b["jobs"]
        assert torch.equal(a["graph"]["job"].x, b["graph"]["job"].x)


def test_generate_events_differs_across_seeds(cpu_config, seeded, fake_vroom):
    first, _  = generate_events(batch_size=2, seed=1, config=cpu_config)
    second, _ = generate_events(batch_size=2, seed=2, config=cpu_config)

    assert any(a["jobs"] != b["jobs"] for a, b in zip(first, second))


def test_generate_events_without_profiling_returns_none_stats(cpu_config, seeded, fake_vroom):
    _, profile_stats = generate_events(batch_size=1, seed=3, config=cpu_config)

    assert profile_stats is None


def test_generate_events_with_profiling_returns_pstats(cpu_config, seeded, fake_vroom):
    _, profile_stats = generate_events(batch_size=1, seed=3, config=cpu_config, enable_profiling=True)

    assert isinstance(profile_stats, pstats.Stats)


def test_generate_events_task_unpacks_tuple(cpu_config, seeded, fake_vroom):
    batch, profile_stats = generate_events_task((2, 9, cpu_config, False))

    assert len(batch) == 2
    assert profile_stats is None


def test_environment_reconstructs_dataset_item(cpu_config, seeded, fake_vroom, batch):
    item        = batch[0]
    environment = Environment(cpu_config)

    environment.load_from_dataset(item)

    assert [job.to_dict() for job in environment.jobs] == item["jobs"]
    assert environment.current_state.to_payload() == item["state"]

    graph, mask_info = environment.observe()

    assert mask_info == item["mask_info"]
    assert torch.equal(graph["job"].x, item["graph"]["job"].x)
