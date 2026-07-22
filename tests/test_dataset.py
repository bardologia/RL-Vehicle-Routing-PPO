import pytest
import torch

from core.dataset import Dataset, generate_events
from core.shared import Environment, RoutingState


@pytest.fixture
def batch(cpu_config, seeded, fake_vroom):
    items, profile = generate_events(batch_size=3, seed=11, config=cpu_config)
    return items


def test_generate_events_item_schema(batch):
    for item in batch:
        assert set(item.keys()) == {"state", "graph", "mask_info", "jobs", "vehicles"}
        assert item["state"]["schema"] == "routing-state-v1"
        assert len(item["jobs"]) > 0
        assert len(item["vehicles"]) > 0
        assert set(item["mask_info"].keys()) == {"unassigned_job_indices", "vehicles_with_jobs_indices", "vehicle_to_job_indices"}


def test_generate_events_graphs_are_cpu_and_complete(batch):
    for item in batch:
        graph = item["graph"]
        assert graph["job"].x.shape[1] == 7
        assert graph["vehicle"].x.shape[1] == 5
        assert graph["job"].x.device.type == "cpu"


def test_generate_events_is_deterministic_per_seed(cpu_config, seeded, fake_vroom):
    first, _  = generate_events(batch_size=2, seed=5, config=cpu_config)
    second, _ = generate_events(batch_size=2, seed=5, config=cpu_config)

    for a, b in zip(first, second):
        assert a["state"] == b["state"]
        assert a["jobs"] == b["jobs"]
        assert torch.equal(a["graph"]["job"].x, b["graph"]["job"].x)


def test_environment_reconstructs_dataset_item(cpu_config, seeded, fake_vroom, batch):
    item        = batch[0]
    environment = Environment(cpu_config)

    environment.load_from_dataset(item)

    assert [job.to_dict() for job in environment.jobs] == item["jobs"]
    assert environment.current_state.to_payload() == item["state"]

    graph, mask_info = environment.observe()

    assert mask_info == item["mask_info"]
    assert torch.equal(graph["job"].x, item["graph"]["job"].x)


def test_chunk_write_and_iterate_round_trip(cpu_config, seeded, fake_vroom, tmp_path, batch):
    torch.save(batch, tmp_path / "chunk_00000.pt")

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    items = list(dataset)

    assert len(items) == len(batch)
    assert len(dataset) == len(batch)
    for stored, loaded in zip(batch, items):
        assert stored["state"] == loaded["state"]


def test_dataset_resume_state_skips_consumed_items(cpu_config, seeded, fake_vroom, tmp_path, batch):
    torch.save(batch, tmp_path / "chunk_00000.pt")

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)
    dataset.set_state({"current_chunk_idx": 0, "current_item_idx": 1, "total_items_yielded": 1})

    items = list(dataset)

    assert len(items) == len(batch) - 1
    assert items[0]["state"] == batch[1]["state"]


def test_dataset_set_state_requires_all_keys(cpu_config, tmp_path):
    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    with pytest.raises(KeyError):
        dataset.set_state({"current_chunk_idx": 0})
