import pytest
import torch

from core.dataset import Dataset, generate_events


@pytest.fixture
def batch(cpu_config, seeded, fake_vroom):
    items, _ = generate_events(batch_size=3, seed=11, config=cpu_config)
    return items


def write_chunk(tmp_path, batch, index=0):
    torch.save(batch, tmp_path / f"chunk_{index:05d}.pt")


def test_chunk_write_and_iterate_round_trip(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    items = list(dataset)

    assert len(items) == len(batch)
    assert len(dataset) == len(batch)
    for stored, loaded in zip(batch, items):
        assert stored["state"] == loaded["state"]


def test_dataset_iterates_multiple_chunks_in_order(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch[:2], index=0)
    write_chunk(tmp_path, batch[2:], index=1)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    items = list(dataset)

    assert len(items) == 3


def test_dataset_resume_state_skips_consumed_items(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)
    dataset.set_state({"current_chunk_idx": 0, "current_item_idx": 1, "total_items_yielded": 1})

    items = list(dataset)

    assert len(items) == len(batch) - 1
    assert items[0]["state"] == batch[1]["state"]


def test_dataset_get_state_reports_progress(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch)

    dataset  = Dataset(dataset_dir=str(tmp_path), config=cpu_config)
    consumed = 0

    for _ in dataset:
        consumed += 1
        break

    state = dataset.get_state()

    assert state["total_items_yielded"] == 1
    assert set(state.keys()) == {"current_chunk_idx", "current_item_idx", "total_items_yielded"}


def test_dataset_iteration_resets_cursor_at_end(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    list(dataset)
    second_pass = list(dataset)

    assert len(second_pass) == len(batch)


def test_dataset_set_state_requires_all_keys(cpu_config, tmp_path):
    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    with pytest.raises(KeyError):
        dataset.set_state({"current_chunk_idx": 0})


def test_dataset_total_events_counts_lazily(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config)

    assert dataset._total_events is None
    assert dataset.total_events == len(batch)


def test_dataset_shuffle_keeps_all_chunks(cpu_config, tmp_path, batch):
    write_chunk(tmp_path, batch[:1], index=0)
    write_chunk(tmp_path, batch[1:], index=1)

    dataset = Dataset(dataset_dir=str(tmp_path), config=cpu_config, shuffle_chunks=True)

    assert len(dataset.chunk_paths) == 2
    assert len(list(dataset)) == len(batch)
