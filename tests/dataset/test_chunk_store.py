import pytest
import torch

from core.dataset import ChunkStore, generate_events


@pytest.fixture
def batch(cpu_config, seeded, fake_vroom):
    items, _ = generate_events(batch_size=3, seed=11, config=cpu_config)
    return items


def test_chunk_store_saves_lists_counts_and_loads(tmp_path, batch):
    store = ChunkStore(str(tmp_path))

    store.save(batch, 0)
    store.save(batch, 1)

    chunks = store.existing_chunks()

    assert [chunk.split("/")[-1] for chunk in chunks] == ["chunk_00000.pt", "chunk_00001.pt"]

    total, num_chunks = store.count_events()

    assert num_chunks == 2
    assert total == 2 * len(batch)

    loaded = store.load(chunks[0])

    assert len(loaded) == len(batch)
    assert loaded[0]["state"] == batch[0]["state"]


def test_chunk_path_zero_pads_index(tmp_path):
    store = ChunkStore(str(tmp_path))

    assert store.chunk_path(7).endswith("chunk_00007.pt")
    assert store.chunk_path(12345).endswith("chunk_12345.pt")


def test_existing_chunks_sorts_by_numeric_index(tmp_path, batch):
    store = ChunkStore(str(tmp_path))

    for index in (10, 2, 1):
        store.save(batch, index)

    chunks = [chunk.split("/")[-1] for chunk in store.existing_chunks()]

    assert chunks == ["chunk_00001.pt", "chunk_00002.pt", "chunk_00010.pt"]


def test_chunk_store_ignores_missing_directory(tmp_path):
    store = ChunkStore(str(tmp_path / "absent"))

    assert store.existing_chunks() == []


def test_count_events_on_empty_store_is_zero(tmp_path):
    store = ChunkStore(str(tmp_path))

    total, num_chunks = store.count_events()

    assert total == 0
    assert num_chunks == 0


def test_save_then_reload_preserves_items(tmp_path, batch):
    store = ChunkStore(str(tmp_path))
    store.save(batch, 0)

    reloaded = store.load(store.existing_chunks()[0])

    assert reloaded[0]["state"] == batch[0]["state"]
    assert reloaded[0]["depot"] == batch[0]["depot"]
    assert reloaded[0]["jobs"] == batch[0]["jobs"]
