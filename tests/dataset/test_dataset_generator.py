import os
import pickle

from core.dataset import ChunkStore, DatasetGenerator, generate_events, generate_events_task


def test_resolve_dir_keeps_absolute_and_joins_relative(cpu_config, tmp_path):
    generator = DatasetGenerator(config=cpu_config, repo_root=tmp_path)

    assert generator._resolve_dir("/var/data/chunks") == "/var/data/chunks"
    assert generator._resolve_dir("datasets/chunked") == os.path.join(str(tmp_path), "datasets/chunked")


def test_worker_task_tuple_is_picklable(cpu_config):
    task     = (2, 5, cpu_config, False)
    restored = pickle.loads(pickle.dumps(task))

    assert restored[0] == 2
    assert restored[1] == 5
    assert restored[3] is False


def test_worker_entry_point_is_picklable():
    restored = pickle.loads(pickle.dumps(generate_events_task))

    assert restored is generate_events_task


def test_append_parallel_pool_writes_all_events(cpu_config, seeded, fake_vroom, tmp_path):
    output = str(tmp_path / "chunks")
    cpu_config.io.dataset_dir = output

    generator = DatasetGenerator(config=cpu_config, repo_root=tmp_path)
    generator.append(num_events=6, output_dir=output, seed=3, chunk_size=4, batch_size=2, num_workers=2)

    store             = ChunkStore(output)
    total, num_chunks = store.count_events()

    assert total == 6
    assert num_chunks == 2

    items = store.load(store.existing_chunks()[0])
    assert set(items[0].keys()) == {"state", "depot", "clock", "jobs", "vehicles"}


def test_append_skips_when_enough_events_exist(cpu_config, seeded, fake_vroom, tmp_path):
    output = str(tmp_path / "chunks")
    cpu_config.io.dataset_dir = output
    os.makedirs(output, exist_ok=True)

    batch, _ = generate_events(batch_size=3, seed=7, config=cpu_config)
    ChunkStore(output).save(batch, 0)

    generator = DatasetGenerator(config=cpu_config, repo_root=tmp_path)
    result    = generator.append(num_events=2, output_dir=output, seed=1, chunk_size=4, batch_size=2, num_workers=2)

    assert result == output

    total, num_chunks = ChunkStore(output).count_events()
    assert total == 3
    assert num_chunks == 1
