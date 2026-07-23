import cProfile
import gc
import multiprocessing
import os
import pstats
import random

import numpy as np
import torch
from tqdm import tqdm

from tools.logger import NullLogger
from core.shared import Environment
from .dataset import ChunkStore


def generate_events(batch_size, seed, config, enable_profiling=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    profile_stats = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    simulation_env = Environment(config)
    batch = [None] * batch_size

    for i in range(batch_size):
        simulation_env.reset()
        initial_state = simulation_env.initial_state
        event_type, num_items = simulation_env.generate_event()
        event_state = simulation_env.apply_event(initial_state, event_type, num_items)
        graph, mask_info = simulation_env.observe()

        graph_cpu = graph.clone().detach().cpu()

        batch[i] = {
            "state"     : event_state.to_payload(),
            "graph"     : graph_cpu,
            "mask_info" : mask_info,
            "jobs"      : [job.to_dict() for job in simulation_env.jobs],
            "vehicles"  : [vehicle.to_dict() for vehicle in simulation_env.vehicles],
        }

    if enable_profiling:
        profiler.disable()
        profile_stats = pstats.Stats(profiler)
        profile_stats.stream = None

    return batch, profile_stats


def generate_events_task(task):
    return generate_events(*task)


class DatasetGenerator:
    def __init__(self, config, repo_root, logger=None):
        self.config      = config
        self.repo_root   = repo_root
        self.logger      = logger or NullLogger()
        self.dataset_dir = self._resolve_dir(config.io.dataset_dir)
        self.store       = ChunkStore(self.dataset_dir)

    def _resolve_dir(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(str(self.repo_root), path)

    def append(
        self,
        num_events,
        output_dir,
        seed=None,
        chunk_size=10_000,
        batch_size=100,
        num_workers=None,
        enable_worker_profiling=False,
    ):

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)

        existing_events, num_existing_chunks = self.store.count_events()
        self.logger.info(f"Found {num_existing_chunks} chunks with {existing_events} total events")

        if existing_events >= num_events:
            self.logger.info(f"There are already {existing_events} events. Nothing to do")
            return output_dir

        events_to_create = num_events - existing_events
        num_batches      = int(np.ceil(events_to_create / batch_size))
        num_workers      = num_workers or max(1, (os.cpu_count() or 2) - 2)

        self.logger.info(f"Creating {events_to_create} new events")
        self.logger.info(f"Generating {num_batches} batches with batch_size={batch_size} on {num_workers} workers")

        tasks = []
        for i in range(num_batches):
            batch_seed        = seed + i if seed is not None else random.randint(0, 2**31)
            actual_batch_size = min(batch_size, events_to_create - i * batch_size)
            profile_this      = enable_worker_profiling and i == 0
            tasks.append((actual_batch_size, batch_seed, self.config, profile_this))

        current_chunk       = []
        current_chunk_index = num_existing_chunks

        context = multiprocessing.get_context("fork")
        with context.Pool(processes=num_workers) as pool:
            with tqdm(total=events_to_create, desc="Generating events", ncols=80) as pbar:
                for batch_items, profile_stats in pool.imap(generate_events_task, tasks):
                    if profile_stats is not None:
                        self.logger.save_profiler_results(profile_stats, os.path.join(output_dir, "generation_profile.md"))

                    for item in batch_items:
                        current_chunk.append(item)
                        pbar.update(1)

                        if len(current_chunk) >= chunk_size:
                            chunk_path = self.store.chunk_path(current_chunk_index)
                            self.logger.info(f"Saving chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")

                            self.store.save(current_chunk, current_chunk_index)
                            current_chunk = []
                            current_chunk_index += 1
                            gc.collect()

        if current_chunk:
            chunk_path = self.store.chunk_path(current_chunk_index)
            self.logger.info(f"Saving final chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")
            self.store.save(current_chunk, current_chunk_index)

        total_events, total_chunks = self.store.count_events()
        self.logger.info(f"Dataset complete: {total_chunks} chunks, {total_events} events in {output_dir}")

        return output_dir

    def generate(self):
        return self.append(
            num_events = self.config.io.dataset_num_events,
            output_dir = self.dataset_dir,
            seed       = self.config.io.dataset_seed,
            chunk_size = self.config.io.dataset_chunk_size,
            batch_size = self.config.io.dataset_batch_size,
        )
