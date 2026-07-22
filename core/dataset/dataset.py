import os
import re
import gc
import glob
import random
import multiprocessing
import torch
import numpy as np
from tqdm import tqdm
from tools.logger import NullLogger
from .generation import generate_events_task


class Dataset:
    def __init__(self, dataset_dir, config, shuffle_chunks=False, count_events=False, logger=None):
        self.dataset_dir    = dataset_dir
        self.config         = config
        self.logger         = logger or NullLogger()
        self.chunk_paths    = self.get_existing_chunks(dataset_dir)
        self.shuffle_chunks = shuffle_chunks

        if shuffle_chunks:
            random.shuffle(self.chunk_paths)

        self._total_events = None
        if count_events:
            self._count_events()
            self.logger.info(f"Dataset: {len(self.chunk_paths)} chunks, {self._total_events} events")

        self._current_chunk_idx = 0
        self._current_item_idx = 0
        self._total_items_yielded = 0
    
    def get_state(self):
        return {
            "current_chunk_idx": self._current_chunk_idx,
            "current_item_idx": self._current_item_idx,
            "total_items_yielded": self._total_items_yielded,
        }
    
    def set_state(self, state):
        self._current_chunk_idx = state["current_chunk_idx"]
        self._current_item_idx = state["current_item_idx"]
        self._total_items_yielded = state["total_items_yielded"]
        self.logger.info(f"Dataset: resuming from chunk {self._current_chunk_idx}, item {self._current_item_idx} (total yielded: {self._total_items_yielded})")
             
    def _count_events(self):
        self._total_events = 0
        for path in self.chunk_paths:
            data = torch.load(path, weights_only=False)
            self._total_events += len(data)
            del data
    
    @property
    def total_events(self):
        if self._total_events is None:
            self._count_events()
        return self._total_events
    
    def get_existing_chunks(self, output_dir):
        if not os.path.exists(output_dir):
            return []
        pattern = os.path.join(output_dir, "chunk_*.pt")
        files = glob.glob(pattern)

        def extract_index(path):
            m = re.search(r"chunk_(\d+)\.pt", os.path.basename(path))
            return int(m.group(1)) if m else -1

        return sorted(files, key=extract_index)
    
    def get_chunk_path(self, chunk_index):
        return os.path.join(self.dataset_dir, f"chunk_{chunk_index:05d}.pt")

    def count_existing_events(self):
        chunks = self.get_existing_chunks(self.dataset_dir)
        total = 0
  
        for chunk_path in chunks:
            data = torch.load(chunk_path, weights_only=False, mmap=True)
            total += len(data)
            del data  
            
        return total, len(chunks)

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

        existing_events, num_existing_chunks = self.count_existing_events()
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
                        self.logger.info(f"Worker profile (first batch):\n{profile_stats}")

                    for item in batch_items:
                        current_chunk.append(item)
                        pbar.update(1)

                        if len(current_chunk) >= chunk_size:
                            chunk_path = self.get_chunk_path(current_chunk_index)
                            self.logger.info(f"Saving chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")

                            torch.save(current_chunk, chunk_path)
                            current_chunk = []
                            current_chunk_index += 1
                            gc.collect()

        if current_chunk:
            chunk_path = self.get_chunk_path(current_chunk_index)
            self.logger.info(f"Saving final chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")
            torch.save(current_chunk, chunk_path)

        total_events, total_chunks = self.count_existing_events()
        self.logger.info(f"Dataset complete: {total_chunks} chunks, {total_events} events in {output_dir}")

        return output_dir
    
    def __len__(self):
        return self.total_events
    
    def __iter__(self):
        start_chunk_idx = self._current_chunk_idx
        start_item_idx = self._current_item_idx
        
        for chunk_idx, chunk_path in enumerate(self.chunk_paths):
            # Skip chunks before the resumption point
            if chunk_idx < start_chunk_idx:
                continue
            
            self._current_chunk_idx = chunk_idx
            self.logger.info(f"Loading chunk {chunk_idx + 1}/{len(self.chunk_paths)}: {chunk_path}")
            
            chunk_data = torch.load(chunk_path, weights_only=False)
            
            for item_idx, item in enumerate(chunk_data):
                # Skip items before the resumption point (only for the first chunk)
                if chunk_idx == start_chunk_idx and item_idx < start_item_idx:
                    continue
                
                self._current_item_idx = item_idx
                self._total_items_yielded += 1
                yield item
            
            # Reset item index for next chunk
            self._current_item_idx = 0
            start_item_idx = 0  # Only skip items in the first chunk
            
            del chunk_data
            gc.collect()
            torch.cuda.empty_cache()
        
        # Reset state after complete iteration
        self._current_chunk_idx = 0
        self._current_item_idx = 0
