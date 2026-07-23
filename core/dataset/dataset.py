import gc
import glob
import os
import random
import re

import torch

from tools.logger import NullLogger


class ChunkStore:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def existing_chunks(self):
        if not os.path.exists(self.dataset_dir):
            return []

        pattern = os.path.join(self.dataset_dir, "chunk_*.pt")
        files   = glob.glob(pattern)

        return sorted(files, key=self._chunk_index)

    def _chunk_index(self, path):
        match = re.search(r"chunk_(\d+)\.pt", os.path.basename(path))
        return int(match.group(1)) if match else -1

    def chunk_path(self, chunk_index):
        return os.path.join(self.dataset_dir, f"chunk_{chunk_index:05d}.pt")

    def count_events(self):
        chunks = self.existing_chunks()
        total  = 0

        for chunk_path in chunks:
            data = torch.load(chunk_path, weights_only=False, mmap=True)
            total += len(data)
            del data

        return total, len(chunks)

    def load(self, chunk_path):
        return torch.load(chunk_path, weights_only=False)

    def save(self, items, chunk_index):
        torch.save(items, self.chunk_path(chunk_index))


class Dataset:
    def __init__(self, dataset_dir, config, shuffle_chunks=False, count_events=False, logger=None):
        self.dataset_dir    = dataset_dir
        self.config         = config
        self.logger         = logger or NullLogger()
        self.store          = ChunkStore(dataset_dir)
        self.chunk_paths    = self.store.existing_chunks()
        self.shuffle_chunks = shuffle_chunks

        if shuffle_chunks:
            random.shuffle(self.chunk_paths)

        self._total_events = None
        if count_events:
            self._count_events()
            self.logger.info(f"Dataset: {len(self.chunk_paths)} chunks, {self._total_events} events")

        self._current_chunk_idx   = 0
        self._current_item_idx    = 0
        self._total_items_yielded = 0

    @property
    def total_events(self):
        if self._total_events is None:
            self._count_events()
        return self._total_events

    def _count_events(self):
        self._total_events = 0
        for path in self.chunk_paths:
            data = self.store.load(path)
            self._total_events += len(data)
            del data

    def get_state(self):
        return {
            "current_chunk_idx"   : self._current_chunk_idx,
            "current_item_idx"    : self._current_item_idx,
            "total_items_yielded" : self._total_items_yielded,
        }

    def set_state(self, state):
        self._current_chunk_idx   = state["current_chunk_idx"]
        self._current_item_idx    = state["current_item_idx"]
        self._total_items_yielded = state["total_items_yielded"]
        self.logger.info(f"Dataset: resuming from chunk {self._current_chunk_idx}, item {self._current_item_idx} (total yielded: {self._total_items_yielded})")

    def __len__(self):
        return self.total_events

    def __iter__(self):
        start_chunk_idx = self._current_chunk_idx
        start_item_idx  = self._current_item_idx

        for chunk_idx, chunk_path in enumerate(self.chunk_paths):
            if chunk_idx < start_chunk_idx:
                continue

            self._current_chunk_idx = chunk_idx
            self.logger.info(f"Loading chunk {chunk_idx + 1}/{len(self.chunk_paths)}: {chunk_path}")

            chunk_data = self.store.load(chunk_path)

            for item_idx, item in enumerate(chunk_data):
                if chunk_idx == start_chunk_idx and item_idx < start_item_idx:
                    continue

                self._current_item_idx     = item_idx
                self._total_items_yielded += 1
                yield item

            self._current_item_idx = 0
            start_item_idx         = 0

            del chunk_data
            gc.collect()
            torch.cuda.empty_cache()

        self._current_chunk_idx = 0
        self._current_item_idx  = 0
