import os
import re
import gc
import glob
import random
import torch
import numpy as np
from tqdm import tqdm
from .environment import *
import cProfile
import pstats
from io import StringIO



class Dataset:
    def __init__(self, dataset_dir, config, shuffle_chunks=False, verbose=True, count_events=False):
        self.dataset_dir = dataset_dir
        self.config = config
        self.chunk_paths = self.get_existing_chunks(dataset_dir)
        self.shuffle_chunks = shuffle_chunks
        self.verbose = verbose
        
        if shuffle_chunks:
            import random
            random.shuffle(self.chunk_paths)
        
        self._total_events = None
        if count_events:
            self._count_events()
            print(f"Dataset: {len(self.chunk_paths)} chunks, {self._total_events} events")
        
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
        self._current_chunk_idx = state.get("current_chunk_idx", 0)
        self._current_item_idx = state.get("current_item_idx", 0)
        self._total_items_yielded = state.get("total_items_yielded", 0)
        if self.verbose:
            print(f"Dataset: Resuming from chunk {self._current_chunk_idx}, item {self._current_item_idx} (total yielded: {self._total_items_yielded})")
             
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
        verbose=False,
        chunk_size=10_000,
        num_workers=None,
        batch_size=100,
        enable_worker_profiling=False,
    ):

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        os.makedirs(output_dir, exist_ok=True)

        existing_events, num_existing_chunks = self.count_existing_events()

        if verbose:
            print(f"Found {num_existing_chunks} chunks with {existing_events} total events")

        if existing_events >= num_events:
            if verbose:
                print(f"There are already {existing_events} events. Nothing to do")
            return output_dir

        events_to_create = num_events - existing_events

        if verbose:
            print(f"Creating {events_to_create} new events")
            print(f"Generating with batch_size={batch_size}")
        
        # Generate events sequentially (no Ray / parallelism)
        current_chunk = []
        current_chunk_index = num_existing_chunks

        num_batches = int(np.ceil(events_to_create / batch_size))

        if verbose:
            print(f"Generating {num_batches} batches sequentially, batch_size={batch_size}")

        # Capture config in the main process to ensure consistency
        graph_builder_config = {
            "k_near_paths_for_job": int(self.config.env.max_path_neighbors),
            "k_near_vehicles_for_job": int(self.config.env.max_vehicle_neighbors),
            "k_near_jobs_for_unassigned_vehicle": int(self.config.env.max_vehicle_neighbors),
        }

        with tqdm(total=events_to_create, desc="Generating events", ncols=80) as pbar:
            for i in range(num_batches):
                batch_seed = (seed or 0) + i if seed is not None else random.randint(0, 2**31)
                actual_batch_size = min(batch_size, events_to_create - i * batch_size)
                profile_this = enable_worker_profiling and i == 0

                result = generate_events(actual_batch_size, batch_seed, graph_builder_config, profile_this)
                batch_items, profile_stats = result

                if profile_stats is not None:
                    print(f"\n{'='*80}")
                    print(f"WORKER PROFILE (batch {i}, profiled run)")
                    print(f"{'='*80}")
                    print(profile_stats)
                    print(f"{'='*80}\n")

                for item in batch_items:
                    current_chunk.append(item)
                    pbar.update(1)

                    if len(current_chunk) >= chunk_size:
                        chunk_path = self.get_chunk_path(current_chunk_index)
                        if verbose:
                            print(f"\nSaving chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")

                        torch.save(current_chunk, chunk_path)
                        current_chunk = []
                        current_chunk_index += 1
                        gc.collect()
        
        if current_chunk:
            chunk_path = self.get_chunk_path(current_chunk_index)
            print(f"\nSaving final chunk {current_chunk_index} with {len(current_chunk)} events to {chunk_path}")
            torch.save(current_chunk, chunk_path)

        total_events, total_chunks = self.count_existing_events()
        print(f"\nDataset complete: {total_chunks} chunks, {total_events} events in {output_dir}")

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
            
            if self.verbose:
                print(f"\nLoading chunk {chunk_idx + 1}/{len(self.chunk_paths)}: {chunk_path}")
            
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
    

def generate_events(batch_size, seed, graph_builder_config=None, enable_profiling=False):  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    profile_stats = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    
    simulation_env = Environment()
    batch = [None] * batch_size
    
    for i in range(batch_size):
        simulation_env.reset()
        initial_state = simulation_env.initial_state
        event_type, num_items = simulation_env.generate_event()
        event_state = simulation_env.apply_event(initial_state, event_type, num_items)
        graph, mask_info = simulation_env.observe()
        
        graph_cpu = graph.clone().detach().cpu()
        
        batch[i] = {
            "event_state": event_state.to_dict(),
            "graph": graph_cpu,
            "mask_info": mask_info,
            "jobs": simulation_env.jobs.copy(),
            "vehicles": simulation_env.vehicles.copy(),
        }
        
        if i % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if enable_profiling:
        profiler.disable()
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(50)
        profile_stats = s.getvalue()
    
    return batch, profile_stats
