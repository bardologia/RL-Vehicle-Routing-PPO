from dataclasses import dataclass
from typing      import Optional


@dataclass
class IOConfig:
    runs_dir            : str = "runs"
    run_name            : Optional[str] = None
    logdir              : Optional[str] = None
    dataset_dir         : str = "datasets/chunked"
    checkpoint_filename : str = "graph_ppo_policy.pt"
    resume_from_run     : Optional[str] = None

    dataset_num_events  : int = 1024000
    dataset_chunk_size  : int = 1024
    dataset_batch_size  : int = 128
    dataset_seed        : int = 42
