from dataclasses import dataclass
from typing      import Optional


@dataclass
class IOConfig:
    runs_dir            : str = "runs"
    run_name            : Optional[str] = None
    logdir              : Optional[str] = None
    checkpoint_filename : str = "graph_ppo_policy.pt"
    resume_from_run     : Optional[str] = None
    init_from_run       : Optional[str] = None
