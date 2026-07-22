from dataclasses import dataclass
from typing      import Optional


@dataclass
class TrainingConfig:
    device                : str = "cuda"
    load_checkpoint       : bool = False
    resume_from_run       : Optional[str] = None
    max_steps_per_episode : int = 5
    minibatch_size        : int = 128
    num_epochs            : int = 5
    print_frequency       : int = 5
    log_episode_frequency : int = 5
    use_mixed_precision   : bool = False
    large_negative_value  : float = -1e8
    verbose               : bool = False
