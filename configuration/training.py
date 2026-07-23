from dataclasses import dataclass


@dataclass
class TrainingConfig:
    device                : str = "cuda"
    max_steps_per_episode : int = 5
    minibatch_size        : int = 128
    num_epochs            : int = 5
    print_frequency       : int = 5
    log_episode_frequency : int = 5
    use_mixed_precision   : bool = False
    large_negative_value  : float = -1e8
    verbose               : bool = False
