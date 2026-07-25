from dataclasses import dataclass


@dataclass
class TrainingConfig:
    device                      : str = "cuda"
    max_disruptions_per_episode : int = 5
    max_edits_per_round         : int = 8
    num_updates                 : int = 16
    episodes_per_update         : int = 1024
    minibatch_size              : int = 128
    expected_edits_per_round    : float = 2.0
    num_epochs                  : int = 5
    print_frequency             : int = 5
    log_episode_frequency       : int = 5
    use_mixed_precision         : bool = False
    large_negative_value        : float = -1e8
    verbose                     : bool = False
