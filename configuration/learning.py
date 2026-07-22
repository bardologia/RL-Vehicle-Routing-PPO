from dataclasses import dataclass


@dataclass
class LearningRate:
    lr_operator_actor : float = 3e-4
    lr_vehicle_actor  : float = 3e-4
    lr_critic         : float = 5e-4
    lr_embedding      : float = 2e-4
    lr_job_actor      : float = 3e-4

    lr_warmup_steps: int = 1000
    lr_min: float = 1e-5
    lr_decay_steps: int = 100000

@dataclass
class Entropy:
    entropy_coef         : float = 0.02
    entropy_start        : float = 0.02
    entropy_end          : float = 0.001
    entropy_anneal_steps : int = 50000
