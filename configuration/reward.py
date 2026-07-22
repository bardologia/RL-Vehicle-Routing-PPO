from dataclasses import dataclass


@dataclass
class RewardConfig:
    distance_weight           : float = 1.5
    unassigned_penalty_weight : float = 1.0
    idle_penalty_weight       : float = 0.5
    priority_penalty_weight   : float = 0.5

    invalid_action_penalty    : float = 0.0
    add_job_penalty           : float = 0.5
    remove_job_penalty        : float = 1.5
    reoptimize_penalty        : float = -1.5
    no_action_penalty         : float = 0.0
