from dataclasses import dataclass


@dataclass
class RewardConfig:
    distance_weight           : float = 1.5
    unassigned_penalty_weight : float = 1.0
    idle_penalty_weight       : float = 0.5
    priority_penalty_weight   : float = 0.5

    add_job_cost    : float = 0.1
    remove_job_cost : float = 0.1
    no_action_cost  : float = 0.0
    disruption_cost : float = 0.3
