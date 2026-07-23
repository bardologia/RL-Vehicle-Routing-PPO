from dataclasses import dataclass
from typing      import Tuple


@dataclass
class EnvironmentConfig:
    center: Tuple[float, float] = (-46.63, -23.55)
    radius: float = 25.0

    depot_radius                     : float = 3.0
    depot_service                    : int = 120
    repossession_fraction            : float = 0.5
    repossession_success_probability : float = 0.85
    support_service_min              : int = 600
    support_service_max              : int = 1800
    repossession_service_min         : int = 300
    repossession_service_max         : int = 900

    tick_seconds: int = 900

    min_vehicles : int = 2
    min_jobs     : int = 4

    mean_jobs     : int = 16
    std_jobs      : int = 4
    mean_vehicles : int = 4
    std_vehicles  : int = 1

    reset_max_attempts : int = 32

    outlier_frequency  : int = 8
    outlier_multiplier : int = 2

    job_insert_min: int = 2
    job_insert_max: int = 3
    job_remove_min: int = 1
    job_remove_max: int = 2

    vehicle_insert_min: int = 1
    vehicle_insert_max: int = 1
    vehicle_remove_min: int = 1
    vehicle_remove_max: int = 1

    step_event_probability: float = 0.3

    @property
    def outlier_probability(self):
        return 1.0 / self.outlier_frequency if self.outlier_frequency > 0 else 0.0
