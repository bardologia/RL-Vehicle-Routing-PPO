from dataclasses import dataclass
from typing      import Tuple


@dataclass
class EnvironmentConfig:
    center: Tuple[float, float] = (-46.63, -23.55)
    radius: float = 25.0

    min_vehicles : int = 2
    min_jobs     : int = 4

    mean_jobs     : int = 16
    std_jobs      : int = 4
    mean_vehicles : int = 4
    std_vehicles  : int = 1

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

    @property
    def outlier_probability(self):
        return 1.0 / self.outlier_frequency if self.outlier_frequency > 0 else 0.0
