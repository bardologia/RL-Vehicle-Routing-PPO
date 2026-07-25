import numpy as np

from tools.auxiliary import generate_coords_batch
from .state import EntityPool, Job, Vehicle


class ScenarioSampler:
    def __init__(self, env_config):
        self.env_config = env_config

    def sample_depot(self):
        coords = generate_coords_batch(self.env_config.center, self.env_config.depot_radius, 1, 0, 1)
        return (float(coords[0][0]), float(coords[0][1]))

    def sample_jobs(self, pool: EntityPool, count: int):
        env        = self.env_config
        coords     = generate_coords_batch(env.center, env.radius, count, env.outlier_probability, env.outlier_multiplier)
        priorities = np.random.choice([1, 2, 3, 4, 5], size=count)
        kinds      = np.random.random(count) < env.repossession_fraction
        first_id   = pool.next_id()

        jobs = []
        for i in range(count):
            repossession = bool(kinds[i])

            if repossession:
                service = np.random.randint(env.repossession_service_min, env.repossession_service_max + 1)
            else:
                service = np.random.randint(env.support_service_min, env.support_service_max + 1)

            jobs.append(
                Job(
                    id          = first_id + i,
                    location    = (float(coords[i][0]), float(coords[i][1])),
                    kind        = "repossession" if repossession else "support",
                    service     = int(service),
                    amount      = 1 if repossession else 0,
                    priority    = int(priorities[i]),
                    description = f"Job {first_id + i}",
                )
            )

        return jobs

    def sample_vehicles(self, pool: EntityPool, count: int):
        coords     = generate_coords_batch(self.env_config.center, self.env_config.radius, count, 0, self.env_config.outlier_multiplier)
        capacities = np.random.choice([3, 4, 5], size=count)
        speeds     = np.random.choice([0.9, 1.0, 1.1], size=count)
        first_id   = pool.next_id()

        return [
            Vehicle(
                id           = first_id + i,
                start        = (float(coords[i][0]), float(coords[i][1])),
                capacity     = int(capacities[i]),
                speed_factor = float(speeds[i]),
                description  = f"Vehicle {first_id + i}",
            )
            for i in range(count)
        ]
