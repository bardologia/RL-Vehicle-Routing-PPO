"""Scenario Lab backend: service health, random scenarios, templates and solved plans with KPIs."""

import random

import numpy as np
import requests

from core.inference import EpisodeLedger, PlanMetrics, ScenarioPlanner
from core.shared.sampling import ScenarioSampler
from core.shared.state import EntityPool
from scenario_templates import ScenarioTemplates
from web_errors import ApiError


class ScenarioLab:
    """Builds and solves hand-made scenarios.

    Attributes:
        runtime   : ConsoleRuntime with the base Config and the environment lock.
        geometry  : GeometryCache giving road paths.
        templates : ScenarioTemplates catalog.
    """

    def __init__(self, runtime, geometry, logger):
        self.runtime   = runtime
        self.geometry  = geometry
        self.logger    = logger
        self.templates = ScenarioTemplates()
        self.planner   = ScenarioPlanner()

    def health(self):
        """Return {"osrm_url", "vroom_url", "osrm": bool, "vroom": bool} from a 2 s probe of each service."""
        service = self.runtime.config.service
        status  = {"osrm_url": service.osrm_url, "vroom_url": service.vroom_url}

        try:
            probe          = f"{service.osrm_url}/route/v1/driving/-46.63,-23.55;-46.62,-23.54?overview=false"
            status["osrm"] = requests.get(probe, timeout=2).status_code == 200
        except requests.RequestException:
            status["osrm"] = False

        try:
            status["vroom"] = requests.get(f"{service.vroom_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            status["vroom"] = False

        return status

    def list_templates(self):
        """Return the scenario templates with their structured expectations."""
        return self.templates.catalog()

    def sample(self, num_jobs, num_vehicles, seed):
        """Return a random scenario {"jobs", "vehicles", "depot"} drawn from the environment sampler with a seed."""
        with self.runtime.lock:
            random.seed(seed)
            np.random.seed(seed)

            sampler  = ScenarioSampler(self.runtime.config.env)
            jobs     = sampler.sample_jobs(EntityPool(), num_jobs)
            vehicles = sampler.sample_vehicles(EntityPool(), num_vehicles)
            depot    = sampler.sample_depot()

        return {"jobs": [job.to_dict() for job in jobs], "vehicles": [vehicle.to_dict() for vehicle in vehicles], "depot": list(depot)}

    def solve(self, jobs, vehicles, assignment=None, depot=None):
        """Solve a scenario and return its plan.

        Args:
            jobs       : Job dicts.
            vehicles   : Vehicle dicts.
            assignment : {vehicle_id: [job_id]} pinning jobs to vehicles, or None to let VROOM assign.
            depot      : [lon, lat] of the depot.

        Returns:
            {"plan"} in the trace plan shape, with road paths, distances and KPIs.

        Raises:
            ApiError : The scenario is malformed or has no solution (400), or OSRM is unavailable (503).
        """
        try:
            pools_jobs, pools_vehicles = self.planner.pools(jobs, vehicles)
            depot_point                = (float(depot[0]), float(depot[1])) if depot else None
            clock                      = self.planner.clock(pools_vehicles)

            with self.runtime.lock:
                state = self.planner.initial_state(pools_jobs, pools_vehicles, assignment, depot_point, clock)
        except (ValueError, KeyError, TypeError) as error:
            raise ApiError(400, str(error))

        metrics = PlanMetrics(self.runtime.config.reward)
        ledger  = EpisodeLedger()
        ledger.see_jobs(pools_jobs, clock)

        plan = metrics.plan(pools_jobs, pools_vehicles, state, clock, ledger)
        return {"plan": metrics.add_geometry(plan, self.geometry.route)}
