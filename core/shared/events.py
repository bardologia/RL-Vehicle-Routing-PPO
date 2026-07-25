import random

from .sampling import ScenarioSampler
from .state import RoutingState


class EventGenerator:
    def __init__(self):
        self.samplers = {
            "new_job":        self.sample_new_job,
            "remove_job":     self.sample_remove_job,
            "new_vehicle":    self.sample_new_vehicle,
            "remove_vehicle": self.sample_remove_vehicle,
        }

    def sample_new_job(self, env):
        cfg = env.config.env
        return random.randint(cfg.job_insert_min, cfg.job_insert_max)

    def sample_new_vehicle(self, env):
        cfg = env.config.env
        return random.randint(cfg.vehicle_insert_min, cfg.vehicle_insert_max)

    def sample_remove_job(self, env):
        cfg = env.config.env
        return self._sample_removal(len(env.jobs), cfg.job_remove_min, cfg.job_remove_max)

    def sample_remove_vehicle(self, env):
        cfg = env.config.env
        return self._sample_removal(len(env.vehicles) - 1, cfg.vehicle_remove_min, cfg.vehicle_remove_max)

    def _sample_removal(self, capacity, remove_min, remove_max):
        max_items = min(remove_max, capacity)
        if max_items <= 0:
            return 0

        min_items = min(remove_min, max_items)
        return random.randint(min_items, max_items)

    def generate(self, env):
        event_type = random.choice(list(self.samplers))
        return event_type, self.samplers[event_type](env)


class EventHandler:
    def __init__(self, sampler: ScenarioSampler):
        self.sampler  = sampler
        self.handlers = {
            "new_job":        self.apply_new_job,
            "new_vehicle":    self.apply_new_vehicle,
            "remove_job":     self.apply_remove_job,
            "remove_vehicle": self.apply_remove_vehicle,
        }

    def apply_new_job(self, env, state: RoutingState, num_items: int) -> RoutingState:
        new_jobs = self.sampler.sample_jobs(env.jobs, num_items)
        env.jobs.add(new_jobs)

        new_state = state.copy()
        new_state.add_unassigned({job.id for job in new_jobs})
        return new_state

    def apply_new_vehicle(self, env, state: RoutingState, num_items: int) -> RoutingState:
        env.vehicles.add(self.sampler.sample_vehicles(env.vehicles, num_items))
        return state.copy()

    def apply_remove_job(self, env, state: RoutingState, num_items: int) -> RoutingState:
        if len(env.jobs) == 0:
            return state.copy()

        num_items    = min(num_items, len(env.jobs))
        removed_ids  = set(env.jobs.sample_ids(num_items))
        affected_ids = [route.vehicle_id for route in state.routes if set(route.job_ids) & removed_ids]

        env.jobs.remove(removed_ids)

        new_state = state.copy()

        for vehicle_id in affected_ids:
            self._resolve_vehicle(env, new_state, vehicle_id)

        new_state.discard_unassigned(removed_ids)
        return new_state

    def _resolve_vehicle(self, env, state: RoutingState, vehicle_id: int) -> None:
        if not env.vehicles.contains(vehicle_id):
            return

        vehicle        = env.vehicles.by_id(vehicle_id)
        route          = state.route_of_vehicle(vehicle_id)
        remaining_jobs = [env.jobs.by_id(job_id) for job_id in (route.job_ids if route is not None else []) if env.jobs.contains(job_id)]

        if not remaining_jobs and vehicle.onboard == 0:
            orphaned = state.remove_vehicles({vehicle_id})
            state.add_unassigned({job_id for job_id in orphaned if env.jobs.contains(job_id)})
            return

        new_route = env.action_handler.solve_vehicle_route(env, remaining_jobs, vehicle)
        if new_route is None:
            env.logger.warning(f"Cancellation re-solve failed for vehicle {vehicle_id}")
            state.remove_vehicles({vehicle_id})
            state.add_unassigned({job.id for job in remaining_jobs})
            return

        state.replace_route(new_route)

    def apply_remove_vehicle(self, env, state: RoutingState, num_items: int) -> RoutingState:
        num_items = min(num_items, len(env.vehicles) - 1)
        if num_items <= 0:
            return state.copy()

        removed_ids = set(env.vehicles.sample_ids(num_items))
        env.vehicles.remove(removed_ids)

        new_state = state.copy()
        orphaned  = new_state.remove_vehicles(removed_ids)
        new_state.add_unassigned({job_id for job_id in orphaned if env.jobs.contains(job_id)})
        return new_state

    def apply(self, env, state: RoutingState, event_type, num_items: int) -> RoutingState:
        if event_type not in self.handlers:
            raise ValueError(f"Unknown event type: {event_type!r}")

        return self.handlers[event_type](env, state, num_items)
