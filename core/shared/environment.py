import random

import numpy as np
import torch

from tools.auxiliary import generate_coords_batch
from tools.logger import NullLogger
from .graph import Graph
from .mask import ActionMaskBuilder
from .services import vroom
from .state import EntityPool, Job, RoutingState, Vehicle


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


class ActionHandler:
    def __init__(self, logger):
        self.logger = logger

    def apply_job_insertion(self, env, state: RoutingState, vehicle_id: int, job_id: int) -> RoutingState:
        route           = state.route_of_vehicle(vehicle_id)
        current_job_ids = route.job_ids if route is not None else []

        candidate_ids  = current_job_ids + [job_id]
        candidate_jobs = [env.jobs.by_id(jid) for jid in candidate_ids if env.jobs.contains(jid)]
        vehicle        = env.vehicles.by_id(vehicle_id)

        new_route = self._solve_vehicle_route(env, candidate_jobs, vehicle)
        if new_route is None:
            self.logger.warning(f"Job insertion failed for vehicle_id={vehicle_id} job_id={job_id}")
            return state

        new_state = state.copy()
        new_state.replace_route(new_route)
        return new_state

    def apply_job_removal(self, env, state: RoutingState, vehicle_id: int, job_id: int) -> RoutingState:
        route = state.route_of_vehicle(vehicle_id)
        if route is None or job_id not in route.job_ids:
            return state

        remaining_ids  = [jid for jid in route.job_ids if jid != job_id]
        remaining_jobs = [env.jobs.by_id(jid) for jid in remaining_ids if env.jobs.contains(jid)]
        vehicle        = env.vehicles.by_id(vehicle_id)
        new_state      = state.copy()

        if not remaining_jobs and vehicle.onboard == 0:
            orphaned = new_state.remove_vehicles({vehicle_id})
            new_state.add_unassigned(orphaned)
            return new_state

        new_route = self._solve_vehicle_route(env, remaining_jobs, vehicle)
        if new_route is None:
            self.logger.warning(f"Job removal failed for vehicle_id={vehicle_id} job_id={job_id}")
            return state

        new_state.replace_route(new_route)
        new_state.add_unassigned({job_id})
        return new_state

    def _solve_vehicle_route(self, env, jobs, vehicle: Vehicle):
        solution = vroom.solve(jobs, [vehicle], depot=env.depot, clock=env.clock)
        if solution is None or not solution.routes:
            return None

        required = {job.id for job in jobs}
        if required & solution.unassigned_ids:
            return None

        return solution.routes[0]


class EventHandler:
    def __init__(self, sampler: ScenarioSampler):
        self.sampler = sampler

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

        num_items   = min(num_items, len(env.jobs))
        removed_ids = set(env.jobs.sample_ids(num_items))
        env.jobs.remove(removed_ids)

        new_state = state.copy()
        new_state.remove_jobs(removed_ids)
        return new_state

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


class ExecutionAdvancer:
    def __init__(self, logger):
        self.logger = logger

    def _complete_stop(self, env, vehicle, stop, planned_delivered, summary):
        if stop.kind == "delivery":
            dropped = min(planned_delivered, vehicle.onboard)
            summary["dropped"] += dropped
            vehicle.onboard    -= dropped
            return

        if stop.kind == "pickup":
            if random.random() < env.config.env.repossession_success_probability:
                vehicle.onboard += 1
                summary["served"].append(stop.job_id)
            else:
                summary["failed"].append(stop.job_id)
        else:
            summary["served"].append(stop.job_id)

        if env.jobs.contains(stop.job_id):
            env.jobs.remove({stop.job_id})

    def advance(self, env, state: RoutingState, tick_seconds: int):
        horizon = env.clock + tick_seconds
        summary = {"served": [], "failed": [], "dropped": 0}

        remaining_by_vehicle = {}
        for route in state.routes:
            if not env.vehicles.contains(route.vehicle_id):
                continue

            vehicle   = env.vehicles.by_id(route.vehicle_id)
            completed = [stop for stop in route.stops if stop.completion <= horizon]

            planned_prev = vehicle.onboard
            for stop in completed:
                self._complete_stop(env, vehicle, stop, max(planned_prev - stop.load, 0), summary)
                planned_prev = stop.load

            if completed:
                vehicle.start = completed[-1].location

            remaining_by_vehicle[vehicle.id] = [job_id for job_id in route.job_ids if env.jobs.contains(job_id)]

        env.clock = horizon

        new_state = RoutingState(routes=[], unassigned_ids=set(state.unassigned_ids))
        for vehicle in env.vehicles:
            remaining_ids = remaining_by_vehicle.get(vehicle.id, [])
            if not remaining_ids and vehicle.onboard == 0:
                continue

            if env.clock >= vehicle.time_window[1]:
                new_state.add_unassigned(set(remaining_ids))
                continue

            remaining_jobs = [env.jobs.by_id(job_id) for job_id in remaining_ids]
            solution       = vroom.solve(remaining_jobs, [vehicle], depot=env.depot, clock=env.clock)

            if solution is None:
                self.logger.warning(f"Execution re-solve failed for vehicle {vehicle.id}")
                new_state.add_unassigned(set(remaining_ids))
                continue

            if solution.routes:
                new_state.routes.append(solution.routes[0])
            new_state.add_unassigned(solution.unassigned_ids)

        return new_state, summary


class Environment:
    def __init__(self, config, logger=None):
        self.config   = config
        self.logger   = logger or NullLogger()
        self.jobs     = EntityPool()
        self.vehicles = EntityPool()

        self.depot = None
        self.clock = None

        self.current_state: RoutingState = None
        self.initial_state: RoutingState = None

        self.graph              = Graph(config)
        self.mask_builder       = ActionMaskBuilder()
        self.sampler            = ScenarioSampler(config.env)
        self.event_handler      = EventHandler(self.sampler)
        self.action_handler     = ActionHandler(self.logger)
        self.execution_advancer = ExecutionAdvancer(self.logger)

        self.reset()

    def reset(self):
        env = self.config.env

        for _ in range(env.reset_max_attempts):
            n_jobs     = max(env.min_jobs,     int(np.random.normal(loc=env.mean_jobs,     scale=env.std_jobs)))
            n_vehicles = max(env.min_vehicles, int(np.random.normal(loc=env.mean_vehicles, scale=env.std_vehicles)))

            self.jobs     = EntityPool(self.sampler.sample_jobs(EntityPool(), n_jobs))
            self.vehicles = EntityPool(self.sampler.sample_vehicles(EntityPool(), n_vehicles))
            self.depot    = self.sampler.sample_depot()
            self.clock    = min(vehicle.time_window[0] for vehicle in self.vehicles)

            solution = vroom.solve(list(self.jobs), list(self.vehicles), depot=self.depot, clock=self.clock)
            if solution is not None:
                self.current_state = solution
                self.initial_state = solution.copy()
                return

        raise RuntimeError(f"Environment.reset exhausted {env.reset_max_attempts} attempts without a VROOM solution (mean_jobs={env.mean_jobs}, mean_vehicles={env.mean_vehicles})")

    def sample_episode(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.reset()

        event_type, num_items = self.generate_event()
        self.apply_event(self.initial_state, event_type, num_items)

    def load_scenario(self, item):
        self.jobs          = EntityPool([Job.from_dict(job) for job in item["jobs"]])
        self.vehicles      = EntityPool([Vehicle.from_dict(vehicle) for vehicle in item["vehicles"]])
        self.depot         = (float(item["depot"][0]), float(item["depot"][1]))
        self.clock         = int(item["clock"])
        self.current_state = RoutingState.from_payload(item["state"])

    def observe(self, state=None):
        if state is None:
            state = self.current_state

        graph     = self.graph.build(self.jobs, self.vehicles, state, self.depot, self.clock)
        mask_info = self.mask_builder.build(self.jobs, self.vehicles, state)
        return graph, mask_info

    def generate_event(self):
        env = self.config.env
        event_type = random.choice(["new_job", "remove_job", "new_vehicle", "remove_vehicle"])

        if event_type == "new_job":
            num_items = random.randint(env.job_insert_min, env.job_insert_max)

        elif event_type == "new_vehicle":
            num_items = random.randint(env.vehicle_insert_min, env.vehicle_insert_max)

        elif event_type == "remove_job":
            total_jobs         = len(self.jobs)
            max_items_possible = min(env.job_remove_max, total_jobs)
            min_items_possible = min(env.job_remove_min, max_items_possible)
            num_items          = random.randint(min_items_possible, max_items_possible) if max_items_possible > 0 else 0

        elif event_type == "remove_vehicle":
            total_vehicles     = len(self.vehicles)
            max_items_possible = min(env.vehicle_remove_max, total_vehicles - 1)
            min_items_possible = min(env.vehicle_remove_min, max_items_possible)
            num_items          = random.randint(min_items_possible, max_items_possible) if max_items_possible > 0 else 0

        return event_type, num_items

    def apply_event(self, state: RoutingState, event_type, num_items=1) -> RoutingState:
        if event_type == "new_job":
            event_state = self.event_handler.apply_new_job(self, state, num_items)
        elif event_type == "new_vehicle":
            event_state = self.event_handler.apply_new_vehicle(self, state, num_items)
        elif event_type == "remove_job":
            event_state = self.event_handler.apply_remove_job(self, state, num_items)
        elif event_type == "remove_vehicle":
            event_state = self.event_handler.apply_remove_vehicle(self, state, num_items)
        else:
            raise ValueError(f"Unknown event type: {event_type!r}")

        self.current_state = event_state
        return event_state

    def apply_random_event(self):
        if random.random() >= self.config.env.step_event_probability:
            return

        event_type, num_items = self.generate_event()
        if num_items > 0:
            self.apply_event(self.current_state, event_type, num_items)

    def advance_execution(self):
        tick = self.config.env.tick_seconds
        if tick <= 0:
            return {"served": [], "failed": [], "dropped": 0}

        new_state, summary = self.execution_advancer.advance(self, self.current_state, tick)
        self.current_state = new_state
        return summary

    def apply_action_to(self, state, action):
        operator  = action.operator
        old_state = state.copy()

        if operator in (0, 1):
            vehicle_id = self.vehicles[action.vehicle_index].id
            job_id     = self.jobs[action.job_index].id

        if operator == 0:
            new_state = self.action_handler.apply_job_insertion(self, old_state, vehicle_id, job_id)
        elif operator == 1:
            new_state = self.action_handler.apply_job_removal(self, old_state, vehicle_id, job_id)
        elif operator == 2:
            new_state = old_state
        else:
            raise ValueError(f"Unknown operator index: {operator}")

        return old_state, new_state

    def evaluate_cost(self, state: RoutingState):
        base_cost       = float(state.cost)
        num_unassigned  = state.num_unassigned
        active_vehicles = state.num_routes
        idle_vehicles   = len(self.vehicles) - active_vehicles

        unassigned_priority_sum = sum(
            self.jobs.by_id(job_id).priority
            for job_id in state.unassigned_ids
            if self.jobs.contains(job_id)
        )

        reward_config   = self.config.reward
        distance_cost   = reward_config.distance_weight * base_cost / 1000
        unassigned_cost = reward_config.unassigned_penalty_weight * num_unassigned
        idle_cost       = reward_config.idle_penalty_weight * idle_vehicles
        priority_cost   = reward_config.priority_penalty_weight * unassigned_priority_sum

        return distance_cost, unassigned_cost, idle_cost, priority_cost

    def evaluate_disruption(self, old_state: RoutingState, new_state: RoutingState):
        old_vehicle_by_job = {stop.job_id: route.vehicle_id for route in old_state.routes for stop in route.stops}
        new_vehicle_by_job = {stop.job_id: route.vehicle_id for route in new_state.routes for stop in route.stops}

        return sum(1 for job_id, vehicle_id in old_vehicle_by_job.items() if new_vehicle_by_job.get(job_id) != vehicle_id)

    def step(self, old_state, new_state, operator_index):
        old_distance_cost, old_unassigned_cost, old_idle_cost, old_priority_cost = self.evaluate_cost(old_state)
        new_distance_cost, new_unassigned_cost, new_idle_cost, new_priority_cost = self.evaluate_cost(new_state)

        distance_reward   = -(new_distance_cost   - old_distance_cost)
        unassigned_reward = -(new_unassigned_cost - old_unassigned_cost)
        idle_reward       = -(new_idle_cost       - old_idle_cost)
        priority_reward   = -(new_priority_cost   - old_priority_cost)

        reward_config = self.config.reward
        operator_costs = {
            0: reward_config.add_job_cost,
            1: reward_config.remove_job_cost,
            2: reward_config.no_action_cost,
        }

        disruption    = self.evaluate_disruption(old_state, new_state)
        action_cost   = operator_costs[operator_index] + reward_config.disruption_cost * disruption
        action_reward = -action_cost

        costs = {
            "old_distance_cost"   : old_distance_cost,
            "old_unassigned_cost" : old_unassigned_cost,
            "old_idle_cost"       : old_idle_cost,
            "old_priority_cost"   : old_priority_cost,
            "new_distance_cost"   : new_distance_cost,
            "new_unassigned_cost" : new_unassigned_cost,
            "new_idle_cost"       : new_idle_cost,
            "new_priority_cost"   : new_priority_cost,
            "disruption"          : disruption,
            "action_cost"         : action_cost
        }

        rewards = {
            "distance_reward"   : distance_reward,
            "unassigned_reward" : unassigned_reward,
            "idle_reward"       : idle_reward,
            "priority_reward"   : priority_reward,
            "action_reward"     : action_reward
        }

        return rewards, costs
