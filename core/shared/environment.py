import random
import numpy as np

from tools.auxiliary import generate_coords_batch
from .graph import Graph
from .mask import MaskContext
from .services import vroom
from .state import EntityPool, Job, Route, RoutingState, Vehicle
from tools.logger import NullLogger


class ScenarioSampler:
    def __init__(self, env_config):
        self.env = env_config

    def sample_jobs(self, pool: EntityPool, count: int):
        coords     = generate_coords_batch(self.env.center, self.env.radius, count, self.env.outlier_probability, self.env.outlier_multiplier)
        priorities = np.random.choice([1, 2, 3, 4, 5], size=count)
        first_id   = pool.next_id()

        return [
            Job(
                id          = first_id + i,
                location    = (float(coords[i][0]), float(coords[i][1])),
                priority    = int(priorities[i]),
                description = f"Job {first_id + i}",
            )
            for i in range(count)
        ]

    def sample_vehicles(self, pool: EntityPool, count: int):
        coords     = generate_coords_batch(self.env.center, self.env.radius, count, 0, self.env.outlier_multiplier)
        capacities = np.random.choice([1, 2, 3], size=count)
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

    def _solve_vehicle_route(self, jobs, vehicle: Vehicle):
        solution = vroom.solve(jobs, [vehicle])
        if solution is None or not solution.routes:
            return None

        return solution.routes[0]

    def apply_job_insertion(self, env, state: RoutingState, vehicle_id: int, job_id: int) -> RoutingState:
        route           = state.route_of_vehicle(vehicle_id)
        current_job_ids = route.job_ids if route is not None else []

        vehicle = env.vehicles.by_id(vehicle_id)
        if len(current_job_ids) >= vehicle.capacity:
            vehicle = vehicle.with_capacity(len(current_job_ids) + 1)

        candidate_ids  = current_job_ids + [job_id]
        candidate_jobs = [env.jobs.by_id(jid) for jid in candidate_ids if env.jobs.contains(jid)]

        new_route = self._solve_vehicle_route(candidate_jobs, vehicle)
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

        remaining_ids = [jid for jid in route.job_ids if jid != job_id]
        new_state     = state.copy()

        if not remaining_ids:
            orphaned = new_state.remove_vehicles({vehicle_id})
            new_state.add_unassigned(orphaned)
            return new_state

        remaining_jobs = [env.jobs.by_id(jid) for jid in remaining_ids if env.jobs.contains(jid)]
        vehicle        = env.vehicles.by_id(vehicle_id)

        new_route = self._solve_vehicle_route(remaining_jobs, vehicle)
        if new_route is None:
            self.logger.warning(f"Job removal failed for vehicle_id={vehicle_id} job_id={job_id}")
            return state

        new_state.replace_route(new_route)
        new_state.add_unassigned({job_id})
        return new_state

    def apply_reoptimize(self, env, state: RoutingState) -> RoutingState:
        solution = vroom.solve(list(env.jobs), list(env.vehicles))
        if solution is None:
            self.logger.warning("Reoptimize failed: VROOM returned no solution")
            return state

        return solution


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


class Environment:
    def __init__(self, config, logger=None):
        self.config   = config
        self.logger   = logger or NullLogger()
        self.jobs     = EntityPool()
        self.vehicles = EntityPool()

        self.current_state: RoutingState = None
        self.initial_state: RoutingState = None

        self.graph          = Graph(config)
        self.mask_context   = MaskContext()
        self.sampler        = ScenarioSampler(config.env)
        self.event_handler  = EventHandler(self.sampler)
        self.action_handler = ActionHandler(self.logger)

        self.reset()

    def reset(self):
        env = self.config.env

        while True:
            n_jobs     = max(env.min_jobs,     int(np.random.normal(loc=env.mean_jobs,     scale=env.std_jobs)))
            n_vehicles = max(env.min_vehicles, int(np.random.normal(loc=env.mean_vehicles, scale=env.std_vehicles)))

            self.jobs     = EntityPool(self.sampler.sample_jobs(EntityPool(), n_jobs))
            self.vehicles = EntityPool(self.sampler.sample_vehicles(EntityPool(), n_vehicles))

            solution = vroom.solve(list(self.jobs), list(self.vehicles))
            if solution is None:
                continue

            self.current_state = solution
            self.initial_state = solution.copy()
            break

    def load_from_dataset(self, item):
        self.jobs          = EntityPool([Job.from_dict(job) for job in item["jobs"]])
        self.vehicles      = EntityPool([Vehicle.from_dict(vehicle) for vehicle in item["vehicles"]])
        self.current_state = RoutingState.from_payload(item["state"])

    def observe(self):
        graph        = self.graph.build(self.jobs, self.vehicles, self.current_state)
        mask_context = self.mask_context.build(self)
        return graph, mask_context

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
            event_state = state.copy()

        self.current_state = event_state
        return event_state

    def apply_action(self, action):
        operator      = action.operator
        vehicle_index = action.vehicle_index
        job_index     = action.job_index

        old_state  = self.current_state.copy()
        vehicle_id = self.vehicles[vehicle_index].id
        job_id     = self.jobs[job_index].id

        if operator == 0:
            new_state = self.action_handler.apply_job_insertion(self, old_state, vehicle_id, job_id)
        elif operator == 1:
            new_state = self.action_handler.apply_job_removal(self, old_state, vehicle_id, job_id)
        elif operator == 2:
            new_state = old_state
        elif operator == 3:
            new_state = self.action_handler.apply_reoptimize(self, old_state)
        else:
            new_state = old_state

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

        reward_cfg      = self.config.reward
        distance_cost   = reward_cfg.distance_weight * base_cost / 1000
        unassigned_cost = reward_cfg.unassigned_penalty_weight * num_unassigned
        idle_cost       = reward_cfg.idle_penalty_weight * idle_vehicles
        priority_cost   = reward_cfg.priority_penalty_weight * unassigned_priority_sum

        return distance_cost, unassigned_cost, idle_cost, priority_cost

    def step(self, old_state, new_state, operator_idx):
        old_distance_cost, old_unassigned_cost, old_idle_cost, old_priority_cost = self.evaluate_cost(old_state)
        new_distance_cost, new_unassigned_cost, new_idle_cost, new_priority_cost = self.evaluate_cost(new_state)

        distance_reward   = -(new_distance_cost   - old_distance_cost)
        unassigned_reward = -(new_unassigned_cost - old_unassigned_cost)
        idle_reward       = -(new_idle_cost       - old_idle_cost)
        priority_reward   = -(new_priority_cost   - old_priority_cost)

        reward_cfg = self.config.reward
        action_penalties = {
            0: reward_cfg.add_job_penalty,
            1: reward_cfg.remove_job_penalty,
            2: reward_cfg.invalid_action_penalty,
            3: reward_cfg.reoptimize_penalty
        }
        action_reward = action_penalties.get(operator_idx, 0)

        costs = {
            "old_distance_cost"   : old_distance_cost,
            "old_unassigned_cost" : old_unassigned_cost,
            "old_idle_cost"       : old_idle_cost,
            "old_priority_cost"   : old_priority_cost,
            "new_distance_cost"   : new_distance_cost,
            "new_unassigned_cost" : new_unassigned_cost,
            "new_idle_cost"       : new_idle_cost,
            "new_priority_cost"   : new_priority_cost
        }

        rewards = {
            "distance_reward"   : distance_reward,
            "unassigned_reward" : unassigned_reward,
            "idle_reward"       : idle_reward,
            "priority_reward"   : priority_reward,
            "action_reward"     : action_reward
        }

        return rewards, costs
