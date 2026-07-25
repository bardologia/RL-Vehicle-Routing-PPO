import random

import numpy as np
import torch

from tools.logger import NullLogger
from .actions import ActionHandler
from .events import EventGenerator, EventHandler
from .execution import ExecutionAdvancer
from .graph import Graph
from .mask import ActionMaskBuilder
from .rewards import RewardCalculator
from .sampling import ScenarioSampler
from .services import vroom
from .state import EntityPool, Job, RoutingState, Vehicle


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
        self.event_generator    = EventGenerator()
        self.event_handler      = EventHandler(self.sampler)
        self.action_handler     = ActionHandler(self.logger)
        self.execution_advancer = ExecutionAdvancer(self.logger)
        self.reward_calculator  = RewardCalculator()

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
        self.apply_disruption(self.initial_state)

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
        return self.event_generator.generate(self)

    def apply_event(self, state: RoutingState, event_type, num_items=1) -> RoutingState:
        event_state        = self.event_handler.apply(self, state, event_type, num_items)
        self.current_state = event_state
        return event_state

    def apply_disruption(self, state):
        for _ in range(self.config.env.reset_max_attempts):
            event_type, num_items = self.generate_event()
            if num_items > 0:
                self.apply_event(state, event_type, num_items)
                return {"type": event_type, "count": num_items}

        num_items = self.config.env.job_insert_min
        self.apply_event(state, "new_job", num_items)
        return {"type": "new_job", "count": num_items}

    def sample_disruption_gap(self):
        env = self.config.env
        gap = np.random.exponential(env.disruption_gap_mean)
        return int(min(max(gap, env.disruption_gap_min), env.disruption_gap_max))

    def advance_execution(self, gap_seconds):
        if gap_seconds <= 0:
            return {"served": [], "failed": [], "dropped": 0}

        new_state, summary = self.execution_advancer.advance(self, self.current_state, gap_seconds)
        self.current_state = new_state
        return summary

    def apply_action_to(self, state, action):
        old_state = state.copy()
        new_state = self.action_handler.apply(self, old_state, action)
        return old_state, new_state

    def step(self, old_state, new_state, operator_index):
        return self.reward_calculator.evaluate(self, old_state, new_state, operator_index)
