import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch

from core.shared import Environment, EpisodeDriver, RoutingState
from model.policy_model import Action, Policy, PolicyCheckpoint


@dataclass
class InferenceStep:
    step_number    : int
    state          : RoutingState
    round_index    : int = 0
    action         : Optional[Action] = None
    action_info    : Optional[Dict] = None
    reward_info    : Optional[Dict] = None
    cost           : int = 0
    num_routes     : int = 0
    num_unassigned : int = 0

    def to_dict(self) -> Dict:
        return {
            "step_number"    : self.step_number,
            "round_index"    : self.round_index,
            "cost"           : self.cost,
            "num_routes"     : self.num_routes,
            "num_unassigned" : self.num_unassigned,
            "operator"       : self.action.operator if self.action else None,
            "vehicle_index"  : self.action.vehicle_index if self.action else None,
            "job_index"      : self.action.job_index if self.action else None,
        }


@dataclass
class InferenceResult:
    steps          : List[InferenceStep] = field(default_factory=list)
    initial_state  : Optional[RoutingState] = None
    final_state    : Optional[RoutingState] = None
    stopped_reason : str = ""
    total_steps    : int = 0
    execution      : Dict = field(default_factory=dict)

    def get_initial_cost(self) -> int:
        return self.initial_state.cost if self.initial_state else 0

    def get_final_cost(self) -> int:
        return self.final_state.cost if self.final_state else 0

    def get_cost_improvement(self) -> int:
        return self.get_initial_cost() - self.get_final_cost()

    def get_cost_improvement_percentage(self) -> float:
        initial = self.get_initial_cost()
        if initial == 0:
            return 0.0
        return (self.get_cost_improvement() / initial) * 100.0

    def summary(self) -> Dict:
        return {
            "total_steps"          : self.total_steps,
            "stopped_reason"       : self.stopped_reason,
            "initial_cost"         : self.get_initial_cost(),
            "final_cost"           : self.get_final_cost(),
            "cost_improvement"     : self.get_cost_improvement(),
            "cost_improvement_pct" : self.get_cost_improvement_percentage(),
            "initial_routes"       : self.initial_state.num_routes if self.initial_state else 0,
            "final_routes"         : self.final_state.num_routes if self.final_state else 0,
            "initial_unassigned"   : self.initial_state.num_unassigned if self.initial_state else 0,
            "final_unassigned"     : self.final_state.num_unassigned if self.final_state else 0,
            "served"               : self.execution.get("served", 0),
            "failed"               : self.execution.get("failed", 0),
            "dropped"              : self.execution.get("dropped", 0),
        }

    def get_trajectory(self) -> List[Dict]:
        return [step.to_dict() for step in self.steps]


class ModelInference:
    OPERATOR_NAMES = {0: "INSERT", 1: "REMOVE", 2: "DO_NOTHING"}

    def __init__(self, model: Policy, environment: Environment, config, device: str = "cuda", greedy: bool = True):
        self.model       = model
        self.environment = environment
        self.driver      = EpisodeDriver(environment, config)
        self.device      = device
        self.greedy      = greedy

        self.model.eval()
        self.model.to(self.device)

    def _select(self, graph, mask_info):
        with torch.no_grad():
            return self.model.select_action(graph, mask_info=mask_info, greedy=self.greedy)["action"]

    def _record(self, result, step, action, state, rewards, costs):
        result.steps.append(
            InferenceStep(
                step_number    = len(result.steps),
                round_index    = step.round_index,
                state          = state.copy(),
                action         = action,
                action_info    = {
                    "operator"      : action.operator,
                    "operator_name" : self.OPERATOR_NAMES.get(action.operator, f"UNKNOWN({action.operator})"),
                    "vehicle_index" : action.vehicle_index,
                    "job_index"     : action.job_index,
                },
                reward_info    = {**rewards, **costs},
                cost           = state.cost,
                num_routes     = state.num_routes,
                num_unassigned = state.num_unassigned,
            )
        )

    def run(self, seed: int):
        result = InferenceResult()

        for step in self.driver.episode(seed):
            if result.initial_state is None:
                result.initial_state = self.environment.current_state.copy()

            action                    = self._select(step.graph, step.mask_info)
            _, new_state, rewards, costs = step.commit(action)

            self._record(result, step, action, new_state, rewards, costs)

        result.final_state    = self.environment.current_state.copy()
        result.total_steps    = len(result.steps)
        result.execution      = dict(self.driver.totals)
        result.stopped_reason = "episode_complete"

        return result


class InferencePipeline:
    def __init__(self, config, repo_root, logger):
        self.config    = config
        self.repo_root = repo_root
        self.logger    = logger

        self.device      = config.training.device
        self.run_dir     = None
        self.environment = None
        self.model       = None
        self.result      = None

    def _resolve_run(self):
        run_name = self.config.io.run_name
        if not run_name:
            raise ValueError("config.io.run_name must name the run directory to load for inference")

        runs_root = self.config.io.runs_dir
        if not os.path.isabs(runs_root):
            runs_root = os.path.join(str(self.repo_root), runs_root)

        self.run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def _build_environment(self):
        self.environment = Environment(self.config, logger=self.logger)

    def _load_model(self):
        self.model = Policy(self.config)
        PolicyCheckpoint().load(self.model, self.config.io.checkpoint_filename, self.run_dir, map_location=self.device)

    def _infer(self):
        inference   = ModelInference(self.model, self.environment, self.config, device=self.device, greedy=self.config.inference.greedy)
        self.result = inference.run(self.config.inference.seed)

    def _report(self):
        self.logger.kv_table(self.result.summary(), title="Inference Summary")

    def run(self):
        self._resolve_run()
        self._build_environment()
        self._load_model()
        self._infer()
        self._report()
        return self.result
