import json
import math
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from tools.logger import Logger
from core.dataset import Dataset
from core.shared import Environment, vroom
from core.training.pretraining import RegretInsertionTeacher
from model.policy_model import Action, Policy, PolicyCheckpoint


class ModelAgent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, environment, graph, mask_info):
        with torch.no_grad():
            return self.policy.select_action(graph, mask_info=mask_info)["action"]


class TeacherAgent:
    def __init__(self, teacher):
        self.teacher = teacher

    def act(self, environment, graph, mask_info):
        return self.teacher.select_action(environment, environment.current_state)


class FixedOperatorAgent:
    def __init__(self, operator):
        self.operator = operator

    def act(self, environment, graph, mask_info):
        return Action(operator=self.operator, vehicle_index=0, job_index=0)


class EpisodeEvaluator:
    def __init__(self, environment, config):
        self.environment = environment
        self.config      = config
        self.max_steps   = config.training.max_steps_per_episode

    def run(self, agent, dataset_item, episode_seed):
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)

        self.environment.load_from_dataset(dataset_item)

        total_reward    = 0.0
        operator_counts = {operator: 0 for operator in range(4)}

        for step_in_episode in range(self.max_steps):
            if step_in_episode > 0:
                self.environment.apply_random_event()

            graph, mask_info = self.environment.observe()
            action           = agent.act(self.environment, graph, mask_info)

            old_state, new_state = self.environment.apply_action(action)
            rewards, _           = self.environment.step(old_state, new_state, action.operator)

            total_reward += float(sum(rewards.values()))
            operator_counts[action.operator] += 1

            self.environment.current_state = new_state

        final_state = self.environment.current_state
        return {
            "total_reward"     : total_reward,
            "final_cost"       : float(final_state.cost),
            "final_unassigned" : int(final_state.num_unassigned),
            "operator_counts"  : operator_counts,
        }


class EvaluationPipeline:
    def __init__(self, config, repo_root, logger=None):
        self.config    = config
        self.repo_root = repo_root
        self.logger    = logger or Logger(name="evaluation")

        self.run_dir     = None
        self.dataset_dir = None
        self.items       = None
        self.environment = None
        self.model       = None
        self.agents      = None
        self.results     = None

    def resolve_run(self):
        run_name = self.config.io.run_name
        if not run_name:
            raise ValueError("config.io.run_name must name the run directory whose checkpoint is evaluated")

        runs_root = self.config.io.runs_dir
        if not os.path.isabs(runs_root):
            runs_root = os.path.join(str(self.repo_root), runs_root)

        self.run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def load_items(self):
        self.dataset_dir = self.config.io.dataset_dir
        if not os.path.isabs(self.dataset_dir):
            self.dataset_dir = os.path.join(str(self.repo_root), self.dataset_dir)

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        requested = self.config.evaluation.episodes
        dataset   = Dataset(dataset_dir=self.dataset_dir, config=self.config, shuffle_chunks=False, logger=self.logger)

        self.items = []
        for item in dataset:
            self.items.append(item)
            if len(self.items) >= requested:
                break

        if len(self.items) < requested:
            raise ValueError(f"Evaluation requested {requested} episodes but dataset {self.dataset_dir} holds only {len(self.items)} items")

    def build_environment(self):
        vroom.logger     = self.logger
        self.environment = Environment(self.config, logger=self.logger)

    def load_model(self):
        self.model = Policy(self.config)
        self.model.to(self.model.device)
        PolicyCheckpoint().load(self.model, self.config.io.checkpoint_filename, self.run_dir, map_location=self.model.device)
        self.model.eval()

    def build_agents(self):
        self.agents = {
            "model"             : ModelAgent(self.model),
            "teacher"           : TeacherAgent(RegretInsertionTeacher(self.config)),
            "insertion_only"    : TeacherAgent(RegretInsertionTeacher(self.config, reoptimize_margin=math.inf)),
            "always_reoptimize" : FixedOperatorAgent(3),
            "do_nothing"        : FixedOperatorAgent(2),
        }

    def aggregate(self, episodes):
        rewards    = [episode["total_reward"] for episode in episodes]
        costs      = [episode["final_cost"] for episode in episodes]
        unassigned = [episode["final_unassigned"] for episode in episodes]

        total_actions   = sum(sum(episode["operator_counts"].values()) for episode in episodes)
        operator_totals = {operator: sum(episode["operator_counts"][operator] for episode in episodes) for operator in range(4)}

        return {
            "episodes"              : len(episodes),
            "mean_reward"           : float(np.mean(rewards)),
            "std_reward"            : float(np.std(rewards)),
            "mean_final_cost"       : float(np.mean(costs)),
            "mean_final_unassigned" : float(np.mean(unassigned)),
            "operator_frequency"    : {f"op{operator}": count / total_actions for operator, count in operator_totals.items()},
        }

    def evaluate(self):
        evaluator = EpisodeEvaluator(self.environment, self.config)
        seed      = self.config.evaluation.seed

        self.results = {}
        for name, agent in self.agents.items():
            episodes = []
            for index, item in enumerate(tqdm(self.items, desc=f"Evaluating {name}", unit="episode")):
                episodes.append(evaluator.run(agent, item, seed + index))

            self.results[name] = self.aggregate(episodes)

        return self.results

    def report(self):
        for name, metrics in self.results.items():
            flat = {key: value for key, value in metrics.items() if key != "operator_frequency"}
            flat.update(metrics["operator_frequency"])
            self.logger.kv_table(flat, title=f"Evaluation: {name}")

        report_path = os.path.join(self.run_dir, "evaluation.json")
        with open(report_path, "w") as handle:
            json.dump(self.results, handle, indent=2)

        self.logger.info(f"Evaluation report saved to {report_path}")

    def run(self):
        self.resolve_run()
        self.load_items()
        self.build_environment()
        self.load_model()
        self.build_agents()
        self.evaluate()
        self.report()
        return self.results
