import json
import os
from dataclasses import asdict

import numpy as np
import torch
from tqdm import tqdm

from tools.logger import Logger
from core.shared import Environment, EpisodeDriver, vroom
from core.training.pretraining import RegretInsertionTeacher
from model.policy_model import Action, Policy, PolicyCheckpoint


class ModelAgent:
    def __init__(self, policy, greedy=False):
        self.policy      = policy
        self.greedy      = greedy
        self.last_output = None

    def act(self, environment, graph, mask_info, remaining_steps):
        with torch.no_grad():
            self.last_output = self.policy.select_action(graph, mask_info=mask_info, greedy=self.greedy)

        return self.last_output["action"]


class TeacherAgent:
    def __init__(self, teacher):
        self.teacher = teacher

    def act(self, environment, graph, mask_info, remaining_steps):
        return self.teacher.select_action(environment, environment.current_state, remaining_steps)


class FixedOperatorAgent:
    def __init__(self, operator):
        self.operator = operator

    def act(self, environment, graph, mask_info, remaining_steps):
        return Action(operator=self.operator, vehicle_index=0, job_index=0)


class EpisodeEvaluator:
    def __init__(self, environment, config):
        self.environment = environment
        self.config      = config
        self.driver      = EpisodeDriver(environment, config)

    def run(self, agent, episode_seed):
        total_reward    = 0.0
        operator_counts = {operator: 0 for operator in range(3)}

        for step in self.driver.episode(episode_seed):
            action = agent.act(self.environment, step.graph, step.mask_info, step.remaining)

            _, _, rewards, _ = step.commit(action)

            total_reward += float(sum(rewards.values()))
            operator_counts[action.operator] += 1

        final_state = self.environment.current_state
        return {
            "total_reward"     : total_reward,
            "final_cost"       : float(final_state.cost),
            "final_unassigned" : int(final_state.num_unassigned),
            "served"           : int(self.driver.totals["served"]),
            "failed"           : int(self.driver.totals["failed"]),
            "dropped"          : int(self.driver.totals["dropped"]),
            "operator_counts"  : operator_counts,
        }


class EvaluationPipeline:
    def __init__(self, config, repo_root, logger=None):
        self.config    = config
        self.repo_root = repo_root
        self.logger    = logger or Logger(name="evaluation")

        self.run_dir     = None
        self.environment = None
        self.model       = None
        self.agents      = None
        self.results     = None

    def _resolve_run(self):
        run_name = self.config.io.run_name
        if not run_name:
            raise ValueError("config.io.run_name must name the run directory whose checkpoint is evaluated")

        runs_root = self.config.io.runs_dir
        if not os.path.isabs(runs_root):
            runs_root = os.path.join(str(self.repo_root), runs_root)

        self.run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

    def _validate_held_out_seeds(self):
        training   = self.config.training
        evaluation = self.config.evaluation

        train_episodes = max(self.config.pretrain.episodes, training.num_updates * training.episodes_per_update)
        train_band     = range(self.config.env.scenario_seed, self.config.env.scenario_seed + train_episodes)
        eval_band      = range(evaluation.seed, evaluation.seed + evaluation.episodes)

        overlap = range(max(train_band.start, eval_band.start), min(train_band.stop, eval_band.stop))
        if len(overlap) > 0:
            raise ValueError(
                f"evaluation seeds [{eval_band.start}, {eval_band.stop}) overlap the training seeds "
                f"[{train_band.start}, {train_band.stop}) on {len(overlap)} episodes; the evaluation set is not held out"
            )

    def _build_environment(self):
        vroom.logger     = self.logger
        self.environment = Environment(self.config, logger=self.logger)

    def _load_model(self):
        self.model = Policy(self.config)
        self.model.to(self.model.device)
        PolicyCheckpoint().load(self.model, self.config.io.checkpoint_filename, self.run_dir, map_location=self.model.device)
        self.model.eval()

    def _build_agents(self):
        self.agents = {
            "model"          : ModelAgent(self.model, greedy=self.config.evaluation.greedy),
            "teacher"        : TeacherAgent(RegretInsertionTeacher(self.config)),
            "insertion_only" : TeacherAgent(RegretInsertionTeacher(self.config, allow_removal=False)),
            "do_nothing"     : FixedOperatorAgent(2),
        }

    def _aggregate(self, episodes):
        rewards    = [episode["total_reward"] for episode in episodes]
        costs      = [episode["final_cost"] for episode in episodes]
        unassigned = [episode["final_unassigned"] for episode in episodes]

        total_actions   = sum(sum(episode["operator_counts"].values()) for episode in episodes)
        operator_totals = {operator: sum(episode["operator_counts"][operator] for episode in episodes) for operator in range(3)}

        served = [episode["served"] for episode in episodes]

        return {
            "episodes"              : len(episodes),
            "mean_reward"           : float(np.mean(rewards)),
            "std_reward"            : float(np.std(rewards)),
            "sem_reward"            : float(np.std(rewards) / max(np.sqrt(len(rewards)), 1.0)),
            "mean_final_cost"       : float(np.mean(costs)),
            "std_final_cost"        : float(np.std(costs)),
            "mean_final_unassigned" : float(np.mean(unassigned)),
            "mean_served"           : float(np.mean(served)),
            "mean_failed"           : float(np.mean([episode["failed"] for episode in episodes])),
            "mean_dropped"          : float(np.mean([episode["dropped"] for episode in episodes])),
            "cost_per_served"       : float(np.sum(costs) / max(np.sum(served), 1.0)),
            "operator_frequency"    : {f"op{operator}": count / total_actions for operator, count in operator_totals.items()},
            "per_episode"           : {
                "reward"     : [float(value) for value in rewards],
                "final_cost" : [float(value) for value in costs],
                "unassigned" : [int(value) for value in unassigned],
                "served"     : [int(value) for value in served],
            },
        }

    def _evaluate(self):
        evaluator      = EpisodeEvaluator(self.environment, self.config)
        seed           = self.config.evaluation.seed
        episodes_count = self.config.evaluation.episodes

        self.results = {}
        for name, agent in self.agents.items():
            episodes = []
            for index in tqdm(range(episodes_count), desc=f"Evaluating {name}", unit="episode"):
                episodes.append(evaluator.run(agent, seed + index))

            self.results[name] = self._aggregate(episodes)
            self._write_report()

        return self.results

    def _meta(self):
        return {
            "run_name"            : self.config.io.run_name,
            "checkpoint_filename" : self.config.io.checkpoint_filename,
            "seed"                : self.config.evaluation.seed,
            "episodes"            : self.config.evaluation.episodes,
            "greedy"              : self.config.evaluation.greedy,
            "scenario_seed"       : self.config.env.scenario_seed,
            "reward"              : asdict(self.config.reward),
            "environment"         : asdict(self.config.env),
            "caps"                : {
                "max_disruptions_per_episode" : self.config.training.max_disruptions_per_episode,
                "max_edits_per_round"         : self.config.training.max_edits_per_round,
            },
        }

    def _write_report(self):
        report_path = os.path.join(self.run_dir, f"evaluation_seed{self.config.evaluation.seed}.json")
        with open(report_path, "w") as handle:
            json.dump({"meta": self._meta(), "agents": self.results}, handle, indent=2)

        return report_path

    def _report(self):
        for name, metrics in self.results.items():
            flat = {key: value for key, value in metrics.items() if key not in ("operator_frequency", "per_episode")}
            flat.update(metrics["operator_frequency"])
            self.logger.kv_table(flat, title=f"Evaluation: {name}")

        self.logger.info(f"Evaluation report saved to {self._write_report()}")

    def run(self):
        self._resolve_run()
        self._validate_held_out_seeds()
        self._build_environment()
        self._load_model()
        self._build_agents()
        self._evaluate()
        self._report()
        return self.results
