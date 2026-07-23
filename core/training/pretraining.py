import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch
from tqdm import tqdm

from tools.logger import Logger, NullLogger
from tools.telemetry import PPOTelemetry
from core.dataset import Dataset
from core.shared import ActionMasker, Environment, vroom
from model.policy_model import Action, Policy, PolicyCheckpoint
from .ppo import PPOMemory
from .session import RunDirectory


class RegretInsertionTeacher:
    def __init__(self, config, reoptimize_margin=None, allow_removal=True):
        self.config            = config
        self.reoptimize_margin = reoptimize_margin
        self.allow_removal     = allow_removal

    def no_op_reward(self, environment, state):
        rewards, _ = environment.step(state, state, 2)
        return sum(rewards.values())

    def insertion_options(self, environment, state):
        assigned_ids = state.assigned_job_ids
        eligible_ids = sorted(job_id for job_id in state.unassigned_ids if environment.jobs.contains(job_id) and job_id not in assigned_ids)

        load_by_vehicle_id = {route.vehicle_id: len(route.stops) for route in state.routes}
        open_vehicles      = [vehicle for vehicle in environment.vehicles if load_by_vehicle_id.get(vehicle.id, 0) < vehicle.capacity]

        options = {}
        for job_id in eligible_ids:
            for vehicle in open_vehicles:
                new_state = environment.action_handler.apply_job_insertion(environment, state, vehicle.id, job_id)
                if new_state is state:
                    continue

                rewards, _ = environment.step(state, new_state, 0)
                options.setdefault(job_id, []).append((sum(rewards.values()), vehicle.id))

        return options

    def best_insertion(self, options, baseline):
        best = None

        for job_id in sorted(options):
            scored = sorted((entry for entry in options[job_id] if entry[0] >= baseline), key=lambda entry: (-entry[0], entry[1]))
            if not scored:
                continue

            top_reward, top_vehicle = scored[0]
            regret = math.inf if len(scored) == 1 else top_reward - scored[1][0]

            if best is None or (regret, top_reward) > (best["regret"], best["reward"]):
                best = {"regret": regret, "reward": top_reward, "job_id": job_id, "vehicle_id": top_vehicle}

        return best

    def refresh_options(self, environment, state, options, inserted_job_id, vehicle_id):
        options.pop(inserted_job_id, None)

        vehicle = environment.vehicles.by_id(vehicle_id)
        route   = state.route_of_vehicle(vehicle_id)
        is_open = (len(route.stops) if route is not None else 0) < vehicle.capacity

        for job_id in list(options):
            entries = [entry for entry in options[job_id] if entry[1] != vehicle_id]

            if is_open:
                new_state = environment.action_handler.apply_job_insertion(environment, state, vehicle_id, job_id)
                if new_state is not state:
                    rewards, _ = environment.step(state, new_state, 0)
                    entries.append((sum(rewards.values()), vehicle_id))

            if entries:
                options[job_id] = entries
            else:
                options.pop(job_id)

        return options

    def insertion_plan(self, environment, state, horizon, baseline):
        plan = {"value": 0.0, "action": None}
        if horizon <= 0:
            return plan

        discount = 1.0
        options  = self.insertion_options(environment, state)

        for _ in range(horizon):
            best = self.best_insertion(options, baseline)
            if best is None:
                break

            new_state = environment.action_handler.apply_job_insertion(environment, state, best["vehicle_id"], best["job_id"])
            if new_state is state:
                break

            if plan["action"] is None:
                plan["action"] = Action(operator=0, vehicle_index=environment.vehicles.index_of(best["vehicle_id"]), job_index=environment.jobs.index_of(best["job_id"]))

            rewards, _     = environment.step(state, new_state, 0)
            plan["value"] += discount * sum(rewards.values())
            discount      *= self.config.ppo.gamma
            state          = new_state
            options        = self.refresh_options(environment, state, options, best["job_id"], best["vehicle_id"])

        return plan

    def removal_options(self, environment, state):
        options = []

        for route in state.routes:
            if environment.vehicles.index_of(route.vehicle_id) is None:
                continue

            for job_id in route.job_ids:
                if not environment.jobs.contains(job_id):
                    continue

                new_state = environment.action_handler.apply_job_removal(environment, state, route.vehicle_id, job_id)
                if new_state is state:
                    continue

                rewards, _ = environment.step(state, new_state, 1)
                options.append({"reward": sum(rewards.values()), "vehicle_id": route.vehicle_id, "job_id": job_id, "state": new_state})

        return options

    def best_removal(self, environment, state, horizon, baseline):
        gamma   = self.config.ppo.gamma
        options = sorted(self.removal_options(environment, state), key=lambda option: (-option["reward"], option["vehicle_id"], option["job_id"]))

        best = None
        for option in options:
            continuation = self.insertion_plan(environment, option["state"], horizon - 1, baseline)
            value        = option["reward"] + gamma * continuation["value"]

            if best is None or value > best["value"]:
                best = {"value": value, "vehicle_id": option["vehicle_id"], "job_id": option["job_id"]}

        return best

    def reoptimize_outcome(self, environment, state):
        new_state = environment.action_handler.apply_reoptimize(environment, state)
        if new_state is state:
            return None, None

        rewards, _ = environment.step(state, new_state, 3)
        return sum(rewards.values()), new_state

    def select_action(self, environment, state):
        baseline = self.no_op_reward(environment, state)
        horizon  = self.config.pretrain.plan_horizon
        gamma    = self.config.ppo.gamma

        plan = self.insertion_plan(environment, state, horizon, baseline)

        chosen_value = baseline
        action       = Action(operator=2, vehicle_index=0, job_index=0)

        if plan["action"] is not None and plan["value"] >= baseline:
            chosen_value = plan["value"]
            action       = plan["action"]

        if self.allow_removal:
            removal = self.best_removal(environment, state, horizon, baseline)
            if removal is not None and removal["value"] > chosen_value:
                chosen_value = removal["value"]
                action       = Action(operator=1, vehicle_index=environment.vehicles.index_of(removal["vehicle_id"]), job_index=environment.jobs.index_of(removal["job_id"]))

        margin = self.reoptimize_margin if self.reoptimize_margin is not None else self.config.pretrain.reoptimize_margin
        if margin == math.inf:
            return action

        reoptimize_reward, reoptimize_state = self.reoptimize_outcome(environment, state)
        if reoptimize_reward is not None:
            continuation     = self.insertion_plan(environment, reoptimize_state, horizon - 1, baseline)
            reoptimize_value = reoptimize_reward + gamma * continuation["value"]

            if reoptimize_value > chosen_value + margin:
                action = Action(operator=3, vehicle_index=0, job_index=0)

        return action


class TeacherRolloutCollector:
    def __init__(self, environment, teacher, config, logger=None):
        self.environment = environment
        self.teacher     = teacher
        self.config      = config
        self.logger      = logger or NullLogger()

        self.max_steps = config.training.max_steps_per_episode
        self.gamma     = config.ppo.gamma

    def attach_returns(self, records):
        running = 0.0
        for record in reversed(records):
            running          = record["reward"] + self.gamma * running
            record["return"] = running

        return records

    def rollout(self, dataset_item):
        self.environment.load_from_dataset(dataset_item)

        records = []
        for step_in_episode in range(self.max_steps):
            if step_in_episode > 0:
                self.environment.apply_random_event()

            graph, mask_info = self.environment.observe()
            action           = self.teacher.select_action(self.environment, self.environment.current_state)

            old_state, new_state = self.environment.apply_action(action)
            rewards, _           = self.environment.step(old_state, new_state, action.operator)

            records.append({
                "graph"     : PPOMemory._clone_detached(graph),
                "mask_info" : mask_info,
                "action"    : action,
                "reward"    : float(sum(rewards.values())),
            })

            self.environment.current_state = new_state

        return self.attach_returns(records)


class BCTrainer:
    def __init__(self, policy, masker, config, telemetry, logger=None):
        self.policy    = policy
        self.masker    = masker
        self.config    = config
        self.telemetry = telemetry
        self.logger    = logger or NullLogger()

        pretrain = config.pretrain

        self.epochs          = pretrain.bc_epochs
        self.minibatch_size  = pretrain.minibatch_size
        self.value_loss_coef = pretrain.value_loss_coef
        self.max_norm        = pretrain.gradient_clip_max_norm
        self.device          = policy.device
        self.optimizer       = optim.Adam(policy.parameters(), lr=pretrain.lr, eps=1e-5)

        self.global_batch_step = 0

    def sample_losses(self, sample, record):
        action    = record["action"]
        mask_info = record["mask_info"]

        logits = self.policy.compute_logits(
            actor_embeddings  = sample["embeddings"],
            global_context    = sample["context"],
            operator_logits   = sample["operator_logits"],
            selected_operator = None,
        )

        masked_operator = self.masker.mask_operator(logits["operator_logits"], mask_info).float()
        masked_vehicle  = self.masker.mask_vehicle(logits["vehicle_logits"][action.operator], mask_info, action.operator).float()
        masked_job      = self.masker.mask_job(logits["job_logits"][action.operator, action.vehicle_index], mask_info, action.operator, action.vehicle_index).float()

        operator_target = torch.tensor(action.operator, device=self.device)
        vehicle_target  = torch.tensor(action.vehicle_index, device=self.device)
        job_target      = torch.tensor(action.job_index, device=self.device)

        operator_loss = -torch.distributions.Categorical(logits=masked_operator).log_prob(operator_target)
        vehicle_loss  = -torch.distributions.Categorical(logits=masked_vehicle).log_prob(vehicle_target)
        job_loss      = -torch.distributions.Categorical(logits=masked_job).log_prob(job_target)

        target_return = torch.tensor(record["return"], dtype=torch.float32, device=self.device)
        value_loss    = (sample["state_value"] - target_return).pow(2)

        hits = {
            "operator" : float(int(masked_operator.argmax().item()) == action.operator),
            "vehicle"  : float(int(masked_vehicle.argmax().item()) == action.vehicle_index),
            "job"      : float(int(masked_job.argmax().item()) == action.job_index),
        }

        return {"operator": operator_loss, "vehicle": vehicle_loss, "job": job_loss, "value": value_loss}, hits

    def minibatch_step(self, batch_records):
        batch_graph = Batch.from_data_list([record["graph"] for record in batch_records])
        per_sample  = self.policy.forward_batch(batch_graph)

        losses = {head: torch.tensor(0.0, device=self.device) for head in ("operator", "vehicle", "job", "value")}
        hits   = {head: 0.0 for head in ("operator", "vehicle", "job")}

        for sample, record in zip(per_sample, batch_records):
            sample_loss, sample_hits = self.sample_losses(sample, record)

            for head in losses:
                losses[head] = losses[head] + sample_loss[head]
            for head in hits:
                hits[head] += sample_hits[head]

        batch_size = len(batch_records)
        total_loss = (losses["operator"] + losses["vehicle"] + losses["job"] + self.value_loss_coef * losses["value"]) / batch_size

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_norm)
        self.optimizer.step()

        self.global_batch_step += 1

        loss_values          = {head: float(value.item()) / batch_size for head, value in losses.items()}
        loss_values["total"] = float(total_loss.item())
        self.telemetry.pretrain_batch(loss_values, self.global_batch_step)

        return loss_values, hits

    def train(self, records):
        self.policy.train()
        indices = np.arange(len(records))

        metrics = None
        for epoch in tqdm(range(self.epochs), desc="BC pretraining", unit="epoch"):
            np.random.shuffle(indices)

            epoch_loss    = 0.0
            epoch_hits    = {head: 0.0 for head in ("operator", "vehicle", "job")}
            epoch_batches = 0

            for start_index in range(0, len(records), self.minibatch_size):
                batch_records = [records[index] for index in indices[start_index:start_index + self.minibatch_size]]

                loss_values, hits = self.minibatch_step(batch_records)

                epoch_loss    += loss_values["total"]
                epoch_batches += 1
                for head in epoch_hits:
                    epoch_hits[head] += hits[head]

            mean_loss = epoch_loss / epoch_batches
            accuracy  = {head: epoch_hits[head] / len(records) for head in epoch_hits}

            self.telemetry.pretrain_epoch(mean_loss, accuracy, epoch)
            self.logger.subsection(f"BC epoch {epoch}: loss={mean_loss:.4f} accuracy operator={accuracy['operator']:.3f} vehicle={accuracy['vehicle']:.3f} job={accuracy['job']:.3f}")

            metrics = {"loss": mean_loss, "accuracy": accuracy}

        self.policy.eval()
        return metrics


class PretrainingPipeline:
    def __init__(self, config, repo_root):
        self.config    = config
        self.repo_root = repo_root

        self.runs_root   = None
        self.dataset_dir = None
        self.session     = None
        self.logger      = None
        self.dataset     = None
        self.telemetry   = None
        self.environment = None
        self.teacher     = None
        self.collector   = None
        self.policy      = None
        self.trainer     = None

        self.num_episodes = 0
        self.num_records  = 0

    def resolve_paths(self):
        self.runs_root   = self._absolute(self.config.io.runs_dir)
        self.dataset_dir = self._absolute(self.config.io.dataset_dir)

    def _absolute(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(str(self.repo_root), path)

    def open_session(self):
        self.session = RunDirectory(self.config, self.runs_root).prepare()

    def build_logger(self):
        self.logger = Logger(log_dir=self.config.io.logdir, name="pretraining", level="INFO")

    def load_dataset(self):
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.config.io.dataset_dir = self.dataset_dir
        self.dataset               = Dataset(dataset_dir=self.dataset_dir, config=self.config, shuffle_chunks=False, logger=self.logger)

    def build_components(self):
        vroom.logger = self.logger

        self.telemetry   = PPOTelemetry(self.session.tracker, self.config)
        self.environment = Environment(self.config, logger=self.logger)
        self.teacher     = RegretInsertionTeacher(self.config)
        self.collector   = TeacherRolloutCollector(self.environment, self.teacher, self.config, logger=self.logger)
        self.policy      = Policy(self.config).to(self.config.training.device)

        masker       = ActionMasker(self.config)
        self.trainer = BCTrainer(self.policy, masker, self.config, self.telemetry, logger=self.logger)

    def collect(self):
        target  = self.config.pretrain.episodes
        records = []

        self.logger.section("[Teacher Rollouts]")
        self.logger.subsection(f"Collecting {target} episodes of {self.config.training.max_steps_per_episode} steps")

        with tqdm(total=target, desc="Teacher rollouts", unit="episode") as progress:
            for item in self.dataset:
                episode_records = self.collector.rollout(item)

                episode_reward  = sum(record["reward"] for record in episode_records)
                operator_counts = {operator: 0 for operator in range(4)}
                for record in episode_records:
                    operator_counts[record["action"].operator] += 1

                self.telemetry.pretrain_rollout(episode_reward, operator_counts, self.num_episodes)

                records.extend(episode_records)
                self.num_episodes += 1
                progress.update(1)

                if self.num_episodes >= target:
                    break

        if not records:
            raise RuntimeError(f"Teacher rollout collection produced no records from dataset {self.dataset_dir}")

        self.num_records = len(records)
        self.logger.subsection(f"Collected {self.num_records} records from {self.num_episodes} episodes \n")
        return records

    def save_checkpoint(self):
        training_state = {
            "phase"    : "pretraining",
            "episodes" : self.num_episodes,
            "records"  : self.num_records,
        }

        PolicyCheckpoint().save(
            policy         = self.policy,
            filename       = self.config.io.checkpoint_filename,
            directory      = self.config.io.logdir,
            training_state = training_state,
        )

        self.logger.subsection(f"Pretrained policy saved to {os.path.join(self.config.io.logdir, self.config.io.checkpoint_filename)}")

    def run(self):
        self.resolve_paths()
        self.open_session()
        self.build_logger()
        self.load_dataset()
        self.build_components()

        records = self.collect()
        metrics = self.trainer.train(records)

        self.save_checkpoint()
        return metrics
