import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from model.policy_model import Policy


class ActionDistribution:
    def __init__(self, config, masker):
        self.config               = config
        self.large_negative_value = config.training.large_negative_value
        self.masker               = masker

    def masked_action_logits(self, vehicle_logits, job_logits, mask_info):
        vehicle_masked = vehicle_logits.clone()
        job_masked     = job_logits.clone()

        if mask_info is None:
            return vehicle_masked, job_masked

        num_vehicles = vehicle_masked.size(1)
        num_jobs     = job_masked.size(2)
        device       = vehicle_masked.device
        neg          = self.large_negative_value

        vehicles_with_jobs = mask_info["vehicles_with_jobs_indices"]
        unassigned_jobs    = mask_info["unassigned_job_indices"]
        vehicle_to_jobs    = mask_info["vehicle_to_job_indices"]

        if len(vehicles_with_jobs) > 0:
            blocked = torch.ones(num_vehicles, dtype=torch.bool, device=device)
            blocked[vehicles_with_jobs] = False
            vehicle_masked[1, blocked] = neg

        if num_vehicles > 1:
            vehicle_masked[2, 1:] = neg
            vehicle_masked[3, 1:] = neg

        if len(unassigned_jobs) > 0:
            blocked = torch.ones(num_jobs, dtype=torch.bool, device=device)
            blocked[unassigned_jobs] = False
            job_masked[0, :, blocked] = neg

        for vehicle_index, job_indices in vehicle_to_jobs.items():
            if len(job_indices) > 0 and int(vehicle_index) < num_vehicles:
                blocked = torch.ones(num_jobs, dtype=torch.bool, device=device)
                blocked[job_indices] = False
                job_masked[1, int(vehicle_index), blocked] = neg

        if num_jobs > 1:
            job_masked[2, :, 1:] = neg
            job_masked[3, :, 1:] = neg

        return vehicle_masked, job_masked

    @staticmethod
    def categorical_kl(old_logits, new_logits):
        old_log_probs     = torch.log_softmax(old_logits, dim=-1)
        new_log_probs     = torch.log_softmax(new_logits, dim=-1)
        old_probabilities = torch.softmax(old_logits, dim=-1)
        return torch.sum(old_probabilities * (old_log_probs - new_log_probs), dim=-1)

    @staticmethod
    def _entropy(logits):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    def compute(
        self,
        old_operator_logits,
        old_vehicle_logits,
        old_job_logits,
        new_operator_logits,
        new_vehicle_logits,
        new_job_logits,
        mask_info,
    ):
        masked_old_operator_logits = self.masker.mask_operator(old_operator_logits, mask_info).float()
        masked_new_operator_logits = self.masker.mask_operator(new_operator_logits, mask_info).float()

        old_vehicle_masked, old_job_masked = self.masked_action_logits(old_vehicle_logits, old_job_logits, mask_info)
        new_vehicle_masked, new_job_masked = self.masked_action_logits(new_vehicle_logits, new_job_logits, mask_info)

        old_vehicle_masked = old_vehicle_masked.float()
        new_vehicle_masked = new_vehicle_masked.float()
        old_job_masked     = old_job_masked.float()
        new_job_masked     = new_job_masked.float()

        old_operator_probs = torch.softmax(masked_old_operator_logits, dim=-1)
        new_operator_probs = torch.softmax(masked_new_operator_logits, dim=-1)

        operator_kl      = self.categorical_kl(masked_old_operator_logits, masked_new_operator_logits)
        operator_entropy = self._entropy(masked_new_operator_logits)

        vehicle_kl_exp      = (old_operator_probs * self.categorical_kl(old_vehicle_masked, new_vehicle_masked)).sum()
        vehicle_entropy_exp = (new_operator_probs * self._entropy(new_vehicle_masked)).sum()

        old_vehicle_probs = torch.softmax(old_vehicle_masked, dim=-1)
        new_vehicle_probs = torch.softmax(new_vehicle_masked, dim=-1)

        joint_old  = old_operator_probs.unsqueeze(-1) * old_vehicle_probs
        joint_new  = new_operator_probs.unsqueeze(-1) * new_vehicle_probs

        job_kl_exp      = (joint_old * self.categorical_kl(old_job_masked, new_job_masked)).sum()
        job_entropy_exp = (joint_new * self._entropy(new_job_masked)).sum()

        total_entropy = operator_entropy + vehicle_entropy_exp + job_entropy_exp
        entropy = {
            "total_entropy"       : total_entropy,
            "operator_entropy"    : operator_entropy,
            "vehicle_entropy_exp" : vehicle_entropy_exp,
            "job_entropy_exp"     : job_entropy_exp,
        }

        total_kl = operator_kl + vehicle_kl_exp + job_kl_exp
        kl ={
            "total_kl"          : float(total_kl.item()),
            "operator_kl"       : float(operator_kl.item()),
            "vehicle_kl_exp"    : float(vehicle_kl_exp.item()),
            "job_kl_exp"        : float(job_kl_exp.item()),
            "mean_kl"           : float(total_kl.item()) / 3
        }

        return entropy, kl


class PPOMemory:
    def __init__(self):
        self.graphs            = []
        self.actions           = []
        self.log_prob_operator = []
        self.log_prob_vehicle  = []
        self.log_prob_job      = []
        self.rewards           = []
        self.state_values      = []
        self.mask_infos        = []
        self.dones             = []
        self.bootstrap_values  = []

        self.old_operator_logits = []
        self.old_vehicle_logits  = []
        self.old_job_logits      = []

    @staticmethod
    def _clone_detached(graph):
        out = HeteroData()
        for ntype, x in graph.x_dict.items():
            out[ntype].x = x.detach().cpu().clone()
        for etype in graph.edge_types:
            out[etype].edge_index = graph[etype].edge_index.detach().cpu().clone()
            out[etype].edge_attr = graph[etype].edge_attr.detach().cpu().clone()
        return out

    def add(
        self,
        graph,
        action,
        log_prob_operator,
        log_prob_vehicle,
        log_prob_job,
        reward,
        state_value,
        mask_info,
        done,
        bootstrap_value=0.0,
        old_operator_logits=None,
        old_vehicle_logits=None,
        old_job_logits=None,
    ):
        self.graphs.append(self._clone_detached(graph))
        self.actions.append(action)

        self.log_prob_operator.append(log_prob_operator.detach().cpu())
        self.log_prob_vehicle.append(log_prob_vehicle.detach().cpu())
        self.log_prob_job.append(log_prob_job.detach().cpu())

        self.rewards.append(reward)
        self.state_values.append(state_value.detach().cpu())
        self.mask_infos.append(mask_info)
        self.dones.append(done)
        self.bootstrap_values.append(float(bootstrap_value))

        if old_operator_logits is not None:
            self.old_operator_logits.append(old_operator_logits.detach().cpu().clone())

        if old_vehicle_logits is not None:
            self.old_vehicle_logits.append(old_vehicle_logits.detach().cpu().clone())

        if old_job_logits is not None:
            self.old_job_logits.append(old_job_logits.detach().cpu().clone())

    def clear(self):
        self.__init__()


class PPO(nn.Module):
    def __init__(self, optimizer, config):
        super().__init__()

        self.config = config
        self.optimizer = optimizer
        self.device = "cuda" if config.training.device.startswith("cuda") else "cpu"
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.training.use_mixed_precision)

        self.lr_scheduler      = None
        self.entropy_scheduler = None
        self.early_stopping    = None
        self.tracker           = None

        self.num_epochs     = config.training.num_epochs
        self.minibatch_size = config.training.minibatch_size

        self.current_entropy_coef = config.entropy.entropy_start
        self.num_operators        = config.model.num_operators

        self.policy       = Policy(config).to(self.device)
        self.memory       = PPOMemory()
        self.logger       = None
        self.telemetry    = None
        self.masker       = None
        self.distribution = None

        self.global_epoch_step  = 0
        self.global_batch_step  = 0
        self.global_update_step = 0
        self.global_sample_step = 0

        self._modules_dict = {
            "operator_actor" : self.policy.operator_actor,
            "vehicle_actor"  : self.policy.vehicle_actor,
            "critic"         : self.policy.critic,
            "embedder_actor" : self.policy.graph_embedding,
            "job_actor"      : self.policy.job_actor,
        }

    def gae(self, rewards, values, dones, bootstrap_values):
        advantages = torch.zeros_like(rewards)
        gamma   = self.config.ppo.gamma
        lambda_ = self.config.ppo.gae_lambda

        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            if dones[t] > 0.5:
                next_value     = bootstrap_values[t]
                next_advantage = 0.0
            else:
                next_value     = values[t + 1]
                next_advantage = last_advantage

            delta          = rewards[t] + gamma * next_value - values[t]
            advantages[t]  = delta + gamma * lambda_ * next_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def prepare_batch(self):
        device                = self.device
        actions               = np.array([[a.operator, a.vehicle_index, a.job_index] for a in self.memory.actions], dtype=np.int64)
        rewards               = torch.tensor(self.memory.rewards, dtype=torch.float32, device=device)
        dones                 = torch.tensor(self.memory.dones, dtype=torch.float32, device=device)
        bootstrap_values      = torch.tensor(self.memory.bootstrap_values, dtype=torch.float32, device=device)
        mask_infos            = self.memory.mask_infos
        old_log_prob_operator = torch.stack(self.memory.log_prob_operator).to(device)
        old_log_prob_vehicle  = torch.stack(self.memory.log_prob_vehicle).to(device)
        old_log_prob_job      = torch.stack(self.memory.log_prob_job).to(device)
        values                = torch.stack(self.memory.state_values).to(device)
        actions_tensor        = torch.from_numpy(actions).to(device)

        advantages, returns = self.gae(rewards, values, dones, bootstrap_values)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = {
            "actions_tensor"        : actions_tensor,
            "mask_infos"            : mask_infos,
            "rewards"               : rewards,
            "old_log_prob_operator" : old_log_prob_operator,
            "old_log_prob_vehicle"  : old_log_prob_vehicle,
            "old_log_prob_job"      : old_log_prob_job,
            "values"                : values,
            "advantages"            : advantages.detach(),
            "normalized_advantages" : normalized_advantages.detach(),
            "returns"               : returns.detach(),
        }

        return batch

    def action_distributions(self, sample_index, batch_data, logits, mask_info):
        operator_logits = logits["operator_logits"]
        vehicle_logits  = logits["vehicle_logits"]
        job_logits      = logits["job_logits"]

        actions_tensor = batch_data["actions_tensor"]

        operator_index = int(actions_tensor[sample_index, 0].item())
        vehicle_index  = int(actions_tensor[sample_index, 1].item())

        masked_operator_logits = self.masker.mask_operator(operator_logits, mask_info)
        operator_distribution  = torch.distributions.Categorical(logits=masked_operator_logits.float())
        operator_log_prob      = operator_distribution.log_prob(actions_tensor[sample_index, 0])

        vehicle_logits_selected = vehicle_logits[operator_index]

        masked_vehicle_logits = self.masker.mask_vehicle(
            vehicle_logits          = vehicle_logits_selected,
            mask_info               = mask_info,
            selected_operator_index = operator_index,
        )

        vehicle_distribution = torch.distributions.Categorical(logits=masked_vehicle_logits.float())
        vehicle_log_prob     = vehicle_distribution.log_prob(actions_tensor[sample_index, 1])

        job_logits_selected = job_logits[operator_index, vehicle_index]

        masked_job_logits = self.masker.mask_job(
            job_logits              = job_logits_selected,
            mask_info               = mask_info,
            selected_operator_index = operator_index,
            selected_vehicle_index  = vehicle_index,
        )

        job_distribution = torch.distributions.Categorical(logits=masked_job_logits.float())
        job_log_prob     = job_distribution.log_prob(actions_tensor[sample_index, 2])

        new_log_probs = {
            "op"    : operator_log_prob,
            "veh"   : vehicle_log_prob,
            "job"   : job_log_prob,
            "total" : (operator_log_prob + vehicle_log_prob + job_log_prob),
        }

        new_distribution = {
            "op"  : operator_distribution,
            "veh" : vehicle_distribution,
            "job" : job_distribution,
        }

        return new_distribution, new_log_probs

    def policy_loss(self, new_log_prob, old_log_prob, advantage):
        prob_ratio    = torch.exp(new_log_prob - old_log_prob)
        unclipped_obj = prob_ratio * advantage

        min_ratio = 1.0 - self.config.ppo.clip_ratio
        max_ratio = 1.0 + self.config.ppo.clip_ratio

        clipped_ratio = torch.clamp(prob_ratio, min_ratio, max_ratio)
        clipped_obj   = clipped_ratio * advantage
        policy_loss   = -torch.min(unclipped_obj, clipped_obj)

        policy_loss_dict = {
            "prob_ratio"    : prob_ratio,
            "unclipped_obj" : unclipped_obj,
            "clipped_obj"   : clipped_obj,
            "policy_loss"   : policy_loss,
        }

        return policy_loss_dict

    def value_loss(self, old_value, pred_state_value, target_return):
        clip_ratio         = self.config.ppo.value_clip_ratio
        value_pred_clipped = old_value + torch.clamp(pred_state_value - old_value, -clip_ratio, clip_ratio)

        val_loss_unclip = (target_return - pred_state_value).pow(2)
        val_loss_clip   = (target_return - value_pred_clipped).pow(2)
        value_loss      = 0.5 * torch.max(val_loss_unclip, val_loss_clip)

        value_loss_dict = {
            "value_loss_unclipped" : val_loss_unclip,
            "value_loss_clipped"   : val_loss_clip,
            "value_loss"           : value_loss,
        }

        return value_loss_dict

    def entropy_and_kl(self, sample_index, logits, mask_info):
        operator_logits = logits["operator_logits"]
        vehicle_logits  = logits["vehicle_logits"]
        job_logits      = logits["job_logits"]

        with torch.no_grad():
            old_operator_logits = self.memory.old_operator_logits[sample_index].to(self.device)
            old_vehicle_logits  = self.memory.old_vehicle_logits[sample_index].to(self.device)
            old_job_logits      = self.memory.old_job_logits[sample_index].to(self.device)

        entropy_dict, kl_dict = self.distribution.compute(
            old_operator_logits = old_operator_logits,
            old_vehicle_logits  = old_vehicle_logits,
            old_job_logits      = old_job_logits,
            new_operator_logits = operator_logits,
            new_vehicle_logits  = vehicle_logits,
            new_job_logits      = job_logits,
            mask_info           = mask_info,
        )

        return entropy_dict, kl_dict

    def process_batch(self, batch_indices, batch_data):
        batch_graph = Batch.from_data_list([self.memory.graphs[idx] for idx in batch_indices])
        per_sample  = self.policy.forward_batch(batch_graph)

        accumulated_loss = torch.tensor(0.0, device=self.device)
        accumulated_kl   = 0.0
        batch_size       = len(batch_indices)

        for i, sample_index in enumerate(batch_indices):
            sample                = per_sample[i]
            advantage             = batch_data["normalized_advantages"][sample_index]
            target_return         = batch_data["returns"][sample_index]
            old_log_prob_operator = batch_data["old_log_prob_operator"][sample_index]
            old_log_prob_vehicle  = batch_data["old_log_prob_vehicle"][sample_index]
            old_log_prob_job      = batch_data["old_log_prob_job"][sample_index]
            old_value             = batch_data["values"][sample_index]
            mask_info             = batch_data["mask_infos"][sample_index]

            old_log_prob_dict = {
                "op"    : old_log_prob_operator,
                "veh"   : old_log_prob_vehicle,
                "job"   : old_log_prob_job,
                "total" : old_log_prob_operator + old_log_prob_vehicle + old_log_prob_job,
            }

            pred_state_value = sample["state_value"]

            logits_dict = self.policy.compute_logits(
                actor_embeddings  = sample["embeddings"],
                global_context    = sample["context"],
                operator_logits   = sample["operator_logits"],
                selected_operator = None,
            )

            new_distribution_dict, new_log_prob_dict = self.action_distributions(sample_index, batch_data, logits_dict, mask_info)

            new_log_prob = new_log_prob_dict["total"]
            old_log_prob = old_log_prob_dict["total"]

            policy_loss_dict      = self.policy_loss(new_log_prob, old_log_prob, advantage)
            value_loss_dict       = self.value_loss(old_value, pred_state_value, target_return)
            entropy_dict, kl_dict = self.entropy_and_kl(sample_index, logits_dict, mask_info)

            entropy_loss = self.current_entropy_coef * entropy_dict["total_entropy"]
            total_loss   = policy_loss_dict["policy_loss"] + self.config.ppo.value_loss_coef * value_loss_dict["value_loss"] - entropy_loss

            accumulated_loss = accumulated_loss + total_loss
            accumulated_kl   += kl_dict["mean_kl"]

            self.telemetry.sample(
                sample_step      = self.global_sample_step,
                advantage        = advantage,
                target_return    = target_return,
                old_log_probs    = old_log_prob_dict,
                new_log_probs    = new_log_prob_dict,
                distributions    = new_distribution_dict,
                policy_loss_dict = policy_loss_dict,
                value_loss_dict  = value_loss_dict,
                entropy_dict     = entropy_dict,
                kl_dict          = kl_dict,
                entropy_loss     = entropy_loss.item(),
                total_loss       = total_loss.item(),
            )
            self.global_sample_step += 1

        mean_loss_tensor = accumulated_loss / batch_size
        mean_loss_value  = accumulated_loss.item() / batch_size
        mean_kl          = accumulated_kl / batch_size

        return mean_loss_tensor, mean_loss_value, mean_kl

    def backward(self, total_loss, batch_step=0):
        self.optimizer.zero_grad()

        if self.config.training.use_mixed_precision:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        self.telemetry.gradients(self._modules_dict, batch_step)

        total_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.ppo.gradient_clip_max_norm)
        self.telemetry.grad_norm(total_norm.item(), batch_step)

        if self.config.training.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def minibatch_step(self, batch_indices, batch_data):
        mean_batch_loss_tensor, mean_batch_loss_value, mean_batch_kl = self.process_batch(batch_indices, batch_data)

        self.global_batch_step += 1

        self.backward(mean_batch_loss_tensor, self.global_batch_step)
        self.lr_scheduler.step()
        self.current_entropy_coef = self.entropy_scheduler.step()

        self.telemetry.batch(mean_batch_loss_value, mean_batch_kl, self.global_batch_step)
        self.telemetry.learning_rates(self.optimizer, self.global_batch_step)
        self.telemetry.entropy_coefficient(self.current_entropy_coef, self.global_batch_step)

        return mean_batch_loss_value, mean_batch_kl

    def update(self):
        self.policy.train()
        total_samples  = len(self.memory.rewards)
        batch_data     = self.prepare_batch()
        indices        = np.arange(total_samples)

        self.telemetry.buffer_size(total_samples, self.global_update_step)
        self.telemetry.baseline(batch_data, self.global_update_step)

        self.logger.subsection(f"Starting PPO Update - Max Epochs: {self.num_epochs}, Minibatch Size: {self.minibatch_size}")
        for epoch in tqdm(range(self.num_epochs), desc="PPO Update", unit="epoch"):
            np.random.shuffle(indices)
            epoch_kl_sum   = 0.0
            epoch_loss_sum = 0.0
            epoch_batches  = 0

            for start_index in tqdm(range(0, total_samples, self.minibatch_size), desc="Minibatch", leave=False):
                end_index     = start_index + self.minibatch_size
                batch_indices = indices[start_index:end_index]

                mean_batch_loss_value, mean_batch_kl = self.minibatch_step(batch_indices, batch_data)

                epoch_kl_sum   += mean_batch_kl
                epoch_loss_sum += mean_batch_loss_value
                epoch_batches  += 1

            self.global_epoch_step += 1
            mean_epoch_kl           = epoch_kl_sum / epoch_batches
            mean_epoch_loss         = epoch_loss_sum / epoch_batches

            self.telemetry.epoch(mean_epoch_loss, mean_epoch_kl, self.global_epoch_step)

            if self.early_stopping.should_stop(mean_epoch_kl):
                self.logger.subsection(f"Early stopping PPO update at epoch {epoch} : KL = {mean_epoch_kl:.4f} > {self.early_stopping.threshold}")
                self.telemetry.early_stop(epoch, self.global_epoch_step)
                break

        self.global_update_step += 1
        self.policy.eval()
        self.memory.clear()
