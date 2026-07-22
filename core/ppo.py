import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from .model import Policy
from tqdm import tqdm


class PPOTools:
    def __init__(self, config, tracker):
        self.config  = config
        self.tracker = tracker
    
    def layer_gradients(self, module_name, module, batch_step):
        layer_stats = {}
        
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                layer_key = f'{module_name}/{name}'
                
                layer_stats[f'{layer_key}/norm'] = grad.norm().item()
                layer_stats[f'{layer_key}/mean'] = grad.mean().item()
                layer_stats[f'{layer_key}/std']  = grad.std(unbiased=False).item()
        
        if layer_stats:
            self.tracker.log_dict(f'batch/gradients_layers/{module_name}', layer_stats, batch_step)
    
    def compute_gradient_stats(self, module, module_name):
        stats        = {}
        grad_values  = []
        param_values = []
        
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_values.append(grad.view(-1))
                param_values.append(param.detach().view(-1))
        
        if grad_values:
            all_grads  = torch.cat(grad_values)
            all_params = torch.cat(param_values)
            
            stats[f'{module_name}/grad_mean']     = all_grads.mean().item()
            stats[f'{module_name}/grad_std']      = all_grads.std(unbiased=False).item()
            stats[f'{module_name}/grad_max']      = all_grads.max().item()
            stats[f'{module_name}/grad_min']      = all_grads.min().item()
            stats[f'{module_name}/grad_abs_mean'] = all_grads.abs().mean().item()
            
            param_norm = all_params.norm().item()
            grad_norm  = all_grads.norm().item()
            
            if param_norm > 1e-8:
                stats[f'{module_name}/grad_param_ratio'] = grad_norm / param_norm
            
            stats[f'{module_name}/has_nan'] = torch.isnan(all_grads).any().item()
            stats[f'{module_name}/has_inf'] = torch.isinf(all_grads).any().item()
        
        return stats
    
    def compute_prob_ratios(self, old_log_probs, new_log_probs):
        prob_ratio_op  = torch.exp(new_log_probs["op"]  - old_log_probs["op"])
        prob_ratio_veh = torch.exp(new_log_probs["veh"] - old_log_probs["veh"])
        prob_ratio_job = torch.exp(new_log_probs["job"] - old_log_probs["job"])

        prob_ratios = {
            "operator" : prob_ratio_op,
            "vehicle"  : prob_ratio_veh,
            "job"      : prob_ratio_job,
        }

        return prob_ratios
    
    def compute_component_loss(self, advantage, prob_ratios):
        clipped_ratio_op  = torch.clamp(prob_ratios["operator"], 1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
        clipped_ratio_veh = torch.clamp(prob_ratios["vehicle"],  1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
        clipped_ratio_job = torch.clamp(prob_ratios["job"],      1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)

        operator_loss = -torch.min(prob_ratios["operator"] * advantage, clipped_ratio_op * advantage)
        veh_loss      = -torch.min(prob_ratios["vehicle"]  * advantage, clipped_ratio_veh * advantage)
        job_loss      = -torch.min(prob_ratios["job"]      * advantage, clipped_ratio_job * advantage)

        loss = {
            "operator" : operator_loss.item(),
            "vehicle"  : veh_loss.item(),
            "job"      : job_loss.item(),
        }

        return loss
    
    def compute_clip_fraction(self, prob_ratios):
        operator_clip_frac = (torch.abs(prob_ratios["operator"] - 1.0) > self.config.ppo.clip_ratio).float().mean()
        vehicle_clip_frac  = (torch.abs(prob_ratios["vehicle"]  - 1.0) > self.config.ppo.clip_ratio).float().mean()
        job_clip_frac      = (torch.abs(prob_ratios["job"]      - 1.0) > self.config.ppo.clip_ratio).float().mean()

        clip_fracs = {
            "operator": operator_clip_frac.item(),
            "vehicle": vehicle_clip_frac.item(),
            "job": job_clip_frac.item(),
        }

        return clip_fracs
    
    def log_baseline(self, batch_data, global_step):
        values  = batch_data["values"]
        returns = batch_data["returns"]
        
        explained_var = 0.0
        if returns.std() > 1e-8:
            explained_var = 1.0 - ((returns - values).var() / (returns.var() + 1e-8))
        
        baseline = {
            "advantage_mean"     : float(batch_data["advantages"].mean().item()),
            "advantage_std"      : float(batch_data["advantages"].std().item()),
            "return_mean"        : float(batch_data["returns"].mean().item()),
            "return_std"         : float(batch_data["returns"].std().item()),
            "reward_mean"        : float(batch_data["rewards"].mean().item()),
            "reward_std"         : float(batch_data["rewards"].std().item()),
            "value_mean"         : float(batch_data["values"].mean().item()),
            "value_std"          : float(batch_data["values"].std().item()),
            "explained_variance" : float(explained_var.item()) if torch.is_tensor(explained_var) else float(explained_var),
            "value_target_std"   : float(returns.std().item()),
        }

        self.tracker.log_dict('batch/baseline_stats', baseline, global_step)
        
    def log_action_distribution(self, dist, sample_index):
        with torch.no_grad():
            op_probs      = dist["op"].probs.cpu().detach()
            vehicle_probs = dist["veh"].probs.cpu().detach()
            job_probs     = dist["job"].probs.cpu().detach()
            
            op_stats = {
                'op_max_prob'  : op_probs.max().item(),
                'op_mean_prob' : op_probs.mean().item(),
                'op_min_prob'  : op_probs.min().item(),
            }

            veh_stats = {
                'veh_max_prob'  : vehicle_probs.max().item(),
                'veh_mean_prob' : vehicle_probs.mean().item(),
                'veh_min_prob'  : vehicle_probs.min().item(),
            }
                        
            job_stats = {
                'job_max_prob'  : job_probs.max().item(),
                'job_mean_prob' : job_probs.mean().item(),
                'job_min_prob'  : job_probs.min().item(),
            }
            
            self.tracker.log_dict('sample/op_distribution',  op_stats,  sample_index)
            self.tracker.log_dict('sample/job_distribution', job_stats, sample_index)
            self.tracker.log_dict('sample/veh_distribution', veh_stats, sample_index)

    def log_sample_metrics(self, sample_results, global_sample_step):
        loss_components       = sample_results["loss_components"]
        entropy_dict          = sample_results["entropy_dict"]
        kl_dict               = sample_results["kl_dict"]
        clip_fractions        = sample_results["clip_fractions"]
        old_log_probs         = sample_results["old_log_probs"]
        new_log_probs         = sample_results["new_log_probs"]
        accumulated_loss      = sample_results["accumulated_loss"]
        accumulated_kl        = sample_results["accumulated_kl"]
        new_distribution_dict = sample_results["new_distribution_dict"]
        total_loss            = sample_results["total_loss"]
        entropy_loss          = sample_results["entropy_loss"]
        policy_loss_dict      = sample_results["policy_loss_dict"]
        value_loss_dict       = sample_results["value_loss_dict"]
        advantage             = sample_results["advantage"]
        target_return         = sample_results["target_return"]
        prob_ratios           = sample_results["prob_ratios"]

        prob_ratios_log = {
            "operator" : prob_ratios["operator"].item(),
            "vehicle"  : prob_ratios["vehicle"].item(),
            "job"      : prob_ratios["job"].item(),
        }
        
        old_log_probs_log = {
            "op"    : old_log_probs["op"].item(),
            "veh"   : old_log_probs["veh"].item(),
            "job"   : old_log_probs["job"].item(),
            "total" : old_log_probs["total"].item(),
        }
        
        new_log_probs_log = {
            "op"    : new_log_probs["op"].item(),
            "veh"   : new_log_probs["veh"].item(),
            "job"   : new_log_probs["job"].item(),
            "total" : new_log_probs["total"].item(),
        }
        
        entropy_dict_log = {
            "total_entropy"       : entropy_dict["total_entropy"].item(),
            "operator_entropy"    : entropy_dict["operator_entropy"].item(),
            "vehicle_entropy_exp" : entropy_dict["vehicle_entropy_exp"].item(),
            "job_entropy_exp"     : entropy_dict["job_entropy_exp"].item(),
        }
        
        policy_loss_log = {
            "prob_ratio"    : policy_loss_dict["prob_ratio"].item(),
            "unclipped_obj" : policy_loss_dict["unclipped_obj"].item(),
            "clipped_obj"   : policy_loss_dict["clipped_obj"].item(),
            "policy_loss"   : policy_loss_dict["policy_loss"].item(),
        }
        
        value_loss_log = {
            "value_loss_unclipped" : value_loss_dict["value_loss_unclipped"].item(),
            "value_loss_clipped"   : value_loss_dict["value_loss_clipped"].item(),
            "value_loss"           : value_loss_dict["value_loss"].item(),
        }

        self.tracker.log_dict('sample/loss', loss_components, global_sample_step)
        self.tracker.log_dict('sample/clip_fraction', clip_fractions, global_sample_step)
        self.tracker.log_dict('sample/prob_ratio', prob_ratios_log, global_sample_step)
        self.tracker.log_dict('sample/old_log_probs', old_log_probs_log, global_sample_step)
        self.tracker.log_dict('sample/new_log_probs', new_log_probs_log, global_sample_step)
        self.tracker.log_dict('sample/entropy', entropy_dict_log, global_sample_step)
        self.tracker.log_dict('sample/kl_divergence', kl_dict, global_sample_step)
        self.tracker.log_scalar('sample/accumulated_loss', accumulated_loss, global_sample_step)
        self.tracker.log_scalar('sample/accumulated_kl', accumulated_kl, global_sample_step)
        self.log_action_distribution(new_distribution_dict, global_sample_step)
        self.tracker.log_scalar('sample/total_loss', total_loss, global_sample_step)
        self.tracker.log_scalar('sample/entropy_loss', entropy_loss, global_sample_step)
        self.tracker.log_dict('sample/policy_loss', policy_loss_log, global_sample_step)
        self.tracker.log_dict('sample/value_loss', value_loss_log, global_sample_step)
        self.tracker.log_scalar('sample/advantage', advantage.item(), global_sample_step)
        self.tracker.log_scalar('sample/target_return', target_return.item(), global_sample_step)
    
    def log_gradients(self, modules_dict, batch_step):
        grad_norms = {}
        grad_stats = {}

        for module_name, module in modules_dict.items():
            grad_norms[module_name] = nn.utils.clip_grad_norm_(module.parameters(), max_norm=float('inf')).item()
            grad_stats.update(self.compute_gradient_stats(module, module_name))

            if batch_step % 10 == 0:
                self.layer_gradients(module_name, module, batch_step)

        self.tracker.log_dict('batch/gradients_norms', grad_norms, batch_step)
        self.tracker.log_dict('batch/gradients_stats', grad_stats, batch_step)
    

class PPODistribution:
    def __init__(self, config, masking):
        self.config               = config
        self.large_negative_value = config.training.large_negative_value
        self.masking              = masking

    @staticmethod
    def categorical_kl(old_logits, new_logits):
        old_log_probs     = torch.log_softmax(old_logits, dim=-1)
        new_log_probs     = torch.log_softmax(new_logits, dim=-1)
        old_probabilities = torch.softmax(old_logits, dim=-1)
        return torch.sum(old_probabilities * (old_log_probs - new_log_probs), dim=-1)

    def masked_action_logits(self, veh_logits, job_logits, mask_info):
        veh_masked = veh_logits.clone()
        job_masked = job_logits.clone()

        if mask_info is None:
            return veh_masked, job_masked

        num_vehicles = veh_masked.size(1)
        num_jobs     = job_masked.size(2)
        device       = veh_masked.device
        neg          = self.large_negative_value

        vehicles_with_jobs = mask_info.get("vehicles_with_jobs_indices", [])
        unassigned_jobs    = mask_info.get("unassigned_job_indices", [])
        vehicle_to_jobs    = mask_info.get("vehicle_to_job_indices", {})

        if len(vehicles_with_jobs) > 0:
            blocked = torch.ones(num_vehicles, dtype=torch.bool, device=device)
            blocked[vehicles_with_jobs] = False
            veh_masked[1, blocked] = neg

        if num_vehicles > 1:
            veh_masked[2, 1:] = neg
            veh_masked[3, 1:] = neg

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

        return veh_masked, job_masked

    @staticmethod
    def _entropy(logits):
        log_probs = torch.log_softmax(logits, dim=-1)
        return -(log_probs.exp() * log_probs).sum(dim=-1)

    def compute(
        self,
        old_op_logits,
        old_veh_logits,
        old_job_logits,
        new_op_logits,
        new_veh_logits,
        new_job_logits,
        mask_info,
    ):
        masked_old_op_logits = self.masking.mask_operator(old_op_logits, mask_info).float()
        masked_new_op_logits = self.masking.mask_operator(new_op_logits, mask_info).float()

        old_veh_masked, old_job_masked = self.masked_action_logits(old_veh_logits, old_job_logits, mask_info)
        new_veh_masked, new_job_masked = self.masked_action_logits(new_veh_logits, new_job_logits, mask_info)

        old_veh_masked = old_veh_masked.float()
        new_veh_masked = new_veh_masked.float()
        old_job_masked = old_job_masked.float()
        new_job_masked = new_job_masked.float()

        old_op_probs = torch.softmax(masked_old_op_logits, dim=-1)
        new_op_probs = torch.softmax(masked_new_op_logits, dim=-1)

        operator_kl      = self.categorical_kl(masked_old_op_logits, masked_new_op_logits)
        operator_entropy = self._entropy(masked_new_op_logits)

        veh_kl_exp          = (old_op_probs * self.categorical_kl(old_veh_masked, new_veh_masked)).sum()
        vehicle_entropy_exp = (new_op_probs * self._entropy(new_veh_masked)).sum()

        old_veh_probs = torch.softmax(old_veh_masked, dim=-1)
        new_veh_probs = torch.softmax(new_veh_masked, dim=-1)

        joint_old  = old_op_probs.unsqueeze(-1) * old_veh_probs
        joint_new  = new_op_probs.unsqueeze(-1) * new_veh_probs

        job_kl_exp      = (joint_old * self.categorical_kl(old_job_masked, new_job_masked)).sum()
        job_entropy_exp = (joint_new * self._entropy(new_job_masked)).sum()

        total_entropy = operator_entropy + vehicle_entropy_exp + job_entropy_exp
        entropy = {
            "total_entropy"       : total_entropy,
            "operator_entropy"    : operator_entropy,
            "vehicle_entropy_exp" : vehicle_entropy_exp,
            "job_entropy_exp"     : job_entropy_exp,
        }
        
        total_kl = operator_kl + veh_kl_exp + job_kl_exp
        kl ={
            "total_kl"          : float(total_kl.item()),
            "operator_kl"       : float(operator_kl.item()),
            "vehicle_kl_exp"    : float(veh_kl_exp.item()),
            "job_kl_exp"        : float(job_kl_exp.item()),
            "mean_kl"           : float(total_kl.item()) / 3
        }

        return entropy, kl


class PPOMemory:
    def __init__(self):
        self.graphs           = []
        self.actions          = []
        self.log_prob_op      = []
        self.log_prob_veh     = []
        self.log_prob_job     = []
        self.rewards          = []
        self.state_values     = []
        self.mask_infos       = []
        self.dones            = []
        self.bootstrap_values = []

        self.old_op_logits    = []
        self.old_veh_logits   = []
        self.old_job_logits   = []

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
        log_prob_op,
        log_prob_veh,
        log_prob_job,
        reward,
        state_value,
        mask_info,
        done,
        bootstrap_value=0.0,
        old_op_logits=None,
        old_veh_logits=None,
        old_job_logits=None,
    ):
        self.graphs.append(self._clone_detached(graph))
        self.actions.append(action)

        self.log_prob_op.append(log_prob_op.detach().cpu())
        self.log_prob_veh.append(log_prob_veh.detach().cpu())
        self.log_prob_job.append(log_prob_job.detach().cpu())

        self.rewards.append(reward)
        self.state_values.append(state_value.detach().cpu())
        self.mask_infos.append(mask_info)
        self.dones.append(done)
        self.bootstrap_values.append(float(bootstrap_value))

        if old_op_logits is not None:
            self.old_op_logits.append(old_op_logits.detach().cpu().clone())

        if old_veh_logits is not None:
            self.old_veh_logits.append(old_veh_logits.detach().cpu().clone())

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
        self.tools        = None
        self.masking      = None
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
        device             = self.device
        actions            = np.array([[a.operator, a.vehicle_index, a.job_index] for a in self.memory.actions], dtype=np.int64)
        rewards            = torch.tensor(self.memory.rewards, dtype=torch.float32, device=device)
        dones              = torch.tensor(self.memory.dones, dtype=torch.float32, device=device)
        bootstrap_values   = torch.tensor(self.memory.bootstrap_values, dtype=torch.float32, device=device)
        mask_infos         = self.memory.mask_infos
        old_log_prob_op    = torch.stack(self.memory.log_prob_op).to(device)
        old_log_prob_veh   = torch.stack(self.memory.log_prob_veh).to(device)
        old_log_prob_job   = torch.stack(self.memory.log_prob_job).to(device)
        values             = torch.stack(self.memory.state_values).to(device)
        actions_tensor     = torch.from_numpy(actions).to(device)

        advantages, returns = self.gae(rewards, values, dones, bootstrap_values)
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch = {
            "actions_tensor"        : actions_tensor,
            "mask_infos"            : mask_infos,
            "rewards"               : rewards,
            "old_log_prob_op"       : old_log_prob_op,
            "old_log_prob_veh"      : old_log_prob_veh,
            "old_log_prob_job"      : old_log_prob_job,
            "values"                : values,
            "advantages"            : advantages.detach(),
            "normalized_advantages" : normalized_advantages.detach(),
            "returns"               : returns.detach(),
        }

        return batch
               
    def action_distributions(self, sample_index, batch_data, logits, mask_info):
        op_logits  = logits["op_logits"]
        veh_logits = logits["veh_logits"]
        job_logits = logits["job_logits"]
        
        actions_tensor = batch_data["actions_tensor"]
        
        op_idx  = int(actions_tensor[sample_index, 0].item())
        veh_idx = int(actions_tensor[sample_index, 1].item())
        
        masked_op_logits = self.masking.mask_operator(op_logits, mask_info)
        op_dist          = torch.distributions.Categorical(logits=masked_op_logits.float())
        op_log_prob      = op_dist.log_prob(actions_tensor[sample_index, 0])
                
        veh_logits_sel      = veh_logits[op_idx]
        
        masked_veh_logits = self.masking.mask_vehicle(
            veh_logits      = veh_logits_sel,
            mask_info       = mask_info,
            selected_op_idx = op_idx,
        )
        
        veh_dist     = torch.distributions.Categorical(logits=masked_veh_logits.float())
        veh_log_prob = veh_dist.log_prob(actions_tensor[sample_index, 1])
            
        job_logits_sel = job_logits[op_idx, veh_idx]
        
        masked_job_logits = self.masking.mask_job(
            job_logits       = job_logits_sel,
            mask_info        = mask_info,
            selected_op_idx  = op_idx,
            selected_veh_idx = veh_idx,
        )
        
        job_dist     = torch.distributions.Categorical(logits=masked_job_logits.float())
        job_log_prob = job_dist.log_prob(actions_tensor[sample_index, 2])
        
        new_log_probs = {
            "op"    : op_log_prob,
            "veh"   : veh_log_prob,
            "job"   : job_log_prob,
            "total" : (op_log_prob + veh_log_prob + job_log_prob),
        }

        new_distribution = {
            "op"  : op_dist,
            "veh" : veh_dist,
            "job" : job_dist,
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
    
    def entropy_and_kl(self, sample_idx, logits, mask_info):
        op_logits  = logits["op_logits"]
        veh_logits = logits["veh_logits"]
        job_logits = logits["job_logits"]

        with torch.no_grad():
            old_op_logits  = self.memory.old_op_logits[sample_idx].to(self.device)
            old_veh_logits = self.memory.old_veh_logits[sample_idx].to(self.device)
            old_job_logits = self.memory.old_job_logits[sample_idx].to(self.device)
        
        entropy_dict, kl_dict = self.distribution.compute(
            old_op_logits  = old_op_logits,
            old_veh_logits = old_veh_logits,
            old_job_logits = old_job_logits,
            new_op_logits  = op_logits,
            new_veh_logits = veh_logits,
            new_job_logits = job_logits,
            mask_info      = mask_info,
        )
        
        return entropy_dict, kl_dict

    def backward(self, total_loss, batch_step=0):
        self.optimizer.zero_grad()

        if self.config.training.use_mixed_precision:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        self.tools.log_gradients(self._modules_dict, batch_step)

        total_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.ppo.gradient_clip_max_norm)
        self.tracker.log_scalar('batch/grad_norm_before_clip', total_norm.item(), batch_step)

        if self.config.training.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def process_batch(self, batch_indices, batch_data):
        graphs = [self.memory.graphs[idx].to(self.device) for idx in batch_indices]
        
        accumulated_loss = torch.tensor(0.0, device=self.device)
        accumulated_kl   = 0.0
        batch_size       = len(batch_indices)
        
        for i, sample_idx in enumerate(batch_indices):
            graph            = graphs[i]
            advantage        = batch_data["normalized_advantages"][sample_idx]
            target_return    = batch_data["returns"][sample_idx]
            old_log_prob_op  = batch_data["old_log_prob_op"][sample_idx]
            old_log_prob_veh = batch_data["old_log_prob_veh"][sample_idx]
            old_log_prob_job = batch_data["old_log_prob_job"][sample_idx]
            old_value        = batch_data["values"][sample_idx]
            mask_info        = batch_data["mask_infos"][sample_idx]
        
            old_log_prob_dict = {
                "op"    : old_log_prob_op,
                "veh"   : old_log_prob_veh,
                "job"   : old_log_prob_job,
                "total" : old_log_prob_op + old_log_prob_veh + old_log_prob_job,
            }

            actor_embeddings, actor_global_ctx, op_logits, pred_state_value = self.policy(graph)
            
            logits_dict = self.policy.compute_logits(
                actor_embeddings = actor_embeddings,
                actor_global_ctx = actor_global_ctx,
                op_logits        = op_logits,
                selected_op      = None,
            )
                
            new_distribution_dict, new_log_prob_dict = self.action_distributions(sample_idx, batch_data, logits_dict, mask_info)
            
            new_log_prob = new_log_prob_dict["total"]
            old_log_prob = old_log_prob_dict["total"]
            
            policy_loss_dict      = self.policy_loss(new_log_prob, old_log_prob, advantage)
            value_loss_dict       = self.value_loss(old_value, pred_state_value, target_return)
            entropy_dict, kl_dict = self.entropy_and_kl(sample_idx, logits_dict, mask_info)
 
            entropy_loss = self.current_entropy_coef * entropy_dict["total_entropy"]
            total_loss   = policy_loss_dict["policy_loss"] + self.config.ppo.value_loss_coef * value_loss_dict["value_loss"] - entropy_loss
            
            accumulated_loss = accumulated_loss + total_loss
            accumulated_kl   += kl_dict["mean_kl"]
        
            prob_ratios     = self.tools.compute_prob_ratios(old_log_prob_dict, new_log_prob_dict)
            clip_fractions  = self.tools.compute_clip_fraction(prob_ratios)
            loss_components = self.tools.compute_component_loss(advantage, prob_ratios)
            
            sample_results = {
                "sample_index"          : sample_idx,
                "policy_loss_dict"      : policy_loss_dict,
                "value_loss_dict"       : value_loss_dict,
                "entropy_loss"          : entropy_loss.item(),
                "total_loss"            : total_loss.item(),
                "kl_dict"               : kl_dict,
                "entropy_dict"          : entropy_dict,
                "old_log_probs"         : old_log_prob_dict,
                "new_log_probs"         : new_log_prob_dict,
                "accumulated_loss"      : accumulated_loss.item(),
                "accumulated_kl"        : accumulated_kl,
                "new_distribution_dict" : new_distribution_dict,
                "clip_fractions"        : clip_fractions,
                "loss_components"       : loss_components,
                "prob_ratios"           : prob_ratios,
                "advantage"             : advantage,
                "target_return"         : target_return,
            }

            self.tools.log_sample_metrics(sample_results, self.global_sample_step)
            self.global_sample_step += 1

        mean_loss_tensor = accumulated_loss / batch_size
        mean_loss_value  = accumulated_loss.item() / batch_size
        mean_kl          = accumulated_kl / batch_size
        
        return mean_loss_tensor, mean_loss_value, mean_kl

    def update(self):
        self.policy.train()
        total_samples  = len(self.memory.rewards)
        batch_data     = self.prepare_batch()
        indices        = np.arange(total_samples)
        
        self.logger.subsection("Logging Baseline Statistics")
        self.tools.log_baseline(batch_data, self.global_update_step)
  
        self.logger.subsection(f"Starting PPO Update - Max Epochs: {self.num_epochs}, Minibatch Size: {self.minibatch_size}")
        for epoch in tqdm(range(self.num_epochs), desc="PPO Update", unit="epoch"):
            np.random.shuffle(indices)
            epoch_kl_sum   = 0.0
            epoch_loss_sum = 0.0
            epoch_batches  = 0

            for start_index in tqdm(range(0, total_samples, self.minibatch_size), desc="Minibatch", leave=False):
                end_index = start_index + self.minibatch_size
                batch_indices = indices[start_index:end_index]
                
                mean_batch_loss_tensor, mean_batch_loss_value, mean_batch_kl = self.process_batch(batch_indices, batch_data)
                
                self.tracker.log_scalar('batch/mean_loss', mean_batch_loss_value, self.global_batch_step)
                self.tracker.log_scalar('batch/mean_kl', mean_batch_kl, self.global_batch_step)

                epoch_kl_sum            += mean_batch_kl
                epoch_loss_sum          += mean_batch_loss_value
                epoch_batches           += 1
                self.global_batch_step  += 1
                
                self.backward(mean_batch_loss_tensor, self.global_batch_step)
                self.lr_scheduler.step()
                self.current_entropy_coef = self.entropy_scheduler.step()

            self.global_epoch_step += 1
            mean_epoch_kl           = epoch_kl_sum / epoch_batches
            mean_epoch_loss         = epoch_loss_sum / epoch_batches
            
            self.tracker.log_scalar('epoch/mean_loss', mean_epoch_loss, self.global_epoch_step)
            self.tracker.log_scalar('epoch/mean_kl', mean_epoch_kl, self.global_epoch_step)
          
            if self.early_stopping.should_stop(mean_epoch_kl, epoch, self.global_epoch_step):
                break

        self.global_update_step += 1
        self.policy.eval()
        self.memory.clear()