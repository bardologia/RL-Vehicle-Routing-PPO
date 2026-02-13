import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData # type: ignore
from .embedding import GNN, PointerNetwork
from tqdm import tqdm
from core.mask import PPOMasking


class Action:
    def __init__(self, operator, vehicle_index, job_index):
        self.operator = operator
        self.vehicle_index = vehicle_index
        self.job_index = job_index


class PPODistribution:
    @staticmethod
    def categorical_kl(old_logits, new_logits):
        old_log_probabilities = torch.log_softmax(old_logits, dim=-1)
        new_log_probabilities = torch.log_softmax(new_logits, dim=-1)
        old_probabilities = torch.softmax(old_logits, dim=-1)
        return torch.sum(old_probabilities * (old_log_probabilities - new_log_probabilities), dim=-1)

    @staticmethod
    def compute(
        old_operator_logits,
        old_vehicle_logits_by_operator,
        old_job_logits_by_operator_vehicle,
        new_operator_logits,
        new_vehicle_logits_by_operator,
        new_job_logits_by_operator_vehicle,
        mask_info,
        large_negative_value,
    ):
        number_of_operators = new_operator_logits.size(0)
        number_of_vehicles = new_vehicle_logits_by_operator.size(1)

        masked_old_operator_logits = PPOMasking.mask_operator(old_operator_logits, mask_info, large_negative_value)
        masked_new_operator_logits = PPOMasking.mask_operator(new_operator_logits, mask_info, large_negative_value)

        operator_kl = PPODistribution.categorical_kl(masked_old_operator_logits, masked_new_operator_logits)

        new_operator_distribution = torch.distributions.Categorical(logits=masked_new_operator_logits.float())
        operator_entropy = new_operator_distribution.entropy()
        new_operator_probs = new_operator_distribution.probs
        
        old_operator_distribution = torch.distributions.Categorical(logits=masked_old_operator_logits.float())
        old_operator_probs = old_operator_distribution.probs

        vehicle_kl_expectation = torch.tensor(0.0, device=new_operator_logits.device)
        vehicle_entropy_expectation = torch.tensor(0.0, device=new_operator_logits.device)
        
        job_kl_expectation = torch.tensor(0.0, device=new_operator_logits.device)
        job_entropy_expectation = torch.tensor(0.0, device=new_operator_logits.device)

        for operator_index in range(number_of_operators):
            old_vehicle_logits = old_vehicle_logits_by_operator[operator_index]
            new_vehicle_logits = new_vehicle_logits_by_operator[operator_index]

            masked_old_vehicle_logits = PPOMasking.mask_vehicle(
                vehicle_logits=old_vehicle_logits,
                mask_info=mask_info,
                selected_operator_index=operator_index,
                large_negative_value=large_negative_value,
            )
            
            masked_new_vehicle_logits = PPOMasking.mask_vehicle(
                vehicle_logits=new_vehicle_logits,
                mask_info=mask_info,
                selected_operator_index=operator_index,
                large_negative_value=large_negative_value,
            )

            vehicle_kl = PPODistribution.categorical_kl(masked_old_vehicle_logits, masked_new_vehicle_logits)
            vehicle_kl_expectation += old_operator_probs[operator_index] * vehicle_kl

            new_vehicle_dist = torch.distributions.Categorical(logits=masked_new_vehicle_logits.float())
            vehicle_entropy = new_vehicle_dist.entropy()
            vehicle_entropy_expectation += new_operator_probs[operator_index] * vehicle_entropy
            
            new_vehicle_probs = new_vehicle_dist.probs
            
            old_vehicle_dist = torch.distributions.Categorical(logits=masked_old_vehicle_logits.float())
            old_vehicle_probs = old_vehicle_dist.probs

            for vehicle_index in range(number_of_vehicles):
                old_job_logits = old_job_logits_by_operator_vehicle[operator_index, vehicle_index]
                new_job_logits = new_job_logits_by_operator_vehicle[operator_index, vehicle_index]

                masked_old_job_logits = PPOMasking.mask_job(
                    job_logits=old_job_logits,
                    mask_info=mask_info,
                    selected_operator_index=operator_index,
                    selected_vehicle_index=vehicle_index,
                    large_negative_value=large_negative_value,
                )
                masked_new_job_logits = PPOMasking.mask_job(
                    job_logits=new_job_logits,
                    mask_info=mask_info,
                    selected_operator_index=operator_index,
                    selected_vehicle_index=vehicle_index,
                    large_negative_value=large_negative_value,
                )

                job_kl = PPODistribution.categorical_kl(masked_old_job_logits, masked_new_job_logits)
                joint_prob_old = old_operator_probs[operator_index] * old_vehicle_probs[vehicle_index]
                job_kl_expectation += joint_prob_old * job_kl

                new_job_dist = torch.distributions.Categorical(logits=masked_new_job_logits.float())
                job_entropy = new_job_dist.entropy()
                joint_prob_new = new_operator_probs[operator_index] * new_vehicle_probs[vehicle_index]
                job_entropy_expectation += joint_prob_new * job_entropy

        total_entropy = operator_entropy + vehicle_entropy_expectation + job_entropy_expectation
        entropy = {
            "total_entropy": total_entropy,
            "operator_entropy": operator_entropy,
            "vehicle_entropy_expectation": vehicle_entropy_expectation,
            "job_entropy_expectation": job_entropy_expectation,
        }
        
        total_kl = operator_kl + vehicle_kl_expectation + job_kl_expectation
        kl ={
            "total_kl": float(total_kl.item()),
            "operator_kl": float(operator_kl.item()),
            "vehicle_kl_expectation": float(vehicle_kl_expectation.item()),
            "job_kl_expectation": float(job_kl_expectation.item()),
            "mean_kl": float(total_kl.item()) / 3
        }

        return entropy, kl


class PPOMemory:
    def __init__(self):
        self.graphs = []
        self.actions = []
        self.prob_operator = []
        self.prob_vehicle = []
        self.prob_job = []
        self.rewards = []
        self.state_values = []
        self.mask_infos = []
        self.dones = []
        self.true_vehicle_ids = []
        self.true_job_ids = []
        
        self.old_operator_logits = []
        self.old_vehicle_logits_by_operator = []
        self.old_job_logits_by_operator_vehicle = []

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
        prob_operator,
        prob_vehicle,
        prob_job,
        reward,
        state_value,
        mask_info,
        done,
        old_operator_logits=None,
        old_vehicle_logits_by_operator=None,
        old_job_logits_by_operator_vehicle=None,
        true_vehicle_id = None,                
        true_job_id = None,
    ):
        self.graphs.append(self._clone_detached(graph))
        self.actions.append(action)

        self.prob_operator.append(prob_operator.detach().cpu())
        self.prob_vehicle.append(prob_vehicle.detach().cpu())
        self.prob_job.append(prob_job.detach().cpu())

        self.rewards.append(reward)
        self.state_values.append(state_value.detach().cpu())
        self.mask_infos.append(mask_info)
        self.dones.append(done)

        self.true_job_ids.append(true_job_id)
        self.true_vehicle_ids.append(true_vehicle_id)

        if old_operator_logits is not None:
            self.old_operator_logits.append(old_operator_logits.detach().cpu().clone())

        if old_vehicle_logits_by_operator is not None:
            self.old_vehicle_logits_by_operator.append(old_vehicle_logits_by_operator.detach().cpu().clone())

        if old_job_logits_by_operator_vehicle is not None:
            self.old_job_logits_by_operator_vehicle.append(old_job_logits_by_operator_vehicle.detach().cpu().clone())

    def clear(self):
        self.__init__()


class GraphPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model = config.model
        
        self.device = "cuda" if config.training.device.startswith("cuda") else "cpu"
        self.num_operators = model.num_operators

        self.graph_embedder_actor  = GNN(
            model                    = model,
            job_input_dimension      = model.job_input_dim,
            vehicle_input_dimension  = model.vehicle_input_dim,
            path_input_dimension     = model.path_input_dim,
            hidden_channels          = model.policy_gnn_hidden_channels,
            output_channels          = model.policy_embedding_dim,
            num_layers               = model.gnn_num_layers,
            edge_attribute_dimension = model.edge_attr_dim,
            mlp_hidden_channels      = model.policy_gnn_mlp_hidden_channels,
            edge_dropout             = model.gnn_edge_dropout,
        )

        self.operator_emb = nn.Embedding(model.num_operators, model.operator_embedding_dim)

        self.operator_actor = nn.Sequential(
            self._layer_init(nn.Linear(3 * model.policy_embedding_dim, model.policy_actor_hidden_1)),   
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, model.num_operators), std=0.01),
        )
        
        vehicle_actor_input_dim = model.policy_embedding_dim + 3 * model.policy_embedding_dim + model.operator_embedding_dim
       
        self.vehicle_actor = nn.Sequential(
            self._layer_init(nn.Linear(vehicle_actor_input_dim, model.policy_actor_hidden_1)),  
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, 1), std=0.01),
        )

        pointer_context_input_dim = 3 * model.policy_embedding_dim + model.operator_embedding_dim + model.policy_embedding_dim
        
        self.pointer_context_proj = nn.Sequential(
            self._layer_init(nn.Linear(pointer_context_input_dim, model.pointer_hidden_dim)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.pointer_hidden_dim, model.policy_embedding_dim)),
        )
        
        self.job_pointer = PointerNetwork(
            hidden_dim=model.policy_embedding_dim,
            num_heads=model.pointer_num_heads,
            tanh_clipping=model.pointer_tanh_clipping,
        )

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(3 * model.value_embedding_dim, model.value_critic_hidden_1)),   
            nn.Tanh(),
            self._layer_init(nn.Linear(model.value_critic_hidden_1, model.value_critic_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.value_critic_hidden_2, 1), std=1.0),
        )

    @staticmethod
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, graph):
        device = self.device
        graph = graph.to(device)

        with torch.amp.autocast(device, enabled=self.config.training.use_mixed_precision):
            actor_embeddings, actor_job_context, actor_vehicle_context, actor_path_context = self.graph_embedder_actor(graph)
            actor_global_context = torch.cat([actor_job_context, actor_vehicle_context, actor_path_context], dim=-1)
            operator_logits = self.operator_actor(actor_global_context)
        
            critic_context = actor_global_context.detach()
            state_value = self.critic(critic_context).squeeze(-1)

        return actor_embeddings, actor_global_context, operator_logits, state_value

    def compute_vehicle_scores(self, vehicle_embeddings, actor_global_context, selected_operator_index):
        device = vehicle_embeddings.device
        
        operator_index_tensor       = torch.tensor(selected_operator_index, device=device, dtype=torch.long)
        operator_embedding          = self.operator_emb(operator_index_tensor)
        
        expanded_global_context     = actor_global_context.unsqueeze(0).expand(vehicle_embeddings.size(0), -1)
        expanded_operator_embedding = operator_embedding.unsqueeze(0).expand(vehicle_embeddings.size(0), -1)
        
        vehicle_actor_input = torch.cat([vehicle_embeddings, expanded_global_context, expanded_operator_embedding], dim=-1)
        vehicle_scores      = self.vehicle_actor(vehicle_actor_input).squeeze(-1)
        return vehicle_scores

    def compute_job_scores(self, job_embeddings, actor_global_context, selected_operator_index, selected_vehicle_embedding):
        device = job_embeddings.device
        
        operator_index_tensor       = torch.tensor(selected_operator_index, device=device, dtype=torch.long)
        operator_embedding          = self.operator_emb(operator_index_tensor)
        selected_vehicle_context    = selected_vehicle_embedding
        
        pointer_context = torch.cat([
            actor_global_context,
            operator_embedding,
            selected_vehicle_context,
        ], dim=-1)
        
        pointer_query = self.pointer_context_proj(pointer_context)
        
        job_scores = self.job_pointer(
            query=pointer_query,
            keys=job_embeddings,
            mask=None,
            return_attention=False,
        )
        
        return job_scores

    def act(self, graph, mask_info=None):
        self.eval()
        large_negative_value = self.config.training.large_negative_value
        with torch.no_grad():
            actor_embeddings, actor_global_context, operator_logits, state_value = self.forward(graph)

            masked_operator_logits = PPOMasking.mask_operator(operator_logits, mask_info, large_negative_value)
            operator_distribution  = torch.distributions.Categorical(logits=masked_operator_logits.float())

            selected_operator       = operator_distribution.sample()
            selected_operator_index = int(selected_operator.item())
            operator_log_probability = operator_distribution.log_prob(selected_operator)

            num_ops = self.num_operators
            veh_emb = actor_embeddings["vehicle"] 
            num_vehs = veh_emb.size(0)
            
            op_indices = torch.arange(num_ops, device=self.device)
            op_embs = self.operator_emb(op_indices) 
            
            global_exp = actor_global_context.view(1, 1, -1).expand(num_ops, num_vehs, -1)
            veh_exp = veh_emb.unsqueeze(0).expand(num_ops, num_vehs, -1)
            op_exp  = op_embs.unsqueeze(1).expand(num_ops, num_vehs, -1)
            
            veh_input = torch.cat([veh_exp, global_exp, op_exp], dim=-1)
            
            vehicle_logits_by_operator = self.vehicle_actor(veh_input).squeeze(-1) # (O, V)
            
            vehicle_logits_conditioned = vehicle_logits_by_operator[selected_operator_index]

            masked_vehicle_logits = PPOMasking.mask_vehicle(
                vehicle_logits=vehicle_logits_conditioned,
                mask_info=mask_info,
                selected_operator_index=selected_operator_index,
                large_negative_value=large_negative_value,
            )
            vehicle_distribution = torch.distributions.Categorical(logits=masked_vehicle_logits.float())

            selected_vehicle        = vehicle_distribution.sample()
            selected_vehicle_index  = int(selected_vehicle.item())
            vehicle_log_probability = vehicle_distribution.log_prob(selected_vehicle)

            pointer_context = torch.cat([global_exp, op_exp, veh_exp], dim=-1)
            pointer_query = self.pointer_context_proj(pointer_context) 
            
            job_emb = actor_embeddings["job"] 
        
            job_logits_by_operator_vehicle, attention_info = self.job_pointer(
                query=pointer_query,
                keys=job_emb,
                mask=None,
                return_attention=True
            )
            
            job_logits_conditioned = job_logits_by_operator_vehicle[selected_operator_index, selected_vehicle_index]

            masked_job_logits = PPOMasking.mask_job(
                job_logits=job_logits_conditioned,
                mask_info=mask_info,
                selected_operator_index=selected_operator_index,
                selected_vehicle_index=selected_vehicle_index,
                large_negative_value=large_negative_value,
            )
            job_distribution = torch.distributions.Categorical(logits=masked_job_logits.float())

            selected_job       = job_distribution.sample()
            selected_job_index = int(selected_job.item())
            job_log_probability = job_distribution.log_prob(selected_job)
            
            pointer_weights_selected = attention_info['pointer_weights'][selected_operator_index, selected_vehicle_index]
            glimpse_weights_selected = attention_info['glimpse_weights'][selected_operator_index, selected_vehicle_index]

        action = Action(
            operator=selected_operator_index,
            vehicle_index=selected_vehicle_index,
            job_index=selected_job_index,
        )

        results = {
            "action"                             : action,
            "state_value"                        : state_value,
            "pointer_weights_selected"           : pointer_weights_selected,
            "glimpse_weights_selected"           : glimpse_weights_selected,
            "pointer_entropy"                    : -(pointer_weights_selected * torch.log(pointer_weights_selected + 1e-10)).sum(),
            "glimpse_entropy"                    : -(glimpse_weights_selected * torch.log(glimpse_weights_selected + 1e-10)).sum(),
            "pointer_max_weight"                 : pointer_weights_selected.max(),
            "glimpse_max_weight"                 : glimpse_weights_selected.max(),
            "prob_operator"                      : operator_log_probability,
            "prob_vehicle"                       : vehicle_log_probability,
            "prob_job"                           : job_log_probability,
            "operator_log_probability"           : operator_log_probability,
            "vehicle_log_probability"            : vehicle_log_probability,
            "job_log_probability"                : job_log_probability,
            "old_operator_logits"                : operator_logits,
            "old_vehicle_logits_by_operator"     : vehicle_logits_by_operator,
            "old_job_logits_by_operator_vehicle" : job_logits_by_operator_vehicle,
            "masked_operator_logits"             : masked_operator_logits,
            "vehicle_logits_conditioned"         : vehicle_logits_conditioned,
            "job_logits_conditioned"             : job_logits_conditioned,
            "vehicle_logits_by_operator"         : vehicle_logits_by_operator,
            "job_logits_by_operator_vehicle"     : job_logits_by_operator_vehicle,
        }

        return results

    def load(self, filename, directory):
        filepath = os.path.join(directory, filename)
        checkpoint = torch.load(filepath, map_location=self.config.training.device)

        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint.get("training_state", None)
        else:
            self.load_state_dict(checkpoint)
            return None

    def checkpoint(self, filename, directory, training_state=None):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "training_state": training_state,
        }
        
        torch.save(checkpoint, filepath)


class PPO(nn.Module):
    def __init__(self, optimizer, config):
        super().__init__()

        self.config = config
        self.optimizer = optimizer
        self.device = "cuda" if config.training.device.startswith("cuda") else "cpu"
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.training.use_mixed_precision)
        
        self.lr_scheduler      = None
        self.entropy_scheduler = None
        self.tracker           = None  
        self.current_entropy_coef = config.ppo.ppo_entropy_coef
        self.num_operators        = config.model.num_operators

        self.policy = GraphPolicy(config).to(self.device)
        self.memory = PPOMemory()
        self.logger = None

        self._modules_dict = {
            "operator_actor"  : self.policy.operator_actor,
            "vehicle_actor"   : self.policy.vehicle_actor,
            "critic"          : self.policy.critic,
            "embedder_actor"  : self.policy.graph_embedder_actor,
            "job_pointer"     : self.policy.job_pointer,
            "pointer_context" : self.policy.pointer_context_proj,
        }
    
    def _track_layer_gradients(self, module_name, module, batch_step):
        layer_stats = {}
        
        for name, param in module.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                layer_key = f'{module_name}/{name}'
                
                layer_stats[f'{layer_key}/norm'] = grad.norm().item()
                layer_stats[f'{layer_key}/mean'] = grad.mean().item()
                layer_stats[f'{layer_key}/std'] = grad.std(unbiased=False).item()
        
        if layer_stats:
            self.tracker.log_dict(f'batch/gradients_layers/{module_name}', layer_stats, batch_step)

    def _compute_gae(self, rewards, values, dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 0.0
                next_value = 0.0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.ppo.gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + self.config.ppo.gamma * self.config.ppo.gae_lambda * next_non_terminal * last_advantage
            advantages[t] = last_advantage
            
        returns = advantages + values
        return advantages, returns

    def _prepare_batch(self):
        device = self.device
        actions = np.array([[a.operator, a.vehicle_index, a.job_index] for a in self.memory.actions], dtype=np.int64)
        rewards            = torch.tensor(self.memory.rewards, dtype=torch.float32, device=device)
        old_prob_operator  = torch.stack(self.memory.prob_operator).to(device)
        old_prob_vehicle   = torch.stack(self.memory.prob_vehicle).to(device)
        old_prob_job       = torch.stack(self.memory.prob_job).to(device)
        values             = torch.stack(self.memory.state_values).to(device)
        dones              = torch.tensor(self.memory.dones, dtype=torch.float32, device=device)
        actions_tensor     = torch.from_numpy(actions).to(device)

        advantage_estimates, return_estimates = self._compute_gae(rewards, values, dones)
        
        advantage_estimates_raw = advantage_estimates.view(-1).detach()
        advantage_estimates_normalized = (advantage_estimates_raw - advantage_estimates_raw.mean()) / (advantage_estimates_raw.std() + 1e-8)
        return_estimates = return_estimates.view(-1).detach()
        
        return {
            "actions_tensor"          : actions_tensor,
            "rewards"                 : rewards,
            "old_prob_operator"       : old_prob_operator,
            "old_prob_vehicle"        : old_prob_vehicle,
            "old_prob_job"            : old_prob_job,
            "values"                  : values,
            "advantage_estimates_raw" : advantage_estimates_raw,
            "advantage_estimates"     : advantage_estimates,
            "advantage_normalized"    : advantage_estimates_normalized, 
            "return_estimates"        : return_estimates,
        }

    def _compute_baseline(self, batch_data):
        values = batch_data["values"]
        returns = batch_data["return_estimates"]
        
        explained_var = 0.0
        if returns.std() > 1e-8:
            explained_var = 1.0 - ((returns - values).var() / (returns.var() + 1e-8))
        
        return {
            "advantage_raw_mean" : float(batch_data["advantage_estimates_raw"].mean().item()),
            "advantage_raw_std"  : float(batch_data["advantage_estimates_raw"].std().item()),
            "advantage_mean"     : float(batch_data["advantage_estimates"].mean().item()),
            "advantage_std"      : float(batch_data["advantage_estimates"].std().item()),
            "return_mean"        : float(batch_data["return_estimates"].mean().item()),
            "return_std"         : float(batch_data["return_estimates"].std().item()),
            "reward_mean"        : float(batch_data["rewards"].mean().item()),
            "reward_std"         : float(batch_data["rewards"].std().item()),
            "value_mean"         : float(batch_data["values"].mean().item()),
            "value_std"          : float(batch_data["values"].std().item()),
            "explained_variance" : float(explained_var.item()) if torch.is_tensor(explained_var) else float(explained_var),
            "value_target_std"   : float(returns.std().item()),
        }

    def _compute_logits(self, actor_embeddings, actor_global_context, number_of_operators):
        num_ops = number_of_operators
        veh_emb = actor_embeddings["vehicle"]
        num_vehs = veh_emb.size(0)
        
        op_indices = torch.arange(num_ops, device=self.device)
        op_embs = self.policy.operator_emb(op_indices) 
        
        global_exp = actor_global_context.view(1, 1, -1).expand(num_ops, num_vehs, -1)
        veh_exp = veh_emb.unsqueeze(0).expand(num_ops, num_vehs, -1)
        op_exp = op_embs.unsqueeze(1).expand(num_ops, num_vehs, -1)
        
        veh_input = torch.cat([veh_exp, global_exp, op_exp], dim=-1)
        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            vehicle_logits_by_operator = self.policy.vehicle_actor(veh_input).squeeze(-1) 
        
        pointer_context = torch.cat([global_exp, op_exp, veh_exp], dim=-1)
        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            pointer_query = self.policy.pointer_context_proj(pointer_context) 
            job_emb = actor_embeddings["job"] 
            
            job_logits_by_operator_vehicle = self.policy.job_pointer(
                query=pointer_query,
                keys=job_emb,
                mask=None,
                return_attention=False
            ) 
        
        return vehicle_logits_by_operator, job_logits_by_operator_vehicle

    def _compute_prob_ratios(self, log_probs):
        
        probability_ratio_operator = torch.exp(log_probs["new_operator"] - log_probs["old_operator"])
        probability_ratio_vehicle  = torch.exp(log_probs["new_vehicle"]  - log_probs["old_vehicle"])
        probability_ratio_job      = torch.exp(log_probs["new_job"]      - log_probs["old_job"])

        prob_ratios = {
            "operator" : probability_ratio_operator,
            "vehicle"  : probability_ratio_vehicle,
            "job"      : probability_ratio_job,
        }

        return prob_ratios

    def _compute_component_loss(self, advantage, prob_ratios):

        clipped_ratio_operator = torch.clamp(prob_ratios["operator"], 1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
        clipped_ratio_vehicle  = torch.clamp(prob_ratios["vehicle"],  1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
        clipped_ratio_job      = torch.clamp(prob_ratios["job"],      1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)

        operator_loss = -torch.min(prob_ratios["operator"] * advantage, clipped_ratio_operator * advantage)
        veh_loss      = -torch.min(prob_ratios["vehicle"]  * advantage, clipped_ratio_vehicle  * advantage)
        job_loss      = -torch.min(prob_ratios["job"]      * advantage, clipped_ratio_job      * advantage)

        loss = {
            "operator": operator_loss.item(),
            "vehicle": veh_loss.item(),
            "job": job_loss.item(),
        }

        return loss

    def _compute_clip_fraction(self, prob_ratios):
        
        operator_clip_frac = (torch.abs(prob_ratios["operator"] - 1.0) > self.config.ppo.clip_ratio).float().mean()
        vehicle_clip_frac  = (torch.abs(prob_ratios["vehicle"]  - 1.0) > self.config.ppo.clip_ratio).float().mean()
        job_clip_frac      = (torch.abs(prob_ratios["job"]      - 1.0) > self.config.ppo.clip_ratio).float().mean()

        clip_fracs = {
            "operator": operator_clip_frac.item(),
            "vehicle": vehicle_clip_frac.item(),
            "job": job_clip_frac.item(),
        }

        return clip_fracs

    def _process_sample(self, sample_index, batch_data):
        device = self.device

        actions_tensor      = batch_data["actions_tensor"]
        advantage_estimates = batch_data["advantage_normalized"]
        return_estimates    = batch_data["return_estimates"]

        old_log_probability_operator = batch_data["old_prob_operator"]
        old_log_probability_vehicle  = batch_data["old_prob_vehicle"]
        old_log_probability_job      = batch_data["old_prob_job"]

        graph = self.memory.graphs[sample_index].to(device)
        mask_info = self.memory.mask_infos[sample_index]

        selected_operator_index = int(actions_tensor[sample_index, 0].item())
        selected_vehicle_index  = int(actions_tensor[sample_index, 1].item())
        selected_job_index      = int(actions_tensor[sample_index, 2].item())

        actor_embeddings, actor_global_context, operator_logits, predicted_state_value = self.policy(graph)

        vehicle_logits_by_operator, job_logits_by_operator_vehicle = self._compute_logits(
            actor_embeddings=actor_embeddings,
            actor_global_context=actor_global_context,
            number_of_operators=self.num_operators,
        )

        masked_operator_logits   = PPOMasking.mask_operator(operator_logits, mask_info, self.config.training.large_negative_value)
        operator_distribution    = torch.distributions.Categorical(logits=masked_operator_logits.float())
        operator_log_probability = operator_distribution.log_prob(actions_tensor[sample_index, 0])

        with torch.no_grad():
            operator_probs = operator_distribution.probs
            operator_stats = {
                'operator_max_prob': operator_probs.max().item(),
                'operator_mean_prob': operator_probs.mean().item(),
            }
            self.tracker.log_dict('batch/operator_distribution', operator_stats, sample_index)

        vehicle_logits_conditioned_taken = vehicle_logits_by_operator[selected_operator_index]
        
        masked_vehicle_logits_taken = PPOMasking.mask_vehicle(
            vehicle_logits=vehicle_logits_conditioned_taken,
            mask_info=mask_info,
            selected_operator_index=selected_operator_index,
            large_negative_value=self.config.training.large_negative_value,
        )
        
        vehicle_distribution    = torch.distributions.Categorical(logits=masked_vehicle_logits_taken.float())
        vehicle_log_probability = vehicle_distribution.log_prob(actions_tensor[sample_index, 1])

        with torch.no_grad():
            vehicle_probs = vehicle_distribution.probs
            vehicle_stats = {
                'vehicle_max_prob': vehicle_probs.max().item(),
                'vehicle_mean_prob': vehicle_probs.mean().item(),
            }
            self.tracker.log_dict('batch/vehicle_distribution', vehicle_stats, sample_index)

        job_logits_conditioned_taken = job_logits_by_operator_vehicle[selected_operator_index, selected_vehicle_index]

        masked_job_logits_taken = PPOMasking.mask_job(
            job_logits=job_logits_conditioned_taken,
            mask_info=mask_info,
            selected_operator_index=selected_operator_index,
            selected_vehicle_index=selected_vehicle_index,
            large_negative_value=self.config.training.large_negative_value,
        )
        
        job_distribution    = torch.distributions.Categorical(logits=masked_job_logits_taken.float())
        job_log_probability = job_distribution.log_prob(actions_tensor[sample_index, 2])
        
        with torch.no_grad():
            job_probs         = job_distribution.probs
            pointer_entropy   = -(job_probs * torch.log(job_probs + 1e-10)).sum()
            pointer_max_prob  = job_probs.max()
            pointer_mean_prob = job_probs.mean()
            
            attention_stats = {
                'pointer_entropy': pointer_entropy.item(),
                'pointer_max_prob': pointer_max_prob.item(),
                'pointer_mean_prob': pointer_mean_prob.item(),
            }
            self.tracker.log_dict('batch/pointer_attention', attention_stats, sample_index)

        new_log_probability = operator_log_probability + vehicle_log_probability + job_log_probability
        old_log_probability = (old_log_probability_operator[sample_index] + old_log_probability_vehicle[sample_index] + old_log_probability_job[sample_index])

        advantage     = advantage_estimates[sample_index]
        target_return = return_estimates[sample_index]

        self.tracker.log_scalar('batch/target_return', target_return.item(), sample_index)
        self.tracker.log_scalar('batch/advantage', advantage, sample_index)

        probability_ratio   = torch.exp(new_log_probability - old_log_probability)
        unclipped_objective = probability_ratio * advantage

        clipped_ratio     = torch.clamp(probability_ratio, 1.0 - self.config.ppo.clip_ratio, 1.0 + self.config.ppo.clip_ratio)
        clipped_objective = clipped_ratio * advantage
        
        obj_dict = {'unclipped': unclipped_objective, 'clipped': clipped_objective}
        self.tracker.log_dict('batch/objective', obj_dict, sample_index)

        policy_loss_sample = -torch.min(unclipped_objective, clipped_objective)
        
        old_value = batch_data["values"][sample_index]
        value_pred_clipped = old_value + torch.clamp(
            predicted_state_value - old_value,
            -self.config.ppo.clip_ratio,
            self.config.ppo.clip_ratio
        )
        
        value_loss_unclipped = (target_return.float() - predicted_state_value.float()).pow(2)
        value_loss_clipped   = (target_return.float() - value_pred_clipped.float()).pow(2)
        value_loss_sample    = torch.max(value_loss_unclipped, value_loss_clipped)
        
        with torch.no_grad():
            old_operator_logits                = self.memory.old_operator_logits[sample_index].to(self.device)
            old_vehicle_logits_by_operator     = self.memory.old_vehicle_logits_by_operator[sample_index].to(self.device)
            old_job_logits_by_operator_vehicle = self.memory.old_job_logits_by_operator_vehicle[sample_index].to(self.device)
        
        entropy_components, kl_components = PPODistribution.compute(
            old_operator_logits=old_operator_logits,
            old_vehicle_logits_by_operator=old_vehicle_logits_by_operator,
            old_job_logits_by_operator_vehicle=old_job_logits_by_operator_vehicle,
            new_operator_logits=operator_logits,
            new_vehicle_logits_by_operator=vehicle_logits_by_operator,
            new_job_logits_by_operator_vehicle=job_logits_by_operator_vehicle,
            mask_info=mask_info,
            large_negative_value=self.config.training.large_negative_value,
        )

        entropy_loss_sample = self.current_entropy_coef * entropy_components["total_entropy"]
        total_loss_sample   = policy_loss_sample + self.config.ppo.value_loss_coef * value_loss_sample - entropy_loss_sample
       
        log_probs = {
            "new_operator" : operator_log_probability,
            "new_vehicle"  : vehicle_log_probability,
            "new_job"      : job_log_probability,
            "old_operator" : old_log_probability_operator[sample_index],
            "old_vehicle"  : old_log_probability_vehicle[sample_index],
            "old_job"      : old_log_probability_job[sample_index],
        }

        prob_ratios     = self._compute_prob_ratios(log_probs)
        clip_fractions  = self._compute_clip_fraction(prob_ratios)
        loss_components = self._compute_component_loss(advantage, prob_ratios)
        
        loss_components.update({
            "total_loss"   : total_loss_sample.item(), 
            "policy_loss"  : policy_loss_sample.item(), 
            "value_loss"   : value_loss_sample.item(), 
            "entropy_loss" : entropy_loss_sample.item()
        })

        prob_ratios_for_logging = {
            "operator" : prob_ratios["operator"].item(),
            "vehicle"  : prob_ratios["vehicle"].item(),
            "job"      : prob_ratios["job"].item(),
        }

        self.tracker.log_dict('batch/entropy', entropy_components, sample_index)
        self.tracker.log_dict('batch/kl', kl_components, sample_index)
        self.tracker.log_dict('batch/loss', loss_components, sample_index)
        self.tracker.log_dict('batch/clip_fraction', clip_fractions, sample_index)
        self.tracker.log_dict('batch/prob_ratio', prob_ratios_for_logging, sample_index)
        self.tracker.log_split_dict('batch/log_prob', log_probs, sample_index, ('new', 'old'))

        mean_kl = kl_components["mean_kl"]
        return total_loss_sample, mean_kl

    def _compute_gradient_stats(self, module, module_name):
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
    
    def _backward(self, total_loss, batch_step=0):
        self.optimizer.zero_grad()
        if self.config.training.use_mixed_precision:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()

        before_grad_clip = {}
        grad_stats = {}
        
        for module_name, module in self._modules_dict.items():
            grad_norm = nn.utils.clip_grad_norm_(module.parameters(), max_norm=float('inf'))
            before_grad_clip[module_name] = grad_norm.item()
            
            module_stats = self._compute_gradient_stats(module, module_name)
            grad_stats.update(module_stats)
            
            if batch_step % 10 == 0:
                self._track_layer_gradients(module_name, module, batch_step)
        
        total_grad_norm = nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.config.ppo.gradient_clip_max_norm)
        grad_stats['total_grad_norm_before_clip'] = total_grad_norm.item()
        
        if self.config.training.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.lr_scheduler.step()
        self.current_entropy_coef = self.entropy_scheduler.step()

        after_grad_clip = {}
        for module_name, module in self._modules_dict.items():
            grad_norm = nn.utils.clip_grad_norm_(module.parameters(), max_norm=float('inf'))
            after_grad_clip[module_name] = grad_norm.item()
        
        self.tracker.log_comparison('batch/gradients_norm', before_grad_clip, after_grad_clip, batch_step, 'before_clip', 'after_clip')
        self.tracker.log_dict('batch/gradients_flow', grad_stats, batch_step)
        
        clip_ratios = {}
        for module_name in before_grad_clip.keys():
            if before_grad_clip[module_name] > 1e-8:
                clip_ratios[module_name] = after_grad_clip[module_name] / before_grad_clip[module_name]
        
        if clip_ratios:
            self.tracker.log_dict('batch/gradients_clip_ratio', clip_ratios, batch_step)

    def update(self, number_of_epochs, minibatch_size, global_step):
        self.train()
        total_samples  = len(self.memory.rewards)
        self.logger.info(f"Preparing Batch Data for PPO Update - Total Samples: {total_samples}")
        batch_data     = self._prepare_batch()
        indices        = np.arange(total_samples)
        
        self.logger.info("Logging Baseline Statistics before PPO Update")
        baseline_stats = self._compute_baseline(batch_data)
        self.tracker.log_grouped_dict('batch/baseline', baseline_stats, global_step, ['advantage', 'return'])
        
        batch_step_counter = 0
 
        self.logger.info(f"Starting PPO Update - Max Epochs: {number_of_epochs}, Minibatch Size: {minibatch_size}")
        for epoch in tqdm(range(number_of_epochs), desc="PPO Update", unit="epoch"):
            np.random.shuffle(indices)
            epoch_kl_sum = 0.0
            epoch_loss_sum = 0.0
            epoch_batches = 0

            for start_index in range(0, total_samples, minibatch_size):
                end_index = start_index + minibatch_size
                batch_indices = indices[start_index:end_index]
        
                batch_kl = 0
                batch_loss = 0
                sample_cont = 0
                
                for sample_index in batch_indices:
                    total_loss, sample_kl = self._process_sample(sample_index, batch_data)
                    batch_loss  += total_loss
                    batch_kl    += sample_kl
                    sample_cont += 1

                mean_batch_loss = batch_loss / sample_cont
                mean_batch_kl   = batch_kl / sample_cont
                epoch_kl_sum   += mean_batch_kl
                epoch_loss_sum += mean_batch_loss
                epoch_batches  += 1
                self._backward(mean_batch_loss, batch_step_counter)
                batch_step_counter += 1

            global_step += 1
            epoch_kl   = epoch_kl_sum / epoch_batches
            epoch_loss = epoch_loss_sum / epoch_batches
            self.tracker.log_scalar('batch/epoch_loss', epoch_loss, global_step)
            self.tracker.log_scalar('batch/epoch_kl_divergence', epoch_kl, global_step)
          
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if epoch_kl > self.config.ppo.kl_divergence_threshold:
                self.tracker.log_scalar('batch/epoch_early_stop_epoch', epoch, global_step)
                self.logger.info(f"Early stopping PPO update at epoch {epoch} due to KL = {epoch_kl:.4f} exceeding threshold of {self.config.ppo.kl_divergence_threshold}")
                break

        self.logger.info("Checkpointing PPO policy")
        self.policy.checkpoint(filename="graph_ppo_policy.pt", directory=self.config.io.logdir)
        self.memory.clear()

    def checkpoint(self, filename, directory, training_state=None):
        self.policy.checkpoint(filename=filename, directory=directory, training_state=training_state)
