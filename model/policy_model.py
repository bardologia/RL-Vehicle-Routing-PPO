import os

import numpy as np
import torch
import torch.nn as nn

from core.shared.mask import ActionMasker
from model.gnn_model import GNN


class Action:
    def __init__(self, operator, vehicle_index, job_index):
        self.operator = operator
        self.vehicle_index = vehicle_index
        self.job_index = job_index


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.masker = ActionMasker(config)

        model = config.model

        self.device        = "cuda" if config.training.device.startswith("cuda") else "cpu"
        self.num_operators = model.num_operators

        embedding_dim = model.policy_embedding_dim
        context_dim   = 2 * embedding_dim
        operator_dim  = model.operator_embedding_dim

        self.graph_embedding    = GNN(model)
        self.operator_embedding = nn.Embedding(model.num_operators, operator_dim)

        self.operator_actor = nn.Sequential(
            self._layer_init(nn.Linear(context_dim, model.policy_actor_hidden_1)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, model.num_operators), std=0.01),
        )

        self.vehicle_actor = nn.Sequential(
            self._layer_init(nn.Linear(embedding_dim + context_dim + operator_dim, model.policy_actor_hidden_1)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, 1), std=0.01),
        )

        self.job_actor = nn.Sequential(
            self._layer_init(nn.Linear(2 * embedding_dim + context_dim + operator_dim, model.policy_actor_hidden_1)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, 1), std=0.01),
        )

        self.critic = nn.Sequential(
            self._layer_init(nn.Linear(context_dim, model.value_critic_hidden_1)),
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
        graph = graph.to(self.device)

        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            embeddings, context = self.graph_embedding(graph)
            context             = context.squeeze(0)
            operator_logits     = self.operator_actor(context)
            state_value         = self.critic(context).squeeze(-1)

        return embeddings, context, operator_logits, state_value

    def forward_batch(self, batch_graph):
        batch_graph = batch_graph.to(self.device)

        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            embeddings, context = self.graph_embedding(batch_graph)
            operator_logits     = self.operator_actor(context)
            state_values        = self.critic(context).squeeze(-1)

        job_batch     = batch_graph["job"].batch
        vehicle_batch = batch_graph["vehicle"].batch

        per_sample = []
        for graph_index in range(batch_graph.num_graphs):
            per_sample.append(
                {
                    "embeddings" : {
                        "job"     : embeddings["job"][job_batch == graph_index],
                        "vehicle" : embeddings["vehicle"][vehicle_batch == graph_index],
                    },
                    "context"         : context[graph_index],
                    "operator_logits" : operator_logits[graph_index],
                    "state_value"     : state_values[graph_index],
                }
            )

        return per_sample

    def compute_logits(self, actor_embeddings, global_context, operator_logits, selected_operator=None):
        vehicle_embedding = actor_embeddings["vehicle"]
        job_embedding     = actor_embeddings["job"]

        num_vehicles = vehicle_embedding.size(0)
        num_jobs     = job_embedding.size(0)

        if selected_operator is not None:
            operator_indices = torch.tensor([selected_operator], device=self.device)
        else:
            operator_indices = torch.arange(self.num_operators, device=self.device)

        operator_embeddings   = self.operator_embedding(operator_indices)
        num_scored_operators  = operator_embeddings.size(0)

        context_expanded  = global_context.view(1, 1, -1).expand(num_scored_operators, num_vehicles, -1)
        vehicle_expanded  = vehicle_embedding.unsqueeze(0).expand(num_scored_operators, -1, -1)
        operator_expanded = operator_embeddings.unsqueeze(1).expand(-1, num_vehicles, -1)

        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            vehicle_logits = self.vehicle_actor(torch.cat([vehicle_expanded, context_expanded, operator_expanded], dim=-1)).squeeze(-1)

            job_expanded      = job_embedding.view(1, 1, num_jobs, -1).expand(num_scored_operators, num_vehicles, -1, -1)
            vehicle_for_job   = vehicle_expanded.unsqueeze(2).expand(-1, -1, num_jobs, -1)
            context_for_job   = context_expanded.unsqueeze(2).expand(-1, -1, num_jobs, -1)
            operator_for_job  = operator_expanded.unsqueeze(2).expand(-1, -1, num_jobs, -1)

            job_logits = self.job_actor(torch.cat([job_expanded, vehicle_for_job, context_for_job, operator_for_job], dim=-1)).squeeze(-1)

        if selected_operator is not None:
            vehicle_logits = vehicle_logits.squeeze(0)
            job_logits     = job_logits.squeeze(0)

        return {
            "operator_logits" : operator_logits,
            "vehicle_logits"  : vehicle_logits,
            "job_logits"      : job_logits,
        }

    def select_action(self, graph, mask_info=None, greedy=False):
        self.eval()
        with torch.no_grad():
            actor_embeddings, global_context, operator_logits, state_value = self.forward(graph)

            masked_operator_logits = self.masker.mask_operator(operator_logits, mask_info)
            operator_distribution  = torch.distributions.Categorical(logits=masked_operator_logits.float())
            selected_operator      = masked_operator_logits.argmax(dim=-1) if greedy else operator_distribution.sample()

            operator_index    = int(selected_operator.item())
            operator_log_prob = operator_distribution.log_prob(selected_operator)

            logits_result = self.compute_logits(
                actor_embeddings  = actor_embeddings,
                global_context    = global_context,
                operator_logits   = operator_logits,
                selected_operator = None,
            )

            vehicle_logits_by_operator     = logits_result["vehicle_logits"]
            job_logits_by_operator_vehicle = logits_result["job_logits"]

            masked_vehicle_logits = self.masker.mask_vehicle(
                vehicle_logits          = vehicle_logits_by_operator[operator_index],
                mask_info               = mask_info,
                selected_operator_index = operator_index,
            )
            vehicle_distribution = torch.distributions.Categorical(logits=masked_vehicle_logits.float())
            selected_vehicle     = masked_vehicle_logits.argmax(dim=-1) if greedy else vehicle_distribution.sample()
            vehicle_index        = int(selected_vehicle.item())
            vehicle_log_prob     = vehicle_distribution.log_prob(selected_vehicle)

            masked_job_logits = self.masker.mask_job(
                job_logits              = job_logits_by_operator_vehicle[operator_index, vehicle_index],
                mask_info               = mask_info,
                selected_operator_index = operator_index,
                selected_vehicle_index  = vehicle_index,
            )
            job_distribution = torch.distributions.Categorical(logits=masked_job_logits.float())
            selected_job     = masked_job_logits.argmax(dim=-1) if greedy else job_distribution.sample()
            job_index        = int(selected_job.item())
            job_log_prob     = job_distribution.log_prob(selected_job)

        action = Action(
            operator=operator_index,
            vehicle_index=vehicle_index,
            job_index=job_index,
        )

        results = {
            "action"                 : action,
            "state_value"            : state_value,
            "log_prob_operator"      : operator_log_prob,
            "log_prob_vehicle"       : vehicle_log_prob,
            "log_prob_job"           : job_log_prob,
            "operator_log_prob"      : operator_log_prob,
            "vehicle_log_prob"       : vehicle_log_prob,
            "job_log_prob"           : job_log_prob,
            "old_operator_logits"    : operator_logits,
            "old_vehicle_logits"     : vehicle_logits_by_operator,
            "old_job_logits"         : job_logits_by_operator_vehicle,
            "masked_operator_logits" : masked_operator_logits,
        }

        self.train()
        return results


class PolicyCheckpoint:
    def save(self, policy, filename, directory, training_state=None, optimizer=None):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        checkpoint = {
            "model_state_dict" : policy.state_dict(),
            "training_state"   : training_state,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)

    def read(self, filename, directory, map_location):
        filepath = os.path.join(directory, filename)
        return torch.load(filepath, map_location=map_location, weights_only=False)

    def apply(self, policy, checkpoint):
        policy.load_state_dict(checkpoint["model_state_dict"])

    def load(self, policy, filename, directory, map_location):
        checkpoint = self.read(filename, directory, map_location)
        self.apply(policy, checkpoint)
        return checkpoint
