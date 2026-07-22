import os
import torch
import torch.nn as nn
import numpy as np
from core.shared.mask import PPOMasking
from model.gnn_model import GNN


class Action:
    def __init__(self, operator, vehicle_index, job_index):
        self.operator = operator
        self.vehicle_index = vehicle_index
        self.job_index = job_index


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.masking = PPOMasking(config)

        model = config.model

        self.device        = "cuda" if config.training.device.startswith("cuda") else "cpu"
        self.num_operators = model.num_operators

        embed_dim   = model.policy_embedding_dim
        context_dim = 2 * embed_dim
        op_dim      = model.operator_embedding_dim

        self.graph_embedding = GNN(model)
        self.operator_emb    = nn.Embedding(model.num_operators, op_dim)

        self.operator_actor = nn.Sequential(
            self._layer_init(nn.Linear(context_dim, model.policy_actor_hidden_1)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, model.num_operators), std=0.01),
        )

        self.vehicle_actor = nn.Sequential(
            self._layer_init(nn.Linear(embed_dim + context_dim + op_dim, model.policy_actor_hidden_1)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_1, model.policy_actor_hidden_2)),
            nn.Tanh(),
            self._layer_init(nn.Linear(model.policy_actor_hidden_2, 1), std=0.01),
        )

        self.job_actor = nn.Sequential(
            self._layer_init(nn.Linear(2 * embed_dim + context_dim + op_dim, model.policy_actor_hidden_1)),
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
            op_logits           = self.operator_actor(context)
            state_value         = self.critic(context).squeeze(-1)

        return embeddings, context, op_logits, state_value

    def forward_batch(self, batch_graph):
        batch_graph = batch_graph.to(self.device)

        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            embeddings, context = self.graph_embedding(batch_graph)
            op_logits           = self.operator_actor(context)
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
                    "context"     : context[graph_index],
                    "op_logits"   : op_logits[graph_index],
                    "state_value" : state_values[graph_index],
                }
            )

        return per_sample

    def compute_logits(self, actor_embeddings, actor_global_ctx, op_logits, selected_op=None):
        veh_emb = actor_embeddings["vehicle"]
        job_emb = actor_embeddings["job"]

        num_vehicles = veh_emb.size(0)
        num_jobs     = job_emb.size(0)

        if selected_op is not None:
            op_indices = torch.tensor([selected_op], device=self.device)
        else:
            op_indices = torch.arange(self.num_operators, device=self.device)

        op_embs = self.operator_emb(op_indices)
        num_ops = op_embs.size(0)

        context_exp = actor_global_ctx.view(1, 1, -1).expand(num_ops, num_vehicles, -1)
        veh_exp     = veh_emb.unsqueeze(0).expand(num_ops, -1, -1)
        op_exp      = op_embs.unsqueeze(1).expand(-1, num_vehicles, -1)

        with torch.amp.autocast(self.device, enabled=self.config.training.use_mixed_precision):
            veh_logits = self.vehicle_actor(torch.cat([veh_exp, context_exp, op_exp], dim=-1)).squeeze(-1)

            job_exp     = job_emb.view(1, 1, num_jobs, -1).expand(num_ops, num_vehicles, -1, -1)
            veh_for_job = veh_exp.unsqueeze(2).expand(-1, -1, num_jobs, -1)
            ctx_for_job = context_exp.unsqueeze(2).expand(-1, -1, num_jobs, -1)
            op_for_job  = op_exp.unsqueeze(2).expand(-1, -1, num_jobs, -1)

            job_logits = self.job_actor(torch.cat([job_exp, veh_for_job, ctx_for_job, op_for_job], dim=-1)).squeeze(-1)

        if selected_op is not None:
            veh_logits = veh_logits.squeeze(0)
            job_logits = job_logits.squeeze(0)

        return {
            "op_logits"  : op_logits,
            "veh_logits" : veh_logits,
            "job_logits" : job_logits,
        }

    def act(self, graph, mask_info=None):
        self.eval()
        with torch.no_grad():
            actor_embeddings, actor_global_ctx, op_logits, state_value = self.forward(graph)

            masked_op_logits = self.masking.mask_operator(op_logits, mask_info)
            op_distribution  = torch.distributions.Categorical(logits=masked_op_logits.float())
            selected_op      = op_distribution.sample()

            op_idx      = int(selected_op.item())
            op_log_prob = op_distribution.log_prob(selected_op)

            logits_result = self.compute_logits(
                actor_embeddings = actor_embeddings,
                actor_global_ctx = actor_global_ctx,
                op_logits        = op_logits,
                selected_op      = None,
            )

            veh_logits_by_op     = logits_result["veh_logits"]
            job_logits_by_op_veh = logits_result["job_logits"]

            masked_veh_logits = self.masking.mask_vehicle(
                veh_logits      = veh_logits_by_op[op_idx],
                mask_info       = mask_info,
                selected_op_idx = op_idx,
            )
            vehicle_distribution = torch.distributions.Categorical(logits=masked_veh_logits.float())
            selected_vehicle     = vehicle_distribution.sample()
            veh_idx              = int(selected_vehicle.item())
            veh_log_prob         = vehicle_distribution.log_prob(selected_vehicle)

            masked_job_logits = self.masking.mask_job(
                job_logits       = job_logits_by_op_veh[op_idx, veh_idx],
                mask_info        = mask_info,
                selected_op_idx  = op_idx,
                selected_veh_idx = veh_idx,
            )
            job_distribution = torch.distributions.Categorical(logits=masked_job_logits.float())
            selected_job     = job_distribution.sample()
            job_idx          = int(selected_job.item())
            job_log_prob     = job_distribution.log_prob(selected_job)

        action = Action(
            operator=op_idx,
            vehicle_index=veh_idx,
            job_index=job_idx,
        )

        results = {
            "action"           : action,
            "state_value"      : state_value,
            "log_prob_op"      : op_log_prob,
            "log_prob_veh"     : veh_log_prob,
            "log_prob_job"     : job_log_prob,
            "op_log_prob"      : op_log_prob,
            "veh_log_prob"     : veh_log_prob,
            "job_log_prob"     : job_log_prob,
            "old_op_logits"    : op_logits,
            "old_veh_logits"   : veh_logits_by_op,
            "old_job_logits"   : job_logits_by_op_veh,
            "masked_op_logits" : masked_op_logits,
        }

        self.train()
        return results

    def load(self, filename, directory):
        filepath = os.path.join(directory, filename)
        checkpoint = torch.load(filepath, map_location=self.config.training.device, weights_only=False)

        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint.get("training_state", None)
        else:
            self.load_state_dict(checkpoint)
            return None

    def checkpoint(self, filename, directory, training_state=None, optimizer=None):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "training_state": training_state,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
