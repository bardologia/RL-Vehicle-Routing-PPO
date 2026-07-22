import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import HeteroConv, GATv2Conv
from core.mask import PPOMasking


class Action:
    def __init__(self, operator, vehicle_index, job_index):
        self.operator = operator
        self.vehicle_index = vehicle_index
        self.job_index = job_index


class GNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        hidden = model.policy_gnn_hidden_channels
        out    = model.policy_embedding_dim

        self.encoders = nn.ModuleDict({
            "job"     : nn.Linear(model.job_input_dim, hidden),
            "vehicle" : nn.Linear(model.vehicle_input_dim, hidden),
        })

        self.convs = nn.ModuleList()
        for layer_idx in range(model.gnn_num_layers):
            out_dim = out if layer_idx == model.gnn_num_layers - 1 else hidden
            self.convs.append(self._build_conv(hidden, out_dim, model.edge_attr_dim))

        self.activation = nn.LeakyReLU(0.01)

    def _build_conv(self, in_dim, out_dim, edge_dim):
        def conv(in_channels):
            return GATv2Conv(in_channels, out_dim, heads=1, concat=False, add_self_loops=False, edge_dim=edge_dim)

        return HeteroConv(
            {
                ("job", "job_sequence", "job")              : conv(in_dim),
                ("vehicle", "vehicle_assigned", "job")      : conv((in_dim, in_dim)),
                ("job", "vehicle_assigned", "vehicle")      : conv((in_dim, in_dim)),
                ("job", "job_vehicle_proximity", "vehicle") : conv((in_dim, in_dim)),
                ("vehicle", "job_vehicle_proximity", "job") : conv((in_dim, in_dim)),
            },
            aggr="sum",
        )

    def forward(self, graph):
        edge_index_dict = graph.edge_index_dict
        edge_attr_dict  = {relation: graph[relation].edge_attr for relation in graph.edge_types}

        embeddings = {node_type: self.activation(encoder(graph.x_dict[node_type])) for node_type, encoder in self.encoders.items()}

        for conv in self.convs:
            embeddings = conv(embeddings, edge_index_dict, edge_attr_dict=edge_attr_dict)
            embeddings = {node_type: self.activation(features) for node_type, features in embeddings.items()}

        context = torch.cat([embeddings["job"].mean(dim=0), embeddings["vehicle"].mean(dim=0)], dim=-1)
        return embeddings, context


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
            op_logits           = self.operator_actor(context)
            state_value         = self.critic(context).squeeze(-1)

        return embeddings, context, op_logits, state_value

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
