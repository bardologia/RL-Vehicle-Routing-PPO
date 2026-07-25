import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATv2Conv, global_mean_pool


class GNN(nn.Module):
    def __init__(self, model):
        super().__init__()
        hidden = model.policy_gnn_hidden_channels
        out    = model.policy_embedding_dim

        self.encoders = nn.ModuleDict({
            "job"     : nn.Linear(model.job_input_dim, hidden),
            "vehicle" : nn.Linear(model.vehicle_input_dim, hidden),
        })

        self.context_encoder = nn.Linear(model.context_input_dim, model.context_embedding_dim)

        self.convs     = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for layer_idx in range(model.gnn_num_layers):
            out_dim = out if layer_idx == model.gnn_num_layers - 1 else hidden
            self.convs.append(self._build_conv(hidden, out_dim, model.edge_attr_dim))
            self.residuals.append(self._build_residual(hidden, out_dim))

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

    def _build_residual(self, in_dim, out_dim):
        if in_dim == out_dim:
            return nn.ModuleDict({"job": nn.Identity(), "vehicle": nn.Identity()})

        return nn.ModuleDict({"job": nn.Linear(in_dim, out_dim), "vehicle": nn.Linear(in_dim, out_dim)})

    def _combine(self, embeddings, messages, residual):
        combined = {}

        for node_type, features in embeddings.items():
            updated = residual[node_type](features)

            if node_type in messages:
                updated = updated + messages[node_type]

            combined[node_type] = self.activation(updated)

        return combined

    def _batch_size(self, graph):
        store = graph["job"]
        return int(store.ptr.numel()) - 1 if "ptr" in store else 1

    def forward(self, graph):
        edge_index_dict = graph.edge_index_dict
        edge_attr_dict  = {relation: graph[relation].edge_attr for relation in graph.edge_types}

        embeddings = {node_type: self.activation(encoder(graph.x_dict[node_type])) for node_type, encoder in self.encoders.items()}

        for conv, residual in zip(self.convs, self.residuals):
            messages   = conv(embeddings, edge_index_dict, edge_attr_dict=edge_attr_dict)
            embeddings = self._combine(embeddings, messages, residual)

        job_embedding     = embeddings["job"]
        vehicle_embedding = embeddings["vehicle"]

        job_batch     = graph["job"].batch if "batch" in graph["job"] else torch.zeros(job_embedding.size(0), dtype=torch.long, device=job_embedding.device)
        vehicle_batch = graph["vehicle"].batch if "batch" in graph["vehicle"] else torch.zeros(vehicle_embedding.size(0), dtype=torch.long, device=vehicle_embedding.device)

        batch_size = self._batch_size(graph)
        context    = torch.cat(
            [
                global_mean_pool(job_embedding, job_batch, size=batch_size),
                global_mean_pool(vehicle_embedding, vehicle_batch, size=batch_size),
                self.activation(self.context_encoder(graph.context)),
            ],
            dim=-1,
        )

        return embeddings, context
