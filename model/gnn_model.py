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

        job_embedding     = embeddings["job"]
        vehicle_embedding = embeddings["vehicle"]

        job_batch     = graph["job"].batch if "batch" in graph["job"] else torch.zeros(job_embedding.size(0), dtype=torch.long, device=job_embedding.device)
        vehicle_batch = graph["vehicle"].batch if "batch" in graph["vehicle"] else torch.zeros(vehicle_embedding.size(0), dtype=torch.long, device=vehicle_embedding.device)

        context = torch.cat([global_mean_pool(job_embedding, job_batch), global_mean_pool(vehicle_embedding, vehicle_batch)], dim=-1)
        return embeddings, context
