import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv # type: ignore


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 1, dropout: float = 0.0, ffn_multiplier: int = 4):
        super().__init__()
     
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
  
        self.query_vectors = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_vectors)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.constant_(self.w_k.bias, 0.0)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.constant_(self.w_v.bias, 0.0)
        nn.init.xavier_uniform_(self.w_o.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        
        x_normed = self.norm(x)
        
        Q = self.query_vectors.unsqueeze(0).expand(num_nodes, -1, -1)
        K = self.w_k(x_normed).view(num_nodes, self.num_heads, self.head_dim)
        V = self.w_v(x_normed).view(num_nodes, self.num_heads, self.head_dim)

        attn_scores = (Q * K).sum(dim=-1) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=0)
        attn_weights = self.attn_dropout(attn_weights)

        weighted_values = attn_weights.unsqueeze(-1) * V
        pooled = weighted_values.sum(dim=0).view(self.hidden_dim)
        pooled = self.w_o(pooled)
        
        output = pooled + self.ffn(self.norm_ffn(pooled))
        
        return output


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, ffn_multiplier: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        N_q = query.size(0)
        N_kv = key_value.size(0)
        
        q_normed = self.norm_q(query)
        kv_normed = self.norm_kv(key_value)
        
        Q = self.w_q(q_normed).view(N_q, self.num_heads, self.head_dim)
        K = self.w_k(kv_normed).view(N_kv, self.num_heads, self.head_dim)
        V = self.w_v(kv_normed).view(N_kv, self.num_heads, self.head_dim)
        
        attn_scores = torch.einsum('qhd,khd->qhk', Q, K) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attended = torch.einsum('qhk,khd->qhd', attn_weights, V)
        attended = self.w_o(attended.reshape(N_q, self.hidden_dim))
        
        x = query + attended
        output = x + self.ffn(self.norm_ffn(x))
        return output


class PointerNetwork(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, tanh_clipping: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.tanh_clipping = tanh_clipping
        
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.w_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor = None, return_attention: bool = False) -> torch.Tensor:
        if query.dim() == 1:
            query = query.unsqueeze(0)

        embed_dim = query.size(-1)
        batch_shape = query.shape[:-1]
        
        Q = self.w_q(query).view(*batch_shape, self.num_heads, self.head_dim)
        
        num_keys = keys.size(0)
        K = self.w_k(keys).view(num_keys, self.num_heads, self.head_dim)
        V = self.w_v(keys).view(num_keys, self.num_heads, self.head_dim)

        glimpse_scores = torch.einsum('...hd, nhd -> ...hn', Q, K) * self.scale
        
        if mask is not None:
             if mask.dim() == 1:
                  mask = mask.view(1, 1, -1)
             elif mask.dim() == len(batch_shape) + 1:
                  mask = mask.unsqueeze(-2)
             
             glimpse_scores = glimpse_scores.masked_fill(mask, float('-inf'))
        
        glimpse_weights = F.softmax(glimpse_scores, dim=-1)
        
        glimpse = torch.einsum('...hn, nhd -> ...hd', glimpse_weights, V)
        glimpse = glimpse.reshape(*batch_shape, embed_dim)
        glimpse = self.w_o(glimpse)
        
        final_Q = self.w_q(glimpse).view(*batch_shape, self.num_heads, self.head_dim)
        pointer_scores = torch.einsum('...hd, nhd -> ...n', final_Q, K) * self.scale
        
        if self.tanh_clipping > 0:
            pointer_scores = self.tanh_clipping * torch.tanh(pointer_scores)
        
        if mask is not None:
            if mask.dim() > pointer_scores.dim():
                 mask = mask.squeeze(-2)

            pointer_scores = pointer_scores.masked_fill(mask, float('-inf'))
        
        if return_attention:
            pointer_weights = F.softmax(pointer_scores, dim=-1)
            return pointer_scores, {'glimpse_weights': glimpse_weights, 'pointer_weights': pointer_weights}
        
        return pointer_scores


class GNN(nn.Module):
    def __init__(
        self,
        model,  
        job_input_dimension,
        vehicle_input_dimension,
        path_input_dimension,
        hidden_channels,
        output_channels,
        num_layers=4,
        edge_attribute_dimension=4,
        mlp_hidden_channels=32,
        edge_dropout=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.edge_attribute_dimension = int(edge_attribute_dimension)
        self.mlp_hidden_channels = int(mlp_hidden_channels)
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.edge_dropout = nn.Dropout(edge_dropout)
        self.activation = nn.LeakyReLU(0.01)
        self.node_types = ["job", "vehicle", "path"]
        
        self.gat_heads = model.gat_heads
        self.gat_concat = model.gat_concat
        self.gat_dropout = model.gat_dropout
        self.cross_attention_heads = model.cross_attention_heads
        self.cross_attention_dropout = model.cross_attention_dropout
        self.attention_num_heads = model.attention_num_heads
        self.attention_dropout = model.attention_dropout

        self.encoders = nn.ModuleDict({
            "job"     : self._build_encoder(job_input_dimension, hidden_channels),
            "vehicle" : self._build_encoder(vehicle_input_dimension, hidden_channels),
            "path"    : self._build_encoder(path_input_dimension, hidden_channels),
        })

        self.conv_layers = nn.ModuleList()
        self.pre_norms   = nn.ModuleList()
        self.post_norms  = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            in_dim = hidden_channels
            out_dim = output_channels if layer_idx == num_layers - 1 else hidden_channels
            
            self.conv_layers.append(self._build_gat_conv(in_dim, out_dim))
            self.pre_norms.append(self._build_norm_dict(in_dim) if layer_idx > 0 else None)
            self.post_norms.append(self._build_norm_dict(out_dim))

        self.cross_attn_job_vehicle = CrossAttention(
            output_channels, 
            self.cross_attention_heads, 
            self.cross_attention_dropout
        )
        
        self.cross_attn_job_path = CrossAttention(
            output_channels,
            self.cross_attention_heads,
            self.cross_attention_dropout
        )
        
        self.cross_attn_vehicle_job = CrossAttention(
            output_channels,
            self.cross_attention_heads,
            self.cross_attention_dropout
        )

        self.pooling = nn.ModuleDict({
            "job"     : SelfAttention(hidden_dim=output_channels, num_heads=self.attention_num_heads, dropout=self.attention_dropout),
            "vehicle" : SelfAttention(hidden_dim=output_channels, num_heads=self.attention_num_heads, dropout=self.attention_dropout),
            "path"    : SelfAttention(hidden_dim=output_channels, num_heads=self.attention_num_heads, dropout=self.attention_dropout),
        })

        self._initialize_weights()

    def _build_encoder(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def _build_norm_dict(self, dim):
        return nn.ModuleDict({node_type: nn.LayerNorm(dim) for node_type in self.node_types})

    def _build_gatv2conv(self, in_channels, out_channels):
        heads = self.gat_heads
        concat = self.gat_concat
        dropout = self.gat_dropout
        
        if isinstance(in_channels, tuple):
            in_ch = in_channels[0]
        else:
            in_ch = in_channels
        
        if concat:
            assert out_channels % heads == 0, f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            head_dim = out_channels // heads
        else:
            head_dim = out_channels
        
        return GATv2Conv(
            in_channels=in_ch,
            out_channels=head_dim,
            heads=heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=False,
            edge_dim=self.edge_attribute_dimension,
        )

    def _build_gat_conv(self, in_dim, out_dim):

        edge_types = {
            ("job", "job_sequence", "job"): (in_dim, out_dim),
            ("path", "path_sequence", "path"): (in_dim, out_dim),
            ("vehicle", "vehicle_assigned", "job"): ((in_dim, in_dim), out_dim),
            ("job", "vehicle_assigned", "vehicle"): ((in_dim, in_dim), out_dim),
            ("job", "job_near_path", "path"): ((in_dim, in_dim), out_dim),
            ("path", "job_near_path", "job"): ((in_dim, in_dim), out_dim),
            ("job", "job_vehicle_proximity", "vehicle"): ((in_dim, in_dim), out_dim),
            ("vehicle", "job_vehicle_proximity", "job"): ((in_dim, in_dim), out_dim),
        }
        return HeteroConv(
            {edge_type: self._build_gatv2conv(in_channels, out_channels) 
             for edge_type, (in_channels, out_channels) in edge_types.items()},
            aggr="sum",
        )

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, graph):
        edge_index_dict = graph.edge_index_dict
        edge_attr_dict = {k: graph[k].edge_attr for k in graph.edge_types}
        
        if self.training:
            edge_attr_dict = {k: self.edge_dropout(v) for k, v in edge_attr_dict.items()}

        embeddings = {node_type: self.encoders[node_type](graph.x_dict[node_type]) for node_type in self.node_types}

        for layer_idx, (conv, pre_norm, post_norm) in enumerate(zip(self.conv_layers, self.pre_norms, self.post_norms)):
            if pre_norm is not None:
                residual = embeddings
                embeddings = {k: pre_norm[k](v) for k, v in embeddings.items()}
        
            embeddings = conv(embeddings, edge_index_dict, edge_attr_dict=edge_attr_dict)
            
            if layer_idx > 0 and residual[self.node_types[0]].shape[-1] == embeddings[self.node_types[0]].shape[-1]:
                embeddings = {k: embeddings[k] + residual[k] for k in embeddings}
            
            embeddings = {k: self.activation(post_norm[k](v)) for k, v in embeddings.items()}

        job_emb = embeddings["job"]
        vehicle_emb = embeddings["vehicle"]
        path_emb = embeddings["path"]
        
        job_emb_refined = self.cross_attn_job_vehicle(job_emb, vehicle_emb)
        job_emb_refined = self.cross_attn_job_path(job_emb_refined, path_emb)
        
        vehicle_emb_refined = self.cross_attn_vehicle_job(vehicle_emb, job_emb)
        
        embeddings["job"] = job_emb_refined
        embeddings["vehicle"] = vehicle_emb_refined

        context = {node_type: self.pooling[node_type](embeddings[node_type]) for node_type in self.node_types}

        return embeddings, context["job"], context["vehicle"], context["path"]

