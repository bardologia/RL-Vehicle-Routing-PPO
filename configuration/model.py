from dataclasses import dataclass


@dataclass
class ModelConfig:
    job_input_dim     : int = 7
    vehicle_input_dim : int = 5
    edge_attr_dim     : int = 4

    num_operators          : int = 4
    operator_embedding_dim : int = 32

    gnn_num_layers : int = 2

    policy_gnn_hidden_channels : int = 64
    policy_embedding_dim       : int = 64
    policy_actor_hidden_1      : int = 128
    policy_actor_hidden_2      : int = 64

    value_critic_hidden_1 : int = 64
    value_critic_hidden_2 : int = 64
