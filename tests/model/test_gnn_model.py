import torch
from torch_geometric.data import Batch

from core.shared import EntityPool, Graph, RoutingState
from model.gnn_model import GNN
from tests.conftest import make_jobs, make_route, make_vehicles


def build_graph(cpu_config, num_jobs, num_vehicles):
    jobs     = make_jobs(num_jobs)
    vehicles = make_vehicles(num_vehicles)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:2])] if num_jobs >= 2 else [],
        unassigned_ids = {job.id for job in jobs[2:]},
    )
    return Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)


def test_encoders_match_input_dims(cpu_config):
    gnn = GNN(cpu_config.model)

    assert gnn.encoders["job"].in_features == cpu_config.model.job_input_dim
    assert gnn.encoders["vehicle"].in_features == cpu_config.model.vehicle_input_dim


def test_conv_stack_depth_matches_config(cpu_config):
    gnn = GNN(cpu_config.model)

    assert len(gnn.convs) == cpu_config.model.gnn_num_layers


def test_forward_emits_embeddings_and_context(cpu_config, seeded):
    gnn   = GNN(cpu_config.model)
    graph = build_graph(cpu_config, num_jobs=5, num_vehicles=3)

    embeddings, context = gnn(graph)

    embedding_dim = cpu_config.model.policy_embedding_dim

    assert embeddings["job"].shape == (5, embedding_dim)
    assert embeddings["vehicle"].shape == (3, embedding_dim)
    assert context.shape == (1, 2 * embedding_dim)
    assert torch.isfinite(context).all()


def test_forward_context_pools_over_batch(cpu_config, seeded):
    gnn    = GNN(cpu_config.model)
    graphs = [build_graph(cpu_config, num_jobs=4, num_vehicles=2), build_graph(cpu_config, num_jobs=6, num_vehicles=3)]
    batch  = Batch.from_data_list(graphs)

    _, context = gnn(batch)

    assert context.shape == (2, 2 * cpu_config.model.policy_embedding_dim)
