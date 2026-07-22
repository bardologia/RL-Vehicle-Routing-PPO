import pytest
import torch
from torch_geometric.data import Batch

from core.graph import Graph
from core.model import Policy
from core.state import EntityPool, RoutingState
from tests.conftest import make_jobs, make_route, make_vehicles


def build_graph(cpu_config, num_jobs, num_vehicles):
    jobs     = make_jobs(num_jobs)
    vehicles = make_vehicles(num_vehicles)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:2])] if num_jobs >= 2 else [],
        unassigned_ids = {job.id for job in jobs[2:]},
    )
    return Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state)


@pytest.fixture
def policy(cpu_config, seeded):
    return Policy(cpu_config)


def test_forward_contract_shapes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=5, num_vehicles=3)

    embeddings, context, op_logits, state_value = policy(graph)

    embed_dim = cpu_config.model.policy_embedding_dim

    assert embeddings["job"].shape == (5, embed_dim)
    assert embeddings["vehicle"].shape == (3, embed_dim)
    assert context.shape == (2 * embed_dim,)
    assert op_logits.shape == (cpu_config.model.num_operators,)
    assert state_value.ndim == 0


def test_compute_logits_shapes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=5, num_vehicles=3)

    embeddings, context, op_logits, _ = policy(graph)
    logits = policy.compute_logits(embeddings, context, op_logits)

    assert logits["veh_logits"].shape == (4, 3)
    assert logits["job_logits"].shape == (4, 3, 5)


def test_compute_logits_selected_op_squeezes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=4, num_vehicles=2)

    embeddings, context, op_logits, _ = policy(graph)
    logits = policy.compute_logits(embeddings, context, op_logits, selected_op=1)

    assert logits["veh_logits"].shape == (2,)
    assert logits["job_logits"].shape == (2, 4)


def test_forward_batch_matches_single_forward(policy, cpu_config):
    graphs = [build_graph(cpu_config, num_jobs=n, num_vehicles=v) for n, v in [(4, 2), (6, 3), (3, 2)]]

    per_sample = policy.forward_batch(Batch.from_data_list(graphs))

    for graph, sample in zip(graphs, per_sample):
        embeddings, context, op_logits, state_value = policy(graph)

        assert torch.allclose(embeddings["job"], sample["embeddings"]["job"], atol=1e-5)
        assert torch.allclose(embeddings["vehicle"], sample["embeddings"]["vehicle"], atol=1e-5)
        assert torch.allclose(context, sample["context"], atol=1e-5)
        assert torch.allclose(op_logits, sample["op_logits"], atol=1e-5)
        assert torch.allclose(state_value, sample["state_value"], atol=1e-5)


def test_forward_batch_gradients_reach_encoder(policy, cpu_config):
    graphs     = [build_graph(cpu_config, num_jobs=4, num_vehicles=2) for _ in range(3)]
    per_sample = policy.forward_batch(Batch.from_data_list(graphs))

    loss = sum(sample["state_value"] for sample in per_sample)
    loss.backward()

    encoder_grads = [p.grad for p in policy.graph_embedding.parameters() if p.grad is not None]

    assert len(encoder_grads) > 0
    assert all(torch.isfinite(grad).all() for grad in encoder_grads)


def test_act_respects_masks(policy, cpu_config, seeded):
    graph     = build_graph(cpu_config, num_jobs=5, num_vehicles=3)
    mask_info = {
        "unassigned_job_indices"     : [2, 3],
        "vehicles_with_jobs_indices" : [0],
        "vehicle_to_job_indices"     : {0: [0, 1], 1: [], 2: []},
    }

    for _ in range(60):
        action = policy.act(graph, mask_info=mask_info)["action"]

        if action.operator == 0:
            assert action.job_index in {2, 3}
        elif action.operator == 1:
            assert action.vehicle_index == 0
            assert action.job_index in {0, 1}
        else:
            assert action.vehicle_index == 0
            assert action.job_index == 0


def test_act_never_selects_blocked_operators(policy, cpu_config, seeded):
    graph     = build_graph(cpu_config, num_jobs=4, num_vehicles=2)
    mask_info = {
        "unassigned_job_indices"     : [],
        "vehicles_with_jobs_indices" : [],
        "vehicle_to_job_indices"     : {0: [], 1: []},
    }

    operators = {policy.act(graph, mask_info=mask_info)["action"].operator for _ in range(60)}

    assert operators <= {2, 3}


def test_checkpoint_round_trip(policy, cpu_config, tmp_path):
    policy.checkpoint("model.pt", str(tmp_path), training_state={"episode_index": 7})

    restored       = Policy(cpu_config)
    training_state = restored.load("model.pt", str(tmp_path))

    assert training_state == {"episode_index": 7}
    for original, loaded in zip(policy.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, loaded)
