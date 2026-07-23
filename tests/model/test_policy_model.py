import pytest
import torch
from torch_geometric.data import Batch

from core.shared import EntityPool, Graph, RoutingState
from model.policy_model import Action, Policy, PolicyCheckpoint
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


def test_action_holds_indices():
    action = Action(operator=1, vehicle_index=2, job_index=3)

    assert action.operator == 1
    assert action.vehicle_index == 2
    assert action.job_index == 3


def test_operator_embedding_shape(policy, cpu_config):
    weight = policy.operator_embedding.weight

    assert weight.shape == (cpu_config.model.num_operators, cpu_config.model.operator_embedding_dim)


def test_forward_contract_shapes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=5, num_vehicles=3)

    embeddings, context, operator_logits, state_value = policy(graph)

    embedding_dim = cpu_config.model.policy_embedding_dim

    assert embeddings["job"].shape == (5, embedding_dim)
    assert embeddings["vehicle"].shape == (3, embedding_dim)
    assert context.shape == (2 * embedding_dim,)
    assert operator_logits.shape == (cpu_config.model.num_operators,)
    assert state_value.ndim == 0


def test_compute_logits_shapes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=5, num_vehicles=3)

    embeddings, context, operator_logits, _ = policy(graph)
    logits = policy.compute_logits(embeddings, context, operator_logits)

    assert logits["vehicle_logits"].shape == (4, 3)
    assert logits["job_logits"].shape == (4, 3, 5)


def test_compute_logits_selected_operator_squeezes(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=4, num_vehicles=2)

    embeddings, context, operator_logits, _ = policy(graph)
    logits = policy.compute_logits(embeddings, context, operator_logits, selected_operator=1)

    assert logits["vehicle_logits"].shape == (2,)
    assert logits["job_logits"].shape == (2, 4)


def test_compute_logits_selected_matches_full_slice(policy, cpu_config):
    graph = build_graph(cpu_config, num_jobs=4, num_vehicles=3)

    embeddings, context, operator_logits, _ = policy(graph)

    full = policy.compute_logits(embeddings, context, operator_logits)

    for operator_index in range(cpu_config.model.num_operators):
        selected = policy.compute_logits(embeddings, context, operator_logits, selected_operator=operator_index)

        assert torch.allclose(selected["vehicle_logits"], full["vehicle_logits"][operator_index], atol=1e-6)
        assert torch.allclose(selected["job_logits"], full["job_logits"][operator_index], atol=1e-6)


def test_forward_batch_matches_single_forward(policy, cpu_config):
    graphs = [build_graph(cpu_config, num_jobs=n, num_vehicles=v) for n, v in [(4, 2), (6, 3), (3, 2)]]

    per_sample = policy.forward_batch(Batch.from_data_list(graphs))

    for graph, sample in zip(graphs, per_sample):
        embeddings, context, operator_logits, state_value = policy(graph)

        assert torch.allclose(embeddings["job"], sample["embeddings"]["job"], atol=1e-5)
        assert torch.allclose(embeddings["vehicle"], sample["embeddings"]["vehicle"], atol=1e-5)
        assert torch.allclose(context, sample["context"], atol=1e-5)
        assert torch.allclose(operator_logits, sample["operator_logits"], atol=1e-5)
        assert torch.allclose(state_value, sample["state_value"], atol=1e-5)


def test_forward_batch_gradients_reach_encoder(policy, cpu_config):
    graphs     = [build_graph(cpu_config, num_jobs=4, num_vehicles=2) for _ in range(3)]
    per_sample = policy.forward_batch(Batch.from_data_list(graphs))

    loss = sum(sample["state_value"] for sample in per_sample)
    loss.backward()

    encoder_grads = [p.grad for p in policy.graph_embedding.parameters() if p.grad is not None]

    assert len(encoder_grads) > 0
    assert all(torch.isfinite(grad).all() for grad in encoder_grads)


def test_select_action_is_deterministic_under_fixed_seed(policy, cpu_config):
    graph     = build_graph(cpu_config, num_jobs=5, num_vehicles=3)
    mask_info = {
        "unassigned_job_indices"     : [2, 3],
        "vehicles_with_jobs_indices" : [0],
        "vehicle_to_job_indices"     : {0: [0, 1], 1: [], 2: []},
    }

    torch.manual_seed(123)
    first = policy.select_action(graph, mask_info=mask_info)["action"]

    torch.manual_seed(123)
    second = policy.select_action(graph, mask_info=mask_info)["action"]

    assert (first.operator, first.vehicle_index, first.job_index) == (second.operator, second.vehicle_index, second.job_index)


def test_select_action_output_contract(policy, cpu_config):
    graph  = build_graph(cpu_config, num_jobs=4, num_vehicles=2)
    result = policy.select_action(graph, mask_info=None)

    for key in ("action", "state_value", "log_prob_operator", "log_prob_vehicle", "log_prob_job", "old_operator_logits", "old_vehicle_logits", "old_job_logits"):
        assert key in result

    assert result["old_vehicle_logits"].shape == (cpu_config.model.num_operators, 2)
    assert result["old_job_logits"].shape == (cpu_config.model.num_operators, 2, 4)


def test_act_respects_masks(policy, cpu_config, seeded):
    graph     = build_graph(cpu_config, num_jobs=5, num_vehicles=3)
    mask_info = {
        "unassigned_job_indices"     : [2, 3],
        "vehicles_with_jobs_indices" : [0],
        "vehicle_to_job_indices"     : {0: [0, 1], 1: [], 2: []},
    }

    for _ in range(60):
        action = policy.select_action(graph, mask_info=mask_info)["action"]

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

    operators = {policy.select_action(graph, mask_info=mask_info)["action"].operator for _ in range(60)}

    assert operators <= {2, 3}


def test_checkpoint_round_trip(policy, cpu_config, tmp_path):
    policy_checkpoint = PolicyCheckpoint()
    policy_checkpoint.save(policy, "model.pt", str(tmp_path), training_state={"episode_index": 7})

    restored   = Policy(cpu_config)
    checkpoint = policy_checkpoint.load(restored, "model.pt", str(tmp_path), map_location=cpu_config.training.device)

    assert checkpoint["training_state"] == {"episode_index": 7}
    for original, loaded in zip(policy.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(original, loaded)


def test_checkpoint_persists_optimizer_state(policy, cpu_config, tmp_path):
    optimizer         = torch.optim.Adam(policy.parameters(), lr=1e-3)
    policy_checkpoint = PolicyCheckpoint()

    policy_checkpoint.save(policy, "model.pt", str(tmp_path), training_state=None, optimizer=optimizer)

    checkpoint = policy_checkpoint.read("model.pt", str(tmp_path), map_location=cpu_config.training.device)

    assert "optimizer_state_dict" in checkpoint
    assert checkpoint["training_state"] is None


def test_checkpoint_predicts_identically_after_reload(policy, cpu_config, tmp_path):
    graph     = build_graph(cpu_config, num_jobs=5, num_vehicles=3)
    embeddings, context, operator_logits, _ = policy(graph)

    PolicyCheckpoint().save(policy, "model.pt", str(tmp_path))
    restored = Policy(cpu_config)
    PolicyCheckpoint().load(restored, "model.pt", str(tmp_path), map_location=cpu_config.training.device)

    r_embeddings, r_context, r_operator_logits, _ = restored(graph)

    assert torch.allclose(operator_logits, r_operator_logits, atol=1e-6)
    assert torch.allclose(context, r_context, atol=1e-6)
