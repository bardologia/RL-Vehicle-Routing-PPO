import torch

from core.shared import EntityPool, Graph, RoutingState
from core.training import PPOMemory
from model.policy_model import Action
from tests.conftest import make_jobs, make_route, make_vehicles


def build_graph(cpu_config):
    jobs     = make_jobs(4)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2])], unassigned_ids={jobs[2].id, jobs[3].id})
    return Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state)


def add_one(memory, graph, with_old_logits=True):
    memory.add(
        graph               = graph,
        action              = Action(operator=0, vehicle_index=0, job_index=2),
        log_prob_operator   = torch.tensor(-0.1),
        log_prob_vehicle    = torch.tensor(-0.2),
        log_prob_job        = torch.tensor(-0.3),
        reward              = 1.5,
        state_value         = torch.tensor(0.5),
        mask_info           = {"unassigned_job_indices": [2]},
        done                = False,
        bootstrap_value     = 2.0,
        old_operator_logits = torch.zeros(4) if with_old_logits else None,
        old_vehicle_logits  = torch.zeros(3, 2) if with_old_logits else None,
        old_job_logits      = torch.zeros(3, 2, 4) if with_old_logits else None,
    )


def test_add_appends_to_every_parallel_list(cpu_config):
    memory = PPOMemory()
    graph  = build_graph(cpu_config)

    add_one(memory, graph)

    assert len(memory.graphs) == 1
    assert len(memory.actions) == 1
    assert len(memory.rewards) == 1
    assert len(memory.state_values) == 1
    assert len(memory.dones) == 1
    assert len(memory.bootstrap_values) == 1
    assert memory.rewards[0] == 1.5
    assert memory.bootstrap_values[0] == 2.0


def test_add_stores_detached_cpu_graph_copy(cpu_config):
    memory = PPOMemory()
    graph  = build_graph(cpu_config)

    add_one(memory, graph)

    stored = memory.graphs[0]

    assert stored is not graph
    assert stored["job"].x.device.type == "cpu"
    assert stored["job"].x.requires_grad is False
    assert torch.equal(stored["job"].x, graph["job"].x)


def test_add_records_old_logits_when_present(cpu_config):
    memory = PPOMemory()

    add_one(memory, build_graph(cpu_config), with_old_logits=True)

    assert len(memory.old_operator_logits) == 1
    assert len(memory.old_vehicle_logits) == 1
    assert len(memory.old_job_logits) == 1


def test_add_skips_old_logits_when_absent(cpu_config):
    memory = PPOMemory()

    add_one(memory, build_graph(cpu_config), with_old_logits=False)

    assert memory.old_operator_logits == []
    assert memory.old_vehicle_logits == []
    assert memory.old_job_logits == []


def test_clear_resets_all_buffers(cpu_config):
    memory = PPOMemory()
    graph  = build_graph(cpu_config)

    add_one(memory, graph)
    add_one(memory, graph)

    memory.clear()

    assert memory.graphs == []
    assert memory.actions == []
    assert memory.rewards == []
    assert memory.state_values == []
    assert memory.old_operator_logits == []
    assert memory.dones == []
    assert memory.bootstrap_values == []
