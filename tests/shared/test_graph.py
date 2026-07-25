import pytest
import torch

from core.shared import EntityPool, Graph, RelationCompleter, RoutingState, Stop
from tests.conftest import make_jobs, make_route, make_vehicles


def build_graph(cpu_config, num_jobs=4, num_vehicles=2, stops=2, unassigned=2):
    jobs     = make_jobs(num_jobs)
    vehicles = make_vehicles(num_vehicles)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:stops])],
        unassigned_ids = {job.id for job in jobs[stops:stops + unassigned]},
    )

    data = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)
    return data, jobs, vehicles, state


def test_node_feature_shapes(cpu_config):
    data, jobs, vehicles, _ = build_graph(cpu_config)

    assert data["job"].x.shape == (len(jobs), cpu_config.model.job_input_dim)
    assert data["vehicle"].x.shape == (len(vehicles), cpu_config.model.vehicle_input_dim)


def test_all_model_relations_exist(cpu_config):
    data, _, _, _ = build_graph(cpu_config)

    for relation in RelationCompleter.required_relations:
        assert relation in data.edge_types
        assert data[relation].edge_index.shape[0] == 2
        assert data[relation].edge_attr.shape[1] == 4


def test_bipartite_proximity_covers_all_pairs(cpu_config):
    data, jobs, vehicles, _ = build_graph(cpu_config)

    forward  = data[("job", "job_vehicle_proximity", "vehicle")].edge_index.shape[1]
    backward = data[("vehicle", "job_vehicle_proximity", "job")].edge_index.shape[1]

    assert forward == len(jobs) * len(vehicles)
    assert backward == len(jobs) * len(vehicles)


def test_sequence_and_assignment_edge_counts(cpu_config):
    data, _, _, state = build_graph(cpu_config, stops=3)

    stops = len(state.routes[0].stops)

    assert data[("job", "job_sequence", "job")].edge_index.shape[1] == 2 * (stops - 1)
    assert data[("vehicle", "vehicle_assigned", "job")].edge_index.shape[1] == stops
    assert data[("job", "vehicle_assigned", "vehicle")].edge_index.shape[1] == stops


def test_unassigned_and_assignment_flags(cpu_config):
    data, jobs, _, state = build_graph(cpu_config)

    features = data["job"].x

    for index, job in enumerate(jobs):
        is_unassigned = float(job.id in state.unassigned_ids)
        is_assigned   = float(job.id in state.assigned_job_ids)
        assert features[index, 5].item() == is_unassigned
        assert features[index, 6].item() == is_assigned


def test_job_priority_feature_scale(cpu_config):
    data, jobs, _, _ = build_graph(cpu_config)

    features = data["job"].x

    for index, job in enumerate(jobs):
        assert features[index, 2].item() == pytest.approx(job.priority / 5.0)


def test_vehicle_capacity_and_load_features(cpu_config):
    data, _, vehicles, state = build_graph(cpu_config)

    features = data["vehicle"].x
    loaded   = state.routes[0]
    pickups  = sum(1 for stop in loaded.stops if stop.kind == "pickup")

    assert features[0, 4].item() == pytest.approx(vehicles[0].capacity / 10.0)
    assert features[0, 5].item() == pytest.approx(vehicles[0].onboard / vehicles[0].capacity)
    assert features[0, 6].item() == pytest.approx(len(loaded.job_ids) / 10.0)
    assert features[0, 7].item() == pytest.approx((vehicles[0].onboard + pickups) / vehicles[0].capacity)
    assert features[1, 6].item() == 0.0


def test_vehicle_capacity_feature_counts_planned_pickups(cpu_config):
    jobs     = make_jobs(2, kind="repossession")
    vehicles = make_vehicles(1)
    route    = make_route(vehicles[0], jobs)
    for stop in route.stops:
        stop.kind = "pickup"

    state = RoutingState(routes=[route], unassigned_ids=set())
    data  = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    assert data["vehicle"].x[0, 7].item() == pytest.approx(2 / vehicles[0].capacity)


def test_context_features_track_scale(cpu_config):
    small, _, _, _ = build_graph(cpu_config, num_jobs=4, num_vehicles=2)
    large, _, _, _ = build_graph(cpu_config, num_jobs=12, num_vehicles=4)

    assert small.context.shape == (1, cpu_config.model.context_input_dim)
    assert large.context[0, 0].item() > small.context[0, 0].item()
    assert large.context[0, 2].item() > small.context[0, 2].item()


def test_context_features_track_burst_budget(cpu_config):
    jobs     = make_jobs(3)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[], unassigned_ids={job.id for job in jobs})

    data = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800, round_index=2, edits_remaining=3)

    assert data.context[0, 6].item() == pytest.approx(2 / cpu_config.training.max_disruptions_per_episode)
    assert data.context[0, 7].item() == pytest.approx(3 / cpu_config.training.max_edits_per_round)


def test_graph_is_fully_on_cpu(cpu_config):
    data, _, _, _ = build_graph(cpu_config)

    for node_type in data.node_types:
        assert data[node_type].x.device.type == "cpu"
    for relation in data.edge_types:
        assert data[relation].edge_index.device.type == "cpu"
        assert data[relation].edge_attr.device.type == "cpu"


def test_zero_route_state_produces_edge_complete_graph(cpu_config):
    jobs     = make_jobs(3)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[], unassigned_ids={job.id for job in jobs})

    data = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    for relation in RelationCompleter.required_relations:
        assert relation in data.edge_types

    assert data[("job", "job_sequence", "job")].edge_index.shape[1] == 0
    assert data[("job", "job_vehicle_proximity", "vehicle")].edge_index.shape[1] == len(jobs) * len(vehicles)


def test_single_job_single_vehicle_graph(cpu_config):
    jobs     = make_jobs(1)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[], unassigned_ids={jobs[0].id})

    data = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    assert data["job"].x.shape == (1, cpu_config.model.job_input_dim)
    assert data["vehicle"].x.shape == (1, cpu_config.model.vehicle_input_dim)
    assert data[("job", "job_vehicle_proximity", "vehicle")].edge_index.shape[1] == 1


def test_empty_job_pool_produces_typed_empty_features(cpu_config):
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[], unassigned_ids=set())

    data = Graph(cpu_config).build(EntityPool([]), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    assert data["job"].x.shape == (0, cpu_config.model.job_input_dim)
    assert data["vehicle"].x.shape == (len(vehicles), cpu_config.model.vehicle_input_dim)


def test_depot_delivery_stops_are_not_treated_as_job_stops(cpu_config):
    jobs     = make_jobs(2, kind="repossession")
    vehicles = make_vehicles(1)
    route    = make_route(vehicles[0], jobs)
    route.stops[0].kind = "pickup"
    route.stops[1].kind = "pickup"
    route.stops.insert(1, Stop(job_id=jobs[0].id, location=(-46.63, -23.55), kind="delivery", service=120))

    state = RoutingState(routes=[route], unassigned_ids=set())
    data  = Graph(cpu_config).build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    assert data[("job", "job_sequence", "job")].edge_index.shape[1] == 2
    assert data[("vehicle", "vehicle_assigned", "job")].edge_index.shape[1] == 2


def test_edge_attributes_are_finite_and_flagged(cpu_config):
    data, _, _, _ = build_graph(cpu_config)

    for relation in data.edge_types:
        attrs = data[relation].edge_attr
        if attrs.numel() == 0:
            continue
        assert torch.isfinite(attrs).all()
        assert ((attrs[:, 2] == 0.0) | (attrs[:, 2] == 1.0)).all()
        assert ((attrs[:, 3] == 0.0) | (attrs[:, 3] == 1.0)).all()


def test_assigned_proximity_edge_flags_owner_vehicle(cpu_config):
    data, _, _, _ = build_graph(cpu_config, num_jobs=3, num_vehicles=2, stops=2, unassigned=1)

    proximity   = data[("job", "job_vehicle_proximity", "vehicle")]
    assigned    = proximity.edge_attr[:, 3]

    assert assigned.sum().item() == 2.0


def test_mappings_expose_nodes_and_edge_type_ids(cpu_config):
    jobs     = make_jobs(4)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2])], unassigned_ids={job.id for job in jobs[2:]})

    graph_builder = Graph(cpu_config)
    graph_builder.build(EntityPool(jobs), EntityPool(vehicles), state, (-46.63, -23.55), 28800)

    mappings = graph_builder.mappings()

    assert set(mappings.keys()) == {"index_to_node", "edge_type_ids"}
    assert mappings["edge_type_ids"] == graph_builder.edge_names
    assert all("node_type" in node for node in mappings["index_to_node"])


def test_relation_completer_fills_absent_relations_with_empty_tensors(cpu_config):
    from torch_geometric.data import HeteroData

    data      = HeteroData()
    data["job"].x     = torch.zeros((2, cpu_config.model.job_input_dim))
    data["vehicle"].x = torch.zeros((1, cpu_config.model.vehicle_input_dim))

    completed = RelationCompleter(cpu_config).build(data)

    for relation in RelationCompleter.required_relations:
        assert completed[relation].edge_index.shape == (2, 0)
        assert completed[relation].edge_attr.shape == (0, cpu_config.model.edge_attr_dim)
