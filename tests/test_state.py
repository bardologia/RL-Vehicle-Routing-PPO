import pytest

from core.state import EntityPool, Job, Route, RoutingState, Stop, Vehicle
from tests.conftest import load_fixture, make_jobs, make_route, make_vehicles


def test_job_dict_round_trip():
    job = Job(id=3, location=(-46.6, -23.5), service=300, setup=10, amount=2, priority=4, description="Job 3")

    restored = Job.from_dict(job.to_dict())

    assert restored == job


def test_vehicle_dict_round_trip():
    vehicle = Vehicle(id=7, start=(-46.7, -23.6), capacity=3, speed_factor=1.1, time_window=(0, 3600), return_to_depot=True, description="Vehicle 7")

    restored = Vehicle.from_dict(vehicle.to_dict())

    assert restored == vehicle


def test_vehicle_vroom_payload_wraps_capacity_and_amount():
    job     = Job(id=1, location=(-46.6, -23.5), amount=2)
    vehicle = Vehicle(id=2, start=(-46.7, -23.6), capacity=3)

    assert job.vroom_payload()["amount"] == [2]
    assert vehicle.vroom_payload()["capacity"] == [3]


def test_vehicle_with_capacity_returns_modified_copy():
    vehicle = Vehicle(id=1, start=(-46.7, -23.6), capacity=2)
    bumped  = vehicle.with_capacity(5)

    assert bumped.capacity == 5
    assert vehicle.capacity == 2
    assert bumped.id == vehicle.id


def test_route_remove_jobs_adjusts_service_and_end():
    jobs  = make_jobs(3)
    route = make_route(make_vehicles(1)[0], jobs)

    route.remove_jobs({jobs[2].id})

    assert route.job_ids == [jobs[0].id, jobs[1].id]
    assert route.service == jobs[0].service + jobs[1].service
    assert route.end == jobs[1].location


def test_route_locations_orders_start_stops_end():
    vehicle = make_vehicles(1)[0]
    jobs    = make_jobs(2)
    route   = make_route(vehicle, jobs)

    points = route.locations

    assert points[0] == vehicle.start
    assert points[1] == jobs[0].location
    assert points[-1] == route.end


def test_routing_state_totals_always_derive_from_routes():
    vehicles = make_vehicles(2)
    jobs     = make_jobs(4)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2], cost=500), make_route(vehicles[1], jobs[2:], cost=700)])

    assert state.cost == 1200

    state.routes[0].cost = 900

    assert state.cost == 1600


def test_replace_route_moves_displaced_jobs_to_unassigned():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(3)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs)], unassigned_ids=set())

    new_route = make_route(vehicles[0], jobs[:1])
    state.replace_route(new_route)

    assert state.route_of_vehicle(vehicles[0].id).job_ids == [jobs[0].id]
    assert state.unassigned_ids == {jobs[1].id, jobs[2].id}


def test_replace_route_removes_new_jobs_from_unassigned():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    state    = RoutingState(routes=[], unassigned_ids={jobs[0].id, jobs[1].id})

    state.replace_route(make_route(vehicles[0], jobs))

    assert state.unassigned_ids == set()
    assert state.num_routes == 1


def test_replace_route_prunes_empty_route():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs)])

    state.replace_route(Route(vehicle_id=vehicles[0].id))

    assert state.num_routes == 0
    assert state.unassigned_ids == {jobs[0].id, jobs[1].id}


def test_remove_jobs_prunes_emptied_routes_and_unassigned():
    vehicles = make_vehicles(2)
    jobs     = make_jobs(4)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:1]), make_route(vehicles[1], jobs[1:3])],
        unassigned_ids = {jobs[3].id},
    )

    state.remove_jobs({jobs[0].id, jobs[3].id})

    assert state.num_routes == 1
    assert state.route_of_vehicle(vehicles[1].id) is not None
    assert state.unassigned_ids == set()


def test_remove_vehicles_returns_orphaned_job_ids():
    vehicles = make_vehicles(2)
    jobs     = make_jobs(4)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2]), make_route(vehicles[1], jobs[2:])])

    orphaned = state.remove_vehicles({vehicles[0].id})

    assert orphaned == {jobs[0].id, jobs[1].id}
    assert state.vehicle_ids_with_routes == {vehicles[1].id}


def test_copy_is_deep():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs)], unassigned_ids={99})

    clone = state.copy()
    clone.routes[0].stops.pop()
    clone.unassigned_ids.add(100)

    assert len(state.routes[0].stops) == 2
    assert state.unassigned_ids == {99}


def test_payload_round_trip():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs)], unassigned_ids={42})

    restored = RoutingState.from_payload(state.to_payload())

    assert restored.to_payload() == state.to_payload()


def test_payload_rejects_unknown_schema():
    with pytest.raises(ValueError):
        RoutingState.from_payload({"schema": "event_state", "routes": [], "unassigned": []})


@pytest.mark.parametrize("fixture_name", ["vroom_solution_small", "vroom_solution_tight"])
def test_from_vroom_parses_recorded_solutions(fixture_name):
    solution = load_fixture(fixture_name)
    state    = RoutingState.from_vroom(solution)

    assert state.num_routes == len(solution["routes"])
    assert state.num_unassigned == len(solution.get("unassigned", []))
    assert state.cost == sum(route["cost"] for route in solution["routes"])

    for route in state.routes:
        assert isinstance(route.vehicle_id, int)
        assert route.start is not None
        assert all(isinstance(stop.job_id, int) for stop in route.stops)
        assert all(len(stop.location) == 2 for stop in route.stops)


def test_from_vroom_keeps_assigned_and_unassigned_disjoint():
    state = RoutingState.from_vroom(load_fixture("vroom_solution_tight"))

    assert state.assigned_job_ids.isdisjoint(state.unassigned_ids)


def test_entity_pool_preserves_order_and_indices():
    jobs = make_jobs(4)
    pool = EntityPool(jobs)

    assert pool.ids == [0, 1, 2, 3]
    assert pool.index_of(2) == 2
    assert pool.by_id(3) is jobs[3]
    assert pool.contains(1)
    assert not pool.contains(99)
    assert pool.index_of(99) is None


def test_entity_pool_reindexes_after_mutation():
    pool = EntityPool(make_jobs(3))

    pool.remove({1})

    assert pool.ids == [0, 2]
    assert pool.index_of(2) == 1
    assert pool.next_id() == 3

    pool.add(make_jobs(1, first_id=pool.next_id()))

    assert pool.ids == [0, 2, 3]
    assert len(pool) == 3
