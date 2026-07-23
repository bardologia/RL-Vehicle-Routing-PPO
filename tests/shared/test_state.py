import pytest

from core.shared import EntityPool, Job, Route, RoutingState, Stop, Vehicle
from tests.conftest import load_fixture, make_jobs, make_route, make_stops, make_vehicles


def test_job_dict_round_trip():
    job = Job(id=3, location=(-46.6, -23.5), service=300, setup=10, amount=2, priority=4, description="Job 3")

    restored = Job.from_dict(job.to_dict())

    assert restored == job


def test_job_vroom_payload_keys_and_types():
    job     = Job(id=1, location=(-46.6, -23.5), amount=2, priority=3)
    payload = job.vroom_payload()

    assert set(payload.keys()) == {"id", "location", "setup", "service", "amount", "priority", "description"}
    assert payload["amount"] == [2]
    assert payload["location"] == [-46.6, -23.5]


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


def test_stop_dict_round_trip():
    stop = Stop(job_id=4, location=(-46.6, -23.5), arrival=100, duration=50, service=300, load=2)

    restored = Stop.from_dict(stop.to_dict())

    assert restored == stop


def test_route_dict_round_trip_with_geometry_and_path():
    route             = make_route(make_vehicles(1)[0], make_jobs(2))
    route.geometry    = "abc"
    route.path_coords = [(-46.6, -23.5), (-46.5, -23.4)]

    restored = Route.from_dict(route.to_dict())

    assert restored.geometry == "abc"
    assert restored.path_coords == [(-46.6, -23.5), (-46.5, -23.4)]
    assert restored.job_ids == route.job_ids


def test_route_dict_round_trip_with_none_geometry_and_start():
    route = Route(vehicle_id=1, stops=make_stops(make_jobs(1)))

    restored = Route.from_dict(route.to_dict())

    assert restored.start is None
    assert restored.end is None
    assert restored.geometry is None
    assert restored.path_coords is None


def test_route_job_ids_orders_by_stop_sequence():
    jobs  = make_jobs(3)
    route = make_route(make_vehicles(1)[0], jobs)

    assert route.job_ids == [0, 1, 2]


def test_route_remove_jobs_adjusts_service_and_end():
    jobs  = make_jobs(3)
    route = make_route(make_vehicles(1)[0], jobs)

    route.remove_jobs({jobs[2].id})

    assert route.job_ids == [jobs[0].id, jobs[1].id]
    assert route.service == jobs[0].service + jobs[1].service
    assert route.end == jobs[1].location


def test_route_remove_all_jobs_leaves_end_untouched():
    jobs     = make_jobs(2)
    route    = make_route(make_vehicles(1)[0], jobs)
    prev_end = route.end

    route.remove_jobs({jobs[0].id, jobs[1].id})

    assert route.stops == []
    assert route.service == 0
    assert route.end == prev_end


def test_route_locations_orders_start_stops_end():
    vehicle = make_vehicles(1)[0]
    jobs    = make_jobs(2)
    route   = make_route(vehicle, jobs)

    points = route.locations

    assert points[0] == vehicle.start
    assert points[1] == jobs[0].location
    assert points[-1] == route.end


def test_route_locations_omit_missing_start_and_end():
    route = Route(vehicle_id=1, stops=make_stops(make_jobs(2)))

    points = route.locations

    assert len(points) == 2
    assert points[0] == make_jobs(2)[0].location


def test_route_copy_is_independent():
    route = make_route(make_vehicles(1)[0], make_jobs(2))
    clone = route.copy()

    clone.stops.pop()

    assert len(route.stops) == 2


def test_empty_routing_state_totals_are_zero():
    state = RoutingState(routes=[], unassigned_ids=set())

    assert state.num_routes == 0
    assert state.num_unassigned == 0
    assert state.cost == 0
    assert state.duration == 0
    assert state.service == 0
    assert state.distance == 0
    assert state.waiting_time == 0
    assert state.assigned_job_ids == set()
    assert state.vehicle_ids_with_routes == set()


def test_routing_state_totals_always_derive_from_routes():
    vehicles = make_vehicles(2)
    jobs     = make_jobs(4)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2], cost=500), make_route(vehicles[1], jobs[2:], cost=700)])

    assert state.cost == 1200

    state.routes[0].cost = 900

    assert state.cost == 1600


def test_route_of_vehicle_and_job_lookups():
    vehicles = make_vehicles(2)
    jobs     = make_jobs(3)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:2]), make_route(vehicles[1], jobs[2:])])

    assert state.route_of_vehicle(vehicles[1].id).job_ids == [jobs[2].id]
    assert state.route_of_vehicle(999) is None
    assert state.route_of_job(jobs[0].id).vehicle_id == vehicles[0].id
    assert state.route_of_job(999) is None


def test_add_unassigned_grows_the_set():
    state = RoutingState(routes=[], unassigned_ids={1})

    state.add_unassigned({2, 3})

    assert state.unassigned_ids == {1, 2, 3}


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


def test_remove_vehicles_absent_id_orphans_nothing():
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs)])

    orphaned = state.remove_vehicles({999})

    assert orphaned == set()
    assert state.num_routes == 1


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


def test_payload_schema_tag_and_sorted_unassigned():
    state   = RoutingState(routes=[], unassigned_ids={5, 1, 3})
    payload = state.to_payload()

    assert payload["schema"] == "routing-state-v1"
    assert payload["unassigned"] == [1, 3, 5]


def test_payload_rejects_unknown_schema():
    with pytest.raises(ValueError):
        RoutingState.from_payload({"schema": "event_state", "routes": [], "unassigned": []})


def test_payload_rejects_missing_schema():
    with pytest.raises(ValueError):
        RoutingState.from_payload({"routes": [], "unassigned": []})


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


def test_from_vroom_handles_empty_solution():
    state = RoutingState.from_vroom({})

    assert state.num_routes == 0
    assert state.num_unassigned == 0


def test_route_from_vroom_ignores_steps_without_location():
    route = Route.from_vroom({
        "vehicle" : 3,
        "steps"   : [
            {"type": "start", "location": [-46.6, -23.5]},
            {"type": "job", "id": 5},
            {"type": "job", "job": 6, "location": [-46.5, -23.4], "load": [2]},
            {"type": "end", "location": [-46.4, -23.3]},
        ],
    })

    assert route.vehicle_id == 3
    assert route.job_ids == [6]
    assert route.stops[0].load == 2
    assert route.start == (-46.6, -23.5)
    assert route.end == (-46.4, -23.3)


def test_route_from_vroom_defaults_missing_metrics_to_zero():
    route = Route.from_vroom({"vehicle": 0})

    assert route.cost == 0
    assert route.duration == 0
    assert route.stops == []


def test_route_from_vroom_prefers_job_over_id_field():
    route = Route.from_vroom({
        "vehicle" : 0,
        "steps"   : [{"type": "job", "job": 9, "id": 100, "location": [-46.6, -23.5]}],
    })

    assert route.job_ids == [9]


def test_entity_pool_preserves_order_and_indices():
    jobs = make_jobs(4)
    pool = EntityPool(jobs)

    assert pool.ids == [0, 1, 2, 3]
    assert pool.index_of(2) == 2
    assert pool.by_id(3) is jobs[3]
    assert pool.contains(1)
    assert not pool.contains(99)
    assert pool.index_of(99) is None


def test_entity_pool_empty_defaults():
    pool = EntityPool()

    assert pool.ids == []
    assert len(pool) == 0
    assert pool.next_id() == 0
    assert pool.index_of(0) is None


def test_entity_pool_by_id_raises_for_missing():
    pool = EntityPool(make_jobs(2))

    with pytest.raises(KeyError):
        pool.by_id(99)


def test_entity_pool_getitem_and_iteration():
    jobs = make_jobs(3)
    pool = EntityPool(jobs)

    assert pool[1] is jobs[1]
    assert list(pool) == jobs


def test_entity_pool_reindexes_after_mutation():
    pool = EntityPool(make_jobs(3))

    pool.remove({1})

    assert pool.ids == [0, 2]
    assert pool.index_of(2) == 1
    assert pool.next_id() == 3

    pool.add(make_jobs(1, first_id=pool.next_id()))

    assert pool.ids == [0, 2, 3]
    assert len(pool) == 3
