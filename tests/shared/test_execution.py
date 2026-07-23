from core.shared import RoutingState
from tests.conftest import make_jobs, make_vehicles


def load_execution_scenario(environment, jobs, vehicles, state):
    environment.load_scenario({
        "jobs"     : [job.to_dict() for job in jobs],
        "vehicles" : [vehicle.to_dict() for vehicle in vehicles],
        "depot"    : [-46.63, -23.55],
        "clock"    : 28800,
        "state"    : state.to_payload(),
    })


def solved_state(environment, jobs, vehicles):
    from core.shared.environment import vroom
    return vroom.solve(jobs, vehicles, depot=environment.depot, clock=environment.clock)


def test_tick_serves_due_stops_and_moves_vehicle(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)

    load_execution_scenario(environment, jobs, vehicles, RoutingState(routes=[], unassigned_ids={j.id for j in jobs}))
    environment.current_state = solved_state(environment, list(environment.jobs), list(environment.vehicles))

    summary = environment.advance_execution()

    assert environment.clock == 28800 + environment.config.env.tick_seconds
    assert summary["served"] == [jobs[0].id]
    assert not environment.jobs.contains(jobs[0].id)
    assert environment.jobs.contains(jobs[1].id)
    assert environment.vehicles.by_id(vehicles[0].id).start == jobs[0].location


def test_successful_pickup_loads_motorcycle(environment, fake_vroom):
    environment.config.env.repossession_success_probability = 1.0

    jobs     = make_jobs(2, kind="repossession")
    vehicles = make_vehicles(1)

    load_execution_scenario(environment, jobs, vehicles, RoutingState(routes=[], unassigned_ids={j.id for j in jobs}))
    environment.current_state = solved_state(environment, list(environment.jobs), list(environment.vehicles))

    summary = environment.advance_execution()

    assert summary["served"] == [jobs[0].id]
    assert environment.vehicles.by_id(vehicles[0].id).onboard == 1


def test_failed_pickup_completes_job_without_loading(environment, fake_vroom):
    environment.config.env.repossession_success_probability = 0.0

    jobs     = make_jobs(2, kind="repossession")
    vehicles = make_vehicles(1)

    load_execution_scenario(environment, jobs, vehicles, RoutingState(routes=[], unassigned_ids={j.id for j in jobs}))
    environment.current_state = solved_state(environment, list(environment.jobs), list(environment.vehicles))

    summary = environment.advance_execution()

    assert summary["failed"] == [jobs[0].id]
    assert not environment.jobs.contains(jobs[0].id)
    assert environment.vehicles.by_id(vehicles[0].id).onboard == 0


def test_depot_delivery_resets_onboard(environment, fake_vroom):
    vehicles = make_vehicles(1)
    vehicles[0].onboard = 2

    load_execution_scenario(environment, [], vehicles, RoutingState(routes=[], unassigned_ids=set()))
    environment.current_state = solved_state(environment, [], list(environment.vehicles))

    assert any(stop.kind == "delivery" for stop in environment.current_state.routes[0].stops)

    summary = environment.advance_execution()

    assert summary["dropped"] == 2
    assert environment.vehicles.by_id(vehicles[0].id).onboard == 0
    assert environment.current_state.num_routes == 0


def test_partial_delivery_drops_only_planned_amount(environment, fake_vroom):
    from core.shared import Route, Stop

    vehicles = make_vehicles(1)
    vehicles[0].onboard = 2

    load_execution_scenario(environment, [], vehicles, RoutingState(routes=[], unassigned_ids=set()))
    environment.vehicles.by_id(vehicles[0].id).onboard = 2

    partial = Route(
        vehicle_id = vehicles[0].id,
        stops      = [Stop(job_id=-1, location=(-46.63, -23.55), kind="delivery", arrival=29100, service=120, load=1)],
        start      = vehicles[0].start,
        cost       = 100,
    )
    environment.current_state = RoutingState(routes=[partial], unassigned_ids=set())

    summary = environment.advance_execution()

    assert summary["dropped"] == 1
    assert environment.vehicles.by_id(vehicles[0].id).onboard == 1


def test_tick_past_shift_end_orphans_remaining_jobs(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)

    load_execution_scenario(environment, jobs, vehicles, RoutingState(routes=[], unassigned_ids={j.id for j in jobs}))
    environment.clock         = vehicles[0].time_window[1] - 500
    environment.current_state = solved_state(environment, list(environment.jobs), list(environment.vehicles))

    environment.advance_execution()

    assert environment.current_state.num_routes == 0
    assert set(environment.jobs.ids) >= environment.current_state.unassigned_ids
    assert environment.current_state.num_unassigned > 0


def test_zero_tick_disables_execution(environment, fake_vroom):
    environment.config.env.tick_seconds = 0

    before  = environment.current_state
    clock   = environment.clock
    summary = environment.advance_execution()

    assert environment.current_state is before
    assert environment.clock == clock
    assert summary == {"served": [], "failed": [], "dropped": 0}
