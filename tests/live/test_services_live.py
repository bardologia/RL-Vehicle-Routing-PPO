import socket

import pytest

from core.shared import RoutingState, vroom
from core.shared.state import DELIVERY_ID_OFFSET, DROP_JOB_ID_OFFSET
from tests.conftest import make_jobs, make_vehicles


def _service_up(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("localhost", port)) == 0


pytestmark = pytest.mark.skipif(
    not (_service_up(3000) and _service_up(5000)),
    reason="VROOM/OSRM containers not running",
)


def test_vroom_solve_returns_routing_state():
    state = vroom.solve(make_jobs(5), make_vehicles(2))

    assert isinstance(state, RoutingState)
    assert state.num_routes >= 1
    assert state.cost > 0
    assert state.assigned_job_ids | state.unassigned_ids == {0, 1, 2, 3, 4}


def test_vroom_returns_geometry_only_when_asked():
    plain = vroom.solve(make_jobs(4), make_vehicles(1))
    drawn = vroom.solve(make_jobs(4), make_vehicles(1), options={"g": True})

    assert all(route.geometry is None for route in plain.routes)
    assert all(route.geometry is not None for route in drawn.routes)


def test_repossession_round_trips_as_a_shipment_with_a_depot_delivery():
    depot    = (-46.63, -23.55)
    jobs     = make_jobs(2, kind="repossession")
    vehicles = make_vehicles(1)

    state = vroom.solve(jobs, vehicles, depot=depot, clock=28800)

    assert state is not None

    kinds = [stop.kind for stop in state.routes[0].stops]
    assert "pickup" in kinds
    assert "delivery" in kinds
    assert all(stop.job_id < DELIVERY_ID_OFFSET for stop in state.routes[0].stops)


def test_onboard_load_becomes_a_skill_bound_drop_job():
    depot    = (-46.63, -23.55)
    vehicles = make_vehicles(1)
    vehicles[0].onboard = 2

    state = vroom.solve(make_jobs(2), vehicles, depot=depot, clock=28800)

    assert state is not None

    drop_ids = [stop.job_id for stop in state.routes[0].stops if stop.job_id >= DROP_JOB_ID_OFFSET]
    assert drop_ids == [DROP_JOB_ID_OFFSET + vehicles[0].id]
