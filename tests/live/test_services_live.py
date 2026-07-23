import socket

import pytest

from core.shared import RoutingState, osrm, vroom
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


def test_osrm_distance_duration_is_positive_and_cached():
    first  = osrm.distance_duration(-46.63, -23.55, -46.64, -23.56)
    second = osrm.distance_duration(-46.63, -23.55, -46.64, -23.56)

    assert first[0] > 0.0
    assert first[1] > 0.0
    assert first == second


def test_osrm_enrich_polylines_fills_geometry():
    state = vroom.solve(make_jobs(4), make_vehicles(1))
    assert state is not None

    osrm.enrich_polylines(state.routes)

    for route in state.routes:
        assert route.geometry is not None
        assert route.path_coords is not None
        assert len(route.path_coords) >= 2
