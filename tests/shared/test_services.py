import polyline as polyline_module

import core.shared.services as services_module

from core.shared import RoutingState
from core.shared.services import OsrmClient, VroomClient
from core.shared.state import Route
from tests.conftest import CapturingLogger, load_fixture, make_jobs, make_vehicles


class FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload    = payload
        self.text        = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"status {self.status_code}")


def test_vroom_solve_posts_full_payload_shape(cpu_config, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"]     = url
        captured["payload"] = json
        return FakeResponse(200, payload=load_fixture("vroom_solution_small"))

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    client = VroomClient(cpu_config.service)
    state  = client.solve(make_jobs(3), make_vehicles(2))

    assert isinstance(state, RoutingState)
    assert captured["url"] == cpu_config.service.vroom_url

    payload = captured["payload"]
    assert set(payload.keys()) == {"jobs", "vehicles", "options"}
    assert len(payload["jobs"]) == 3
    assert len(payload["vehicles"]) == 2
    assert payload["options"] == cpu_config.service.options
    assert set(payload["jobs"][0].keys()) == {"id", "location", "setup", "service", "amount", "priority", "description"}
    assert set(payload["vehicles"][0].keys()) == {"id", "start", "capacity", "time_window", "speed_factor", "return_to_depot", "description"}


def test_vroom_solve_returns_none_on_non_200(cpu_config, monkeypatch):
    def fake_post(url, json=None, timeout=None):
        return FakeResponse(500, text="boom")

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    logger = CapturingLogger()
    client = VroomClient(cpu_config.service, logger=logger)

    assert client.solve(make_jobs(1), make_vehicles(1)) is None
    assert any("VROOM error" in message for message in logger.warnings)


def test_osrm_distance_duration_caches_by_rounded_key(cpu_config, monkeypatch):
    calls  = {"count": 0}
    client = OsrmClient(cpu_config.service)

    def fake_fetch(lon_a, lat_a, lon_b, lat_b):
        calls["count"] += 1
        return 1234.0, 56.0

    monkeypatch.setattr(client, "_fetch_distance_duration", fake_fetch)

    first  = client.distance_duration(-46.63, -23.55, -46.64, -23.56)
    second = client.distance_duration(-46.63, -23.55, -46.64, -23.56)

    assert first == (1234.0, 56.0)
    assert second == first
    assert calls["count"] == 1


def test_osrm_distance_duration_rounds_beyond_fifth_decimal(cpu_config, monkeypatch):
    calls  = {"count": 0}
    client = OsrmClient(cpu_config.service)

    def fake_fetch(lon_a, lat_a, lon_b, lat_b):
        calls["count"] += 1
        return 1.0, 2.0

    monkeypatch.setattr(client, "_fetch_distance_duration", fake_fetch)

    client.distance_duration(-46.630001, -23.550001, -46.64, -23.56)
    client.distance_duration(-46.630002, -23.550002, -46.64, -23.56)

    assert calls["count"] == 1


def test_osrm_distinct_coordinates_fetch_separately(cpu_config, monkeypatch):
    calls  = {"count": 0}
    client = OsrmClient(cpu_config.service)

    monkeypatch.setattr(client, "_fetch_distance_duration", lambda *args: (calls.__setitem__("count", calls["count"] + 1), (0.0, 0.0))[1])

    client.distance_duration(-46.63, -23.55, -46.64, -23.56)
    client.distance_duration(-46.60, -23.50, -46.64, -23.56)

    assert calls["count"] == 2


def test_osrm_cache_clears_when_bound_exceeded(cpu_config, monkeypatch):
    client = OsrmClient(cpu_config.service)
    client._distance_cache = {(float(i), 0.0, 0.0, 0.0): (0.0, 0.0) for i in range(100_001)}

    monkeypatch.setattr(client, "_fetch_distance_duration", lambda *args: (9.0, 9.0))

    client.distance_duration(-46.63, -23.55, -46.64, -23.56)

    assert len(client._distance_cache) == 1


def test_osrm_enrich_polylines_skips_routes_with_geometry_and_path(cpu_config, monkeypatch):
    calls  = {"count": 0}
    client = OsrmClient(cpu_config.service)

    monkeypatch.setattr(client, "_fetch_geometry", lambda points: (calls.__setitem__("count", calls["count"] + 1), "xxx")[1])

    route             = Route(vehicle_id=0, start=(-46.6, -23.5), end=(-46.5, -23.4))
    route.geometry    = "already"
    route.path_coords = [(-46.6, -23.5), (-46.5, -23.4)]

    client.enrich_polylines([route])

    assert calls["count"] == 0
    assert route.geometry == "already"


def test_osrm_enrich_polylines_fills_missing_geometry(cpu_config, monkeypatch):
    encoded = polyline_module.encode([(-23.55, -46.63), (-23.56, -46.64)])
    client  = OsrmClient(cpu_config.service)

    monkeypatch.setattr(client, "_fetch_geometry", lambda points: encoded)

    route = Route(vehicle_id=0, start=(-46.63, -23.55), end=(-46.64, -23.56))

    client.enrich_polylines([route])

    assert route.geometry == encoded
    assert route.path_coords == polyline_module.decode(encoded)


def test_osrm_enrich_polylines_skips_routes_without_points(cpu_config, monkeypatch):
    calls  = {"count": 0}
    client = OsrmClient(cpu_config.service)

    monkeypatch.setattr(client, "_fetch_geometry", lambda points: (calls.__setitem__("count", calls["count"] + 1), "xxx")[1])

    client.enrich_polylines([Route(vehicle_id=0)])

    assert calls["count"] == 0
