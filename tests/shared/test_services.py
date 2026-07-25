import pytest

import core.shared.services as services_module

from core.shared import RoutingState
from core.shared.services import VroomClient
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
    assert set(payload["jobs"][0].keys()) == {"id", "location", "setup", "service", "delivery", "priority", "description"}
    assert set(payload["vehicles"][0].keys()) == {"id", "start", "capacity", "skills", "time_window", "speed_factor", "return_to_depot", "description"}


def test_vroom_solve_caches_identical_requests(cpu_config, monkeypatch):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(200, payload=load_fixture("vroom_solution_small"))

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    client = VroomClient(cpu_config.service)
    first  = client.solve(make_jobs(3), make_vehicles(2))
    second = client.solve(make_jobs(3), make_vehicles(2))

    assert calls["count"] == 1
    assert first is not second
    assert first.to_payload() == second.to_payload()

    first.add_unassigned({999})
    assert 999 not in second.unassigned_ids
    assert 999 not in client.solve(make_jobs(3), make_vehicles(2)).unassigned_ids


def test_vroom_solve_raises_when_the_service_stays_unavailable(cpu_config, monkeypatch):
    responses = [FakeResponse(500, text="boom"), FakeResponse(500, text="boom"), FakeResponse(500, text="boom"), FakeResponse(200, payload=load_fixture("vroom_solution_small"))]

    def fake_post(url, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    client = VroomClient(cpu_config.service, logger=CapturingLogger())
    client.BACKOFF_FACTOR = 0

    with pytest.raises(RuntimeError, match="VROOM unavailable"):
        client.solve(make_jobs(1), make_vehicles(1))

    assert isinstance(client.solve(make_jobs(1), make_vehicles(1)), RoutingState)


def test_vroom_solve_retries_routing_failures_then_reports_none(cpu_config, monkeypatch):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(500, payload={"code": 3, "error": "unroutable"}, text='{"code":3}')

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    logger = CapturingLogger()
    client = VroomClient(cpu_config.service, logger=logger)
    client.BACKOFF_FACTOR = 0

    assert client.solve(make_jobs(1), make_vehicles(1)) is None
    assert calls["count"] == client.MAX_ATTEMPTS
    assert any("could not route" in message for message in logger.warnings)


def test_solve_cache_key_ignores_job_order(cpu_config, monkeypatch):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(200, payload=load_fixture("vroom_solution_small"))

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    client = VroomClient(cpu_config.service)
    jobs   = make_jobs(3)

    client.solve(jobs, make_vehicles(2))
    client.solve(list(reversed(jobs)), make_vehicles(2))

    assert calls["count"] == 1


def test_vroom_solve_returns_none_on_client_error(cpu_config, monkeypatch):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(400, text="bad request")

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    logger = CapturingLogger()
    client = VroomClient(cpu_config.service, logger=logger)

    assert client.solve(make_jobs(1), make_vehicles(1)) is None
    assert calls["count"] == 1
    assert any("VROOM rejected" in message for message in logger.warnings)


def test_vroom_solve_retries_transient_server_errors(cpu_config, monkeypatch):
    responses = [FakeResponse(503, text="busy"), FakeResponse(200, payload=load_fixture("vroom_solution_small"))]

    def fake_post(url, json=None, timeout=None):
        return responses.pop(0)

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    client = VroomClient(cpu_config.service, logger=CapturingLogger())
    client.BACKOFF_FACTOR = 0

    assert isinstance(client.solve(make_jobs(1), make_vehicles(1)), RoutingState)
    assert not responses


def test_vroom_solve_does_not_retry_solver_error_codes(cpu_config, monkeypatch):
    calls = {"count": 0}

    def fake_post(url, json=None, timeout=None):
        calls["count"] += 1
        return FakeResponse(500, payload={"code": 1, "error": "cannot create std::vector larger than max_size()"}, text='{"code":1}')

    monkeypatch.setattr(services_module.requests, "post", fake_post)

    logger = CapturingLogger()
    client = VroomClient(cpu_config.service, logger=logger)

    assert client.solve(make_jobs(1), make_vehicles(1)) is None
    assert calls["count"] == 1
    assert any("code 1" in message for message in logger.warnings)

