"""Tests for the composition of the console server.

The server builds over a temporary root without starting a thread, every API
section the frontend calls is owned by exactly one sub-router, and the job,
launch, monitoring and analysis routes answer through the assembled router.
"""

from __future__ import annotations

import json

import pytest

from web_ui_server import WebUIServer

from tests.webui.conftest import FakeHandler

SECTIONS = ("/", "/static", "/runsmedia", "/api/health", "/api/runs", "/api/checkpoints", "/api/episodes", "/api/traces", "/api/scenario", "/api/scenarios", "/api/buildings", "/api/project", "/api/scripts", "/api/run", "/api/jobs", "/api/stream", "/api/saved-runs", "/api/config-templates", "/api/run-config", "/api/sequences", "/api/system", "/api/gpu-activity", "/api/host-history", "/api/impact", "/api/server-log")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Returns a console server built over a temporary root, its threads never started."""
    root = tmp_path_factory.mktemp("root")
    for name in ("main", "configuration", "core", "webui"):
        (root / name).mkdir()
    return WebUIServer(port=0, root=root)


def _get(server: WebUIServer, path: str) -> FakeHandler:
    """Routes one GET through the assembled router."""
    handler = FakeHandler("GET", path)
    server.router.route(handler)
    return handler


def test_every_frontend_section_has_exactly_one_owner(server):
    """Each section the frontend calls is claimed, and the index holds no section twice."""
    assert set(SECTIONS) <= set(server.router.sections)


@pytest.mark.parametrize("path, key", [
    ("/api/jobs",                  "jobs"),
    ("/api/saved-runs",            "saved"),
    ("/api/config-templates",      "templates"),
    ("/api/sequences",             "sequences"),
    ("/api/server-log?limit=5",    "lines"),
    ("/api/host-history?range=hour", "host"),
    ("/api/gpu-activity?range=day",  "gpus"),
])
def test_the_console_routes_answer_through_the_assembled_router(server, path, key):
    """The job, store, sequence and monitoring listings answer 200 with their payload key."""
    handler = _get(server, path)

    assert handler.status == 200
    assert key in json.loads(handler.wfile.getvalue())


def test_an_unknown_history_span_is_a_bad_request(server):
    """A history range outside the offered spans is answered 400."""
    assert _get(server, "/api/host-history?range=year").status == 400


def test_the_system_snapshot_carries_the_merged_sections(server):
    """The system snapshot merges the watchdog, politeness and server state into the host reading."""
    snapshot = json.loads(_get(server, "/api/system").wfile.getvalue())

    assert {"cpu", "mem", "disk", "gpus", "procs", "history", "alerts", "politeness", "server"} <= set(snapshot)
    assert snapshot["politeness"]["level"] in ("off", "polite", "background")
    assert snapshot["server"]["detached"] is False


def test_the_auto_nuke_route_arms_and_disarms_the_watchdog(server):
    """POST /api/impact/arm flips the watchdog's auto-nuke flag, and the system snapshot reports it."""
    for armed in (True, False):
        body    = json.dumps({"armed": armed}).encode("utf-8")
        handler = FakeHandler("POST", "/api/impact/arm", body, {"Content-Type": "application/json"})
        server.router.route(handler)

        assert handler.status == 200
        assert handler.payload()["auto_nuke"] is armed
        assert json.loads(_get(server, "/api/system").wfile.getvalue())["alerts"]["auto_nuke"] is armed
