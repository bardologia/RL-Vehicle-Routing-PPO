"""Tests for the run launcher and the job routes that reach it.

Covers the interpreter check against the discovered environments, the choice
between an immediate launch and the queue, the refusal of malformed override
keys, and the /api/run route answering 404 for an unknown entry point and
passing the DLR launch body through.
"""

from __future__ import annotations

import json
import sys

import pytest

from routers.dispatch       import HttpExchange
from routers.launch_routers import JobRouter
from run_launcher           import RunLauncher
from web_logger             import WebLogger

from tests.webui.conftest import FakeHandler


class SpyProcesses:
    """Process manager stand-in recording launches and queue requests.

    Attributes:
        calls : Recorded (method, key, interpreter, overrides, follow_up, detach) tuples.
    """

    def __init__(self) -> None:
        """Starts with no recorded calls."""
        self.calls = []

    def launch(self, key, interpreter, overrides=None, follow_up=None, detach=False) -> dict:
        """Records an immediate launch, refusing a malformed override key like the real manager."""
        if any(" " in path for path in (overrides or {})):
            raise ValueError("invalid override key 'bad key'")
        self.calls.append(("launch", key, interpreter, overrides, follow_up, detach))
        return {"ok": True, "job_id": "job-1"}

    def enqueue(self, key, interpreter, overrides=None, follow_up=None, detach=False) -> dict:
        """Records a queued launch."""
        self.calls.append(("enqueue", key, interpreter, overrides, follow_up, detach))
        return {"ok": True, "job_id": "job-2", "queued": True}

    def adopt_orphans(self) -> int:
        """Adopts nothing."""
        return 0

    def list_jobs(self) -> list:
        """Lists no jobs."""
        return []


class InterpreterPaths:
    """Path registry stand-in discovering the running interpreter only and knowing the routing scripts."""

    def discover_interpreters(self) -> list:
        """Returns the running interpreter as the only environment."""
        return [{"label": "conda:routing-ppo", "path": sys.executable, "current": True}]

    def preferred_interpreter(self, interpreters: list, key: str = "") -> str:
        """Returns the first discovered interpreter."""
        return interpreters[0]["path"]

    def has_script(self, key: str) -> bool:
        """Returns whether the key is a routing entry point."""
        return key in ("pretrain", "train", "evaluate", "infer", "diagnose")


@pytest.fixture
def launcher():
    """Returns a launcher over the spy process manager."""
    return RunLauncher(InterpreterPaths(), WebLogger(), SpyProcesses())


def test_an_unknown_interpreter_is_refused(launcher):
    """A launch naming an interpreter the console did not discover is refused before anything starts."""
    result = launcher.execute("train", "/usr/bin/nope", {}, None, False, False)

    assert result["ok"] is False and "unknown interpreter" in result["error"]
    assert launcher.processes.calls == []


def test_queue_selects_enqueue_and_otherwise_launches(launcher):
    """The queue flag routes the run through enqueue, its absence through launch, with every argument passed on."""
    launcher.execute("train", sys.executable, {"training.num_updates": "4"}, "evaluate", False, False)
    launcher.execute("evaluate", sys.executable, {}, None, True, True)

    assert launcher.processes.calls == [
        ("launch", "train", sys.executable, {"training.num_updates": "4"}, "evaluate", False),
        ("enqueue", "evaluate", sys.executable, {}, None, True),
    ]


def test_a_malformed_override_key_is_a_failed_result(launcher):
    """A malformed override key comes back as a failed result instead of an exception."""
    result = launcher.execute("train", sys.executable, {"bad key": "1"}, None, False, False)

    assert result == {"ok": False, "error": "invalid override key 'bad key'"}


def _post(router: JobRouter, body: dict) -> FakeHandler:
    """Posts a launch body to /api/run through the job router."""
    handler  = FakeHandler("POST", "/api/run", json.dumps(body).encode(), {"Content-Type": "application/json"})
    exchange = HttpExchange.of(handler)
    exchange.read_body()
    assert router.handle(exchange)
    return handler


def test_the_run_route_passes_the_launch_body_through(launcher):
    """The /api/run body reaches the launcher with its interpreter defaulted to the preferred one."""
    router  = JobRouter(InterpreterPaths(), launcher.processes, launcher, None)
    handler = _post(router, {"script_key": "infer", "overrides": {"io.run_name": "remake_ppo"}, "follow_up": "", "detach": True, "queue": False})

    assert handler.status == 200 and handler.payload() == {"ok": True, "job_id": "job-1"}
    assert launcher.processes.calls == [("launch", "infer", sys.executable, {"io.run_name": "remake_ppo"}, None, True)]


def test_the_run_route_refuses_an_unknown_entry_point(launcher):
    """An unknown entry point is answered 404 and nothing launches."""
    router  = JobRouter(InterpreterPaths(), launcher.processes, launcher, None)
    handler = _post(router, {"script_key": "nope"})

    assert handler.status == 404
    assert launcher.processes.calls == []
