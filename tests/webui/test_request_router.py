"""Tests of the request router over the analysis sub-routers.

Every analysis route reaches its service with its path, query and body
arguments parsed, the static router keeps its path guards, a known path asked
with the wrong verb answers 405, an unknown section or path answers 404, an
unreadable or non-object body answers 400, a service's ApiError answers its own
status and any other exception answers 500, and two routers claiming one section
refuse to build.
"""

import io
import json

import pytest

from request_router            import RequestRouter
from routers.analysis_routers  import BuildingRouter, EpisodeRouter, HealthRouter, RunRouter, ScenarioRouter, TraceRouter
from routers.dispatch          import RouteConflict
from routers.static_router     import StaticRouter
from web_errors                import ApiError
from tests.conftest            import CapturingLogger


class Recorder:
    """Stands in for one service, recording every call and answering with its name."""

    def __init__(self, name, log, fail=None):
        self.name = name
        self.log  = log
        self.fail = fail or {}

    def __getattr__(self, method):
        def call(*args, **kwargs):
            self.log.append((self.name, method, args))
            if method in self.fail:
                raise self.fail[method]
            return {"service": self.name, "method": method}

        return call


class FakeRuns(Recorder):
    def media_path(self, raw):
        return None


class FakeLab(Recorder):
    def health(self):
        return {"osrm": True, "vroom": float("nan")}

    def list_templates(self):
        return []


class FakeBuildings(Recorder):
    def tile(self, zoom, x, y):
        self.log.append(("buildings", "tile", (zoom, x, y)))
        return b'{"type":"FeatureCollection","features":[]}'


class FakeHandler:
    """Minimal request handler capturing the status, headers and body of the reply."""

    def __init__(self, command, path, body=None):
        raw           = b"" if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        self.command  = command
        self.path     = path
        self.headers  = {"Content-Length": str(len(raw))}
        self.rfile    = io.BytesIO(raw)
        self.wfile    = io.BytesIO()
        self.status   = None
        self.sent     = {}

    def send_response(self, status):
        self.status = status

    def send_header(self, name, value):
        self.sent[name] = value

    def end_headers(self):
        pass

    def json(self):
        return json.loads(self.wfile.getvalue())


@pytest.fixture
def setup(paths):
    log = []
    (paths.static_dir / "index.html").write_text("<html>console</html>")
    (paths.static_dir / "app.js").write_text("export {}")

    services = {
        "runs"        : FakeRuns("runs", log),
        "lab"         : FakeLab("lab", log),
        "checkpoints" : Recorder("checkpoints", log),
        "episodes"    : Recorder("episodes", log),
        "traces"      : Recorder("traces", log),
        "verdicts"    : Recorder("verdicts", log),
        "findings"    : Recorder("findings", log),
        "scenarios"   : Recorder("scenarios", log),
        "buildings"   : FakeBuildings("buildings", log),
    }

    router = RequestRouter(CapturingLogger(), [
        StaticRouter(paths, services["runs"]),
        HealthRouter(services["lab"]),
        RunRouter(services["runs"], services["checkpoints"], services["findings"]),
        EpisodeRouter(services["episodes"]),
        TraceRouter(services["traces"], services["verdicts"]),
        ScenarioRouter(services["lab"], services["scenarios"]),
        BuildingRouter(services["buildings"]),
    ])

    return router, services, log


def request(router, command, path, body=None):
    handler = FakeHandler(command, path, body)
    router.route(handler)
    return handler


ROUTES = [
    ("GET",    "/api/checkpoints",                          None, ("checkpoints", "list")),
    ("GET",    "/api/runs",                                 None, ("runs", "list_runs")),
    ("GET",    "/api/runs/run_a",                           None, ("runs", "run_detail")),
    ("GET",    "/api/runs/run_a/curves",                    None, ("runs", "run_curves")),
    ("GET",    "/api/runs/run_a/health",                    None, ("runs", "health")),
    ("GET",    "/api/runs/run_a/findings?seed=5",           None, ("findings", "findings")),
    ("GET",    "/api/runs/run_a/evaluation/5",              None, ("runs", "evaluation")),
    ("GET",    "/api/runs/run_a/diagnostics/5",             None, ("runs", "diagnostics")),
    ("GET",    "/api/episodes",                             None, ("episodes", "list")),
    ("POST",   "/api/episodes",                             {},   ("episodes", "create")),
    ("GET",    "/api/episodes/e1",                          None, ("episodes", "status")),
    ("GET",    "/api/episodes/e1/scenario",                 None, ("episodes", "scenario")),
    ("POST",   "/api/episodes/e1/agents",                   {},   ("episodes", "add_agents")),
    ("POST",   "/api/episodes/e1/cancel",                   {},   ("episodes", "cancel")),
    ("GET",    "/api/traces/ep/e1/model~run_a~greedy",      None, ("traces", "read")),
    ("POST",   "/api/traces/ep/e1/teacher/verdict",         {},   ("verdicts", "verdict")),
    ("GET",    "/api/traces/diag/run_a/5/2/model?from=3",   None, ("traces", "read")),
    ("POST",   "/api/traces/diag/run_a/5/2/model/verdict",  {},   ("verdicts", "verdict")),
    ("POST",   "/api/scenario/sample",                      {},   ("lab", "sample")),
    ("POST",   "/api/scenario/solve",                       {},   ("lab", "solve")),
    ("GET",    "/api/scenarios",                            None, ("scenarios", "list")),
    ("GET",    "/api/scenarios/mine",                       None, ("scenarios", "get")),
    ("PUT",    "/api/scenarios/mine",                       {},   ("scenarios", "put")),
    ("DELETE", "/api/scenarios/mine",                       None, ("scenarios", "delete")),
    ("GET",    "/api/buildings/status",                     None, ("buildings", "status")),
    ("GET",    "/api/buildings/16/23280/37160.json",        None, ("buildings", "tile")),
]


@pytest.mark.parametrize("command, path, body, expected", ROUTES)
def test_every_api_route_reaches_its_service(setup, command, path, body, expected):
    """Checks each analysis route answers 200 JSON from the service it names."""
    router, services, log = setup
    handler               = request(router, command, path, body)

    assert handler.status == 200
    assert log[-1][:2] == expected
    assert handler.sent["Content-Type"] == "application/json"


def test_arguments_are_parsed_from_path_query_and_body(setup):
    """Checks path segments are decoded, query integers parsed and bodies passed through."""
    router, services, log = setup

    request(router, "GET", "/api/runs/run_a/findings?seed=5")
    assert log[-1][2] == ("run_a", 5)

    request(router, "GET", "/api/runs/run_a/findings")
    assert log[-1][2] == ("run_a", None)

    request(router, "GET", "/api/runs/run_a/evaluation/1000000/")
    assert log[-1][2] == ("run_a", 1000000)

    request(router, "GET", "/api/traces/diag/run_a/5/2/model_sampled?from=3")
    ref, from_seq = log[-1][2]
    assert (ref.kind, ref.run, ref.seed, ref.index, ref.agent_key, from_seq) == ("diag", "run_a", 5, 2, "model_sampled", 3)

    request(router, "POST", "/api/traces/ep/e1/model%7Erun_a%7Egreedy/verdict", {"seq": 4, "scope": "full"})
    ref, body = log[-1][2]
    assert (ref.kind, ref.episode_id, ref.agent_key, body) == ("ep", "e1", "model~run_a~greedy", {"seq": 4, "scope": "full"})

    request(router, "POST", "/api/scenario/sample", {"num_jobs": 5, "seed": 9})
    assert log[-1][2] == (5, 3, 9)

    request(router, "PUT", "/api/scenarios/mine", {"jobs": []})
    assert log[-1][2] == ("mine", {"jobs": []})

    request(router, "GET", "/api/buildings/16/23280/37160.json")
    assert log[-1][2] == (16, 23280, 37160)


def test_static_and_health_keep_their_shapes(setup):
    """Checks the index, a static asset and the health payload with its non-finite value nulled."""
    router, services, log = setup

    index = request(router, "GET", "/")
    assert index.status == 200 and b"console" in index.wfile.getvalue()
    assert request(router, "GET", "/static/app.js").sent["Cache-Control"] == "no-cache"

    assert request(router, "GET", "/api/health").json() == {"osrm": True, "vroom": None}
    assert request(router, "GET", "/api/scenario/templates").json() == {"templates": []}


def test_error_statuses(setup):
    """Checks the 400, 403, 404 and 405 answers of malformed or unknown requests."""
    router, services, log = setup

    cases = [
        ("GET",    "/api/project",                    None,          404),
        ("POST",   "/api/scenario/run",               {},            404),
        ("GET",    "/api/scenario/checkpoints",       None,          404),
        ("GET",    "/api/runs/run_a/evaluation/x",    None,          404),
        ("GET",    "/api/traces/diag/run_a/x/2/model", None,         404),
        ("GET",    "/api/buildings/16/a/3.json",      None,          404),
        ("GET",    "/runsmedia?path=/etc/passwd",     None,          404),
        ("GET",    "/static/missing.js",              None,          404),
        ("GET",    "/static/../request_router.py",    None,          403),
        ("DELETE", "/api/episodes",                   None,          405),
        ("PUT",    "/api/runs",                       {},            405),
        ("POST",   "/api/episodes",                   b"{broken",    400),
        ("POST",   "/api/episodes",                   b"[1, 2]",     400),
        ("GET",    "/api/runs/run_a/findings?seed=x", None,          400),
        ("POST",   "/api/scenario/sample",            {"seed": "x"}, 400),
    ]

    for command, path, body, status in cases:
        handler = request(router, command, path, body)
        assert handler.status == status, (command, path)
        assert "error" in handler.json()


def test_service_errors_map_to_their_status_and_unexpected_ones_to_500(setup):
    """Checks an ApiError keeps its status and message while any other exception becomes a 500."""
    router, services, log = setup

    services["episodes"].fail = {"status": ApiError(409, "busy"), "create": RuntimeError("boom")}
    services["traces"].fail   = {"read": ApiError(503, "OSRM unavailable")}

    busy = request(router, "GET", "/api/episodes/e1")
    assert (busy.status, busy.json()) == (409, {"error": "busy"})

    boom = request(router, "POST", "/api/episodes", {})
    assert (boom.status, boom.json()) == (500, {"error": "boom"})

    down = request(router, "GET", "/api/traces/ep/e1/teacher")
    assert (down.status, down.json()) == (503, {"error": "OSRM unavailable"})

    bad_key = request(router, "GET", "/api/traces/ep/e1/bad%20key")
    assert bad_key.status == 400


def test_cross_origin_requests_are_refused(setup):
    """Checks a request whose Origin differs from its Host answers 403 and reaches no service."""
    router, services, log = setup

    handler = FakeHandler("POST", "/api/episodes", {})
    handler.headers.update({"Origin": "http://evil.example", "Host": "127.0.0.1:8766"})
    router.route(handler)

    assert handler.status == 403
    assert log == []


def test_two_routers_claiming_one_section_refuse_to_build(setup):
    """Checks the section index rejects a second owner of the same section."""
    with pytest.raises(RouteConflict):
        RequestRouter(CapturingLogger(), [HealthRouter(FakeLab("lab", [])), HealthRouter(FakeLab("lab", []))])
