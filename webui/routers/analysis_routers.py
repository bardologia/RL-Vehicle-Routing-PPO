"""API routes of the routing analysis services: runs, episodes, traces, scenarios, buildings and service health.

Each router owns one URL section and hands its requests to the service that
answers them. Multi-segment paths are matched as wildcards and split inside
the action, and every segment is percent-decoded before it reaches a service.
"""

from __future__ import annotations

from urllib.parse import unquote

from routers.dispatch import HttpExchange, RouteTable, SubRouter
from trace_store      import TraceRef
from web_errors       import ApiError


class PathArgs:
    """Splitting and parsing of wildcard path keys and query parameters."""

    @staticmethod
    def segments(key: str, count: int) -> list[str] | None:
        """Returns the percent-decoded segments of a wildcard key, or None when it does not have exactly `count` of them."""
        parts = key.split("/")
        if len(parts) != count or not all(parts):
            return None
        return [unquote(part) for part in parts]

    @staticmethod
    def query_int(exchange: HttpExchange, name: str, default: int | None) -> int | None:
        """Returns query parameter `name` as an integer, or `default` when it is absent.

        Raises:
            ApiError : The parameter is present but not an integer (400).
        """
        raw = exchange.text(name)
        if not raw:
            return default

        try:
            return int(raw)
        except ValueError:
            raise ApiError(400, f"query parameter '{name}' must be an integer")


class HealthRouter(SubRouter):
    """Serves the health of the OSRM and VROOM services.

    Attributes:
        lab : Scenario lab probing both services.
    """

    def __init__(self, lab) -> None:
        """Stores the scenario lab and declares the route."""
        self.lab = lab

        super().__init__(("/api/health",))

    def declare(self, table: RouteTable) -> None:
        """Registers the health route."""
        table.add("GET", "/api/health", self.health)

    def health(self, exchange: HttpExchange) -> None:
        """Answers with the reachability of OSRM and VROOM."""
        exchange.send_json(self.lab.health())


class RunRouter(SubRouter):
    """Serves the run list, the checkpoint catalog and every per-run payload.

    Attributes:
        runs        : Run browser reading run directories.
        checkpoints : Checkpoint catalog checking loadability.
        findings    : Rule-based findings over a run's reports.
    """

    def __init__(self, runs, checkpoints, findings) -> None:
        """Stores the run services and declares the routes."""
        self.runs        = runs
        self.checkpoints = checkpoints
        self.findings    = findings

        super().__init__(("/api/runs", "/api/checkpoints"))

    def declare(self, table: RouteTable) -> None:
        """Registers the run list, checkpoint list and per-run routes."""
        table.add("GET", "/api/checkpoints", self.checkpoint_list)
        table.add("GET", "/api/runs",        self.run_list)
        table.wildcard("GET", "/api/runs/", "/curves",   self.curves)
        table.wildcard("GET", "/api/runs/", "/health",   self.health)
        table.wildcard("GET", "/api/runs/", "/findings", self.run_findings)
        table.wildcard("GET", "/api/runs/", "",          self.run_item)

    def checkpoint_list(self, exchange: HttpExchange) -> None:
        """Answers with every checkpoint and its loadability."""
        exchange.send_json({"checkpoints": self.checkpoints.list()})

    def run_list(self, exchange: HttpExchange) -> None:
        """Answers with every run directory."""
        exchange.send_json({"runs": self.runs.list_runs()})

    def curves(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the logged training curves of one run."""
        exchange.send_json(self.runs.run_curves(self._name(key)))

    def health(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the training-health payload of one run."""
        exchange.send_json(self.runs.health(self._name(key)))

    def run_findings(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the findings of one run, for the report seed given in the query when there is one."""
        exchange.send_json({"findings": self.findings.findings(self._name(key), PathArgs.query_int(exchange, "seed", None))})

    def run_item(self, exchange: HttpExchange, key: str) -> None:
        """Answers with a run's detail, or one of its evaluation reports or diagnostics indexes.

        Args:
            exchange : Request being answered.
            key      : Wildcard key, either ``<run>`` or ``<run>/evaluation|diagnostics/<seed>``.

        Returns:
            None. Answers 404 for any other shape of key.
        """
        parts = key.split("/")

        if len(parts) == 1 and parts[0]:
            exchange.send_json(self.runs.run_detail(unquote(parts[0])))
            return

        if len(parts) == 3 and parts[0] and parts[2].isdigit() and parts[1] in ("evaluation", "diagnostics"):
            reader = self.runs.evaluation if parts[1] == "evaluation" else self.runs.diagnostics
            exchange.send_json(reader(unquote(parts[0]), int(parts[2])))
            return

        exchange.not_found()

    @staticmethod
    def _name(key: str) -> str:
        """Returns the run name of a single-segment key.

        Raises:
            ApiError : The key spans more than one segment (404).
        """
        parts = PathArgs.segments(key, 1)
        if parts is None:
            raise ApiError(404, "not found")
        return parts[0]


class EpisodeRouter(SubRouter):
    """Serves the interactive episode recordings.

    Attributes:
        episodes : Episode service recording and listing episodes.
    """

    def __init__(self, episodes) -> None:
        """Stores the episode service and declares the routes."""
        self.episodes = episodes

        super().__init__(("/api/episodes",))

    def declare(self, table: RouteTable) -> None:
        """Registers the episode list, creation and per-episode routes."""
        table.add("GET",  "/api/episodes", self.listing)
        table.add("POST", "/api/episodes", self.create)
        table.wildcard("GET",  "/api/episodes/", "/scenario", self.scenario)
        table.wildcard("POST", "/api/episodes/", "/agents",   self.agents)
        table.wildcard("POST", "/api/episodes/", "/cancel",   self.cancel)
        table.wildcard("GET",  "/api/episodes/", "",          self.status)

    def listing(self, exchange: HttpExchange) -> None:
        """Answers with the stored recordings."""
        exchange.send_json({"episodes": self.episodes.list()})

    def create(self, exchange: HttpExchange) -> None:
        """Starts a recording described by the request body."""
        exchange.send_json(self.episodes.create(exchange.body))

    def scenario(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the start scenario of one recording."""
        exchange.send_json(self.episodes.scenario(self._episode(key)))

    def agents(self, exchange: HttpExchange, key: str) -> None:
        """Adds agents to one recording."""
        exchange.send_json(self.episodes.add_agents(self._episode(key), exchange.body))

    def cancel(self, exchange: HttpExchange, key: str) -> None:
        """Cancels the lanes still running in one recording."""
        exchange.send_json(self.episodes.cancel(self._episode(key)))

    def status(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the status of one recording."""
        exchange.send_json(self.episodes.status(self._episode(key)))

    @staticmethod
    def _episode(key: str) -> str:
        """Returns the episode id of a single-segment key.

        Raises:
            ApiError : The key spans more than one segment (404).
        """
        parts = PathArgs.segments(key, 1)
        if parts is None:
            raise ApiError(404, "not found")
        return parts[0]


class TraceRouter(SubRouter):
    """Serves episode and diagnostics traces and the teacher verdicts on their decisions.

    Attributes:
        traces   : Trace store reading recorded traces.
        verdicts : Verdict service computing the teacher table of one decision.
    """

    def __init__(self, traces, verdicts) -> None:
        """Stores the trace services and declares the routes."""
        self.traces   = traces
        self.verdicts = verdicts

        super().__init__(("/api/traces",))

    def declare(self, table: RouteTable) -> None:
        """Registers the trace and verdict routes of both trace kinds."""
        table.wildcard("POST", "/api/traces/ep/",   "/verdict", self.episode_verdict)
        table.wildcard("POST", "/api/traces/diag/", "/verdict", self.diagnostics_verdict)
        table.wildcard("GET",  "/api/traces/ep/",   "",         self.episode_trace)
        table.wildcard("GET",  "/api/traces/diag/", "",         self.diagnostics_trace)

    def episode_trace(self, exchange: HttpExchange, key: str) -> None:
        """Answers with one lane of an episode recording, from the frame given in the query."""
        exchange.send_json(self.traces.read(self._episode_ref(key), PathArgs.query_int(exchange, "from", 0)))

    def diagnostics_trace(self, exchange: HttpExchange, key: str) -> None:
        """Answers with one diagnostics trace, from the frame given in the query."""
        exchange.send_json(self.traces.read(self._diagnostics_ref(key), PathArgs.query_int(exchange, "from", 0)))

    def episode_verdict(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the teacher verdict on one decision of an episode lane."""
        exchange.send_json(self.verdicts.verdict(self._episode_ref(key), exchange.body))

    def diagnostics_verdict(self, exchange: HttpExchange, key: str) -> None:
        """Answers with the teacher verdict on one decision of a diagnostics trace."""
        exchange.send_json(self.verdicts.verdict(self._diagnostics_ref(key), exchange.body))

    @staticmethod
    def _episode_ref(key: str) -> TraceRef:
        """Returns the trace reference of an ``<episode>/<agent>`` key.

        Raises:
            ApiError : The key does not have two segments (404).
        """
        parts = PathArgs.segments(key, 2)
        if parts is None:
            raise ApiError(404, "not found")
        return TraceRef("ep", parts[1], episode_id=parts[0])

    @staticmethod
    def _diagnostics_ref(key: str) -> TraceRef:
        """Returns the trace reference of a ``<run>/<seed>/<index>/<agent>`` key.

        Raises:
            ApiError : The key does not have four segments with integer seed and index (404).
        """
        parts = PathArgs.segments(key, 4)
        if parts is None or not parts[1].isdigit() or not parts[2].isdigit():
            raise ApiError(404, "not found")
        return TraceRef("diag", parts[3], run=parts[0], seed=int(parts[1]), index=int(parts[2]))


class ScenarioRouter(SubRouter):
    """Serves the Lab: templates, sampling, solving and the saved scenarios.

    Attributes:
        lab       : Scenario lab sampling and solving scenarios.
        scenarios : Store of saved scenarios.
    """

    def __init__(self, lab, scenarios) -> None:
        """Stores the Lab services and declares the routes."""
        self.lab       = lab
        self.scenarios = scenarios

        super().__init__(("/api/scenario", "/api/scenarios"))

    def declare(self, table: RouteTable) -> None:
        """Registers the template, sample, solve and saved-scenario routes."""
        table.add("GET",  "/api/scenario/templates", self.templates)
        table.add("POST", "/api/scenario/sample",    self.sample)
        table.add("POST", "/api/scenario/solve",     self.solve)
        table.add("GET",  "/api/scenarios",          self.listing)
        table.wildcard("GET",    "/api/scenarios/", "", self.get)
        table.wildcard("PUT",    "/api/scenarios/", "", self.put)
        table.wildcard("DELETE", "/api/scenarios/", "", self.delete)

    def templates(self, exchange: HttpExchange) -> None:
        """Answers with the scenario templates."""
        exchange.send_json({"templates": self.lab.list_templates()})

    def sample(self, exchange: HttpExchange) -> None:
        """Samples a scenario of the requested size and seed.

        Raises:
            ApiError : A size or the seed is not an integer (400).
        """
        body = exchange.body

        try:
            arguments = (int(body.get("num_jobs", 12)), int(body.get("num_vehicles", 3)), int(body.get("seed", 0)))
        except (TypeError, ValueError):
            raise ApiError(400, "num_jobs, num_vehicles and seed must be integers")

        exchange.send_json(self.lab.sample(*arguments))

    def solve(self, exchange: HttpExchange) -> None:
        """Solves the scenario in the request body."""
        body = exchange.body
        exchange.send_json(self.lab.solve(body.get("jobs") or [], body.get("vehicles") or [], body.get("assignment"), body.get("depot")))

    def listing(self, exchange: HttpExchange) -> None:
        """Answers with the saved scenarios."""
        exchange.send_json({"scenarios": self.scenarios.list()})

    def get(self, exchange: HttpExchange, key: str) -> None:
        """Answers with one saved scenario."""
        exchange.send_json(self.scenarios.get(self._name(key)))

    def put(self, exchange: HttpExchange, key: str) -> None:
        """Saves the scenario in the request body under a name."""
        exchange.send_json(self.scenarios.put(self._name(key), exchange.body))

    def delete(self, exchange: HttpExchange, key: str) -> None:
        """Deletes one saved scenario."""
        exchange.send_json(self.scenarios.delete(self._name(key)))

    @staticmethod
    def _name(key: str) -> str:
        """Returns the scenario name of a single-segment key.

        Raises:
            ApiError : The key spans more than one segment (404).
        """
        parts = PathArgs.segments(key, 1)
        if parts is None:
            raise ApiError(404, "not found")
        return parts[0]


class BuildingRouter(SubRouter):
    """Serves the building tiles of the map and the state of their cache.

    Attributes:
        buildings : Building tile cache.
    """

    def __init__(self, buildings) -> None:
        """Stores the tile cache and declares the routes."""
        self.buildings = buildings

        super().__init__(("/api/buildings",))

    def declare(self, table: RouteTable) -> None:
        """Registers the cache status and tile routes."""
        table.add("GET", "/api/buildings/status", self.status)
        table.wildcard("GET", "/api/buildings/", ".json", self.tile)

    def status(self, exchange: HttpExchange) -> None:
        """Answers with the state of the tile cache."""
        exchange.send_json(self.buildings.status())

    def tile(self, exchange: HttpExchange, key: str) -> None:
        """Answers with one ``<zoom>/<x>/<y>`` building tile as GeoJSON, or 404 for any other key."""
        parts = PathArgs.segments(key, 3)
        if parts is None or not all(part.isdigit() for part in parts):
            exchange.not_found()
            return

        payload = self.buildings.tile(int(parts[0]), int(parts[1]), int(parts[2]))
        exchange.send_payload("application/json", payload, "max-age=86400")
