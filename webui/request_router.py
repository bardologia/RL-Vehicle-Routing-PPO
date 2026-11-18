"""HTTP routing of the console: one table of method, path pattern and handler."""

import json
import math
import mimetypes
import queue
import re
import traceback
from urllib.parse import parse_qs, unquote, urlparse

from trace_store import TraceRef
from web_errors import ApiError


class RequestRouter:
    """Dispatches requests to the console services and writes JSON responses.

    Errors raised as ApiError answer with their status; any other exception answers 500. Both carry `{"error": str}`.

    Attributes:
        routes : List of (method, compiled path pattern, bound handler); the first match wins.
    """

    SEGMENT = r"([^/]+)"

    def __init__(self, paths, catalog, processes, runs, lab, logger, checkpoints, episodes, traces, verdicts, findings, scenarios, buildings):
        self.paths       = paths
        self.catalog     = catalog
        self.processes   = processes
        self.runs        = runs
        self.lab         = lab
        self.logger      = logger
        self.checkpoints = checkpoints
        self.episodes    = episodes
        self.traces      = traces
        self.verdicts    = verdicts
        self.findings    = findings
        self.scenarios   = scenarios
        self.buildings   = buildings
        self.routes      = self._build_routes()

    def _build_routes(self):
        s     = self.SEGMENT
        table = [
            ("GET",    r"/",                                             self._index),
            ("GET",    r"/static/(.+)",                                  self._static),
            ("GET",    r"/runsmedia",                                    self._media),
            ("GET",    r"/api/health",                                   self._health),
            ("GET",    r"/api/scripts",                                  self._scripts),
            ("GET",    rf"/api/scripts/{s}/config",                      self._script_config),
            ("POST",   r"/api/run",                                      self._launch),
            ("GET",    r"/api/jobs",                                     self._jobs),
            ("GET",    rf"/api/jobs/{s}/stream",                         self._job_stream),
            ("POST",   rf"/api/jobs/{s}/stop",                           self._job_stop),
            ("GET",    r"/api/checkpoints",                              self._checkpoints),
            ("GET",    r"/api/runs",                                     self._runs),
            ("GET",    rf"/api/runs/{s}",                                self._run_detail),
            ("GET",    rf"/api/runs/{s}/curves",                         self._run_curves),
            ("GET",    rf"/api/runs/{s}/health",                         self._run_health),
            ("GET",    rf"/api/runs/{s}/findings",                       self._run_findings),
            ("GET",    rf"/api/runs/{s}/evaluation/(\d+)",               self._run_evaluation),
            ("GET",    rf"/api/runs/{s}/diagnostics/(\d+)",              self._run_diagnostics),
            ("GET",    r"/api/episodes",                                 self._episode_list),
            ("POST",   r"/api/episodes",                                 self._episode_create),
            ("GET",    rf"/api/episodes/{s}",                            self._episode_status),
            ("GET",    rf"/api/episodes/{s}/scenario",                   self._episode_scenario),
            ("POST",   rf"/api/episodes/{s}/agents",                     self._episode_agents),
            ("POST",   rf"/api/episodes/{s}/cancel",                     self._episode_cancel),
            ("GET",    rf"/api/traces/ep/{s}/{s}",                       self._trace_episode),
            ("POST",   rf"/api/traces/ep/{s}/{s}/verdict",               self._verdict_episode),
            ("GET",    rf"/api/traces/diag/{s}/(\d+)/(\d+)/{s}",         self._trace_diagnostics),
            ("POST",   rf"/api/traces/diag/{s}/(\d+)/(\d+)/{s}/verdict", self._verdict_diagnostics),
            ("GET",    r"/api/scenario/templates",                       self._templates),
            ("POST",   r"/api/scenario/sample",                          self._scenario_sample),
            ("POST",   r"/api/scenario/solve",                           self._scenario_solve),
            ("GET",    r"/api/scenarios",                                self._scenario_list),
            ("GET",    rf"/api/scenarios/{s}",                           self._scenario_get),
            ("PUT",    rf"/api/scenarios/{s}",                           self._scenario_put),
            ("DELETE", rf"/api/scenarios/{s}",                           self._scenario_delete),
            ("GET",    r"/api/buildings/status",                         self._buildings_status),
            ("GET",    r"/api/buildings/(\d+)/(\d+)/(\d+)\.json",        self._buildings_tile),
        ]

        return [(method, re.compile(f"^{pattern}$"), handler) for method, pattern, handler in table]

    def _index(self, handler, groups, query, body):
        self._serve_static(handler, "index.html")

    def _static(self, handler, groups, query, body):
        self._serve_static(handler, groups[0])

    def _media(self, handler, groups, query, body):
        target = self.runs.media_path((query.get("path") or [""])[0])
        if target is None:
            raise ApiError(404, "not found")

        self._send_file(handler, target, "max-age=60")

    def _health(self, handler, groups, query, body):
        self._send_json(handler, self.lab.health())

    def _scripts(self, handler, groups, query, body):
        self._send_json(handler, {"scripts": self.catalog.list_scripts()})

    def _script_config(self, handler, groups, query, body):
        if not self.catalog.has_script(groups[0]):
            raise ApiError(404, f"unknown script '{groups[0]}'")

        self._send_json(handler, self.catalog.form(groups[0]))

    def _launch(self, handler, groups, query, body):
        result = self.processes.launch(body.get("script", ""), body.get("overrides"), bool(body.get("queue")))
        self._send_json(handler, result, 200 if result.get("ok") else 400)

    def _jobs(self, handler, groups, query, body):
        self._send_json(handler, {"jobs": self.processes.list_jobs()})

    def _job_stream(self, handler, groups, query, body):
        stream = self.processes.get_stream(groups[0])
        if stream is None:
            raise ApiError(404, "unknown job")

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Cache-Control", "no-cache")
        handler.send_header("Connection", "keep-alive")
        handler.end_headers()

        subscriber = stream.subscribe()
        try:
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue

                handler.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event.get("type") == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.unsubscribe(subscriber)

    def _job_stop(self, handler, groups, query, body):
        result = self.processes.stop(groups[0])
        self._send_json(handler, result, 200 if result.get("ok") else 400)

    def _checkpoints(self, handler, groups, query, body):
        self._send_json(handler, {"checkpoints": self.checkpoints.list()})

    def _runs(self, handler, groups, query, body):
        self._send_json(handler, {"runs": self.runs.list_runs()})

    def _run_detail(self, handler, groups, query, body):
        self._send_json(handler, self.runs.run_detail(groups[0]))

    def _run_curves(self, handler, groups, query, body):
        self._send_json(handler, self.runs.run_curves(groups[0]))

    def _run_health(self, handler, groups, query, body):
        self._send_json(handler, self.runs.health(groups[0]))

    def _run_findings(self, handler, groups, query, body):
        seed = self._int_query(query, "seed", None)
        self._send_json(handler, {"findings": self.findings.findings(groups[0], seed)})

    def _run_evaluation(self, handler, groups, query, body):
        self._send_json(handler, self.runs.evaluation(groups[0], int(groups[1])))

    def _run_diagnostics(self, handler, groups, query, body):
        self._send_json(handler, self.runs.diagnostics(groups[0], int(groups[1])))

    def _episode_list(self, handler, groups, query, body):
        self._send_json(handler, {"episodes": self.episodes.list()})

    def _episode_create(self, handler, groups, query, body):
        self._send_json(handler, self.episodes.create(body))

    def _episode_status(self, handler, groups, query, body):
        self._send_json(handler, self.episodes.status(groups[0]))

    def _episode_scenario(self, handler, groups, query, body):
        self._send_json(handler, self.episodes.scenario(groups[0]))

    def _episode_agents(self, handler, groups, query, body):
        self._send_json(handler, self.episodes.add_agents(groups[0], body))

    def _episode_cancel(self, handler, groups, query, body):
        self._send_json(handler, self.episodes.cancel(groups[0]))

    def _episode_ref(self, groups):
        return TraceRef("ep", groups[1], episode_id=groups[0])

    def _diagnostics_ref(self, groups):
        return TraceRef("diag", groups[3], run=groups[0], seed=int(groups[1]), index=int(groups[2]))

    def _trace_episode(self, handler, groups, query, body):
        self._send_json(handler, self.traces.read(self._episode_ref(groups), self._int_query(query, "from", 0)))

    def _trace_diagnostics(self, handler, groups, query, body):
        self._send_json(handler, self.traces.read(self._diagnostics_ref(groups), self._int_query(query, "from", 0)))

    def _verdict_episode(self, handler, groups, query, body):
        self._send_json(handler, self.verdicts.verdict(self._episode_ref(groups), body))

    def _verdict_diagnostics(self, handler, groups, query, body):
        self._send_json(handler, self.verdicts.verdict(self._diagnostics_ref(groups), body))

    def _templates(self, handler, groups, query, body):
        self._send_json(handler, {"templates": self.lab.list_templates()})

    def _scenario_sample(self, handler, groups, query, body):
        try:
            arguments = (int(body.get("num_jobs", 12)), int(body.get("num_vehicles", 3)), int(body.get("seed", 0)))
        except (TypeError, ValueError):
            raise ApiError(400, "num_jobs, num_vehicles and seed must be integers")

        self._send_json(handler, self.lab.sample(*arguments))

    def _scenario_solve(self, handler, groups, query, body):
        self._send_json(handler, self.lab.solve(body.get("jobs") or [], body.get("vehicles") or [], body.get("assignment"), body.get("depot")))

    def _scenario_list(self, handler, groups, query, body):
        self._send_json(handler, {"scenarios": self.scenarios.list()})

    def _scenario_get(self, handler, groups, query, body):
        self._send_json(handler, self.scenarios.get(groups[0]))

    def _scenario_put(self, handler, groups, query, body):
        self._send_json(handler, self.scenarios.put(groups[0], body))

    def _scenario_delete(self, handler, groups, query, body):
        self._send_json(handler, self.scenarios.delete(groups[0]))

    def _buildings_status(self, handler, groups, query, body):
        self._send_json(handler, self.buildings.status())

    def _buildings_tile(self, handler, groups, query, body):
        payload = self.buildings.tile(int(groups[0]), int(groups[1]), int(groups[2]))
        self._send_bytes(handler, payload, "application/json", "max-age=86400")

    def _int_query(self, query, name, default):
        values = query.get(name)
        if not values:
            return default

        try:
            return int(values[0])
        except ValueError:
            raise ApiError(400, f"query parameter '{name}' must be an integer")

    def _read_json(self, handler):
        length = int(handler.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}

        try:
            body = json.loads(handler.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "request body is not valid JSON")

        if not isinstance(body, dict):
            raise ApiError(400, "request body must be a JSON object")

        return body

    @classmethod
    def _jsonsafe(cls, value):
        if isinstance(value, dict):
            return {key: cls._jsonsafe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonsafe(child) for child in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _send_bytes(self, handler, payload, content_type, cache=None, status=200):
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(payload)))
        if cache is not None:
            handler.send_header("Cache-Control", cache)
        handler.end_headers()
        handler.wfile.write(payload)

    def _send_json(self, handler, obj, status=200):
        self._send_bytes(handler, json.dumps(self._jsonsafe(obj)).encode("utf-8"), "application/json", None, status)

    def _send_file(self, handler, target, cache):
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(handler, target.read_bytes(), content_type, cache)

    def _serve_static(self, handler, relative):
        target = (self.paths.static_dir / relative).resolve()
        if not target.is_relative_to(self.paths.static_dir.resolve()):
            raise ApiError(403, "forbidden")
        if not target.is_file():
            raise ApiError(404, "not found")

        self._send_file(handler, target, "no-cache")

    def _dispatch(self, handler, path, query):
        allowed = False

        for method, pattern, target in self.routes:
            match = pattern.match(path)
            if match is None:
                continue

            if method != handler.command:
                allowed = True
                continue

            body = self._read_json(handler) if method in ("POST", "PUT") else {}
            target(handler, [unquote(group) for group in match.groups()], query, body)
            return

        raise ApiError(405 if allowed else 404, "method not allowed" if allowed else "not found")

    def route(self, handler):
        """Answer one request: find the route, run its handler and translate errors into JSON responses."""
        parsed = urlparse(handler.path)
        path   = parsed.path.rstrip("/") or "/"
        query  = parse_qs(parsed.query)

        try:
            self._dispatch(handler, path, query)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ApiError as error:
            self._send_json(handler, {"error": error.message}, error.status)
        except Exception as error:
            self.logger.error(f"{handler.command} {path} failed: {error}\n{traceback.format_exc()}")
            try:
                self._send_json(handler, {"error": str(error)}, 500)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
