import json
import math
import mimetypes
import queue
import traceback
from urllib.parse import parse_qs, urlparse


class RequestRouter:
    def __init__(self, paths, catalog, processes, runs, lab, logger):
        self.paths     = paths
        self.catalog   = catalog
        self.processes = processes
        self.runs      = runs
        self.lab       = lab
        self.logger    = logger

    def _route_get(self, handler, path, query):
        if path == "" or path == "/":
            self._serve_static(handler, "index.html")
            return

        if path.startswith("/static/"):
            self._serve_static(handler, path[len("/static/"):])
            return

        if path == "/runsmedia":
            self._serve_media(handler, (query.get("path") or [""])[0])
            return

        if path == "/api/project":
            self._send_json(handler, {
                "name"      : "RL Vehicle Routing PPO",
                "repo_root" : str(self.paths.repo_root),
                "health"    : self.lab.health(),
            })
            return

        if path == "/api/health":
            self._send_json(handler, self.lab.health())
            return

        if path == "/api/scripts":
            self._send_json(handler, {"scripts": self.catalog.list_scripts()})
            return

        if path.startswith("/api/scripts/") and path.endswith("/config"):
            key = path[len("/api/scripts/"):-len("/config")]
            if not self.catalog.has_script(key):
                self._send_json(handler, {"error": f"unknown script '{key}'"}, 404)
                return
            self._send_json(handler, self.catalog.form(key))
            return

        if path == "/api/jobs":
            self._send_json(handler, {"jobs": self.processes.list_jobs()})
            return

        if path.startswith("/api/jobs/") and path.endswith("/stream"):
            job_id = path[len("/api/jobs/"):-len("/stream")]
            self._stream_job(handler, job_id)
            return

        if path == "/api/runs":
            self._send_json(handler, {"runs": self.runs.list_runs()})
            return

        if path.startswith("/api/runs/") and path.endswith("/curves"):
            name = path[len("/api/runs/"):-len("/curves")]
            self._send_json(handler, self.runs.run_curves(name))
            return

        if path.startswith("/api/runs/"):
            name = path[len("/api/runs/"):]
            self._send_json(handler, self.runs.run_detail(name))
            return

        if path == "/api/datasets":
            self._send_json(handler, {"datasets": self.runs.list_datasets()})
            return

        if path == "/api/scenario/checkpoints":
            self._send_json(handler, {"checkpoints": self.lab.checkpoints()})
            return

        if path == "/api/scenario/templates":
            self._send_json(handler, {"templates": self.lab.list_templates()})
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _route_post(self, handler, path):
        body = self._read_json(handler)

        if path == "/api/run":
            key    = body.get("script", "")
            result = self.processes.launch(key, body.get("overrides"), bool(body.get("queue")))
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job_id = path[len("/api/jobs/"):-len("/stop")]
            result = self.processes.stop(job_id)
            self._send_json(handler, result, 200 if result.get("ok") else 400)
            return

        if path == "/api/scenario/sample":
            result = self.lab.sample(int(body.get("num_jobs", 12)), int(body.get("num_vehicles", 3)), int(body.get("seed", 0)))
            self._send_json(handler, result)
            return

        if path == "/api/scenario/solve":
            result = self.lab.solve(body.get("jobs") or [], body.get("vehicles") or [], body.get("assignment"), body.get("depot"))
            self._send_json(handler, result, 200 if "error" not in result else 400)
            return

        if path == "/api/scenario/run":
            result = self.lab.run(body)
            self._send_json(handler, result, 200 if "error" not in result else 400)
            return

        self._send_json(handler, {"error": "not found"}, 404)

    def _stream_job(self, handler, job_id):
        stream = self.processes.get_stream(job_id)
        if stream is None:
            self._send_json(handler, {"error": "unknown job"}, 404)
            return

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

                payload = json.dumps(event)
                handler.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                handler.wfile.flush()

                if event.get("type") == "end":
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            stream.unsubscribe(subscriber)

    def _read_json(self, handler):
        length = int(handler.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}

        raw = handler.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    @classmethod
    def _jsonsafe(cls, value):
        if isinstance(value, dict):
            return {key: cls._jsonsafe(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._jsonsafe(child) for child in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _send_json(self, handler, obj, status=200):
        payload = json.dumps(self._jsonsafe(obj)).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.write(payload)

    def _send_file(self, handler, target, cache):
        data         = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"

        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(data)))
        handler.send_header("Cache-Control", cache)
        handler.end_headers()
        handler.wfile.write(data)

    def _serve_media(self, handler, raw_path):
        target = self.runs.media_path(raw_path)
        if target is None:
            self._send_json(handler, {"error": "not found"}, 404)
            return

        self._send_file(handler, target, "max-age=60")

    def _serve_static(self, handler, relative):
        target = (self.paths.static_dir / relative).resolve()
        if not target.is_relative_to(self.paths.static_dir.resolve()):
            self._send_json(handler, {"error": "forbidden"}, 403)
            return
        if not target.is_file():
            self._send_json(handler, {"error": "not found"}, 404)
            return

        self._send_file(handler, target, "no-cache")

    def route(self, handler):
        parsed = urlparse(handler.path)
        path   = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        query  = parse_qs(parsed.query)

        try:
            if handler.command == "GET":
                self._route_get(handler, path, query)
            elif handler.command == "POST":
                self._route_post(handler, path)
            else:
                self._send_json(handler, {"error": "method not allowed"}, 405)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.logger.error(f"{handler.command} {path} failed: {exc}")
            traceback.print_exc()
            try:
                self._send_json(handler, {"error": str(exc)}, 500)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
