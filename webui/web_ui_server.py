import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from project_paths   import ProjectPaths
from process_manager import ProcessManager
from request_router  import RequestRouter
from results_browser import RunBrowser
from scenario_lab    import ScenarioLab
from script_catalog  import ScriptCatalog
from web_logger      import WebLogger


class _Server(ThreadingHTTPServer):
    daemon_threads     = True
    request_queue_size = 64
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.router.route(self)

    def do_POST(self):
        self.server.router.route(self)

    def log_message(self, format, *args):
        pass


class WebUIServer:
    def __init__(self, host, port):
        self.host   = host
        self.port   = port
        self.logger = WebLogger()

        self.paths     = ProjectPaths()
        self.catalog   = ScriptCatalog()
        self.processes = ProcessManager(self.paths, self.catalog, self.logger)
        self.runs      = RunBrowser(self.paths)
        self.lab       = ScenarioLab(self.paths, self.logger)
        self.router    = RequestRouter(self.paths, self.catalog, self.processes, self.runs, self.lab, self.logger)

    def serve(self):
        server        = _Server((self.host, self.port), _Handler)
        server.router = self.router

        self.logger.ok(f"routing-ppo webui on http://{self.host}:{self.port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.muted("shutting down")
        finally:
            server.server_close()
