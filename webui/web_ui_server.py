"""The console HTTP server and the wiring of its services."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from building_tiles     import BuildingTiles
from checkpoint_catalog import CheckpointCatalog
from console_runtime    import ConsoleRuntime
from episode_service    import EpisodeService, VerdictService
from findings           import RunFindings
from geometry_cache     import GeometryCache
from project_paths      import ProjectPaths
from process_manager    import ProcessManager
from request_router     import RequestRouter
from results_browser    import RunBrowser
from scenario_lab       import ScenarioLab
from scenario_store     import ScenarioStore
from script_catalog     import ScriptCatalog
from trace_store        import TraceStore
from web_logger         import WebLogger


class _Server(ThreadingHTTPServer):
    daemon_threads      = True
    request_queue_size  = 64
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.server.router.route(self)

    def do_POST(self):
        self.server.router.route(self)

    def do_PUT(self):
        self.server.router.route(self)

    def do_DELETE(self):
        self.server.router.route(self)

    def log_message(self, format, *args):
        pass


class WebUIServer:
    """Builds every console service and serves them over HTTP.

    Attributes:
        host   : Interface to bind.
        port   : TCP port to bind.
        router : RequestRouter holding the route table.
    """

    def __init__(self, host, port, overpass_endpoints=None):
        self.host   = host
        self.port   = port
        self.logger = WebLogger()

        self.paths     = ProjectPaths()
        self.runtime   = ConsoleRuntime(self.paths)
        self.catalog   = ScriptCatalog()
        self.processes = ProcessManager(self.paths, self.catalog, self.logger)

        self.checkpoints = CheckpointCatalog(self.runtime)
        self.runs        = RunBrowser(self.paths, self.runtime.layout, self.checkpoints)
        self.findings    = RunFindings(self.runs, self.checkpoints)
        self.geometry    = GeometryCache(self.runtime.config.service.osrm_url)
        self.lab         = ScenarioLab(self.runtime, self.geometry, self.logger)
        self.episodes    = EpisodeService(self.paths, self.runtime, self.checkpoints, self.runs, self.lab.templates, self.logger)
        self.traces      = TraceStore(self.runtime.layout, self.geometry, self.episodes)
        self.verdicts    = VerdictService(self.runtime, self.traces)
        self.scenarios   = ScenarioStore(self.paths.scenarios_dir)
        self.buildings   = BuildingTiles(self.paths.buildings_dir, overpass_endpoints)

        self.router = RequestRouter(
            paths       = self.paths,
            catalog     = self.catalog,
            processes   = self.processes,
            runs        = self.runs,
            lab         = self.lab,
            logger      = self.logger,
            checkpoints = self.checkpoints,
            episodes    = self.episodes,
            traces      = self.traces,
            verdicts    = self.verdicts,
            findings    = self.findings,
            scenarios   = self.scenarios,
            buildings   = self.buildings,
        )

    def serve(self):
        """Serve requests until interrupted."""
        server        = _Server((self.host, self.port), _Handler)
        server.router = self.router

        self.logger.ok(f"routing-ppo webui on http://{self.host}:{self.port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            self.logger.muted("shutting down")
        finally:
            server.server_close()
