"""Composition root of the routing console HTTP server.

Instantiates every console component, wires them into the request router, and
runs the threaded HTTP server together with the background monitors.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib     import Path

from building_tiles         import BuildingTiles
from checkpoint_catalog     import CheckpointCatalog
from config_registry        import ConfigRegistry
from config_template_store  import ConfigTemplateStore
from console_runtime        import ConsoleRuntime
from energy_meter           import JobEnergyMeter
from episode_service        import EpisodeService, VerdictService
from findings               import RunFindings
from geometry_cache         import GeometryCache
from job_describer          import JobDescriber
from launch_layout          import LaunchLayout
from politeness             import JobPoliteness, JobThrottle
from process_manager        import ProcessManager, ProcessNuke, ServerDetacher
from project_paths          import ProjectPaths
from request_router         import RequestRouter
from resource_watchdog      import ResourceWatchdog
from results_browser        import RunBrowser
from run_config_reader      import RunConfigReader
from run_launcher           import RunLauncher
from saved_run_store        import SavedRunStore
from scenario_lab           import ScenarioLab
from scenario_store         import ScenarioStore
from script_catalog         import ScriptCatalog
from script_config_resolver import ScriptConfigResolver
from sequence_decisions     import SequenceDecisions
from sequence_runner        import SequenceRunner
from sequence_store         import SequenceStore
from stream_hub             import StreamHub
from system_monitor         import SystemMonitor
from trace_store            import TraceStore
from web_logger             import WebLogger

from routers.analysis_routers import BuildingRouter, EpisodeRouter, HealthRouter, RunRouter, ScenarioRouter, TraceRouter
from routers.launch_routers   import CatalogRouter, ConfigTemplateRouter, JobRouter, RunConfigRouter, SavedRunRouter
from routers.sequence_router  import SequenceRouter
from routers.static_router    import StaticRouter
from routers.system_router    import SystemRouter


class _Server(ThreadingHTTPServer):
    """Threading HTTP server carrying the request router on the server object."""

    request_queue_size  = 64
    daemon_threads      = True
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    """HTTP/1.1 request handler that delegates every method to the router."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """Routes a GET request through the server's request router."""
        self.server.router.route(self)

    def do_POST(self) -> None:
        """Routes a POST request through the server's request router."""
        self.server.router.route(self)

    def do_PUT(self) -> None:
        """Routes a PUT request through the server's request router."""
        self.server.router.route(self)

    def do_DELETE(self) -> None:
        """Routes a DELETE request through the server's request router."""
        self.server.router.route(self)

    def log_message(self, fmt: str, *args) -> None:
        """Suppresses the default per-request access logging."""
        return


class WebUIServer:
    """Builds and runs the routing console: components, routes and background loops.

    Attributes:
        host   : Interface the HTTP server binds to.
        port   : TCP port the HTTP server binds to.
        logger : Console logger shared by every component.
        paths  : Project path resolver rooted at the repository.
        router : Request router holding the sub-routers.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, root: Path | None = None, overpass_endpoints: list[str] | None = None) -> None:
        """Constructs every console component and assembles the request router.

        Args:
            host               : Interface to bind the HTTP server to.
            port               : TCP port to bind the HTTP server to.
            root               : Root the console operates on; the repository holding this file when None.
            overpass_endpoints : Overpass interpreter URLs for building data, the built-in list when None.
        """
        self.host   = host
        self.port   = port
        self.logger = WebLogger()
        self.paths  = ProjectPaths(root)

        self.resolver    = ScriptConfigResolver(self.paths)
        self.catalog     = ScriptCatalog(self.paths, self.resolver)
        self.layout      = LaunchLayout()
        self.configs     = ConfigRegistry(self.paths)
        self.describer   = JobDescriber(self.paths, self.resolver)
        self.politeness  = JobPoliteness(self.paths, self.logger)
        self.processes   = ProcessManager(self.paths, self.logger, self.describer, self.politeness, self.catalog.titles())
        self.stream_hub  = StreamHub(self.logger)
        self.saved_runs  = SavedRunStore(self.paths, self.logger)
        self.templates   = ConfigTemplateStore(self.paths, self.logger)
        self.nuke        = ProcessNuke(self.processes, self.logger)
        self.throttle    = JobThrottle(self.nuke, self.logger)
        self.detacher    = ServerDetacher(self.paths, self.logger)
        self.system      = SystemMonitor(self.paths, self.logger)
        self.watchdog    = ResourceWatchdog(self.processes, self.politeness, self.throttle, self.logger, self.paths)
        self.energy      = JobEnergyMeter(self.system, self.processes, self.logger)
        self.launcher    = RunLauncher(self.paths, self.logger, self.processes)
        self.run_configs = RunConfigReader(self.logger)
        self.decisions   = SequenceDecisions(self.resolver, self.launcher, self.logger)
        self.sequences   = SequenceStore(self.paths, self.logger, self.decisions)
        self.sequencer   = SequenceRunner(self.sequences, self.processes, self.launcher, self.logger)

        self.runtime     = ConsoleRuntime(self.paths)
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

        self.router = RequestRouter(self.logger, [
            StaticRouter(self.paths, self.runs),
            HealthRouter(self.lab),
            RunRouter(self.runs, self.checkpoints, self.findings),
            EpisodeRouter(self.episodes),
            TraceRouter(self.traces, self.verdicts),
            ScenarioRouter(self.lab, self.scenarios),
            BuildingRouter(self.buildings),
            CatalogRouter(self.paths, self.catalog, self.resolver, self.layout, self.configs, self.launcher),
            JobRouter(self.paths, self.processes, self.launcher, self.stream_hub),
            SavedRunRouter(self.paths, self.saved_runs, self.launcher),
            ConfigTemplateRouter(self.templates),
            RunConfigRouter(self.run_configs),
            SequenceRouter(self.sequences, self.sequencer),
            SystemRouter(self.system, self.watchdog, self.politeness, self.detacher, self.nuke, self.logger),
        ])

    def _report_ready(self) -> None:
        """Prints the startup banner with the URL, root, script count and interpreter."""
        interpreters = self.paths.discover_interpreters()

        self.logger.banner("Routing console", [
            f"URL          http://{self.host}:{self.port}",
            f"repo root    {self.paths.repo_root}",
            f"runs root    {self.paths.runs_dir}",
            f"scripts      {len(self.catalog.list_scripts())} entry points",
            f"interpreter  {self.paths.preferred_interpreter(interpreters)}",
            f"envs found   {len(interpreters)}",
        ])

    def serve(self) -> None:
        """Serves the console until interrupted.

        Starts the host monitor, the resource watchdog, the job energy meter and
        the sequence runner, runs the HTTP server on a worker thread, and blocks
        in the detacher wait loop until a keyboard interrupt or a shutdown
        request arrives; running jobs are their own session leaders and survive.
        """
        server        = _Server((self.host, self.port), _Handler)
        server.router = self.router

        self._report_ready()
        self.system.start()
        self.watchdog.start()
        self.energy.start()
        self.sequencer.start()

        worker = threading.Thread(target=server.serve_forever, name="HttpServer", daemon=True)
        worker.start()

        try:
            self.detacher.wait_loop()
        except KeyboardInterrupt:
            self.logger.warning("shutting down")
        finally:
            server.shutdown()
            server.server_close()
