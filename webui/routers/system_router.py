"""API routes for host monitoring and the console's own lifecycle.

Serves the machine snapshot enriched with the watchdog, politeness and server
state, the GPU activity and host histories, the server's own log, and the
nuke, auto-nuke, keep-alive and shutdown controls.
"""

from __future__ import annotations

from politeness        import JobPoliteness
from process_manager   import ProcessNuke, ServerDetacher
from resource_watchdog import ResourceWatchdog
from routers.dispatch  import HttpExchange, RouteTable, SubRouter
from system_monitor    import SystemMonitor
from web_errors        import ApiError
from web_logger        import WebLogger


class SystemRouter(SubRouter):
    """Serves the host snapshot, the histories and the lifecycle controls.

    Attributes:
        system     : Host monitor producing the machine snapshot and the histories.
        watchdog   : Resource watchdog contributing the alert state.
        politeness : Launch politeness level and auto-throttle switch.
        detacher   : Server detach and shutdown controller.
        nuke       : Process nuke that clears the queue and stops every job process.
        logger     : Console logger whose in-memory ring backs the server-log view.
    """

    HISTORY_SPANS  = ("hour", "day", "week", "month")
    ACTIVITY_SPANS = ("day", "week", "month")

    def __init__(self, system: SystemMonitor, watchdog: ResourceWatchdog, politeness: JobPoliteness, detacher: ServerDetacher, nuke: ProcessNuke, logger: WebLogger) -> None:
        """Stores the monitoring and lifecycle services, then declares the routes."""
        self.system     = system
        self.watchdog   = watchdog
        self.politeness = politeness
        self.detacher   = detacher
        self.nuke       = nuke
        self.logger     = logger

        super().__init__(("/api/system", "/api/gpu-activity", "/api/host-history", "/api/impact", "/api/server-log"))

    def declare(self, table: RouteTable) -> None:
        """Registers the snapshot, history, server-log, lifecycle, politeness and auto-nuke routes."""
        table.add("GET",  "/api/system",            self.snapshot)
        table.add("GET",  "/api/gpu-activity",      self.gpu_activity)
        table.add("GET",  "/api/host-history",      self.host_history)
        table.add("GET",  "/api/server-log",        self.server_log)
        table.add("POST", "/api/system/nuke",       self.nuke_jobs)
        table.add("POST", "/api/system/detach",     self.detach)
        table.add("POST", "/api/system/shutdown",   self.shutdown)
        table.add("POST", "/api/impact/politeness", self.configure_politeness)
        table.add("POST", "/api/impact/arm",        self.arm_nuke)

    def snapshot(self, exchange: HttpExchange) -> None:
        """Answers with the host snapshot enriched with the alert, politeness and server state."""
        payload               = self.system.snapshot()
        payload["alerts"]     = self.watchdog.state()
        payload["politeness"] = self.politeness.state()
        payload["server"]     = self.detacher.state()

        exchange.send_json(payload)

    def gpu_activity(self, exchange: HttpExchange) -> None:
        """Answers with the bucketed per-GPU occupancy over the requested day, week or month span."""
        exchange.send_json(self.system.activity.state(self._span(exchange, self.ACTIVITY_SPANS, "day")))

    def host_history(self, exchange: HttpExchange) -> None:
        """Answers with the host telemetry over the requested hour, day, week or month span."""
        exchange.send_json(self.system.host_hist.state(self._span(exchange, self.HISTORY_SPANS, "day")))

    @staticmethod
    def _span(exchange: HttpExchange, spans: tuple, default: str) -> str:
        """Returns the requested range parameter.

        Raises:
            ApiError : The range is not one of the spans the history offers (400).
        """
        span = exchange.text("range", default)
        if span not in spans:
            raise ApiError(400, f"range must be one of {', '.join(spans)}")
        return span

    def server_log(self, exchange: HttpExchange) -> None:
        """Answers with the newest lines of the server's own console log."""
        exchange.send_json({"lines": self.logger.recent(exchange.integer("limit", "200"))})

    def nuke_jobs(self, exchange: HttpExchange) -> None:
        """Clears the launch queue and stops every job process and project entry script, never anything else."""
        exchange.send_result(self.nuke.nuke())

    def detach(self, exchange: HttpExchange) -> None:
        """Detaches the console server from the current terminal."""
        exchange.send_result(self.detacher.detach(), 500)

    def shutdown(self, exchange: HttpExchange) -> None:
        """Force-kills every other console on this root, answers with the server state, then shuts this console down."""
        siblings = self.detacher.kill_siblings()
        exchange.send_json({"ok": True, "siblings": siblings, **self.detacher.state()})
        self.detacher.shutdown()

    def configure_politeness(self, exchange: HttpExchange) -> None:
        """Sets the launch politeness level and the auto-throttle flag from the request body."""
        exchange.send_result(self.politeness.configure(exchange.body or {}))

    def arm_nuke(self, exchange: HttpExchange) -> None:
        """Arms or disarms the watchdog's automatic nuke from the request body's "armed" flag and answers with the watchdog state."""
        exchange.send_json(self.watchdog.arm(bool((exchange.body or {}).get("armed"))))
