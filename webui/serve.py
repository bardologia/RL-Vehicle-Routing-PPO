"""Command-line entry point that starts the routing console."""

import argparse
import faulthandler
import signal
import sys
from pathlib import Path

WEBUI_ROOT = Path(__file__).resolve().parent
REPO_ROOT  = WEBUI_ROOT.parent

for entry in (str(REPO_ROOT), str(WEBUI_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from web_ui_server import WebUIServer


class ServeEntry:
    """Parses the console command line and starts the web UI server.

    Attributes:
        parser : Argument parser holding the host, port, root and Overpass options.
    """

    def __init__(self) -> None:
        """Builds the argument parser."""
        self.parser = argparse.ArgumentParser(description="RL Vehicle Routing PPO web console")
        self.parser.add_argument("--host", default="127.0.0.1")
        self.parser.add_argument("--port", type=int, default=8766)
        self.parser.add_argument("--root", type=Path, default=None, help="project root the console operates on (jobs, logs, runs); defaults to the repository holding this file")
        self.parser.add_argument("--overpass", default=None, help="comma-separated Overpass interpreter URLs for building data; the built-in list when omitted")

    def run(self) -> None:
        """Parses the command line and serves the console until it is stopped.

        SIGUSR1 dumps every thread's stack to stderr, so a request that hangs can
        be located with ``kill -USR1 <pid>`` while the console keeps running.
        """
        faulthandler.register(signal.SIGUSR1, all_threads=True)

        args      = self.parser.parse_args()
        endpoints = [url.strip() for url in args.overpass.split(",") if url.strip()] if args.overpass else None

        WebUIServer(host=args.host, port=args.port, root=args.root, overpass_endpoints=endpoints).serve()


if __name__ == "__main__":
    ServeEntry().run()
