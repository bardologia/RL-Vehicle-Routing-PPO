import argparse
import sys
from pathlib import Path

WEBUI_ROOT = Path(__file__).resolve().parent
REPO_ROOT  = WEBUI_ROOT.parent

for entry in (str(REPO_ROOT), str(WEBUI_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from web_ui_server import WebUIServer


class ServeEntry:
    def _parse(self):
        parser = argparse.ArgumentParser(description="RL Vehicle Routing PPO web console")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8766)
        parser.add_argument("--overpass", default=None, help="Comma-separated Overpass interpreter URLs for building data; the built-in list when omitted")
        return parser.parse_args()

    def run(self):
        args      = self._parse()
        endpoints = [url.strip() for url in args.overpass.split(",") if url.strip()] if args.overpass else None

        WebUIServer(args.host, args.port, endpoints).serve()


ServeEntry().run()
