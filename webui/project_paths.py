import sys
from pathlib import Path


class ProjectPaths:
    def __init__(self):
        self.webui_dir    = Path(__file__).resolve().parent
        self.repo_root    = self.webui_dir.parent
        self.static_dir   = self.webui_dir / "static"
        self.main_dir     = self.repo_root / "main"
        self.runs_dir     = self.repo_root / "runs"
        self.datasets_dir = self.repo_root / "datasets"
        self.interpreter  = sys.executable

    def script_path(self, key):
        return self.main_dir / f"{key}.py"

    def run_dir(self, name):
        return self.runs_dir / name
