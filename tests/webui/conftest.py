import io
import json
import sys
import time
from pathlib import Path

import pytest

WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))

import core.shared.services as services_module

from politeness      import JobPoliteness
from process_manager import ProcessManager
from system_monitor  import SystemMonitor
from web_logger      import WebLogger

WEBUI_ROOT = WEBUI_DIR
SLEEP_LONG = "import time\ntime.sleep(30)\n"
ARGS_DUMP  = "import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n"


class FakeHandler:
    """Stand-in for a BaseHTTPRequestHandler capturing the response it is sent.

    Attributes:
        command : HTTP method of the simulated request.
        path    : Request path including any query string.
        headers : Request headers, always carrying Content-Length.
        rfile   : Readable stream over the request body.
        wfile   : Writable stream collecting the response body.
        status  : Status code passed to send_response.
        sent    : Response headers passed to send_header, keyed by name.
    """

    def __init__(self, command: str, path: str, body: bytes = b"", headers: dict | None = None) -> None:
        """Builds a simulated request with the given method, path, raw body and extra headers."""
        self.command = command
        self.path    = path
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.rfile   = io.BytesIO(body)
        self.wfile   = io.BytesIO()
        self.status  = 0
        self.sent    = {}

    def send_response(self, status: int) -> None:
        """Records the response status code."""
        self.status = status

    def send_header(self, name: str, value: str) -> None:
        """Records a response header."""
        self.sent[name] = value

    def end_headers(self) -> None:
        """Ends the header block, which the fake does not need to act on."""
        return

    def payload(self) -> dict:
        """Returns the response body decoded as JSON."""
        return json.loads(self.wfile.getvalue().decode("utf-8"))


class StubPaths:
    """Minimal stand-in for the console path registry rooted at a temporary directory.

    Attributes:
        repo_root      : Root of the fake repository.
        webui_dir      : The real webui directory, where the value renderer lives.
        main_dir       : Directory holding the entry-point scripts.
        config_dir     : Directory standing in for the configuration package.
        runs_dir       : Root of the run-output tree.
        logs_dir       : Root of the console's state tree.
        jobs_dir       : Directory of the attached jobs' transcripts.
        telemetry_dir  : Directory of the persisted histories.
        saved_runs_dir : Directory for saved run definitions.
        templates_dir  : Directory for config templates.
        sequences_dir  : Directory for experiment sequence definitions.
    """

    def __init__(self, root: Path) -> None:
        """Derives the standard subdirectories from a root path."""
        self.repo_root      = root
        self.webui_dir      = WEBUI_DIR
        self.main_dir       = root / "main"
        self.config_dir     = root / "configuration"
        self.runs_dir       = root / "runs"
        self.logs_dir       = root / "logs" / "webui"
        self.jobs_dir       = self.logs_dir / "jobs"
        self.telemetry_dir  = self.logs_dir / "telemetry"
        self.saved_runs_dir = self.logs_dir / "saved_runs"
        self.templates_dir  = self.logs_dir / "config_templates"
        self.sequences_dir  = self.logs_dir / "sequences"

    def has_script(self, key: str) -> bool:
        """Returns whether an entry script exists for the given key."""
        return (self.main_dir / f"{key}.py").exists()

    def script_entry(self, key: str) -> dict:
        """Returns the absolute path and repo-relative path of an entry script."""
        return {"path": self.main_dir / f"{key}.py", "rel": f"main/{key}.py", "config_module": "configuration", "config_class": "Config"}


class StubDescriber:
    """Job describer stand-in returning a fixed sentence per script key."""

    def describe(self, key: str, interpreter: str, overrides: dict | None) -> str:
        """Returns a constant description mentioning the script key."""
        return f"stub description for {key}"


def job_record(manager: ProcessManager, job_id: str) -> dict:
    """Returns a snapshot copy of a job record taken under the manager lock."""
    with manager.lock:
        return dict(manager.jobs[job_id])


def wait_for_status(manager: ProcessManager, job_id: str, status: str, timeout: float = 10.0) -> bool:
    """Polls until a job reaches a given status, returning whether it did before the timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if job_record(manager, job_id)["status"] == status:
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def make_manager(tmp_path):
    """Yields a factory building a ProcessManager over named throwaway scripts, stopping all jobs on teardown."""
    built = []

    def build(scripts: dict) -> ProcessManager:
        directory = tmp_path / "main"
        directory.mkdir(parents=True, exist_ok=True)

        for name, body in scripts.items():
            (directory / f"{name}.py").write_text(body)

        paths   = StubPaths(tmp_path)
        logger  = WebLogger()
        manager = ProcessManager(paths, logger, StubDescriber(), JobPoliteness(paths, logger), {name: name.replace("_", " ").title() for name in scripts})
        built.append(manager)

        return manager

    yield build

    for manager in built:
        manager.stop_all()


@pytest.fixture
def monitor(tmp_path, monkeypatch):
    """Returns a SystemMonitor over a temporary root; its sampling threads are never started."""
    return SystemMonitor(StubPaths(tmp_path), WebLogger())


class TempPaths:
    def __init__(self, root):
        self.repo_root     = root
        self.static_dir    = root / "static"
        self.runs_dir      = root / "runs"
        self.episodes_dir  = root / "state" / "episodes"
        self.scenarios_dir = root / "state" / "scenarios"
        self.buildings_dir = root / "state" / "buildings"

        self.runs_dir.mkdir(parents=True)
        self.static_dir.mkdir(parents=True)

    def run_dir(self, name):
        return self.runs_dir / name


class FakeGeometry:
    def __init__(self):
        self.calls = 0

    def route(self, points):
        self.calls += 1
        return {"path": [[point[1], point[0]] for point in points], "distance_m": 1000 * (len(points) - 1)}


class StubCatalog:
    def __init__(self, loadable=True, reason=None, mtime=100.0):
        self.result = {"has_checkpoint": True, "loadable": loadable, "reason": reason, "mtime": mtime}

    def status(self, run_dir):
        return dict(self.result)


def agent_block(rewards, unassigned, served, costs, operator_frequency):
    count = len(rewards)
    mean  = sum(rewards) / count
    std   = (sum((value - mean) ** 2 for value in rewards) / count) ** 0.5

    return {
        "episodes"              : count,
        "mean_reward"           : mean,
        "std_reward"            : std,
        "sem_reward"            : std / count ** 0.5,
        "mean_final_cost"       : sum(costs) / count,
        "std_final_cost"        : 0.0,
        "mean_final_unassigned" : sum(unassigned) / count,
        "mean_served"           : sum(served) / count,
        "mean_failed"           : 0.0,
        "mean_dropped"          : 0.0,
        "cost_per_served"       : sum(costs) / max(sum(served), 1),
        "operator_frequency"    : dict(zip(("op0", "op1", "op2"), operator_frequency)),
        "per_episode"           : {"reward": rewards, "final_cost": costs, "unassigned": unassigned, "served": served},
    }


def write_report(run_dir, agents, seed=1000, greedy=True):
    episodes = len(next(iter(agents.values()))["per_episode"]["reward"]) if agents else 0
    meta     = {"run_name": run_dir.name, "seed": seed, "episodes": episodes, "greedy": greedy, "reward": {}, "environment": {}, "caps": {"max_disruptions_per_episode": 3, "max_edits_per_round": 8}}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"evaluation_seed{seed}.json").write_text(json.dumps({"meta": meta, "agents": agents}))


@pytest.fixture
def paths(tmp_path):
    return TempPaths(tmp_path)


@pytest.fixture
def service_vroom(fake_vroom, monkeypatch):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)
    return fake_vroom
