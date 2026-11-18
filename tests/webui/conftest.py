import json
import sys
from pathlib import Path

import pytest

WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"
if str(WEBUI_DIR) not in sys.path:
    sys.path.insert(0, str(WEBUI_DIR))

import core.shared.services as services_module


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
