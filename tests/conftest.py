import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configuration import Config
from core.shared import Job, Route, RoutingState, Stop, Vehicle

FIXTURES = Path(__file__).resolve().parent / "fixtures"


FAKE_DEPOT = (-46.63, -23.55)


class FakeVroom:
    def __init__(self):
        self.calls = 0

    def _vehicle_route(self, vehicle, taken, depot, clock):
        stops = []
        load  = vehicle.onboard
        t     = max(int(clock or 0), int(vehicle.time_window[0]))

        if load > 0:
            t += 300
            stops.append(Stop(job_id=-1, location=depot, kind="delivery", arrival=t, service=120, load=0))
            t   += 120
            load = 0

        for job in taken:
            t += 300
            if job.kind == "repossession":
                stops.append(Stop(job_id=job.id, location=job.location, kind="pickup", arrival=t, service=job.service, load=load + 1))
                t   += job.setup + job.service
                load = load + 1
                if load >= vehicle.capacity:
                    t += 300
                    stops.append(Stop(job_id=job.id, location=depot, kind="delivery", arrival=t, service=120, load=0))
                    t   += 120
                    load = 0
            else:
                stops.append(Stop(job_id=job.id, location=job.location, kind="job", arrival=t, service=job.service, load=load))
                t += job.setup + job.service

        if load > 0:
            t += 300
            stops.append(Stop(job_id=-1, location=depot, kind="delivery", arrival=t, service=120, load=0))
            t += 120

        return Route(
            vehicle_id = vehicle.id,
            stops      = stops,
            start      = vehicle.start,
            end        = stops[-1].location,
            cost       = 100 * len(stops),
            duration   = 60 * len(stops),
            service    = sum(stop.service for stop in stops),
        )

    def solve(self, jobs, vehicles, depot=None, clock=None):
        self.calls += 1
        jobs     = list(jobs)
        vehicles = list(vehicles)
        depot    = depot or FAKE_DEPOT

        routes    = []
        remaining = list(jobs)

        for vehicle in vehicles:
            taken     = remaining if len(vehicles) == 1 else remaining[:vehicle.capacity]
            remaining = [] if len(vehicles) == 1 else remaining[vehicle.capacity:]
            if not taken and vehicle.onboard == 0:
                continue

            routes.append(self._vehicle_route(vehicle, taken, depot, clock))

        return RoutingState(routes=routes, unassigned_ids={job.id for job in remaining})


class FakeWriter:
    def __init__(self):
        self.scalars    = []
        self.histograms = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        self.histograms.append((tag, step))

    def flush(self):
        pass

    def close(self):
        pass


class CapturingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class RecordingTracker:
    active = True

    def __init__(self):
        self.scalars    = []
        self.metrics    = []
        self.histograms = []
        self._step      = 0

    @property
    def current_step(self):
        return self._step

    def set_step(self, step):
        self._step = int(step)

    def log_scalar(self, tag, value, step=None):
        self.scalars.append((tag, value, step))

    def log_metrics(self, prefix, values, step=None):
        self.metrics.append((prefix, dict(values), step))

    def log_histogram(self, tag, values, step=None, bins="auto"):
        self.histograms.append((tag, step))

    @contextmanager
    def scope(self, name):
        yield self

    def scalar_tags(self):
        return [tag for tag, _, _ in self.scalars]

    def metric_prefixes(self):
        return [prefix for prefix, _, _ in self.metrics]


class ScriptedPolicy:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls   = 0

    def eval(self):
        return self

    def to(self, device):
        return self

    def select_action(self, graph, mask_info=None):
        action      = self.actions[min(self.calls, len(self.actions) - 1)]
        self.calls += 1
        return {"action": action}


def load_fixture(name):
    return json.load(open(FIXTURES / f"{name}.json"))


def make_jobs(count, first_id=0, kind="support"):
    return [
        Job(id=first_id + i, location=(-46.63 + 0.01 * i, -23.55 - 0.004 * i), kind=kind, amount=1 if kind == "repossession" else 0, priority=(i % 5) + 1)
        for i in range(count)
    ]


def make_vehicles(count, first_id=0):
    return [
        Vehicle(id=first_id + i, start=(-46.66 + 0.02 * i, -23.57), capacity=2)
        for i in range(count)
    ]


def make_stops(jobs):
    return [Stop(job_id=job.id, location=job.location, service=job.service) for job in jobs]


def make_route(vehicle, jobs, cost=1000):
    stops = make_stops(jobs)
    return Route(
        vehicle_id = vehicle.id,
        stops      = stops,
        start      = vehicle.start,
        end        = stops[-1].location if stops else None,
        cost       = cost,
        duration   = 60 * len(stops),
        service    = sum(stop.service for stop in stops),
    )


@pytest.fixture
def cpu_config():
    config = Config()
    config.training.device = "cpu"
    config.device.device   = "cpu"
    return config


@pytest.fixture
def seeded():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def fake_vroom(monkeypatch):
    fake = FakeVroom()
    monkeypatch.setattr("core.shared.environment.vroom", fake)
    return fake


@pytest.fixture
def environment(cpu_config, seeded, fake_vroom):
    from core.shared import Environment
    return Environment(cpu_config)
