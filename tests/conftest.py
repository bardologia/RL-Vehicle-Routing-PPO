import json
import random
import sys
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


class FakeVroom:
    def __init__(self):
        self.calls = 0

    def solve(self, jobs, vehicles):
        self.calls += 1
        jobs     = list(jobs)
        vehicles = list(vehicles)

        if not jobs or not vehicles:
            return RoutingState(routes=[], unassigned_ids={job.id for job in jobs})

        stops = [Stop(job_id=job.id, location=job.location, service=job.service, load=1) for job in jobs]
        route = Route(
            vehicle_id = vehicles[0].id,
            stops      = stops,
            start      = vehicles[0].start,
            end        = stops[-1].location,
            cost       = 100 * len(stops),
            duration   = 60 * len(stops),
            service    = sum(stop.service for stop in stops),
        )
        return RoutingState(routes=[route], unassigned_ids=set())


def load_fixture(name):
    return json.load(open(FIXTURES / f"{name}.json"))


def make_jobs(count, first_id=0):
    return [
        Job(id=first_id + i, location=(-46.63 + 0.01 * i, -23.55 - 0.004 * i), priority=(i % 5) + 1)
        for i in range(count)
    ]


def make_vehicles(count, first_id=0):
    return [
        Vehicle(id=first_id + i, start=(-46.66 + 0.02 * i, -23.57), capacity=2)
        for i in range(count)
    ]


def make_route(vehicle, jobs, cost=1000):
    stops = [Stop(job_id=job.id, location=job.location, service=job.service) for job in jobs]
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
