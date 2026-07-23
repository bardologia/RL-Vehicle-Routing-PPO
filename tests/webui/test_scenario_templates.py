import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webui"))

import core.shared.services as services_module
from core.shared.state import EntityPool, Job, Vehicle
from scenario_lab import ScenarioLab
from scenario_templates import ScenarioTemplates
from tests.conftest import FakeVroom, make_jobs, make_vehicles


LON_BOUNDS = (-47.2, -46.0)
LAT_BOUNDS = (-24.2, -23.0)


def catalog():
    return ScenarioTemplates().catalog()


def lab():
    return ScenarioLab(paths=None, logger=None)


def test_catalog_keys_and_required_fields():
    templates = catalog()
    assert templates

    keys = [template["key"] for template in templates]
    assert len(keys) == len(set(keys))

    for template in templates:
        assert template["title"]
        assert template["description"]
        assert template["expected"]
        assert template["jobs"]
        assert template["vehicles"]


def test_catalog_entities_parse_and_stay_in_bounds():
    for template in catalog():
        job_ids     = [job["id"] for job in template["jobs"]]
        vehicle_ids = [vehicle["id"] for vehicle in template["vehicles"]]
        assert len(job_ids) == len(set(job_ids))
        assert len(vehicle_ids) == len(set(vehicle_ids))

        for job in template["jobs"]:
            parsed = Job.from_dict(job)
            assert 1 <= parsed.priority <= 5
            assert LON_BOUNDS[0] <= parsed.location[0] <= LON_BOUNDS[1]
            assert LAT_BOUNDS[0] <= parsed.location[1] <= LAT_BOUNDS[1]

        for vehicle in template["vehicles"]:
            parsed = Vehicle.from_dict(vehicle)
            assert parsed.capacity >= 1
            assert LON_BOUNDS[0] <= parsed.start[0] <= LON_BOUNDS[1]
            assert LAT_BOUNDS[0] <= parsed.start[1] <= LAT_BOUNDS[1]


def test_catalog_assignments_reference_known_entities_within_capacity():
    for template in catalog():
        assignment = template.get("assignment")
        if not assignment:
            continue

        job_ids      = {job["id"] for job in template["jobs"]}
        capacity_by  = {vehicle["id"]: vehicle["capacity"] for vehicle in template["vehicles"]}
        assigned_ids = []

        for vehicle_key, assigned in assignment.items():
            assert int(vehicle_key) in capacity_by
            assert len(assigned) <= capacity_by[int(vehicle_key)]
            assert set(assigned) <= job_ids
            assigned_ids.extend(assigned)

        assert len(assigned_ids) == len(set(assigned_ids))


def test_assigned_state_builds_pinned_routes_and_leftovers(monkeypatch):
    monkeypatch.setattr(services_module, "vroom", FakeVroom())

    jobs     = EntityPool(make_jobs(4))
    vehicles = EntityPool(make_vehicles(2))

    state, error = lab()._assigned_state(jobs, vehicles, {"0": [0, 1], "1": [2]})

    assert error is None
    assert state.route_of_vehicle(0).job_ids == [0, 1]
    assert state.route_of_vehicle(1).job_ids == [2]
    assert state.unassigned_ids == {3}


def test_assigned_state_rejects_bad_assignments(monkeypatch):
    monkeypatch.setattr(services_module, "vroom", FakeVroom())

    jobs     = EntityPool(make_jobs(3))
    vehicles = EntityPool(make_vehicles(1))

    cases = [
        ({"9": [0]},         "unknown vehicle"),
        ({"0": [7]},         "unknown jobs"),
        ({"0": [0, 1, 2]},   "capacity"),
    ]

    for assignment, fragment in cases:
        state, error = lab()._assigned_state(jobs, vehicles, assignment)
        assert state is None
        assert fragment in error


def test_assigned_state_rejects_duplicate_jobs(monkeypatch):
    monkeypatch.setattr(services_module, "vroom", FakeVroom())

    jobs     = EntityPool(make_jobs(2))
    vehicles = EntityPool(make_vehicles(2))

    state, error = lab()._assigned_state(jobs, vehicles, {"0": [0], "1": [0]})

    assert state is None
    assert "repeats" in error
