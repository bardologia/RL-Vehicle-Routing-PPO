import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "webui"))

from console_runtime import ConsoleRuntime
from core.inference import ExpectationCheck, ScenarioPlanner
from core.shared.state import EntityPool, Job, Vehicle
from scenario_lab import ScenarioLab
from scenario_templates import ScenarioTemplates
from web_errors import ApiError
from tests.conftest import make_jobs, make_vehicles
from tests.webui.conftest import FakeGeometry


LON_BOUNDS = (-47.2, -46.0)
LAT_BOUNDS = (-24.2, -23.0)


def catalog():
    return ScenarioTemplates().catalog()


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
        assert LON_BOUNDS[0] <= template["depot"][0] <= LON_BOUNDS[1]
        assert LAT_BOUNDS[0] <= template["depot"][1] <= LAT_BOUNDS[1]


def test_catalog_entities_parse_and_stay_in_bounds():
    for template in catalog():
        job_ids     = [job["id"] for job in template["jobs"]]
        vehicle_ids = [vehicle["id"] for vehicle in template["vehicles"]]
        assert len(job_ids) == len(set(job_ids))
        assert len(vehicle_ids) == len(set(vehicle_ids))

        for job in template["jobs"]:
            parsed = Job.from_dict(job)
            assert parsed.kind in ("support", "repossession")
            assert parsed.amount == (1 if parsed.kind == "repossession" else 0)
            assert 1 <= parsed.priority <= 5
            assert LON_BOUNDS[0] <= parsed.location[0] <= LON_BOUNDS[1]
            assert LAT_BOUNDS[0] <= parsed.location[1] <= LAT_BOUNDS[1]

        for vehicle in template["vehicles"]:
            parsed = Vehicle.from_dict(vehicle)
            assert parsed.capacity >= 1
            assert LON_BOUNDS[0] <= parsed.start[0] <= LON_BOUNDS[1]
            assert LAT_BOUNDS[0] <= parsed.start[1] <= LAT_BOUNDS[1]


def test_catalog_assignments_reference_known_entities_exactly_once():
    for template in catalog():
        assignment = template.get("assignment")
        if not assignment:
            continue

        job_ids      = {job["id"] for job in template["jobs"]}
        vehicle_ids  = {vehicle["id"] for vehicle in template["vehicles"]}
        assigned_ids = []

        for vehicle_key, assigned in assignment.items():
            assert int(vehicle_key) in vehicle_ids
            assert set(assigned) <= job_ids
            assigned_ids.extend(assigned)

        assert len(assigned_ids) == len(set(assigned_ids))


def test_every_template_declares_a_valid_expectation():
    for template in catalog():
        job_ids     = {job["id"] for job in template["jobs"]}
        vehicle_ids = {vehicle["id"] for vehicle in template["vehicles"]}

        assert template["expectation"]

        for check in template["expectation"]:
            assert set(check) == {"kind", "job_id", "vehicle_id", "count"}
            assert check["kind"] in ExpectationCheck.KINDS

            if check["kind"] == "max_edits":
                assert check["count"] >= 0
            else:
                assert check["job_id"] in job_ids
            if check["kind"] in ("job_on_vehicle", "job_not_on_vehicle"):
                assert check["vehicle_id"] in vehicle_ids


def test_expectations_are_not_met_by_the_pinned_starting_plan():
    for template in catalog():
        assignment     = template.get("assignment") or {}
        vehicle_by_job = {job_id: int(vehicle_id) for vehicle_id, job_ids in assignment.items() for job_id in job_ids}
        unassigned     = {job["id"] for job in template["jobs"]} - set(vehicle_by_job)
        placement      = [check for check in template["expectation"] if check["kind"] != "max_edits"]

        if all(check["kind"] == "max_edits" for check in template["expectation"]):
            continue
        if all(check["kind"] == "job_unassigned" and check["job_id"] in unassigned for check in placement):
            continue

        assert not all(ExpectationCheck()._holds(check, vehicle_by_job, unassigned, 0) for check in placement), template["key"]


def test_planner_builds_pinned_routes_and_leftovers(service_vroom):
    jobs     = EntityPool(make_jobs(4))
    vehicles = EntityPool(make_vehicles(2))

    state = ScenarioPlanner().initial_state(jobs, vehicles, {"0": [0, 1], "1": [2]}, (-46.63, -23.55), 28800)

    assert state.route_of_vehicle(0).job_ids == [0, 1]
    assert state.route_of_vehicle(1).job_ids == [2]
    assert state.unassigned_ids == {3}


def test_planner_rejects_bad_assignments(service_vroom):
    jobs     = EntityPool(make_jobs(3))
    vehicles = EntityPool(make_vehicles(2))

    cases = [({"9": [0]}, "unknown vehicle"), ({"0": [7]}, "unknown jobs"), ({"0": [0], "1": [0]}, "repeats")]

    for assignment, fragment in cases:
        with pytest.raises(ValueError, match=fragment):
            ScenarioPlanner().initial_state(jobs, vehicles, assignment, (-46.63, -23.55), 28800)

    with pytest.raises(ValueError, match="depot"):
        ScenarioPlanner().initial_state(jobs, vehicles, None, None, 28800)


def test_lab_solve_returns_a_plan_with_geometry_and_kpis(service_vroom, paths):
    lab      = ScenarioLab(ConsoleRuntime(paths), FakeGeometry(), logger=None)
    template = catalog()[0]

    plan = lab.solve(template["jobs"], template["vehicles"], template["assignment"], template["depot"])["plan"]

    assert [route["vehicle_id"] for route in plan["routes"]] == [0]
    assert plan["routes"][0]["path"]
    assert plan["routes"][0]["distance_m"] > 0
    assert plan["kpis"]["distance_m"] == plan["routes"][0]["distance_m"]
    assert [entry["job_id"] for entry in plan["unassigned"]] == [2]
    assert plan["kpis"]["unassigned"] == 1


def test_lab_solve_answers_400_on_a_bad_scenario(service_vroom, paths):
    lab      = ScenarioLab(ConsoleRuntime(paths), FakeGeometry(), logger=None)
    template = catalog()[0]

    for arguments in (([], template["vehicles"], None, template["depot"]), (template["jobs"], template["vehicles"], {"9": [0]}, template["depot"]), (template["jobs"], template["vehicles"], None, None)):
        with pytest.raises(ApiError) as raised:
            lab.solve(*arguments)
        assert raised.value.status == 400


def test_lab_sample_is_reproducible_per_seed(paths):
    lab = ScenarioLab(ConsoleRuntime(paths), FakeGeometry(), logger=None)

    assert lab.sample(5, 2, 3) == lab.sample(5, 2, 3)
    assert lab.sample(5, 2, 3) != lab.sample(5, 2, 4)
    assert len(lab.sample(5, 2, 3)["jobs"]) == 5
