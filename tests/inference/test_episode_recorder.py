import json

import pytest

import core.shared.services as services_module
from core.inference import EpisodeEvaluator, EpisodeRecorder, FixedOperatorAgent, ModelAgent, TeacherAgent, TraceFile
from core.inference.recording import REWARD_TERMS
from core.shared import Environment
from core.training.pretraining import RegretInsertionTeacher
from model.policy_model import Policy
from tests.conftest import make_jobs, make_vehicles


META = {"agent_key": "model", "agent": "model", "run_name": "fake", "greedy": True}


@pytest.fixture
def setup(cpu_config, seeded, fake_vroom, monkeypatch):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)

    cpu_config.training.max_disruptions_per_episode = 3
    cpu_config.training.max_edits_per_round         = 3

    environment = Environment(cpu_config)
    return cpu_config, environment, EpisodeRecorder(cpu_config, environment)


def decisions_of(trace):
    return [frame for frame in trace["frames"] if frame["kind"] == "decision"]


def test_trace_has_the_contract_keys_and_survives_json(setup):
    config, environment, recorder = setup
    trace                         = recorder.record(ModelAgent(Policy(config), greedy=True), META, {"kind": "seed", "seed": 5}, verdict_scope="insertions")

    json.dumps(trace, allow_nan=False)

    assert set(trace) == {"meta", "depot", "entities", "frames", "snapshots", "summary", "expectation"}
    assert trace["meta"]["complete"] is True
    assert trace["meta"]["caps"] == {"rounds": 3, "edits": 3}
    assert [frame["seq"] for frame in trace["frames"]] == list(range(len(trace["frames"])))
    assert trace["frames"][0]["kind"] == "round_open"
    assert trace["frames"][0]["round_open"]["event"]["type"] in ("new_job", "remove_job", "new_vehicle", "remove_vehicle")

    for frame in decisions_of(trace):
        decision = frame["decision"]

        assert set(decision["rewards"]) == set(REWARD_TERMS)
        assert decision["reward"] == pytest.approx(sum(decision["rewards"].values()))
        assert decision["features"]["context"]["raw"][6] == frame["round"]
        assert decision["features"]["context"]["raw"][7] == decision["edits_remaining"]
        assert decision["verdict"]["scope"] == "insertions"
        assert str(frame["seq"]) in trace["snapshots"]
        assert set(frame["plan"]) == {"routes", "idle", "idle_vehicle_ids", "unassigned", "vehicles", "kpis"}


def test_policy_block_is_a_consistent_distribution(setup):
    config, environment, recorder = setup
    trace                         = recorder.record(ModelAgent(Policy(config), greedy=False), META, {"kind": "seed", "seed": 6})

    for frame in decisions_of(trace):
        policy = frame["decision"]["policy"]
        joint  = policy["operator"]["probs"][2]

        assert sum(policy["operator"]["probs"]) == pytest.approx(1.0, abs=1e-5)
        assert sum(policy["operator"]["raw_probs"]) == pytest.approx(1.0, abs=1e-5)

        for operator in (0, 1):
            vehicle_row = policy["vehicle_probs"][operator]

            if policy["operator"]["masked"][operator]:
                assert all(value is None for value in vehicle_row)
                continue

            assert sum(value for value in vehicle_row if value is not None) == pytest.approx(1.0, abs=1e-5)

            for vehicle_probability, job_row in zip(vehicle_row, policy["job_probs"][operator]):
                if vehicle_probability is None:
                    assert all(value is None for value in job_row)
                    continue

                assert sum(value for value in job_row if value is not None) == pytest.approx(1.0, abs=1e-5)
                joint += policy["operator"]["probs"][operator] * vehicle_probability

        assert joint == pytest.approx(1.0, abs=1e-4)
        assert 0.0 < policy["chosen_joint_prob"] <= 1.0
        assert policy["chosen_rank"] >= 1
        assert all(0.0 <= value <= 1.0 + 1e-6 for value in policy["entropy"].values())


def test_cumulative_reward_equals_the_evaluator_total(setup):
    config, environment, recorder = setup

    for agent in (FixedOperatorAgent(2), TeacherAgent(RegretInsertionTeacher(config, allow_removal=False)), TeacherAgent(RegretInsertionTeacher(config))):
        trace    = recorder.record(agent, META, {"kind": "seed", "seed": 9})
        expected = EpisodeEvaluator(environment, config).run(agent, 9)

        assert trace["summary"]["total_reward"] == pytest.approx(expected["total_reward"], abs=1e-9)
        assert trace["summary"]["served"] == expected["served"]
        assert trace["summary"]["failed"] == expected["failed"]
        assert trace["summary"]["dropped"] == expected["dropped"]
        assert trace["summary"]["final_unassigned"] == expected["final_unassigned"]
        assert trace["summary"]["operator_counts"] == [expected["operator_counts"][operator] for operator in range(3)]
        assert decisions_of(trace)[-1]["decision"]["cumulative"] == pytest.approx(expected["total_reward"], abs=1e-9)
        assert decisions_of(trace)[0]["decision"]["return_to_go"] == pytest.approx(expected["total_reward"], abs=1e-9)


def test_holding_is_attributed_to_the_previous_commit(setup):
    config, environment, recorder = setup
    trace                         = recorder.record(FixedOperatorAgent(2), META, {"kind": "seed", "seed": 4})

    opens     = [frame for frame in trace["frames"] if frame["kind"] == "round_open"]
    decisions = decisions_of(trace)

    assert opens[0]["round_open"]["holding_for_seq"] is None
    assert opens[0]["round_open"]["gap_s"] == 0

    for frame in opens[1:]:
        previous = [decision for decision in decisions if decision["round"] == frame["round"] - 1][-1]
        first    = [decision for decision in decisions if decision["round"] == frame["round"]][0]

        assert frame["round_open"]["holding_for_seq"] == previous["seq"]
        assert frame["round_open"]["holding_charged"] == pytest.approx(first["decision"]["rewards"]["holding_reward"])
        assert frame["round_open"]["gap_s"] > 0


def test_round_open_lists_the_ids_the_event_touched(setup):
    config, environment, recorder = setup
    seen                          = set()

    for seed in range(12):
        trace = recorder.record(FixedOperatorAgent(2), META, {"kind": "seed", "seed": seed})

        for frame in trace["frames"]:
            if frame["kind"] != "round_open":
                continue

            event = frame["round_open"]["event"]
            seen.add(event["type"])

            if event["type"] == "new_job":
                assert len(event["added_job_ids"]) == event["count"]
                assert set(event["added_job_ids"]) <= {entry["job_id"] for entry in frame["plan"]["unassigned"]}
            if event["type"] == "remove_vehicle":
                assert len(event["removed_vehicle_ids"]) == event["count"]
                assert set(event["orphaned_job_ids"]) <= {entry["job_id"] for entry in frame["plan"]["unassigned"]}
            if event["type"] == "new_vehicle":
                assert len(event["added_vehicle_ids"]) == event["count"]
            if event["type"] == "remove_job":
                assert event["removed_job_ids"]

            for entry in frame["round_open"]["executed"]["served"]:
                assert set(entry) == {"job_id", "vehicle_id", "priority", "kind"}

    assert seen == {"new_job", "remove_job", "new_vehicle", "remove_vehicle"}


def test_snapshot_round_trips_through_load_scenario(setup):
    config, environment, recorder = setup
    trace                         = recorder.record(TeacherAgent(RegretInsertionTeacher(config, allow_removal=False)), META, {"kind": "seed", "seed": 3}, verdict_scope="insertions")

    scratch = Environment(config)

    for frame in decisions_of(trace):
        snapshot = trace["snapshots"][str(frame["seq"])]
        scratch.load_scenario(snapshot)

        verdict = recorder.verdicts.build(scratch, scratch.current_state, snapshot["edits_remaining"], "insertions", frame["decision"]["action"], frame["decision"]["rewards"])

        assert verdict == frame["decision"]["verdict"]
        assert verdict["match"]["operator"] is True
        assert verdict["one_step_regret"] >= -1e-9


def test_scenario_source_runs_the_pinned_plan_and_checks_the_expectation(setup):
    config, environment, recorder = setup

    source = {
        "kind"               : "scenario",
        "name"               : None,
        "template"           : "hand",
        "jobs"               : [job.to_dict() for job in make_jobs(3)],
        "vehicles"           : [vehicle.to_dict() for vehicle in make_vehicles(1)],
        "depot"              : [-46.63, -23.55],
        "assignment"         : {"0": [0, 1]},
        "seed"               : 1,
        "initial_disruption" : False,
        "expectation"        : [{"kind": "job_on_vehicle", "job_id": 2, "vehicle_id": 0, "count": None}],
    }

    teacher = recorder.record(TeacherAgent(RegretInsertionTeacher(config, allow_removal=False)), META, source)
    idle    = recorder.record(FixedOperatorAgent(2), META, source)

    assert teacher["frames"][0]["round_open"]["event"] is None
    assert [entry["job_id"] for entry in teacher["frames"][0]["plan"]["unassigned"]] == [2]
    assert teacher["expectation"]["status"] == "PASS"
    assert idle["expectation"]["status"] == "FAIL"
    assert "jobs" not in teacher["meta"]["source"]


def test_should_stop_ends_the_recording_as_cancelled(setup):
    config, environment, recorder = setup
    seen                          = []

    trace = recorder.record(FixedOperatorAgent(2), META, {"kind": "seed", "seed": 2}, should_stop=lambda: len(seen) >= 2, on_frame=seen.append)

    assert trace["summary"]["stopped_reason"] == "cancelled"
    assert len(decisions_of(trace)) == 1


def test_trace_file_round_trip(setup, tmp_path):
    config, environment, recorder = setup
    trace                         = recorder.record(FixedOperatorAgent(2), META, {"kind": "seed", "seed": 2})

    TraceFile().write(tmp_path / "nested" / "trace.json.gz", trace)

    assert TraceFile().read(tmp_path / "nested" / "trace.json.gz") == json.loads(json.dumps(trace))
