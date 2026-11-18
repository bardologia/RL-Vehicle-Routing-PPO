import copy

import pytest

import core.shared.services as services_module
from core.inference import DivergenceCheck, EpisodeRecorder, ExpectationCheck, FixedOperatorAgent, TeacherAgent, TraceFindings, TraceSummary
from core.shared import Environment
from core.training.pretraining import RegretInsertionTeacher


META = {"agent_key": "insertion_only", "agent": "insertion_only", "run_name": None, "greedy": True}


@pytest.fixture
def trace(cpu_config, seeded, fake_vroom, monkeypatch):
    monkeypatch.setattr(services_module, "vroom", fake_vroom)

    cpu_config.training.max_disruptions_per_episode = 4
    cpu_config.training.max_edits_per_round         = 4

    recorder = EpisodeRecorder(cpu_config, Environment(cpu_config))
    return recorder.record(TeacherAgent(RegretInsertionTeacher(cpu_config, allow_removal=False)), META, {"kind": "seed", "seed": 8}, verdict_scope="insertions")


def by_id(findings):
    return {finding["id"]: finding for finding in findings}


def decisions_of(trace):
    return [frame for frame in trace["frames"] if frame["kind"] == "decision"]


def test_every_rule_reports_once_with_a_known_status(trace):
    findings = TraceFindings().evaluate(trace)

    assert [finding["id"] for finding in findings] == [f"E{number}" for number in range(1, 10)]
    assert all(finding["status"] in ("PASS", "WARN", "FAIL", "NO DATA") for finding in findings)
    assert by_id(findings)["E6"]["status"] == "NO DATA"
    assert by_id(findings)["E7"]["status"] == "PASS"
    assert by_id(findings)["E9"]["status"] == "NO DATA"


def test_wasted_edit_and_early_commit_rules_fire_on_marked_frames(trace):
    frame                                = decisions_of(trace)[0]
    frame["decision"]["wasted"]          = True
    frame["decision"]["is_commit"]       = True
    frame["decision"]["edits_remaining"] = 3
    frame["plan"]["kpis"]["unassigned"]  = 2

    findings = by_id(TraceFindings().evaluate(trace))

    assert findings["E1"]["status"] == "WARN"
    assert findings["E1"]["seqs"] == [frame["seq"]]
    assert findings["E2"]["status"] == "WARN"
    assert frame["seq"] in findings["E2"]["seqs"]


def test_holding_share_rule_uses_the_term_sums(trace):
    trace["summary"]["reward_by_term"] = {"distance_reward": -1.0, "unassigned_reward": 1.0, "idle_reward": 0.0, "priority_reward": 0.0, "action_reward": 0.0, "holding_reward": -6.0}

    finding = by_id(TraceFindings().evaluate(trace))["E3"]

    assert finding["status"] == "WARN"
    assert finding["value"] == pytest.approx(0.75)


def test_confident_and_wrong_needs_policy_and_verdict(trace):
    frame                        = decisions_of(trace)[0]
    frame["decision"]["policy"]  = {"chosen_joint_prob": 0.9, "value": 0.0}
    frame["decision"]["verdict"] = {**frame["decision"]["verdict"], "one_step_regret": 2.0}

    findings = by_id(TraceFindings().evaluate(trace))

    assert findings["E5"]["status"] == "WARN"
    assert findings["E5"]["seqs"] == [frame["seq"]]
    assert findings["E6"]["status"] == "PASS"


def test_summary_separates_relocations_from_abandonments(trace):
    decisions = decisions_of(trace)[:3]
    for frame, (operator, job_id) in zip(decisions, [(1, 7), (0, 7), (1, 9)]):
        frame["round"]                         = 0
        frame["decision"]["action"]            = {"operator": operator, "operator_name": "X", "vehicle_id": 0, "job_id": job_id}
        frame["decision"]["wasted"]            = False
        frame["decision"]["reward"]            = -1.0
        frame["decision"]["rewards"]["holding_reward"] = 0.0

    summary = TraceSummary().summarise(trace, "complete")

    assert summary["relocations"] == [{"remove_seq": decisions[0]["seq"], "insert_seq": decisions[1]["seq"], "job_id": 7, "net_reward": -2.0}]
    assert summary["abandonments"] == [{"seq": decisions[2]["seq"], "job_id": 9}]

    trace["summary"] = summary
    assert by_id(TraceFindings().evaluate(trace))["E4"]["status"] == "WARN"


def test_expectation_checks_every_kind(trace):
    frames = [frame for frame in trace["frames"] if frame["round"] == 0]
    plan   = decisions_of(trace)[len([frame for frame in frames if frame["kind"] == "decision"]) - 1]["plan"]
    route  = plan["routes"][0]
    job_id = next(stop["job_id"] for stop in route["stops"] if stop["kind"] in ("job", "pickup"))

    checks = [
        {"kind": "job_on_vehicle", "job_id": job_id, "vehicle_id": route["vehicle_id"], "count": None},
        {"kind": "job_assigned", "job_id": job_id, "vehicle_id": None, "count": None},
        {"kind": "job_not_on_vehicle", "job_id": job_id, "vehicle_id": route["vehicle_id"] + 99, "count": None},
        {"kind": "max_edits", "job_id": None, "vehicle_id": None, "count": 99},
    ]

    assert ExpectationCheck().evaluate(checks, trace["frames"])["status"] == "PASS"
    assert ExpectationCheck().evaluate(checks + [{"kind": "job_unassigned", "job_id": job_id, "vehicle_id": None, "count": None}], trace["frames"])["status"] == "FAIL"

    with pytest.raises(ValueError):
        ExpectationCheck().evaluate([{"kind": "nonsense", "job_id": None, "vehicle_id": None, "count": None}], trace["frames"])


def test_divergence_flags_rounds_whose_event_ids_or_failures_differ(trace):
    other = copy.deepcopy(trace)
    opens = [frame for frame in other["frames"] if frame["kind"] == "round_open"]

    opens[1]["round_open"]["event"]["added_job_ids"] = [999]
    opens[2]["round_open"]["executed"]["failed"]     = [{"job_id": 5, "vehicle_id": 0, "priority": 1, "kind": "repossession"}] + opens[2]["round_open"]["executed"]["failed"]

    found = DivergenceCheck().compare({"a": trace, "b": other})

    assert {"round": 1, "kind": "event_ids", "agents": ["b"]} in found
    assert {"round": 2, "kind": "execution", "agents": ["b"]} in found
    assert DivergenceCheck().compare({"a": trace}) == []
    assert DivergenceCheck().compare({"a": trace, "b": copy.deepcopy(trace)}) == []
