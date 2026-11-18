import socket

import pytest

from core.inference import EpisodeEvaluator, EpisodeRecorder, FixedOperatorAgent, TeacherAgent
from core.shared import Environment
from core.training.pretraining import RegretInsertionTeacher


def _service_up(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("localhost", port)) == 0


pytestmark = pytest.mark.skipif(
    not (_service_up(3000) and _service_up(5000)),
    reason="VROOM/OSRM containers not running",
)


META = {"agent_key": "probe", "agent": "probe", "run_name": None, "greedy": True}


def test_recorded_return_equals_the_evaluator_on_real_vroom(cpu_config):
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 3

    environment = Environment(cpu_config)
    recorder    = EpisodeRecorder(cpu_config, environment)

    for agent in (FixedOperatorAgent(2), TeacherAgent(RegretInsertionTeacher(cpu_config, allow_removal=False))):
        trace    = recorder.record(agent, META, {"kind": "seed", "seed": 1_000_003})
        expected = EpisodeEvaluator(environment, cpu_config).run(agent, 1_000_003)

        assert trace["summary"]["total_reward"] == pytest.approx(expected["total_reward"], abs=1e-9)
        assert trace["summary"]["final_drive_s"] == expected["final_cost"]
        assert trace["summary"]["final_unassigned"] == expected["final_unassigned"]


def test_verdict_on_real_vroom_agrees_with_the_insertion_teacher(cpu_config):
    cpu_config.training.max_disruptions_per_episode = 1
    cpu_config.training.max_edits_per_round         = 3

    recorder = EpisodeRecorder(cpu_config, Environment(cpu_config))
    trace    = recorder.record(TeacherAgent(RegretInsertionTeacher(cpu_config, allow_removal=False)), META, {"kind": "seed", "seed": 1_000_004}, verdict_scope="insertions")

    decisions = [frame["decision"] for frame in trace["frames"] if frame["kind"] == "decision"]

    assert decisions
    assert all(decision["verdict"]["match"]["operator"] for decision in decisions)
    assert all(entry["added_drive_s"] == pytest.approx((-entry["terms"]["distance_reward"]) / 1.5 * 1000, abs=1) for decision in decisions for entry in decision["verdict"]["insertions"])
