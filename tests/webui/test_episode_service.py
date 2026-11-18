import threading
import time

import pytest

from console_runtime import ConsoleRuntime
from core.inference import EpisodeRecorder
from core.shared import Environment
from episode_service import EpisodeService
from project_paths import RunLayout
from results_browser import RunBrowser
from scenario_templates import ScenarioTemplates
from trace_store import TraceRef, TraceStore
from web_errors import ApiError
from tests.conftest import CapturingLogger
from tests.webui.conftest import FakeGeometry, StubCatalog, agent_block, write_report


class GatedRecorder:
    def __init__(self, service):
        self.service = service
        self.guard   = threading.RLock()
        self.trace   = None

    def _frame(self, seq, kind, round_index):
        round_open = {"gap_s": 600 * round_index, "event": {"type": "new_job", "count": 2, "added_job_ids": [], "removed_job_ids": [], "added_vehicle_ids": [], "removed_vehicle_ids": []}, "executed": {"failed": []}} if kind == "round_open" else None
        return {"seq": seq, "kind": kind, "round": round_index, "edit": None if kind == "round_open" else 0, "round_open": round_open, "decision": None}

    def record(self, agent, meta, source, should_stop=None, verdict_scope="none", on_frame=None):
        self.trace = {"meta": {**meta, "complete": False}, "frames": [], "summary": None}
        reason     = "complete"

        if meta["agent_key"] in self.service.fail:
            raise RuntimeError("recorder exploded\nsecond line")

        for seq, (kind, round_index) in enumerate([("round_open", 0), ("decision", 0), ("round_open", 1), ("decision", 1)]):
            self.service.gate.wait(5)
            if should_stop():
                reason = "cancelled"
                break

            with self.guard:
                self.trace["frames"].append(self._frame(seq, kind, round_index))
            on_frame(self.trace["frames"][-1])

        self.trace["summary"]          = {"total_reward": 1.5, "stopped_reason": reason}
        self.trace["meta"]["complete"] = True
        return self.trace


@pytest.fixture
def service(paths):
    runtime = ConsoleRuntime(paths)
    catalog = StubCatalog()
    runs    = RunBrowser(paths, RunLayout(paths), catalog)

    built      = EpisodeService(paths, runtime, catalog, runs, ScenarioTemplates(), CapturingLogger())
    built.gate = threading.Event()
    built.fail = set()
    built.gate.set()

    built.recorder_factory = lambda config, guard, step_guard: GatedRecorder(built)
    built._build_agent     = lambda spec, config: object()
    return built


def wait_for(condition, limit=5.0):
    started = time.time()
    while time.time() - started < limit:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def finished(service, episode_id):
    return service.status(episode_id)["status"] != "running"


SEED_BODY = {"source": {"kind": "seed", "seed": 7}, "caps": {"rounds": 2, "edits": 3}, "agents": [{"agent": "model", "run_name": "run_a", "greedy": False}, {"agent": "insertion_only"}, {"agent": "do_nothing"}]}


def test_create_queues_lanes_and_records_them_in_order(service):
    created = service.create(SEED_BODY)

    assert created["agent_keys"] == ["model~run_a~sampled", "insertion_only", "do_nothing"]

    wait_for(lambda: finished(service, created["episode_id"]))
    status = service.status(created["episode_id"])

    assert status["status"] == "finished"
    assert status["caps"] == {"rounds": 2, "edits": 3}
    assert status["source"] == {"kind": "seed", "seed": 7}
    assert status["shared"] == {"gaps": [0, 600], "events": [{"type": "new_job", "count": 2}, {"type": "new_job", "count": 2}]}

    for key in created["agent_keys"]:
        lane = status["lanes"][key]
        assert (lane["status"], lane["frames"], lane["decisions"], lane["round"], lane["total_reward"]) == ("finished", 4, 2, 1, 1.5)
        assert service.trace_path(created["episode_id"], key).exists()

    assert status["lanes"]["model~run_a~sampled"]["greedy"] is False
    assert [entry["episode_id"] for entry in service.list()] == [created["episode_id"]]
    assert service.list()[0]["agents"]["do_nothing"] == {"status": "finished", "total_reward": 1.5}


def test_progress_is_visible_while_a_lane_records_and_a_second_request_waits(service):
    service.gate.clear()

    first  = service.create(SEED_BODY)
    second = service.create({"source": {"kind": "seed", "seed": 8}, "agents": [{"agent": "do_nothing"}]})

    wait_for(lambda: service.status(first["episode_id"])["lanes"]["model~run_a~sampled"]["status"] == "running")

    assert service.live_trace(first["episode_id"], "model~run_a~sampled") is not None
    assert service.status(first["episode_id"])["lanes"]["insertion_only"]["status"] == "queued"
    assert service.status(second["episode_id"])["lanes"]["do_nothing"]["status"] == "queued"
    assert service.status(second["episode_id"])["caps"] == {"rounds": 5, "edits": 8}

    with pytest.raises(ApiError) as raised:
        service.add_agents(first["episode_id"], {"agents": [{"agent": "insertion_only"}]})
    assert raised.value.status == 409

    service.gate.set()
    wait_for(lambda: finished(service, second["episode_id"]))

    assert service.status(first["episode_id"])["status"] == "finished"
    assert service.live_trace(first["episode_id"], "model~run_a~sampled") is None


def test_cancel_stops_the_running_lane_and_drops_the_queued_ones(service):
    service.gate.clear()
    created = service.create(SEED_BODY)

    wait_for(lambda: service.status(created["episode_id"])["lanes"]["model~run_a~sampled"]["status"] == "running")

    assert service.cancel(created["episode_id"]) == {"ok": True}
    service.gate.set()
    wait_for(lambda: finished(service, created["episode_id"]))

    status = service.status(created["episode_id"])
    assert status["status"] == "cancelled"
    assert {lane["status"] for lane in status["lanes"].values()} == {"cancelled"}


def test_a_failing_lane_does_not_stop_the_others(service):
    service.fail = {"insertion_only"}
    created      = service.create(SEED_BODY)

    wait_for(lambda: finished(service, created["episode_id"]))
    lanes = service.status(created["episode_id"])["lanes"]

    assert lanes["insertion_only"]["status"] == "failed"
    assert lanes["insertion_only"]["error"] == "recorder exploded"
    assert "RuntimeError" in lanes["insertion_only"]["traceback"]
    assert lanes["model~run_a~sampled"]["status"] == "finished"
    assert lanes["do_nothing"]["status"] == "finished"
    assert service.status(created["episode_id"])["status"] == "finished"

    service.fail = set()
    again        = service.add_agents(created["episode_id"], {"agents": [{"agent": "insertion_only"}, {"agent": "teacher"}]})
    wait_for(lambda: finished(service, created["episode_id"]))

    lanes = service.status(created["episode_id"])["lanes"]
    assert again["agent_keys"] == ["insertion_only", "teacher"]
    assert lanes["insertion_only"]["status"] == "finished"
    assert lanes["teacher"]["status"] == "finished"


def test_status_survives_a_restart_and_marks_interrupted_lanes(service, paths):
    service.gate.clear()
    created = service.create(SEED_BODY)
    wait_for(lambda: service.status(created["episode_id"])["lanes"]["model~run_a~sampled"]["frames"] >= 0 and (paths.episodes_dir / created["episode_id"] / "episode.json").exists())

    restarted = EpisodeService(paths, service.runtime, service.catalog, service.runs, ScenarioTemplates(), CapturingLogger())
    status    = restarted.status(created["episode_id"])

    assert status["status"] == "cancelled"
    assert all(lane["status"] == "cancelled" and "restarted" in lane["error"] for lane in status["lanes"].values())

    service.gate.set()
    wait_for(lambda: finished(service, created["episode_id"]))


def test_evaluation_source_takes_seed_and_caps_from_the_report(service, paths):
    write_report(paths.run_dir("run_a"), {"model": agent_block([1.0, 2.0], [0, 0], [1, 1], [10.0, 10.0], [0.5, 0.0, 0.5])}, seed=500)

    created = service.create({"source": {"kind": "evaluation", "run": "run_a", "report_seed": 500, "index": 1}, "caps": {"rounds": 9, "edits": 1}, "agents": [{"agent": "do_nothing"}]})
    wait_for(lambda: finished(service, created["episode_id"]))
    status = service.status(created["episode_id"])

    assert status["source"] == {"kind": "evaluation", "run": "run_a", "report_seed": 500, "index": 1, "seed": 501}
    assert status["caps"] == {"rounds": 3, "edits": 8}

    for source, code in (({"kind": "evaluation", "run": "run_a", "report_seed": 500, "index": 2}, 400), ({"kind": "evaluation", "run": "run_a", "report_seed": 9, "index": 0}, 404), ({"kind": "evaluation", "run": "nope", "report_seed": 500, "index": 0}, 404)):
        with pytest.raises(ApiError) as raised:
            service.create({"source": source, "agents": [{"agent": "do_nothing"}]})
        assert raised.value.status == code


def test_scenario_source_carries_the_template_expectation_and_can_be_read_back(service):
    template = ScenarioTemplates().catalog()[0]
    source   = {"kind": "scenario", "name": "mine", "template": template["key"], "jobs": template["jobs"], "vehicles": template["vehicles"], "depot": template["depot"], "assignment": template["assignment"], "seed": 3}

    created = service.create({"source": source, "agents": [{"agent": "teacher"}]})
    wait_for(lambda: finished(service, created["episode_id"]))

    assert service.episodes[created["episode_id"]]["source"]["expectation"] == template["expectation"]
    assert service.status(created["episode_id"])["source"] == {"kind": "scenario", "name": "mine", "template": template["key"], "seed": 3, "initial_disruption": False}
    assert service.status(created["episode_id"])["caps"] == {"rounds": 2, "edits": 8}
    assert service.scenario(created["episode_id"])["jobs"] == template["jobs"]


def test_malformed_requests_answer_400_and_unknown_episodes_404(service):
    bodies = [
        {"source": {"kind": "seed", "seed": 1}, "agents": []},
        {"source": {"kind": "seed", "seed": 1}, "agents": [{"agent": "oracle"}]},
        {"source": {"kind": "seed", "seed": 1}, "agents": [{"agent": "model"}]},
        {"source": {"kind": "seed", "seed": 1}, "agents": [{"agent": "teacher"}, {"agent": "teacher"}]},
        {"source": {"kind": "seed", "seed": "1"}, "agents": [{"agent": "teacher"}]},
        {"source": {"kind": "seed", "seed": 1}, "caps": {"rounds": 11}, "agents": [{"agent": "teacher"}]},
        {"source": {"kind": "seed", "seed": 1}, "caps": {"edits": 0}, "agents": [{"agent": "teacher"}]},
        {"source": {"kind": "scenario", "jobs": [], "vehicles": [], "depot": None}, "agents": [{"agent": "teacher"}]},
        {"source": {"kind": "scenario", "template": "nope", "jobs": [{}], "vehicles": [{}], "depot": [0, 0]}, "agents": [{"agent": "teacher"}]},
        {"source": {"kind": "replay"}, "agents": [{"agent": "teacher"}]},
    ]

    for body in bodies:
        with pytest.raises(ApiError) as raised:
            service.create(body)
        assert raised.value.status == 400, body

    for call in (lambda: service.status("missing"), lambda: service.cancel("../x"), lambda: service.add_agents("missing", {"agents": [{"agent": "teacher"}]})):
        with pytest.raises(ApiError) as raised:
            call()
        assert raised.value.status == 404


def test_real_recorder_runs_through_the_service_and_the_trace_store_reads_live_and_stored(paths, cpu_config, seeded, service_vroom):
    runtime = ConsoleRuntime(paths)
    catalog = StubCatalog()
    runs    = RunBrowser(paths, RunLayout(paths), catalog)
    service = EpisodeService(paths, runtime, catalog, runs, ScenarioTemplates(), CapturingLogger())
    store   = TraceStore(runtime.layout, FakeGeometry(), service)

    created = service.create({"source": {"kind": "seed", "seed": 4}, "caps": {"rounds": 2, "edits": 2}, "agents": [{"agent": "insertion_only"}, {"agent": "do_nothing"}]})
    wait_for(lambda: finished(service, created["episode_id"]), limit=20)

    status = service.status(created["episode_id"])
    assert {lane["status"] for lane in status["lanes"].values()} == {"finished"}
    assert len(status["shared"]["gaps"]) == 2
    assert status["divergence"] == [] or all(set(entry) == {"round", "kind", "agents"} for entry in status["divergence"])

    trace = store.read(TraceRef("ep", "insertion_only", episode_id=created["episode_id"]))
    assert trace["meta"]["complete"] is True
    assert trace["summary"]["total_reward"] == pytest.approx(status["lanes"]["insertion_only"]["total_reward"])
    assert trace["meta"]["caps"] == {"rounds": 2, "edits": 2}

    recorder     = EpisodeRecorder(runtime.episode_config(rounds=2, edits=2), Environment(runtime.episode_config(rounds=2, edits=2)))
    live_service = type("Live", (), {"live_trace": lambda self, episode_id, agent_key: (recorder, recorder.guard), "trace_path": service.trace_path})()
    recorder.record(service._build_agent({"agent": "do_nothing"}, None), {"agent_key": "do_nothing", "agent": "do_nothing", "run_name": None, "greedy": True}, {"kind": "seed", "seed": 4})

    live = TraceStore(runtime.layout, FakeGeometry(), live_service).read(TraceRef("ep", "do_nothing", episode_id="live"), from_seq=1)
    assert live["frames"][0]["seq"] == 1
    assert "snapshots" not in live
