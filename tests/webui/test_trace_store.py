import polyline
import pytest
import requests

from console_runtime import ConsoleRuntime
from core.inference import EpisodeRecorder, FixedOperatorAgent, TeacherAgent, TraceFile
from core.shared import Environment
from core.training.pretraining import RegretInsertionTeacher
from episode_service import VerdictService
from geometry_cache import GeometryCache
from project_paths import RunLayout
from trace_store import TraceRef, TraceStore
from web_errors import ApiError
from tests.webui.conftest import FakeGeometry


META = {"agent_key": "insertion_only", "agent": "insertion_only", "run_name": None, "greedy": True}


class NoEpisodes:
    def __init__(self, root):
        self.root = root

    def live_trace(self, episode_id, agent_key):
        return None

    def trace_path(self, episode_id, agent_key):
        return self.root / episode_id / f"{agent_key}.json.gz"


class FakeOsrmResponse:
    def __init__(self, status_code, points):
        self.status_code = status_code
        self.points      = points

    def json(self):
        return {"routes": [{"geometry": polyline.encode([(lat, lon) for lon, lat in self.points]), "distance": 1234.4}]}


class FakeOsrm:
    def __init__(self, status_code=200, fail=False):
        self.status_code = status_code
        self.fail        = fail
        self.urls        = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if self.fail:
            raise requests.ConnectionError("down")

        pairs = url.split("/driving/")[1].split("?")[0].split(";")
        return FakeOsrmResponse(self.status_code, [tuple(float(value) for value in pair.split(",")) for pair in pairs])


@pytest.fixture
def stored(paths, cpu_config, seeded, service_vroom):
    runtime = ConsoleRuntime(paths)
    config  = runtime.episode_config(rounds=3, edits=3)

    recorder = EpisodeRecorder(config, Environment(config))
    trace    = recorder.record(TeacherAgent(RegretInsertionTeacher(config, allow_removal=False)), META, {"kind": "seed", "seed": 5})

    layout  = RunLayout(paths)
    run_dir = paths.run_dir("run")
    run_dir.mkdir()
    TraceFile().write(layout.trace_path(run_dir, 1000, 2, "insertion_only"), trace)

    geometry = FakeGeometry()
    store    = TraceStore(layout, geometry, NoEpisodes(paths.episodes_dir))
    return runtime, store, geometry, TraceRef("diag", "insertion_only", run="run", seed=1000, index=2), trace


def test_read_adds_paths_distances_findings_and_hides_snapshots(stored):
    runtime, store, geometry, ref, raw = stored
    trace                              = store.read(ref)

    assert set(trace) == {"meta", "depot", "entities", "paths", "frames", "positions", "summary", "expectation", "findings"}
    assert len(trace["frames"]) == len(raw["frames"])
    assert [finding["id"] for finding in trace["findings"]] == [f"E{number}" for number in range(1, 10)]

    for frame in trace["frames"]:
        total = 0
        for route in frame["plan"]["routes"]:
            assert "path" not in route
            assert len(trace["paths"][route["path_id"]]) == len(route["stops"]) + 1
            assert route["distance_m"] == 1000 * len(route["stops"])
            assert route["kpis"]["detour"] > 0
            total += route["distance_m"]

        assert frame["plan"]["kpis"]["distance_m"] == total


def test_from_returns_only_later_frames_and_their_paths(stored):
    runtime, store, geometry, ref, raw = stored

    full    = store.read(ref)
    partial = store.read(ref, from_seq=3)

    assert [frame["seq"] for frame in partial["frames"]] == [frame["seq"] for frame in full["frames"][3:]]
    assert partial["positions"] == full["positions"]
    assert set(partial["paths"]) == {route["path_id"] for frame in partial["frames"] for route in frame["plan"]["routes"]}


def test_positions_resolve_rounds_edits_and_sequence_numbers(stored):
    runtime, store, geometry, ref, raw = stored
    frames                             = raw["frames"]

    assert store.seq_of(frames, "r0") == 0
    assert store.seq_of(frames, "r0e0") == 1
    assert store.seq_of(frames, 4) == 4
    assert frames[store.seq_of(frames, "r2")]["kind"] == "round_open"
    assert frames[store.seq_of(frames, "r1e0")]["round"] == 1
    assert set(store.positions(frames).values()) == set(range(len(frames)))

    for position, status in (("r9", 404), ("r0e9", 404), ("999", 404), ("x1", 400), ("r1e", 400)):
        with pytest.raises(ApiError) as raised:
            store.seq_of(frames, position)
        assert raised.value.status == status


def test_missing_trace_and_run_answer_404(stored):
    runtime, store, geometry, ref, raw = stored

    for missing in (TraceRef("diag", "model", run="run", seed=1000, index=2), TraceRef("diag", "insertion_only", run="nope", seed=1000, index=2), TraceRef("ep", "model", episode_id="abc")):
        with pytest.raises(ApiError) as raised:
            store.read(missing)
        assert raised.value.status == 404

    with pytest.raises(ApiError) as raised:
        TraceRef("diag", "../x", run="run", seed=1, index=0)
    assert raised.value.status == 400


def test_verdict_is_computed_once_then_served_from_the_file(stored, service_vroom):
    runtime, store, geometry, ref, raw = stored
    service                            = VerdictService(runtime, store)
    seq                                = next(frame["seq"] for frame in raw["frames"] if frame["kind"] == "decision")

    first = service.verdict(ref, {"seq": seq, "scope": "insertions"})
    calls = service_vroom.calls

    assert first["scope"] == "insertions"
    assert first["removals"] is None
    assert first["match"]["operator"] is True
    assert store.load(ref)["frames"][seq]["decision"]["verdict"] == first
    assert store.read(ref)["frames"][seq]["decision"]["verdict"] == first

    assert service.verdict(ref, {"seq": seq, "scope": "insertions"}) == first
    assert service_vroom.calls == calls

    full = service.verdict(ref, {"seq": seq, "scope": "full"})
    assert full["scope"] == "full"
    assert full["removals"] is not None
    assert service.verdict(ref, {"seq": seq, "scope": "insertions"}) == full


def test_verdict_rejects_bad_requests(stored):
    runtime, store, geometry, ref, raw = stored
    service                            = VerdictService(runtime, store)

    for body, status in (({"seq": 0, "scope": "insertions"}, 400), ({"seq": 1, "scope": "everything"}, 400), ({"seq": "1"}, 400), ({"seq": 999}, 404)):
        with pytest.raises(ApiError) as raised:
            service.verdict(ref, body)
        assert raised.value.status == status

    service.busy.acquire()
    with pytest.raises(ApiError) as raised:
        service.verdict(ref, {"seq": 1, "scope": "insertions"})
    assert raised.value.status == 409


def test_geometry_cache_asks_osrm_once_per_stop_sequence():
    osrm  = FakeOsrm()
    cache = GeometryCache("http://osrm.test/", osrm)

    points = [[-46.6, -23.5], [-46.61, -23.51], [-46.62, -23.52]]
    first  = cache.route(points)
    again  = cache.route([list(point) for point in points])

    assert len(osrm.urls) == 1
    assert osrm.urls[0].startswith("http://osrm.test/route/v1/driving/-46.6,-23.5;-46.61,-23.51;-46.62,-23.52?")
    assert again is first
    assert first["distance_m"] == 1234
    assert first["path"] == [[pytest.approx(lat, abs=1e-5), pytest.approx(lon, abs=1e-5)] for lon, lat in points]

    assert cache.route([[-46.6, -23.5]]) == {"path": [[-23.5, -46.6]], "distance_m": 0}
    cache.route(points[:2])
    assert len(osrm.urls) == 2


def test_geometry_cache_answers_503_when_osrm_is_down():
    for osrm in (FakeOsrm(fail=True), FakeOsrm(status_code=400)):
        with pytest.raises(ApiError) as raised:
            GeometryCache("http://osrm.test", osrm).route([[0, 0], [1, 1]])
        assert raised.value.status == 503
