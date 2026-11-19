"""Tests for the sequence runner over a real process manager running throwaway scripts.

Covers the launch of ready blocks in dependency order, independent blocks in
parallel, the redirect of ``io.runs_dir`` into the isolated workspace, the
checkpoint hold and its release, the skip and bypass marks, the skipping of
descendants of a failed block, stopping a running sequence, the state file a
resume scan reads back, and the refusals of an empty or already running
sequence.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from run_launcher    import RunLauncher
from sequence_runner import SequenceRunner
from sequence_store  import SequenceStore
from web_logger      import WebLogger

from tests.webui.conftest import SLEEP_LONG, decision_engine

QUICK   = "import pathlib, sys\nargs = sys.argv[1:]\nname = pathlib.Path(__file__).stem\npathlib.Path('order.txt').open('a').write(name + ' ' + ' '.join(args) + '\\n')\n"
FAILING = "import sys\nsys.exit(2)\n"


class LocalLauncher(RunLauncher):
    """Run launcher accepting the running interpreter without interpreter discovery."""

    def interpreter_error(self, interpreter: str) -> str:
        """Accepts every interpreter."""
        return ""

    def preferred_interpreter(self, script_key: str = "") -> str:
        """Returns the interpreter running the tests."""
        return sys.executable


@pytest.fixture
def setup(make_manager, tmp_path):
    """Returns a store and runner over a manager whose scripts record their order and arguments."""
    manager = make_manager({"first": QUICK, "second": QUICK, "third": QUICK, "failing": FAILING, "sleep_long": SLEEP_LONG})
    store   = SequenceStore(manager.paths, WebLogger(), decision_engine())
    runner  = SequenceRunner(store, manager, LocalLauncher(manager.paths, WebLogger(), manager), WebLogger())
    return store, runner, manager


def _sequence(store: SequenceStore, blocks: list, isolate: bool = False, detach: bool = False) -> str:
    """Creates and saves a sequence with the given blocks and returns its id."""
    sequence_id = store.create("test")["entry"]["sequence_id"]
    result      = store.save(sequence_id, {"name": "test", "detach": detach, "isolate": isolate, "blocks": blocks})
    assert result["ok"], result.get("error")
    return sequence_id


def _drive(runner: SequenceRunner, sequence_id: str, until: tuple = ("done", "failed", "stopped", "held"), timeout: float = 20.0) -> dict:
    """Scans until the sequence reaches one of the given statuses and returns its state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        runner.scan()
        state = runner.state(sequence_id)
        if state["status"] in until:
            return state
        time.sleep(0.1)
    return runner.state(sequence_id)


def _order(tmp_path: Path) -> list[str]:
    """Returns the lines the quick scripts recorded, in execution order."""
    path = tmp_path / "order.txt"
    return path.read_text().splitlines() if path.exists() else []


def test_blocks_run_after_their_predecessors(setup, tmp_path):
    """A chain runs strictly in dependency order and the sequence ends done."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "c", "script": "third", "predecessors": ["b"]}, {"block_id": "b", "script": "second", "predecessors": ["a"]}, {"block_id": "a", "script": "first"}])

    assert runner.start_sequence(sequence_id)["ok"]
    state = _drive(runner, sequence_id)

    assert state["status"] == "done"
    assert [line.split()[0] for line in _order(tmp_path)] == ["first", "second", "third"]


def test_an_isolated_sequence_redirects_the_runs_directory(setup, tmp_path):
    """Every block of an isolated sequence writes into the execution's own workspace under runs/sequences."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "first", "overrides": {"io.run_name": "r1"}}], isolate=True)

    runner.start_sequence(sequence_id)
    state = _drive(runner, sequence_id)
    line  = _order(tmp_path)[0]

    assert Path(state["run_dir"]).parent == tmp_path / "runs" / "sequences"
    assert f"--io.runs_dir {state['run_dir']}" in line and "--io.run_name r1" in line
    assert (Path(state["run_dir"]) / SequenceRunner.STATE_FILENAME).is_file()


def test_an_external_runs_directory_is_left_alone(setup, tmp_path):
    """A block that names its own runs directory keeps it inside an isolated sequence."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "first", "overrides": {"io.runs_dir": "/data/elsewhere"}}], isolate=True)

    runner.start_sequence(sequence_id)
    _drive(runner, sequence_id)

    assert "--io.runs_dir /data/elsewhere" in _order(tmp_path)[0]


def test_a_checkpoint_holds_until_released(setup, tmp_path):
    """A checkpoint block waits held until released, then runs."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "first"}, {"block_id": "b", "script": "second", "predecessors": ["a"], "checkpoint": True}])

    runner.start_sequence(sequence_id)
    held = _drive(runner, sequence_id)

    assert held["status"] == "held" and held["blocks"]["b"]["status"] == "held"
    assert runner.release(sequence_id, "a")["ok"] is False
    assert runner.release(sequence_id, "b")["ok"] is True
    assert _drive(runner, sequence_id, until=("done", "failed"))["status"] == "done"


def test_skip_and_bypass_satisfy_successors_without_running(setup, tmp_path):
    """A skipped block and a block bypassed for this execution never run but let their successors start."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "first", "skip": True}, {"block_id": "b", "script": "second", "predecessors": ["a"]}, {"block_id": "c", "script": "third", "predecessors": ["b"]}])

    runner.start_sequence(sequence_id, bypass=["b"])
    state = _drive(runner, sequence_id)

    assert (state["blocks"]["a"]["status"], state["blocks"]["b"]["status"], state["blocks"]["c"]["status"]) == ("bypassed", "bypassed", "done")
    assert [line.split()[0] for line in _order(tmp_path)] == ["third"]


def test_a_failed_block_skips_its_descendants(setup):
    """The descendants of a failed block are skipped and the sequence ends failed."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "failing"}, {"block_id": "b", "script": "second", "predecessors": ["a"]}])

    runner.start_sequence(sequence_id)
    state = _drive(runner, sequence_id)

    assert state["status"] == "failed"
    assert state["blocks"]["a"]["status"] == "failed" and "ended as failed" in state["blocks"]["a"]["error"]
    assert state["blocks"]["b"]["status"] == "skipped"


def test_stopping_cancels_the_running_block_and_skips_the_rest(setup):
    """A stop cancels the running block's job and skips the pending blocks."""
    store, runner, manager = setup
    sequence_id            = _sequence(store, [{"block_id": "a", "script": "sleep_long"}, {"block_id": "b", "script": "second", "predecessors": ["a"]}])

    runner.start_sequence(sequence_id)
    job_id = runner.state(sequence_id)["blocks"]["a"]["job_id"]
    result = runner.stop_sequence(sequence_id)

    assert result["ok"] is True
    assert (result["state"]["blocks"]["a"]["status"], result["state"]["blocks"]["b"]["status"]) == ("cancelled", "skipped")
    assert manager.job_record(job_id)["stopped"] is True
    assert runner.clear_state(sequence_id) == {"ok": True, "cleared": True}


def test_a_resume_scan_reads_what_the_workspace_recorded(setup):
    """A resume scan of a finished workspace proposes its done blocks for bypassing."""
    store, runner, _ = setup
    sequence_id      = _sequence(store, [{"block_id": "a", "script": "first"}, {"block_id": "b", "script": "failing", "predecessors": ["a"]}], isolate=True)

    runner.start_sequence(sequence_id)
    state = _drive(runner, sequence_id)
    scan  = runner.resume_scan(sequence_id, state["run_dir"])

    assert scan["recorded"] is True
    assert scan["done"] == ["a"]
    assert scan["statuses"]["b"] == "failed"
    assert runner.resume_scan(sequence_id, "/tmp")["ok"] is False


def test_an_empty_or_running_sequence_cannot_start(setup):
    """An empty sequence and one already running are refused."""
    store, runner, _ = setup
    empty            = store.create("empty")["entry"]["sequence_id"]
    busy             = _sequence(store, [{"block_id": "a", "script": "sleep_long"}])

    assert runner.start_sequence(empty) == {"ok": False, "error": "the sequence has no blocks"}
    assert runner.start_sequence(busy)["ok"] is True
    assert runner.start_sequence(busy) == {"ok": False, "error": "the sequence is already running"}

    runner.stop_sequence(busy)
