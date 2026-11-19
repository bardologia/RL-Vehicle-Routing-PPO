"""Tests for the scope of the emergency stop.

The nuke and the throttle act on the console's jobs, their descendants and the
project's entry scripts only: an unrelated process of the same user, the test
runner and its ancestors are never targets, and a nuke clears the queue,
stops the targets and leaves the outsider running.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from process_manager import ProcessNuke
from web_logger      import WebLogger

from tests.webui.conftest import SLEEP_LONG, job_record, wait_for_status

SLEEPER = "import time\ntime.sleep(60)\n"
SPAWNER = "import subprocess, sys, time\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\ntime.sleep(60)\n"


@pytest.fixture
def manager(make_manager):
    """Returns a process manager with a long sleeper and a script that spawns a grandchild."""
    return make_manager({"sleep_long": SLEEP_LONG, "spawner": SPAWNER})


@pytest.fixture
def spawned():
    """Yields a list of Popen handles killed and reaped at teardown."""
    procs = []
    yield procs
    for proc in procs:
        proc.kill()
        proc.wait()


def _children(pid: int) -> list[int]:
    """Returns the direct children of a pid read from /proc."""
    found = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            raw = open(f"/proc/{entry}/stat").read()
        except OSError:
            continue
        if int(raw[raw.rindex(")") + 2:].split()[1]) == pid:
            found.append(int(entry))
    return found


def test_an_unrelated_process_is_never_a_target(manager, spawned):
    """A sleeping Python process of this user that is neither a job nor an entry script is not targeted."""
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    spawned.append(outsider)
    time.sleep(0.3)

    targets = ProcessNuke(manager, WebLogger()).targets()

    assert outsider.pid not in targets
    assert os.getpid() not in targets
    assert os.getppid() not in targets


def test_a_project_entry_script_is_a_target(manager, spawned, tmp_path):
    """A process running a script under main/ is targeted although the console never launched it."""
    script = tmp_path / "main" / "loose_run.py"
    script.write_text(SLEEPER)
    process = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
    spawned.append(process)
    time.sleep(0.3)

    assert process.pid in ProcessNuke(manager, WebLogger()).targets()


def test_a_running_job_and_its_descendants_are_targets(manager):
    """A console job and the grandchild it spawned are both targeted."""
    job_id = manager.launch("spawner", sys.executable)["job_id"]
    pid    = job_record(manager, job_id)["pid"]

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not _children(pid):
        time.sleep(0.05)

    targets = ProcessNuke(manager, WebLogger()).targets()

    assert pid in targets
    assert set(_children(pid)) <= set(targets)


def test_the_nuke_stops_the_jobs_clears_the_queue_and_spares_the_outsider(manager, spawned):
    """A nuke cancels the queued job, stops the running one and leaves an unrelated process alive."""
    outsider = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    spawned.append(outsider)

    running = manager.launch("sleep_long", sys.executable)["job_id"]
    queued  = manager.enqueue("sleep_long", sys.executable)["job_id"]

    result = ProcessNuke(manager, WebLogger()).nuke(grace=2.0)

    assert result["ok"] is True
    assert result["cancelled"] == 1
    assert result["signalled"] >= 1
    assert job_record(manager, queued)["status"] == "cancelled"
    assert wait_for_status(manager, running, "failed")
    assert job_record(manager, running)["stopped"] is True
    assert outsider.poll() is None
