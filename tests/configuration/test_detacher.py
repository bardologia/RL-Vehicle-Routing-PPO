"""Tests for the detach flag of the entry scripts and the runner that records a detached run's exit code.

ConfigCli strips --detach and --nohup from the override pairs and hands
control to the detacher only after every override parsed; the detached runner
writes the script's exit code, negative for a death by signal, into the marker
beside the log and exits with the same status.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from configuration          import Config
from configuration.cli      import ConfigCli
from tools.runtime.detacher import Detacher

RUNNER = Path(__file__).resolve().parents[2] / "tools" / "runtime" / "detacher.py"


class SpyDetacher:
    """Detacher stand-in counting how often it was asked to detach.

    Attributes:
        calls : Number of ensure calls.
    """

    def __init__(self) -> None:
        """Starts with no calls."""
        self.calls = 0

    def ensure(self) -> None:
        """Records the call instead of relaunching the process."""
        self.calls += 1


def test_the_detach_flag_is_stripped_and_honoured_after_the_overrides():
    """--detach leaves the override pairs intact and triggers the detacher once the overrides applied."""
    spy    = SpyDetacher()
    config = ConfigCli(Config(), ["--env.scenario_seed", "7", "--detach"], spy).apply()

    assert config.env.scenario_seed == 7
    assert spy.calls == 1


def test_without_the_flag_nothing_detaches():
    """A launch without a detach flag never calls the detacher."""
    spy = SpyDetacher()
    ConfigCli(Config(), ["--nohup", "--env.scenario_seed", "3"], spy).apply()
    ConfigCli(Config(), ["--env.scenario_seed", "3"], spy).apply()

    assert spy.calls == 1


def test_a_bad_override_fails_before_detaching():
    """A malformed override raises in the foreground and the detacher is never reached."""
    spy = SpyDetacher()

    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--nope.field", "1", "--detach"], spy).apply()

    assert spy.calls == 0


def test_the_marker_sits_beside_the_log():
    """The exit marker shares the log's stem under the .exit suffix."""
    detacher = Detacher("logs")
    log      = detacher.log_path("train", "20261119_101010")

    assert log == Path("logs/train_20261119_101010.out")
    assert detacher.marker_path(log) == Path("logs/train_20261119_101010.exit")


def test_the_runner_records_a_failing_exit_code(tmp_path):
    """The runner writes the script's exit code to the marker and exits with it."""
    marker = tmp_path / "run.exit"
    done   = subprocess.run([sys.executable, str(RUNNER), str(marker), sys.executable, "-c", "import sys; sys.exit(4)"], timeout=30)

    assert done.returncode == 4
    assert marker.read_text() == "4\n"


def test_the_runner_records_a_clean_exit(tmp_path):
    """A clean script leaves a marker of 0."""
    marker = tmp_path / "run.exit"
    done   = subprocess.run([sys.executable, str(RUNNER), str(marker), sys.executable, "-c", "print('ok')"], capture_output=True, timeout=30)

    assert done.returncode == 0
    assert marker.read_text() == "0\n"


def test_a_terminated_run_is_recorded_as_killed_by_its_signal(tmp_path):
    """SIGTERM to the runner reaches the script, whose death by signal lands as a negative code."""
    marker = tmp_path / "run.exit"
    runner = subprocess.Popen([sys.executable, str(RUNNER), str(marker), sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    time.sleep(0.8)

    runner.send_signal(signal.SIGTERM)
    runner.wait(timeout=15)

    assert marker.read_text() == "-15\n"
    assert runner.returncode == 128 + 15
