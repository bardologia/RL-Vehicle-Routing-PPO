"""Relaunch of an entry point as a hangup-immune background process with a recorded exit code.

When --detach is present the script re-execs itself in a new session with its
output redirected to a timestamped log, so long runs survive the terminal or the
console that started them. The relaunch goes through DetachedRunner, which runs
the script as its own child and writes the script's exit code to a marker file
beside the log once it ends, so whoever follows the run later can tell a clean
finish from a crash.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib  import Path


class Detacher:
    """Re-execs the running script detached from its controlling terminal.

    Attributes:
        log_dir  : Directory the detached process's output log and exit marker are written to.
        ENV_FLAG : Environment variable marking an already-detached process.
        FLAGS    : Command-line flags that request detachment.
    """

    ENV_FLAG      = "ROUTING_DETACHED"
    FLAGS         = ("--detach", "--nohup")
    MARKER_SUFFIX = ".exit"

    def __init__(self, log_dir: str = "logs") -> None:
        """Initializes the detacher with the directory that receives output logs."""
        self.log_dir = Path(log_dir)

    def requested(self, argv: list[str] | None = None) -> bool:
        """Returns whether a detach flag is present in the given or process arguments."""
        argv = sys.argv[1:] if argv is None else argv
        return any(flag in argv for flag in self.FLAGS)

    def active(self) -> bool:
        """Returns whether this process is already the detached child."""
        return os.environ.get(self.ENV_FLAG) == "1"

    def log_path(self, script: str, stamp: str) -> Path:
        """Returns the output log of a detached run of `script` started at `stamp`."""
        return self.log_dir / f"{script}_{stamp}.out"

    def marker_path(self, log_path: Path) -> Path:
        """Returns the exit marker written beside an output log."""
        return log_path.with_suffix(self.MARKER_SUFFIX)

    def ensure(self) -> None:
        """Detaches the process when requested, or arms the child against hangups.

        In the detached child SIGHUP is ignored and control returns. In the
        parent the script is relaunched through DetachedRunner in a new session
        with output appended to logs/<script>_<stamp>.out, the runner pid and
        the log path are printed for the console to pick up, and the parent
        exits.

        Raises:
            SystemExit : In the parent, with status 0, once the runner is running.
        """
        if self.active():
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            return

        if not self.requested():
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        name     = Path(sys.argv[0]).stem
        log_path = self.log_path(name, datetime.now().strftime("%Y%m%d_%H%M%S"))
        marker   = self.marker_path(log_path)

        env                     = dict(os.environ)
        env[self.ENV_FLAG]      = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("FORCE_COLOR", "1")
        env.setdefault("COLUMNS", "120")

        with open(log_path, "ab") as sink:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), str(marker.resolve()), sys.executable, "-u", *sys.argv],
                cwd               = os.getcwd(),
                stdout            = sink,
                stderr            = subprocess.STDOUT,
                stdin             = subprocess.DEVNULL,
                env               = env,
                start_new_session = True,
            )

        print(f"detached {name} as pid {process.pid}, immune to hangup")
        print(f"output: {log_path}")
        print(f"follow: tail -f {log_path}")

        raise SystemExit(0)


class DetachedRunner:
    """Runs a detached script as its own child and records the child's exit code.

    The runner is the session leader the console tracks. It ignores SIGHUP,
    forwards SIGTERM and SIGINT to the script, waits for it, writes the exit
    code to the marker file (negative for a death by signal) and exits with the
    same code.

    Attributes:
        marker : File the exit code is written to.
        argv   : Command line of the script to run.
        child  : The running script, None before it is started.
    """

    FORWARDED = (signal.SIGTERM, signal.SIGINT)

    def __init__(self, marker: str, argv: list[str]) -> None:
        """Stores the marker path and the command line of the script."""
        self.marker = Path(marker)
        self.argv   = list(argv)
        self.child  = None

    def _forward(self, signum: int, frame) -> None:
        """Passes a termination signal on to the script."""
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signum)

    def _arm(self) -> None:
        """Ignores hangups and forwards termination signals to the script."""
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        for signum in self.FORWARDED:
            signal.signal(signum, self._forward)

    def _wait(self) -> int:
        """Starts the script, waits for it and returns its exit code."""
        self.child = subprocess.Popen(self.argv)

        while True:
            try:
                return self.child.wait()
            except InterruptedError:
                continue

    def _record(self, code: int) -> None:
        """Writes the exit code to the marker file."""
        self.marker.write_text(f"{code}\n")

    def run(self) -> None:
        """Arms the signals, runs the script, records its exit code and exits with it.

        Raises:
            SystemExit : Always, with the script's exit code clipped to the 0-255 range of a process status.
        """
        self._arm()
        code = self._wait()
        self._record(code)

        raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    DetachedRunner(sys.argv[1], sys.argv[2:]).run()
