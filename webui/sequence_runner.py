"""Execution engine that walks experiment sequences over the process manager.

Holds the live state of every started sequence, launches a block as soon as all
of its predecessor blocks have finished, launches independent blocks in
parallel, holds checkpoint blocks until they are explicitly released, and skips
the descendants of a failed block.

An isolated sequence gets a fresh workspace directory per execution under the
repository's ``runs/sequences`` tree; every block's ``io.runs_dir`` is redirected
into it when it points at the project's runs tree, so repeated executions never
collide and a block whose ``io.run_name``, ``io.init_from_run`` or
``io.resume_from_run`` names a predecessor's run reads exactly the run this
execution produced. A start may instead name the workspace of an earlier
execution to resume it. Every isolated execution mirrors its live state into a
JSON file inside the workspace whenever it changes; the in-memory state dies
with the server, but the file survives, and a resume scan reads it to report
which blocks already finished so the resumed start can bypass them.

A block whose skip toggle is set is bypassed instead of launched: it counts as
satisfied for its successors. The toggle is read fresh from the store on every
scan, so it can be flipped mid-run for any block that has not launched yet.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from web_logger import WebLogger


class SequenceRunner:
    """Advances started sequences by polling job states and launching ready blocks.

    Attributes:
        store     : Sequence store supplying the current definition of each sequence.
        processes : Process manager whose job table supplies the block job states.
        launcher  : Run launcher executing the blocks.
        logger    : Console logger.
        active    : Live sequence states keyed by sequence id.
        released  : Released checkpoint block ids per sequence id.
        persisted : Last state-file text written per sequence id, to skip unchanged writes.
        lock      : Guards the active table, the released sets, and the state-file staleness check and write.
    """

    INTERVAL_S     = 2.0
    ENDED_OK       = ("finished",)
    ENDED_BAD      = ("failed", "cancelled")
    STATE_FILENAME = "sequence_state.json"
    RUNS_FIELD     = "io.runs_dir"

    def __init__(self, store, processes, launcher, logger: WebLogger) -> None:
        """Stores the collaborators and starts with no active sequences."""
        self.store     = store
        self.processes = processes
        self.launcher  = launcher
        self.logger    = logger
        self.active    = {}
        self.released  = {}
        self.persisted = {}
        self.lock      = threading.Lock()

    def start_sequence(self, sequence_id: str, resume_dir: str = "", bypass: list | None = None) -> dict:
        """Starts one sequence, in a fresh workspace or resuming a previous one.

        Args:
            sequence_id : Identifier of the sequence to start.
            resume_dir  : Workspace directory of the execution to resume, or empty to start fresh.
            bypass      : Identifiers of blocks bypassed for this execution only, typically the ones a resume scan reported finished.

        Returns:
            Result with the fresh state, or ``ok`` False when the sequence is unknown, empty, already running, or the resume workspace is invalid.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}
        if not entry["blocks"]:
            return {"ok": False, "error": "the sequence has no blocks"}

        run_dir, error = self._resume_workspace(entry, resume_dir) if resume_dir else ("", "")
        if error:
            return {"ok": False, "error": error}

        with self.lock:
            state = self.active.get(sequence_id)
            if state is not None and state["status"] in ("running", "held"):
                return {"ok": False, "error": "the sequence is already running"}

            self.active[sequence_id] = {
                "sequence_id" : sequence_id,
                "status"      : "running",
                "started_at"  : time.strftime("%Y-%m-%dT%H:%M:%S"),
                "run_dir"     : run_dir or (self._make_workspace(entry) if entry["isolate"] else ""),
                "bypass"      : sorted(str(block_id) for block_id in (bypass or [])),
                "blocks"      : {block["block_id"]: self._fresh_block_state() for block in entry["blocks"]},
            }
            self.released[sequence_id] = set()

        self.logger.ok(f"started sequence '{entry['name']}' ({sequence_id}) with {len(entry['blocks'])} blocks" + (f", resuming workspace {run_dir}" if run_dir else ""))
        self.scan()
        return {"ok": True, "state": self.state(sequence_id)}

    def resume_scan(self, sequence_id: str, resume_dir: str) -> dict:
        """Reads the execution state a previous run recorded into its workspace.

        A block the recorded execution finished, or bypassed as part of its own
        resume list, is proposed for bypassing; a block bypassed by a manual skip
        toggle is not, since nothing shows it ever ran.

        Args:
            sequence_id : Identifier of the sequence being resumed.
            resume_dir  : Workspace directory of the execution to resume.

        Returns:
            Result with ``recorded`` False when the workspace holds no state file, or the recorded per-block statuses plus the ids of the blocks that finished and still exist in the sequence; ``ok`` False when the sequence is unknown or the workspace is invalid.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            return {"ok": False, "error": "sequence not found"}

        run_dir, error = self._resume_workspace(entry, resume_dir)
        if error:
            return {"ok": False, "error": error}

        path = Path(run_dir) / self.STATE_FILENAME
        if not path.is_file():
            return {"ok": True, "recorded": False, "run_dir": run_dir, "statuses": {}, "done": []}

        recorded = json.loads(path.read_text(encoding="utf-8"))
        statuses = {block_id: block["status"] for block_id, block in recorded["blocks"].items()}
        current  = {block["block_id"] for block in entry["blocks"]}
        resumed  = set(recorded["bypass"])
        done     = sorted(block_id for block_id, status in statuses.items() if (status == "done" or (status == "bypassed" and block_id in resumed)) and block_id in current)

        return {"ok": True, "recorded": True, "run_dir": run_dir, "statuses": statuses, "done": done}

    def release(self, sequence_id: str, block_id: str) -> dict:
        """Releases one held checkpoint block so the next scan launches it.

        Args:
            sequence_id : Identifier of the running sequence.
            block_id    : Identifier of the held block.

        Returns:
            Result with the updated state, or ``ok`` False when the sequence is not active or the block is not currently held.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None:
                return {"ok": False, "error": "sequence is not running"}

            block = state["blocks"].get(block_id)
            if block is None or block["status"] != "held":
                return {"ok": False, "error": "block is not held at a checkpoint"}

            self.released[sequence_id].add(block_id)

        self.logger.ok(f"released checkpoint block {block_id} of sequence {sequence_id}")
        self.scan()
        return {"ok": True, "state": self.state(sequence_id)}

    def stop_sequence(self, sequence_id: str) -> dict:
        """Stops one sequence, stopping its running blocks and skipping the rest.

        Args:
            sequence_id : Identifier of the sequence to stop.

        Returns:
            Result with the stopped state, or ``ok`` False when it is not active.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None or state["status"] not in ("running", "held"):
                return {"ok": False, "error": "sequence is not running"}

            state["status"] = "stopped"
            running = []
            for block in state["blocks"].values():
                if block["status"] in ("running", "launching"):
                    if block["job_id"]:
                        running.append(block["job_id"])
                    block["status"] = "cancelled"
                elif block["status"] in ("pending", "held"):
                    block["status"] = "skipped"

        for job_id in running:
            self.processes.stop(job_id)

        self._persist_states()
        self.logger.warning(f"stopped sequence {sequence_id}, {len(running)} running block(s) stopped")
        return {"ok": True, "state": self.state(sequence_id)}

    def clear_state(self, sequence_id: str) -> dict:
        """Discards one sequence's finished execution state so the board returns to idle.

        Args:
            sequence_id : Identifier of the sequence whose state is discarded.

        Returns:
            Result with ``cleared`` True when a state was dropped, False when there was none, or ``ok`` False while the sequence is still active.
        """
        with self.lock:
            state = self.active.get(sequence_id)
            if state is not None and state["status"] in ("running", "held"):
                return {"ok": False, "error": "stop the sequence before clearing its state"}

            cleared = self.active.pop(sequence_id, None) is not None
            self.released.pop(sequence_id, None)
            self.persisted.pop(sequence_id, None)

        if cleared:
            self.logger.muted(f"cleared execution state of sequence {sequence_id}")
        return {"ok": True, "cleared": cleared}

    def state(self, sequence_id: str) -> dict | None:
        """Returns a snapshot of one sequence's live state, or None when never started."""
        with self.lock:
            state = self.active.get(sequence_id)
            if state is None:
                return None
            return {**state, "blocks": {block_id: dict(block) for block_id, block in state["blocks"].items()}}

    def states(self) -> dict:
        """Returns a snapshot of every live sequence state keyed by sequence id."""
        with self.lock:
            return {sequence_id: {**state, "blocks": {block_id: dict(block) for block_id, block in state["blocks"].items()}} for sequence_id, state in self.active.items()}

    @staticmethod
    def _fresh_block_state() -> dict:
        """Returns the initial live state of one block."""
        return {"status": "pending", "job_id": None, "error": ""}

    def _resume_workspace(self, entry: dict, resume_dir: str) -> tuple:
        """Validates the workspace of a previous execution against the sequences tree.

        Returns:
            Pair of the normalised workspace path and an error message; exactly one of the two is non-empty.
        """
        if not entry["isolate"]:
            return "", "the sequence does not isolate its executions, so there is no workspace to resume; turn 'Isolated dir' on"

        workspace = Path(resume_dir).expanduser()
        tree      = self.store.paths.runs_dir / "sequences"
        if workspace.parent != tree:
            return "", f"the resume workspace must be a directory directly under {tree}"
        if not workspace.is_dir():
            return "", f"the resume workspace does not exist: {workspace}"

        return str(workspace), ""

    def _make_workspace(self, entry: dict) -> str:
        """Creates and returns a fresh per-execution run directory for an isolated sequence.

        The directory lives under ``<repo_root>/runs/sequences`` and is named after
        the sequence and a start stamp; a numeric suffix keeps it unique when the
        same sequence starts twice within one second.
        """
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", entry["name"]).strip("_").lower() or "sequence"
        base = self.store.paths.runs_dir / "sequences" / f"{slug}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

        candidate = base
        counter   = 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}_{counter}")
            counter  += 1

        candidate.mkdir(parents=True)
        self.logger.ok(f"sequence workspace {candidate}")
        return str(candidate)

    def _job_status(self, job_id: str) -> str:
        """Returns the process manager's status for one job, or "failed" when it is gone."""
        record = self.processes.job_record(job_id)
        return record["status"] if record else "failed"

    def _launch_overrides(self, block: dict, run_dir: str) -> dict:
        """Returns the block's overrides with the run-output root redirected into the workspace.

        The redirect applies only while the block leaves ``io.runs_dir`` at the
        project's runs tree, so a deliberately external directory is left alone.

        Args:
            block   : Block definition supplying the stored overrides.
            run_dir : Isolated workspace directory of this execution, or empty.
        """
        overrides = dict(block["overrides"])
        current   = overrides.get(self.RUNS_FIELD, "runs")

        if run_dir and current in ("runs", "runs/", str(self.store.paths.runs_dir)):
            overrides[self.RUNS_FIELD] = run_dir

        return overrides

    def _launch_block(self, entry: dict, block: dict, run_dir: str) -> None:
        """Launches one ready block outside the runner lock and records the outcome.

        Args:
            entry   : Sequence definition the block belongs to.
            block   : Block definition to launch, read fresh from the store.
            run_dir : Isolated workspace directory of this execution, or empty.
        """
        interpreter = block["interpreter"] or self.launcher.preferred_interpreter(block["script"])
        result      = self.launcher.execute(block["script"], interpreter, self._launch_overrides(block, run_dir), None, entry["detach"], False)

        if not result.get("ok"):
            self._record_launch(entry, block, "", result.get("error", "launch failed"))
            return

        self._record_launch(entry, block, result["job_id"], "")

    def _record_launch(self, entry: dict, block: dict, job_id: str, error: str) -> None:
        """Writes a finished launch back into the live state, stopping the job when the block was stopped meanwhile.

        Args:
            entry  : Sequence definition the block belongs to.
            block  : Block definition that was launched.
            job_id : Console job id of the started job, or empty on failure.
            error  : Failure description, or empty when the launch succeeded.
        """
        with self.lock:
            state       = self.active.get(entry["sequence_id"])
            block_state = state["blocks"].get(block["block_id"]) if state else None
            stale       = block_state is None or block_state["status"] != "launching"

            if not stale and error:
                block_state["status"] = "failed"
                block_state["error"]  = error
            elif not stale:
                block_state["status"] = "running"
                block_state["job_id"] = job_id

        if stale:
            if job_id:
                self.processes.stop(job_id)
            self.logger.warning(f"sequence {entry['sequence_id']}: block '{block['title']}' was stopped while launching, {'job ' + job_id + ' stopped' if job_id else 'no job started'}")
        elif error:
            self.logger.error(f"sequence {entry['sequence_id']}: block '{block['title']}' failed to launch: {error}")
        else:
            self.logger.ok(f"sequence {entry['sequence_id']}: launched block '{block['title']}' as job {job_id}")

    def _advance(self, sequence_id: str, state: dict) -> list:
        """Advances one running sequence: syncs block jobs and marks ready blocks for launch.

        A block that is ready to run is marked "launching" and returned instead
        of being launched here, so the caller can launch it with the runner lock
        released. A block marked skip, or bypassed for this execution, turns
        bypassed only once its own predecessors are satisfied.

        Args:
            sequence_id : Identifier of the sequence being advanced.
            state       : Its live state, mutated in place under the runner lock.

        Returns:
            List of (entry, block, run_dir) tuples to launch outside the lock.
        """
        entry = self.store.get(sequence_id)
        if entry is None:
            state["status"] = "stopped"
            return []

        blocks = {block["block_id"]: block for block in entry["blocks"]}
        for block_id in [block_id for block_id in state["blocks"] if block_id not in blocks]:
            state["blocks"].pop(block_id)
        for block_id in blocks:
            state["blocks"].setdefault(block_id, self._fresh_block_state())

        for block_id, block_state in state["blocks"].items():
            if block_state["status"] != "running":
                continue
            status = self._job_status(block_state["job_id"])
            if status in self.ENDED_OK:
                block_state["status"] = "done"
            elif status in self.ENDED_BAD:
                block_state["status"] = "failed"
                block_state["error"]  = f"job {block_state['job_id']} ended as {status}"

        launches = []
        changed  = True
        while changed:
            changed = False
            for block_id, block in blocks.items():
                block_state = state["blocks"][block_id]
                if block_state["status"] not in ("pending", "held"):
                    continue

                pred_states = [state["blocks"][pred]["status"] for pred in block["predecessors"]]
                if any(pred in ("failed", "skipped", "cancelled") for pred in pred_states):
                    block_state["status"] = "skipped"
                    changed = True
                    continue

                if any(pred not in ("done", "bypassed") for pred in pred_states):
                    continue

                if block["skip"] or block_id in state["bypass"]:
                    block_state["status"] = "bypassed"
                    changed = True
                    continue

                if block["checkpoint"] and block_id not in self.released[sequence_id]:
                    block_state["status"] = "held"
                    continue

                block_state["status"] = "launching"
                launches.append((entry, block, state["run_dir"]))
                changed = True

        statuses = [block_state["status"] for block_state in state["blocks"].values()]
        if any(status in ("running", "launching") for status in statuses):
            state["status"] = "running"
        elif any(status == "held" for status in statuses):
            state["status"] = "held"
        elif any(status == "pending" for status in statuses):
            state["status"] = "running"
        elif any(status in ("failed", "skipped", "cancelled") for status in statuses):
            state["status"] = "failed"
        else:
            state["status"] = "done"

        return launches

    def _persist_states(self) -> None:
        """Mirrors every isolated sequence's live state into its workspace when it changed, through a staging file."""
        with self.lock:
            for sequence_id, state in self.active.items():
                if not state["run_dir"]:
                    continue

                snapshot = {**state, "blocks": {block_id: dict(block) for block_id, block in state["blocks"].items()}}
                text     = json.dumps(snapshot, indent=2) + "\n"
                if self.persisted.get(sequence_id) == text:
                    continue

                path    = Path(state["run_dir"]) / self.STATE_FILENAME
                staging = path.with_suffix(".tmp")
                staging.write_text(text, encoding="utf-8")
                staging.replace(path)
                self.persisted[sequence_id] = text

    def scan(self) -> None:
        """Runs one advance pass over every running or held sequence, then launches what it marked."""
        with self.lock:
            launches = []
            for sequence_id, state in self.active.items():
                if state["status"] in ("running", "held"):
                    launches += self._advance(sequence_id, state)

        for entry, block, run_dir in launches:
            self._launch_block(entry, block, run_dir)

        self._persist_states()

    def start(self) -> None:
        """Starts the background polling thread that advances sequences forever."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()

    def _watch(self) -> None:
        """Scans on the fixed interval forever, logging any scan failure instead of exiting."""
        while True:
            time.sleep(self.INTERVAL_S)
            try:
                self.scan()
            except Exception as exc:
                self.logger.error(f"sequence runner error: {exc}")
