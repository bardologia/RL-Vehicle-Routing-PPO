import codecs
import os
import queue
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime    import datetime


class JobStream:
    def __init__(self):
        self.buffer      = deque(maxlen=4000)
        self.subscribers = []
        self.lock        = threading.Lock()

    def publish(self, event):
        with self.lock:
            self.buffer.append(event)
            for subscriber in list(self.subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self):
        subscriber = queue.Queue(maxsize=8000)
        with self.lock:
            for event in self.buffer:
                subscriber.put_nowait(event)
            self.subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber):
        with self.lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)


class ProcessManager:
    OVERRIDE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

    def __init__(self, paths, catalog, logger):
        self.paths        = paths
        self.catalog      = catalog
        self.logger       = logger
        self.jobs         = {}
        self.streams      = {}
        self.launch_queue = deque()
        self.lock         = threading.Lock()

    def launch(self, key, overrides=None, queued=False):
        if not self.catalog.has_script(key):
            return {"ok": False, "error": f"unknown script '{key}'"}
        if not self.paths.script_path(key).exists():
            return {"ok": False, "error": f"script file missing for '{key}'"}

        try:
            cleaned = self._clean_overrides(key, overrides)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

        record = self._make_record(key, cleaned)
        stream = JobStream()

        with self.lock:
            self.jobs[record["job_id"]]    = record
            self.streams[record["job_id"]] = stream

        if queued:
            with self.lock:
                record["status"] = "queued"
                self.launch_queue.append(record["job_id"])
                position = len(self.launch_queue)

            stream.publish({"type": "status", "status": "queued", "position": position})
            self.logger.muted(f"queued {key} as job {record['job_id']} at position {position}")
            self._advance_queue()

            with self.lock:
                still_queued = record["status"] == "queued"
            return {"ok": True, "job_id": record["job_id"], "queued": still_queued}

        error = self._start(record, stream)
        if error is not None:
            with self.lock:
                self.jobs.pop(record["job_id"], None)
                self.streams.pop(record["job_id"], None)
            return {"ok": False, "error": error}

        return {"ok": True, "job_id": record["job_id"], "queued": False}

    def _clean_overrides(self, key, overrides):
        known   = self.catalog.known_paths(key)
        cleaned = {}

        for path, value in (overrides or {}).items():
            if not isinstance(path, str) or not self.OVERRIDE_NAME.match(path):
                raise ValueError(f"invalid override key '{path}'")
            if path not in known:
                raise ValueError(f"override '{path}' is not a form field of '{key}'")
            cleaned[path] = str(value)

        return cleaned

    def _make_record(self, key, overrides):
        return {
            "job_id"    : uuid.uuid4().hex[:12],
            "script"    : key,
            "title"     : self.catalog.SCRIPTS[key]["title"],
            "command"   : self._render_command(key, overrides),
            "overrides" : overrides,
            "status"    : "pending",
            "pid"       : None,
            "started"   : datetime.now().isoformat(timespec="seconds"),
            "ended"     : None,
            "exit_code" : None,
        }

    def _render_command(self, key, overrides):
        parts = ["python", "-u", f"main/{key}.py"]
        for path, value in overrides.items():
            parts += [f"--{path}", shlex.quote(value)]
        return " ".join(parts)

    def _runtime_env(self):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"]      = "1"
        env["COLUMNS"]          = "120"
        return env

    def _start(self, record, stream):
        argv = [self.paths.interpreter, "-u", str(self.paths.script_path(record["script"]))]
        for path, value in record["overrides"].items():
            argv += [f"--{path}", value]

        try:
            process = subprocess.Popen(
                argv,
                cwd               = str(self.paths.repo_root),
                stdout            = subprocess.PIPE,
                stderr            = subprocess.STDOUT,
                env               = self._runtime_env(),
                start_new_session = True,
            )
        except OSError as exc:
            return str(exc)

        with self.lock:
            record["status"]  = "running"
            record["pid"]     = process.pid
            record["started"] = datetime.now().isoformat(timespec="seconds")

        self.logger.ok(f"launched {record['script']} as job {record['job_id']} (pid {process.pid})")
        stream.publish({"type": "status", "status": "running", "pid": process.pid})

        worker = threading.Thread(target=self._pump, args=(record["job_id"], process, stream), daemon=True)
        worker.start()
        return None

    def _pump(self, job_id, process, stream):
        fd      = process.stdout.fileno()
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                stream.publish({"type": "chunk", "data": text})

        tail = decoder.decode(b"", final=True)
        if tail:
            stream.publish({"type": "chunk", "data": tail})

        process.wait()
        code = process.returncode

        with self.lock:
            record = self.jobs.get(job_id)
            if record is not None:
                record["status"]    = "finished" if code == 0 else "failed"
                record["exit_code"] = code
                record["ended"]     = datetime.now().isoformat(timespec="seconds")
                status              = record["status"]

        self.logger.muted(f"job {job_id} exited with code {code}")
        stream.publish({"type": "status", "status": status, "code": code})
        stream.publish({"type": "end"})

        self._advance_queue()

    def _advance_queue(self):
        while True:
            with self.lock:
                busy = any(record["status"] in ("pending", "running") for record in self.jobs.values())
                if busy or not self.launch_queue:
                    return

                job_id = self.launch_queue.popleft()
                record = self.jobs.get(job_id)
                stream = self.streams.get(job_id)
                if record is None or stream is None or record["status"] != "queued":
                    continue
                record["status"] = "pending"

            error = self._start(record, stream)
            if error is None:
                return

            with self.lock:
                record["status"] = "failed"
                record["ended"]  = datetime.now().isoformat(timespec="seconds")

            stream.publish({"type": "status", "status": "failed", "code": None})
            stream.publish({"type": "end"})
            self.logger.error(f"queued job {job_id} failed to start: {error}")

    def stop(self, job_id):
        with self.lock:
            record = self.jobs.get(job_id)
            stream = self.streams.get(job_id)
            if record is None:
                return {"ok": False, "error": "unknown job"}

            if record["status"] == "queued":
                if job_id in self.launch_queue:
                    self.launch_queue.remove(job_id)
                record["status"] = "cancelled"
                record["ended"]  = datetime.now().isoformat(timespec="seconds")
                cancelled        = True
            else:
                cancelled = False
                pid       = record["pid"]

        if cancelled:
            stream.publish({"type": "status", "status": "cancelled", "code": None})
            stream.publish({"type": "end"})
            self._advance_queue()
            return {"ok": True}

        if pid is None or record["status"] not in ("running", "pending"):
            return {"ok": False, "error": "job is not running"}

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            return {"ok": True}

        killer = threading.Thread(target=self._force_kill, args=(pid,), daemon=True)
        killer.start()
        self.logger.warning(f"stopping job {job_id} (pid {pid})")
        return {"ok": True}

    def _force_kill(self, pid):
        deadline = time.time() + 10.0
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    def get_stream(self, job_id):
        with self.lock:
            return self.streams.get(job_id)

    def list_jobs(self):
        with self.lock:
            records = [dict(record) for record in self.jobs.values()]

        records.sort(key=lambda record: record["started"], reverse=True)
        return records
