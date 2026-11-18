"""Interactive episode recordings and on-demand teacher verdicts.

One worker thread records the queued lanes of every episode in order; requests return at once and poll for progress.
"""

import json
import queue
import threading
import time
import traceback
import uuid

from core.inference import DivergenceCheck, EpisodeRecorder, FixedOperatorAgent, ModelAgent, PlanMetrics, TeacherAgent, TraceFile, VerdictBuilder
from core.shared import Environment
from core.training.pretraining import RegretInsertionTeacher
from web_errors import ApiError


class EpisodeService:
    """Records agents on one identical episode each, as lanes of an interactive recording.

    Attributes:
        paths     : ProjectPaths; recordings live under `episodes_dir/<episode_id>/`.
        runtime   : ConsoleRuntime with the environment lock, episode configs and the model cache.
        catalog   : CheckpointCatalog used to refuse checkpoints that do not load.
        runs      : RunBrowser giving evaluation report metas.
        templates : ScenarioTemplates giving template expectations.
    """

    AGENTS     = ("model", "teacher", "insertion_only", "do_nothing")
    MAX_ROUNDS = 10
    MAX_EDITS  = 8
    ACTIVE     = ("queued", "running")

    def __init__(self, paths, runtime, catalog, runs, templates, logger, recorder_factory=None):
        self.paths            = paths
        self.runtime          = runtime
        self.catalog          = catalog
        self.runs             = runs
        self.templates        = templates
        self.logger           = logger
        self.recorder_factory = recorder_factory or self._build_recorder

        self.files      = TraceFile()
        self.divergence = DivergenceCheck()
        self.episodes   = {}
        self.state_lock = threading.RLock()
        self.pending    = queue.Queue()
        self.worker     = None

    def _episode_dir(self, episode_id):
        return self.paths.episodes_dir / episode_id

    def trace_path(self, episode_id, agent_key):
        """Return the file a lane's trace is stored in; the file may not exist."""
        return self._episode_dir(episode_id) / f"{agent_key}.json.gz"

    def _int(self, value, name, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ApiError(400, f"{name} must be an integer between {low} and {high}")

        return value

    def _caps(self, body, default_rounds):
        caps = body.get("caps") or {}
        return {
            "rounds" : self._int(caps.get("rounds", default_rounds), "caps.rounds", 1, self.MAX_ROUNDS),
            "edits"  : self._int(caps.get("edits", self.MAX_EDITS), "caps.edits", 1, self.MAX_EDITS),
        }

    def _seed_source(self, source, body):
        seed = self._int(source.get("seed"), "source.seed", 0, 2 ** 31 - 1)
        caps = self._caps(body, 5)

        return {"kind": "seed", "seed": seed}, caps, {"environment": None, "reward": None}

    def _evaluation_source(self, source):
        run    = str(source.get("run", ""))
        seed   = self._int(source.get("report_seed"), "source.report_seed", 0, 2 ** 31 - 1)
        meta   = self.runs.read_report(run, seed)["meta"]
        index  = self._int(source.get("index"), "source.index", 0, meta["episodes"] - 1)
        caps   = {"rounds": meta["caps"]["max_disruptions_per_episode"], "edits": meta["caps"]["max_edits_per_round"]}
        public = {"kind": "evaluation", "run": run, "report_seed": seed, "index": index, "seed": meta["seed"] + index}

        return public, caps, {"environment": meta["environment"], "reward": meta["reward"]}

    def _scenario_source(self, source, body):
        template    = source.get("template")
        expectation = None

        if template is not None:
            known = {entry["key"]: entry for entry in self.templates.catalog()}
            if template not in known:
                raise ApiError(400, f"unknown template '{template}'")
            expectation = known[template]["expectation"]

        if not source.get("jobs") or not source.get("vehicles") or not source.get("depot"):
            raise ApiError(400, "scenario needs jobs, vehicles and a depot")

        public = {
            "kind"               : "scenario",
            "name"               : source.get("name"),
            "template"           : template,
            "jobs"               : source["jobs"],
            "vehicles"           : source["vehicles"],
            "depot"              : source["depot"],
            "assignment"         : source.get("assignment"),
            "seed"               : self._int(source.get("seed", 0), "source.seed", 0, 2 ** 31 - 1),
            "initial_disruption" : bool(source.get("initial_disruption", False)),
            "expectation"        : expectation,
        }

        return public, self._caps(body, 2), {"environment": None, "reward": None}

    def _parse_source(self, body):
        source = body.get("source") or {}
        kind   = source.get("kind")

        if kind == "seed":
            return self._seed_source(source, body)
        if kind == "evaluation":
            return self._evaluation_source(source)
        if kind == "scenario":
            return self._scenario_source(source, body)

        raise ApiError(400, "source.kind must be one of seed, evaluation, scenario")

    def _parse_agents(self, body):
        specs = body.get("agents") or []
        if not specs:
            raise ApiError(400, "at least one agent is required")

        parsed = []
        for spec in specs:
            agent = spec.get("agent")
            if agent not in self.AGENTS:
                raise ApiError(400, f"unknown agent '{agent}'; expected one of {', '.join(self.AGENTS)}")

            greedy   = bool(spec.get("greedy", True))
            run_name = spec.get("run_name")

            if agent == "model":
                if not run_name or "~" in run_name:
                    raise ApiError(400, "agent 'model' needs a run_name")
                key = f"model~{run_name}~{'greedy' if greedy else 'sampled'}"
            else:
                run_name, greedy, key = None, True, agent

            parsed.append({"agent_key": key, "agent": agent, "run_name": run_name, "greedy": greedy})

        keys = [spec["agent_key"] for spec in parsed]
        if len(keys) != len(set(keys)):
            raise ApiError(400, "the same agent was requested twice")

        return parsed

    def _new_lane(self, spec):
        return {"spec": spec, "status": "queued", "round": 0, "decisions": 0, "frames": 0, "started": None, "elapsed_s": 0.0, "error": None, "traceback": None, "total_reward": None}

    def _persist(self, episode):
        with self.state_lock:
            stored = {key: episode[key] for key in ("episode_id", "created", "source", "caps", "settings", "shared", "divergence")}
            stored["lanes"] = {key: {name: value for name, value in lane.items()} for key, lane in episode["lanes"].items()}

        directory = self._episode_dir(episode["episode_id"])
        directory.mkdir(parents=True, exist_ok=True)

        temporary = directory / "episode.json.tmp"
        temporary.write_text(json.dumps(stored))
        temporary.replace(directory / "episode.json")

    def _load(self, episode_id):
        with self.state_lock:
            if episode_id in self.episodes:
                return self.episodes[episode_id]

        path = self._episode_dir(episode_id) / "episode.json" if episode_id.isalnum() else None
        if path is None or not path.exists():
            raise ApiError(404, f"unknown episode '{episode_id}'")

        episode = json.loads(path.read_text())
        for lane in episode["lanes"].values():
            if lane["status"] in self.ACTIVE:
                lane["status"] = "cancelled"
                lane["error"]  = "the console restarted while this lane was recording"

        episode["recorders"] = {}
        episode["cancel"]    = False

        with self.state_lock:
            return self.episodes.setdefault(episode_id, episode)

    def _ensure_worker(self):
        with self.state_lock:
            if self.worker is None or not self.worker.is_alive():
                self.worker = threading.Thread(target=self._work, name="episode-recorder", daemon=True)
                self.worker.start()

    def _enqueue(self, episode, specs):
        with self.state_lock:
            episode["cancel"] = False
            for spec in specs:
                episode["lanes"][spec["agent_key"]] = self._new_lane(spec)

        self._persist(episode)

        for spec in specs:
            self.pending.put((episode["episode_id"], spec["agent_key"]))

        self._ensure_worker()

    def create(self, body):
        """Start a recording.

        Args:
            body : {"source": seed, evaluation or scenario source, "caps": {"rounds", "edits"}, "agents": [{"agent", "run_name", "greedy"}]}.

        Returns:
            {"episode_id", "agent_keys"}; the lanes are queued on the worker thread.

        Raises:
            ApiError : The body is malformed (400) or names an unknown run or report (404).
        """
        source, caps, settings = self._parse_source(body)
        specs                  = self._parse_agents(body)

        self.runtime.episode_config(settings["environment"], settings["reward"], caps["rounds"], caps["edits"])

        episode = {
            "episode_id" : uuid.uuid4().hex[:12],
            "created"    : time.time(),
            "source"     : source,
            "caps"       : caps,
            "settings"   : settings,
            "shared"     : {"gaps": [], "events": []},
            "divergence" : [],
            "lanes"      : {},
            "recorders"  : {},
            "cancel"     : False,
        }

        with self.state_lock:
            self.episodes[episode["episode_id"]] = episode

        self._enqueue(episode, specs)
        return {"episode_id": episode["episode_id"], "agent_keys": [spec["agent_key"] for spec in specs]}

    def add_agents(self, episode_id, body):
        """Add lanes to an existing episode; same response as `create`. 409 when a requested lane is already queued or recording."""
        episode = self._load(episode_id)
        specs   = self._parse_agents(body)

        with self.state_lock:
            busy = [spec["agent_key"] for spec in specs if spec["agent_key"] in episode["lanes"] and episode["lanes"][spec["agent_key"]]["status"] in self.ACTIVE]
        if busy:
            raise ApiError(409, f"lane {busy[0]} is still recording")

        self._enqueue(episode, specs)
        return {"episode_id": episode_id, "agent_keys": [spec["agent_key"] for spec in specs]}

    def _public_source(self, source):
        return {key: value for key, value in source.items() if key not in ("jobs", "vehicles", "assignment", "expectation", "depot")}

    def _episode_status(self, lanes):
        statuses = [lane["status"] for lane in lanes.values()]

        if any(status in self.ACTIVE for status in statuses):
            return "running"
        if statuses and all(status == "failed" for status in statuses):
            return "failed"
        if statuses and all(status == "cancelled" for status in statuses):
            return "cancelled"

        return "finished"

    def _lane_view(self, lane):
        elapsed = time.time() - lane["started"] if lane["status"] == "running" else lane["elapsed_s"]
        return {key: lane[key] for key in ("status", "round", "decisions", "frames", "error", "traceback", "total_reward")} | {"elapsed_s": elapsed, "agent": lane["spec"]["agent"], "run_name": lane["spec"]["run_name"], "greedy": lane["spec"]["greedy"]}

    def status(self, episode_id):
        """Return the progress of an episode.

        Returns:
            {"episode_id", "created", "status", "source", "caps", "lanes": {agent_key: {"status", "round", "decisions", "frames", "elapsed_s", "error", "traceback", "total_reward", "agent", "run_name", "greedy"}}, "shared": {"gaps", "events"}, "divergence"}.

        Raises:
            ApiError : The episode does not exist (404).
        """
        episode = self._load(episode_id)

        with self.state_lock:
            return {
                "episode_id" : episode["episode_id"],
                "created"    : episode["created"],
                "status"     : self._episode_status(episode["lanes"]),
                "source"     : self._public_source(episode["source"]),
                "caps"       : episode["caps"],
                "lanes"      : {key: self._lane_view(lane) for key, lane in episode["lanes"].items()},
                "shared"     : episode["shared"],
                "divergence" : episode["divergence"],
            }

    def scenario(self, episode_id):
        """Return the scenario an episode was recorded on, {"name", "template", "jobs", "vehicles", "depot", "assignment"}; 404 for other sources."""
        source = self._load(episode_id)["source"]
        if source["kind"] != "scenario":
            raise ApiError(404, "this episode was not recorded on a scenario")

        return {key: source[key] for key in ("name", "template", "jobs", "vehicles", "depot", "assignment")}

    def list(self):
        """Return the stored recordings, newest first, as {"episode_id", "created", "source", "agents": {agent_key: {"status", "total_reward"}}}."""
        if not self.paths.episodes_dir.is_dir():
            return []

        entries = []
        for directory in self.paths.episodes_dir.iterdir():
            if not (directory / "episode.json").exists():
                continue

            status = self.status(directory.name)
            entries.append({
                "episode_id" : status["episode_id"],
                "created"    : status["created"],
                "status"     : status["status"],
                "source"     : status["source"],
                "agents"     : {key: {"status": lane["status"], "total_reward": lane["total_reward"]} for key, lane in status["lanes"].items()},
            })

        entries.sort(key=lambda entry: entry["created"], reverse=True)
        return entries

    def cancel(self, episode_id):
        """Cancel the queued and running lanes of an episode; a running lane stops before its next decision."""
        episode = self._load(episode_id)

        with self.state_lock:
            episode["cancel"] = True
            for lane in episode["lanes"].values():
                if lane["status"] == "queued":
                    lane["status"] = "cancelled"

        self._persist(episode)
        return {"ok": True}

    def live_trace(self, episode_id, agent_key):
        """Return (recorder, guard) of a lane that is recording right now, or None when the lane is not running."""
        with self.state_lock:
            episode = self.episodes.get(episode_id)
            if episode is None or agent_key not in episode["recorders"]:
                return None

            return episode["recorders"][agent_key]

    def _build_agent(self, spec, config):
        if spec["agent"] == "model":
            run_dir = self.runtime.layout.run_dir(spec["run_name"])
            if run_dir is None:
                raise ValueError(f"unknown run '{spec['run_name']}'")

            status = self.catalog.status(run_dir)
            if not status["has_checkpoint"]:
                raise ValueError(f"run '{spec['run_name']}' has no checkpoint")
            if not status["loadable"]:
                raise ValueError(f"checkpoint of '{spec['run_name']}' does not load: {status['reason']}")

            return ModelAgent(self.runtime.model(spec["run_name"]), greedy=spec["greedy"])

        if spec["agent"] == "teacher":
            return TeacherAgent(RegretInsertionTeacher(config))
        if spec["agent"] == "insertion_only":
            return TeacherAgent(RegretInsertionTeacher(config, allow_removal=False))

        return FixedOperatorAgent(2)

    def _build_recorder(self, config, guard, step_guard):
        with step_guard:
            environment = Environment(config)

        return EpisodeRecorder(config, environment, guard=guard, step_guard=step_guard)

    def _on_frame(self, episode, key, recorder, frame):
        lane = episode["lanes"][key]

        with self.state_lock:
            lane["frames"] = frame["seq"] + 1
            lane["round"]  = frame["round"]

            if frame["kind"] == "decision":
                lane["decisions"] += 1
            elif frame["round"] >= len(episode["shared"]["gaps"]):
                event = frame["round_open"]["event"]
                episode["shared"]["gaps"].append(frame["round_open"]["gap_s"])
                episode["shared"]["events"].append(None if event is None else {"type": event["type"], "count": event["count"]})

        if frame["kind"] == "round_open":
            with recorder.guard:
                self.files.write(self.trace_path(episode["episode_id"], key), recorder.trace)
            self._persist(episode)

    def _finished_traces(self, episode):
        traces = {}
        for key, lane in episode["lanes"].items():
            if lane["status"] == "finished" and self.trace_path(episode["episode_id"], key).exists():
                traces[key] = self.files.read(self.trace_path(episode["episode_id"], key))

        return traces

    def _record_lane(self, episode, key):
        lane = episode["lanes"][key]
        spec = lane["spec"]

        config     = self.runtime.episode_config(episode["settings"]["environment"], episode["settings"]["reward"], episode["caps"]["rounds"], episode["caps"]["edits"])
        guard      = threading.RLock()
        step_guard = self.runtime.rng_guard()

        with self.runtime.lock:
            agent = self._build_agent(spec, config)

        recorder = self.recorder_factory(config, guard, step_guard)
        with self.state_lock:
            episode["recorders"][key] = (recorder, guard)

        try:
            trace = recorder.record(
                agent,
                {"agent_key": key, "agent": spec["agent"], "run_name": spec["run_name"], "greedy": spec["greedy"]},
                episode["source"],
                should_stop = lambda: episode["cancel"],
                on_frame    = lambda frame: self._on_frame(episode, key, recorder, frame),
            )
        finally:
            if recorder.trace is not None and recorder.trace["frames"]:
                with guard:
                    self.files.write(self.trace_path(episode["episode_id"], key), recorder.trace)

            with self.state_lock:
                episode["recorders"].pop(key, None)

        return trace

    def _run_lane(self, episode_id, key):
        episode = self.episodes[episode_id]
        lane    = episode["lanes"][key]

        with self.state_lock:
            if lane["status"] != "queued":
                return

            lane["status"]  = "running"
            lane["started"] = time.time()

        try:
            trace = self._record_lane(episode, key)

            with self.state_lock:
                lane["status"]       = "cancelled" if trace["summary"]["stopped_reason"] == "cancelled" else "finished"
                lane["total_reward"] = trace["summary"]["total_reward"]
        except Exception as error:
            self.logger.error(f"episode {episode_id} lane {key} failed: {error}")

            with self.state_lock:
                lane["status"]    = "failed"
                lane["error"]     = str(error).splitlines()[0][:300] if str(error) else type(error).__name__
                lane["traceback"] = traceback.format_exc()

        with self.state_lock:
            lane["elapsed_s"] = time.time() - lane["started"]

        divergence = self.divergence.compare(self._finished_traces(episode))
        with self.state_lock:
            episode["divergence"] = divergence

        self._persist(episode)

    def _work(self):
        while True:
            episode_id, key = self.pending.get()
            self._run_lane(episode_id, key)


class VerdictService:
    """Computes the teacher verdict of one recorded decision on demand and caches it in the trace file.

    Attributes:
        runtime : ConsoleRuntime with the environment lock and episode configs.
        traces  : TraceStore locating and rewriting trace files.
    """

    SCOPE_RANK = {"insertions": 1, "full": 2}

    def __init__(self, runtime, traces):
        self.runtime      = runtime
        self.traces       = traces
        self.busy         = threading.Lock()
        self.environments = {}

    def _environment(self, meta):
        key = json.dumps({name: meta[name] for name in ("environment", "reward", "caps")}, sort_keys=True)

        if key not in self.environments:
            config                 = self.runtime.episode_config(meta["environment"], meta["reward"], meta["caps"]["rounds"], meta["caps"]["edits"])
            self.environments[key] = (config, Environment(config))

        return self.environments[key]

    def _compute(self, trace, frame, scope):
        snapshot = trace["snapshots"][str(frame["seq"])]

        with self.runtime.lock:
            config, environment = self._environment(trace["meta"])
            environment.load_scenario(snapshot)

            builder = VerdictBuilder(config, PlanMetrics(config.reward))
            return builder.build(environment, environment.current_state, snapshot["edits_remaining"], scope, frame["decision"]["action"], frame["decision"]["rewards"])

    def verdict(self, ref, body):
        """Return the teacher verdict of one decision, computing and storing it when it is not cached.

        Args:
            ref  : TraceRef of the trace.
            body : {"seq": frame sequence number, "scope": "insertions" or "full"}.

        Returns:
            The verdict dict of `VerdictBuilder.build`. A stored verdict of equal or larger scope is returned without computing.

        Raises:
            ApiError : Malformed body or a frame that is no decision (400), unknown trace or frame (404), lane still recording or another verdict running (409).
        """
        scope = body.get("scope", "insertions")
        seq   = body.get("seq")

        if scope not in self.SCOPE_RANK:
            raise ApiError(400, "scope must be insertions or full")
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ApiError(400, "seq must be an integer")
        if ref.kind == "ep" and self.traces.episodes.live_trace(ref.episode_id, ref.agent_key) is not None:
            raise ApiError(409, "this lane is still recording")

        trace = self.traces.load(ref)
        if not 0 <= seq < len(trace["frames"]):
            raise ApiError(404, f"no frame {seq}")

        frame = trace["frames"][seq]
        if frame["kind"] != "decision":
            raise ApiError(400, f"frame {seq} is not a decision")

        stored = frame["decision"]["verdict"]
        if stored is not None and self.SCOPE_RANK[stored["scope"]] >= self.SCOPE_RANK[scope]:
            return stored

        if not self.busy.acquire(blocking=False):
            raise ApiError(409, "another verdict is being computed")

        try:
            verdict = self._compute(trace, frame, scope)
            self.traces.store_verdict(ref, seq, verdict)
            return verdict
        finally:
            self.busy.release()
