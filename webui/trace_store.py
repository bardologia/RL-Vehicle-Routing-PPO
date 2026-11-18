"""Read side of recorded traces: locating them, adding road geometry and findings, and caching teacher verdicts."""

import copy
import re

from configuration import RewardConfig
from core.inference import PlanMetrics, TraceFile, TraceFindings
from web_errors import ApiError


class TraceRef:
    """Address of one trace.

    Attributes:
        kind       : "ep" for an interactive recording, "diag" for a diagnostics trace.
        episode_id : Episode id of an interactive recording, else None.
        run        : Run name of a diagnostics trace, else None.
        seed       : Diagnostics seed, else None.
        index      : Episode index within the diagnostics run, else None.
        agent_key  : Agent key of the trace.
    """

    AGENT_KEY = re.compile(r"^[A-Za-z0-9_.~\-]{1,160}$")

    def __init__(self, kind, agent_key, episode_id=None, run=None, seed=None, index=None):
        if not self.AGENT_KEY.match(agent_key):
            raise ApiError(400, f"malformed agent key '{agent_key}'")

        self.kind       = kind
        self.agent_key  = agent_key
        self.episode_id = episode_id
        self.run        = run
        self.seed       = seed
        self.index      = index


class TraceStore:
    """Serves traces in the shape the browser reads.

    Attributes:
        layout   : RunLayout locating diagnostics traces.
        geometry : GeometryCache giving road paths and distances.
        episodes : EpisodeService holding interactive recordings.
    """

    POSITION = re.compile(r"^r(\d+)(?:e(\d+))?$")

    def __init__(self, layout, geometry, episodes):
        self.layout   = layout
        self.geometry = geometry
        self.episodes = episodes
        self.files    = TraceFile()
        self.findings = TraceFindings()

    def path(self, ref):
        """Return the file a trace is stored in.

        Raises:
            ApiError : The run, episode or trace file does not exist (404).
        """
        if ref.kind == "diag":
            run_dir = self.layout.run_dir(ref.run)
            if run_dir is None:
                raise ApiError(404, f"unknown run '{ref.run}'")

            path = self.layout.trace_path(run_dir, ref.seed, ref.index, ref.agent_key)
        else:
            path = self.episodes.trace_path(ref.episode_id, ref.agent_key)

        if not path.exists():
            raise ApiError(404, f"no trace for agent '{ref.agent_key}'")

        return path

    def load(self, ref):
        """Return the stored trace dict of a reference, snapshots included."""
        return self.files.read(self.path(ref))

    def positions(self, frames):
        """Return the map from position strings to frame sequence numbers.

        Args:
            frames : Frames of a trace.

        Returns:
            {"r<round>": seq of the round-open frame, "r<round>e<edit>": seq of that decision}, both zero-based.
        """
        index = {}
        for frame in frames:
            key        = f"r{frame['round']}" if frame["kind"] == "round_open" else f"r{frame['round']}e{frame['edit']}"
            index[key] = frame["seq"]

        return index

    def seq_of(self, frames, position):
        """Resolve a position, either a frame sequence number or "r<round>[e<edit>]", to a sequence number.

        Raises:
            ApiError : The position is malformed (400) or names no frame (404).
        """
        text = str(position)

        if text.isdigit():
            if int(text) >= len(frames):
                raise ApiError(404, f"no frame {text}")
            return int(text)

        if not self.POSITION.match(text):
            raise ApiError(400, f"malformed position '{text}'")

        index = self.positions(frames)
        if text not in index:
            raise ApiError(404, f"no frame at {text}")

        return index[text]

    def _present(self, trace, from_seq):
        metrics = PlanMetrics(RewardConfig(**trace["meta"]["reward"]))
        frames  = [frame for frame in trace["frames"] if frame["seq"] >= from_seq]

        paths   = {}
        ids     = {}

        for frame in frames:
            metrics.add_geometry(frame["plan"], self.geometry.route)

            for route in frame["plan"]["routes"]:
                path             = route.pop("path")
                route["path_id"] = ids.setdefault(id(path), f"p{len(ids)}")
                paths[route["path_id"]] = path

        return {
            "meta"        : trace["meta"],
            "depot"       : trace["depot"],
            "entities"    : trace["entities"],
            "paths"       : paths,
            "frames"      : frames,
            "positions"   : self.positions(trace["frames"]),
            "summary"     : trace["summary"],
            "expectation" : trace["expectation"],
            "findings"    : self.findings.evaluate(trace) if trace["meta"]["complete"] else [],
        }

    def read(self, ref, from_seq=0):
        """Return a trace for the browser.

        Args:
            ref      : TraceRef of the trace.
            from_seq : Only frames with a sequence number at or above this are returned.

        Returns:
            {"meta", "depot", "entities", "paths", "frames", "positions", "summary", "expectation", "findings"}; every route carries its road distance and a "path_id" into "paths" ({path_id: [[lat, lon]]}), so a path shared by many frames is sent once. A lane still recording is read from memory.

        Raises:
            ApiError : The trace does not exist (404) or OSRM is unavailable (503).
        """
        live = self.episodes.live_trace(ref.episode_id, ref.agent_key) if ref.kind == "ep" else None
        if live is None:
            return self._present(self.load(ref), from_seq)

        recorder, guard = live
        with guard:
            trace = {key: copy.deepcopy(value) if key != "frames" else copy.deepcopy([frame for frame in value if frame["seq"] >= from_seq]) for key, value in recorder.trace.items() if key != "snapshots"}

        return self._present(trace, from_seq)

    def store_verdict(self, ref, seq, verdict):
        """Write a verdict into the stored trace of a reference and return the updated trace."""
        path  = self.path(ref)
        trace = self.files.read(path)

        trace["frames"][seq]["decision"]["verdict"] = verdict
        self.files.write(path, trace)
        return trace
