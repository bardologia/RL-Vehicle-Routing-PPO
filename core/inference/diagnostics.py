"""Batch diagnostics over held-out episodes: trace findings, agent divergence, the diagnostics index and its pipeline."""

import json
import os
from dataclasses import asdict, fields

import numpy as np
from scipy import stats

from tools.logger import Logger
from core.shared import Environment, vroom
from core.training.pretraining import RegretInsertionTeacher
from model.policy_model import Policy, PolicyCheckpoint
from .evaluation import FixedOperatorAgent, ModelAgent, TeacherAgent
from .recording import EpisodeRecorder, EpisodeSettings, TraceFile, TraceSummary


class TraceFindings:
    """Applies the fixed episode rules E1 to E9 to one trace."""

    HOLDING_SHARE    = 0.5
    CONFIDENT_PROB   = 0.8
    CONFIDENT_REGRET = 0.5

    def _finding(self, identifier, status, title, text, value=None, threshold=None, seqs=None):
        return {"id": identifier, "status": status, "title": title, "text": text, "value": value, "threshold": threshold, "seqs": seqs or []}

    def _wasted(self, decisions):
        seqs = [frame["seq"] for frame in decisions if frame["decision"]["wasted"]]
        return self._finding("E1", "WARN" if seqs else "PASS", "Wasted edits", f"{len(seqs)} INSERT left the plan unchanged and still paid the edit cost.", len(seqs), 0, seqs)

    def _early_commits(self, decisions):
        frames   = [frame for frame in decisions if frame["decision"]["is_commit"] and frame["decision"]["edits_remaining"] > 1 and frame["plan"]["kpis"]["unassigned"] > 0]
        verdicts = [frame["decision"]["verdict"] for frame in frames if frame["decision"]["verdict"] is not None]
        text     = f"{len(frames)} commits left unassigned jobs with edits remaining."

        if verdicts:
            count = sum(verdict["profitable_left"]["count"] for verdict in verdicts)
            total = sum(verdict["profitable_left"]["reward_sum"] for verdict in verdicts)
            text += f" On {len(verdicts)} of them the teacher table shows {count} profitable insertions worth {total:.2f} reward points."

        return self._finding("E2", "WARN" if frames else "PASS", "Commits with work left", text, len(frames), 0, [frame["seq"] for frame in frames])

    def _holding(self, trace):
        share  = TraceSummary().holding_share(trace["summary"]["reward_by_term"])
        status = "WARN" if share > self.HOLDING_SHARE else "PASS"
        return self._finding("E3", status, "Holding share", f"Holding is {share:.0%} of the summed absolute reward terms.", share, self.HOLDING_SHARE)

    def _removals(self, trace):
        summary = trace["summary"]
        chains  = ", ".join(f"job {entry['job_id']} net {entry['net_reward']:.2f}" for entry in summary["relocations"])
        text    = f"{len(summary['abandonments'])} removals were not reinserted in their round; {len(summary['relocations'])} relocation chains" + (f" ({chains})." if chains else ".")
        return self._finding("E4", "WARN" if summary["abandonments"] else "PASS", "Abandoned removals", text, len(summary["abandonments"]), 0, [entry["seq"] for entry in summary["abandonments"]])

    def _confident_wrong(self, decisions):
        judged = [frame for frame in decisions if frame["decision"]["verdict"] is not None and frame["decision"]["policy"] is not None]
        if not judged:
            return self._finding("E5", "NO DATA", "Confident and wrong", "Needs a model decision with a teacher verdict.")

        frames = [frame for frame in judged if frame["decision"]["policy"]["chosen_joint_prob"] >= self.CONFIDENT_PROB and frame["decision"]["verdict"]["one_step_regret"] > self.CONFIDENT_REGRET]
        return self._finding("E5", "WARN" if frames else "PASS", "Confident and wrong", f"{len(frames)} of {len(judged)} judged decisions had probability at least {self.CONFIDENT_PROB} and one-step regret above {self.CONFIDENT_REGRET}.", len(frames), 0, [frame["seq"] for frame in frames])

    def _critic(self, decisions):
        valued = [frame for frame in decisions if frame["decision"]["policy"] is not None]
        if not valued:
            return self._finding("E6", "NO DATA", "Critic error", "Rule-based agent: no value estimate.")

        errors = [abs(frame["decision"]["policy"]["value"] - frame["decision"]["return_to_go_discounted"]) for frame in valued]
        worst  = valued[int(np.argmax(errors))]
        return self._finding("E6", "PASS", "Critic error", f"Mean absolute error {float(np.mean(errors)):.2f} over {len(valued)} decisions; worst {max(errors):.2f}.", float(np.mean(errors)), None, [worst["seq"]])

    def _teacher_agreement(self, decisions):
        judged = [frame for frame in decisions if frame["decision"]["verdict"] is not None]
        if not judged:
            return self._finding("E7", "NO DATA", "Operator against the teacher", "Needs teacher verdicts.")

        differ = [frame for frame in judged if not frame["decision"]["verdict"]["match"]["operator"]]
        rate   = 1.0 - len(differ) / len(judged)
        return self._finding("E7", "WARN" if differ else "PASS", "Operator against the teacher", f"Operator agreement {rate:.0%} over {len(judged)} judged decisions.", rate, 1.0, [frame["seq"] for frame in differ])

    def _failed_pickups(self, trace):
        opens  = [frame for frame in trace["frames"] if frame["kind"] == "round_open"]
        failed = [len(frame["round_open"]["executed"]["failed"]) for frame in opens]
        return self._finding("E8", "PASS", "Failed pickups", "Failed pickups per round: " + ", ".join(str(count) for count in failed) + ".", sum(failed), None, [frame["seq"] for frame, count in zip(opens, failed) if count > 0])

    def _expectation(self, trace):
        expectation = trace["expectation"]
        if expectation is None:
            return self._finding("E9", "NO DATA", "Template expectation", "The episode source declares no expectation.")

        held = sum(1 for check in expectation["checks"] if check["ok"])
        return self._finding("E9", expectation["status"], "Template expectation", f"{held} of {len(expectation['checks'])} checks hold on the plan at the round-1 commit.", held, len(expectation["checks"]))

    def evaluate(self, trace):
        """Return the episode findings of a finished trace.

        Args:
            trace : Trace dict with frames, summary and expectation.

        Returns:
            List of {"id", "status", "title", "text", "value", "threshold", "seqs"} for E1 to E9; status is PASS, WARN, FAIL or NO DATA.
        """
        decisions = [frame for frame in trace["frames"] if frame["kind"] == "decision"]

        return [
            self._wasted(decisions),
            self._early_commits(decisions),
            self._holding(trace),
            self._removals(trace),
            self._confident_wrong(decisions),
            self._critic(decisions),
            self._teacher_agreement(decisions),
            self._failed_pickups(trace),
            self._expectation(trace),
        ]


class DivergenceCheck:
    """Finds the rounds where agents on the same seed saw different entities or different pickup outcomes."""

    def _round_marks(self, trace):
        marks = {}
        for frame in trace["frames"]:
            if frame["kind"] != "round_open":
                continue

            event = frame["round_open"]["event"] or {}
            marks[frame["round"]] = {
                "event_ids" : tuple(tuple(event.get(key, [])) for key in ("added_job_ids", "removed_job_ids", "added_vehicle_ids", "removed_vehicle_ids")),
                "execution" : tuple(sorted(entry["job_id"] for entry in frame["round_open"]["executed"]["failed"])),
            }

        return marks

    def compare(self, traces):
        """Compare the round openings of several traces of one episode.

        Args:
            traces : Mapping {agent_key: trace}; the first key is the reference.

        Returns:
            List of {"round", "kind": "event_ids" or "execution", "agents": [agent_key]} naming the agents that differ from the reference in that round.
        """
        keys = list(traces)
        if len(keys) < 2:
            return []

        marks     = {key: self._round_marks(traces[key]) for key in keys}
        reference = marks[keys[0]]
        found     = []

        for round_index in sorted(reference):
            for kind in ("event_ids", "execution"):
                differing = [key for key in keys[1:] if round_index in marks[key] and marks[key][round_index][kind] != reference[round_index][kind]]
                if differing:
                    found.append({"round": round_index, "kind": kind, "agents": differing})

        return found


class DiagnosticsIndex:
    """Builds the index of a diagnostics run from the traces recorded so far.

    Attributes:
        meta     : Index meta block.
        episodes : Per episode index, {"seed", "jobs", "vehicles", "events", "agents": {agent_key: digest}, "divergent_rounds"}.
        model    : Per episode index, the decision rows and (priority, rank) pairs of the greedy model trace.
    """

    EVENT_WORDS = {"new_job": "new job", "remove_job": "remove job", "new_vehicle": "new vehicle", "remove_vehicle": "remove vehicle"}

    def __init__(self, meta, agent_keys):
        self.meta       = meta
        self.agent_keys = list(agent_keys)
        self.episodes   = {}
        self.model      = {}
        self.summary    = TraceSummary()
        self.divergence = DivergenceCheck()

    def _job_ranks(self, trace, decision):
        policy   = decision["policy"]
        eligible = decision["mask"]["insert_job_ids"]
        if policy["operator"]["masked"][0] or len(eligible) < 2:
            return None, []

        marginal = {}
        for vehicle_probability, job_row in zip(policy["vehicle_probs"][0], policy["job_probs"][0]):
            if vehicle_probability is None:
                continue
            for job_id, job_probability in zip(policy["job_ids"], job_row):
                if job_probability is not None:
                    marginal[job_id] = marginal.get(job_id, 0.0) + vehicle_probability * job_probability

        ordered = sorted(marginal, key=lambda job_id: -marginal[job_id])
        pairs   = [(trace["entities"]["jobs"][str(job_id)]["priority"], rank + 1) for rank, job_id in enumerate(ordered)]

        by_priority = {}
        for priority, rank in pairs:
            by_priority.setdefault(str(priority), []).append(rank)

        return {priority: float(np.mean(ranks)) for priority, ranks in by_priority.items()}, pairs

    def _model_rows(self, index, trace):
        rows  = []
        pairs = []

        for frame in trace["frames"]:
            if frame["kind"] != "decision":
                continue

            decision          = frame["decision"]
            policy            = decision["policy"]
            ranks, rank_pairs = self._job_ranks(trace, decision)
            pairs.extend(rank_pairs)

            rows.append({
                "episode"              : index,
                "seq"                  : frame["seq"],
                "round"                : frame["round"],
                "edit"                 : frame["edit"],
                "operator"             : decision["action"]["operator"],
                "value"                : policy["value"],
                "return_to_go"         : decision["return_to_go_discounted"],
                "chosen_joint_prob"    : policy["chosen_joint_prob"],
                "entropy"              : policy["entropy"],
                "p_do_nothing"         : policy["operator"]["probs"][2],
                "unassigned"           : len(decision["mask"]["insert_job_ids"]),
                "job_rank_by_priority" : ranks,
                "match"                : None if decision["verdict"] is None else decision["verdict"]["match"],
                "teacher_operator"     : None if decision["verdict"] is None else decision["verdict"]["teacher_action"]["operator"],
            })

        return {"rows": rows, "pairs": pairs}

    def add(self, index, agent_key, trace, traces):
        """Add one finished trace.

        Args:
            index     : Episode index within the run.
            agent_key : Agent key of the trace.
            trace     : The finished trace dict.
            traces    : Mapping {agent_key: trace} of every trace recorded for this episode so far, including this one.
        """
        opening = trace["frames"][0]["plan"]
        opens   = [frame["round_open"] for frame in trace["frames"] if frame["kind"] == "round_open"]

        episode = self.episodes.setdefault(index, {
            "index"            : index,
            "seed"             : trace["meta"]["seed"],
            "jobs"             : len(opening["unassigned"]) + sum(route["kpis"]["stops"] for route in opening["routes"]),
            "vehicles"         : len(opening["vehicles"]),
            "events"           : [self.EVENT_WORDS[entry["event"]["type"]] for entry in opens if entry["event"] is not None],
            "agents"           : {},
            "divergent_rounds" : [],
        })

        episode["agents"][agent_key] = self.summary.digest(trace)
        episode["divergent_rounds"]  = sorted({entry["round"] for entry in self.divergence.compare(traces)})

        if agent_key == "model":
            self.model[index] = self._model_rows(index, trace)

    def _critic(self, rows):
        if len(rows) < 2:
            return {"explained_variance": None, "bias": None, "mae": None, "n": len(rows)}

        values   = np.array([row["value"] for row in rows])
        returns  = np.array([row["return_to_go"] for row in rows])
        variance = float(np.var(returns))

        return {
            "explained_variance" : 1.0 - float(np.var(returns - values)) / variance if variance > 0 else None,
            "bias"               : float(np.mean(values - returns)),
            "mae"                : float(np.mean(np.abs(values - returns))),
            "n"                  : len(rows),
        }

    def _job_ranking(self, pairs):
        priorities = [pair[0] for pair in pairs]
        ranks      = [pair[1] for pair in pairs]

        if len(set(priorities)) < 2 or len(set(ranks)) < 2:
            return {"spearman": None, "n": len(pairs)}

        return {"spearman": float(stats.spearmanr(priorities, ranks).statistic), "n": len(pairs)}

    def _agreement(self, rows):
        judged = [row for row in rows if row["match"] is not None]
        if not judged:
            return None

        confusion = [[0, 0, 0] for _ in range(3)]
        for row in judged:
            confusion[row["operator"]][row["teacher_operator"]] += 1

        parameterised = [row for row in judged if row["match"]["vehicle"] is not None]

        return {
            "operator"  : float(np.mean([row["match"]["operator"] for row in judged])),
            "vehicle"   : float(np.mean([row["match"]["vehicle"] for row in parameterised])) if parameterised else None,
            "job"       : float(np.mean([row["match"]["job"] for row in parameterised])) if parameterised else None,
            "n"         : len(judged),
            "confusion" : confusion,
        }

    def _pickup_failure(self):
        rates = {}
        for agent_key in self.agent_keys:
            digests = [episode["agents"][agent_key] for episode in self.episodes.values() if agent_key in episode["agents"]]
            served  = sum(digest["pickups"]["served"] for digest in digests)
            failed  = sum(digest["pickups"]["failed"] for digest in digests)

            rates[agent_key] = failed / (served + failed) if served + failed > 0 else None

        return rates

    def build(self):
        """Return the index dict: "meta", "coverage", "episodes", "decisions" and "aggregates"."""
        rows  = [row for index in sorted(self.model) for row in self.model[index]["rows"]]
        pairs = [pair for index in sorted(self.model) for pair in self.model[index]["pairs"]]

        return {
            "meta"       : self.meta,
            "coverage"   : {agent_key: {"done": sum(1 for episode in self.episodes.values() if agent_key in episode["agents"]), "total": self.meta["episodes"]} for agent_key in self.agent_keys},
            "episodes"   : [self.episodes[index] for index in sorted(self.episodes)],
            "decisions"  : rows,
            "aggregates" : {
                "critic"              : self._critic(rows),
                "job_ranking"         : self._job_ranking(pairs),
                "agreement"           : self._agreement(rows),
                "pickup_failure_rate" : self._pickup_failure(),
            },
        }


class DiagnosticsPipeline:
    """Records a trace of every configured agent on a band of held-out episodes and indexes them.

    Attributes:
        config  : Config; `diagnostics` selects seeds, agents and the verdict scope.
        run_dir : Run directory whose checkpoint is diagnosed.
        out_dir : `<run_dir>/diagnostics/seed<seed>`, holding one trace per episode and agent plus `index.json`.
        index   : DiagnosticsIndex of the run.
    """

    AGENTS   = ("model", "teacher", "insertion_only", "do_nothing")
    VERDICTS = ("none", "insertions", "full")

    def __init__(self, config, repo_root, logger=None):
        self.config    = config
        self.repo_root = repo_root
        self.logger    = logger or Logger(name="diagnostics")

        self.run_dir     = None
        self.out_dir     = None
        self.environment = None
        self.model       = None
        self.agents      = None
        self.index       = None
        self.files       = TraceFile()

    def _resolve_run(self):
        run_name    = self.config.io.run_name
        diagnostics = self.config.diagnostics

        if not run_name:
            raise ValueError("config.io.run_name must name the run directory whose checkpoint is diagnosed")
        if diagnostics.teacher_verdict not in self.VERDICTS:
            raise ValueError(f"diagnostics.teacher_verdict must be one of {self.VERDICTS}, got {diagnostics.teacher_verdict!r}")

        unknown = [name for name in diagnostics.agents if name not in self.AGENTS]
        if unknown:
            raise ValueError(f"diagnostics.agents holds unknown agents {unknown}; expected a subset of {self.AGENTS}")

        runs_root = self.config.io.runs_dir
        if not os.path.isabs(runs_root):
            runs_root = os.path.join(str(self.repo_root), runs_root)

        self.run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(self.run_dir):
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self.out_dir = os.path.join(self.run_dir, "diagnostics", f"seed{diagnostics.seed}")
        os.makedirs(self.out_dir, exist_ok=True)

    def _check_section(self, name, section, reported):
        defaults = type(section)()

        for spec in fields(section):
            if not spec.init or spec.name not in reported:
                continue

            current  = getattr(section, spec.name)
            expected = tuple(reported[spec.name]) if isinstance(reported[spec.name], list) else reported[spec.name]

            if current != getattr(defaults, spec.name) and current != expected:
                raise ValueError(f"{name}.{spec.name}={current!r} contradicts the evaluation report of seed {self.config.diagnostics.seed}, which ran with {expected!r}")

    def _apply_report_meta(self):
        report_path = os.path.join(self.run_dir, f"evaluation_seed{self.config.diagnostics.seed}.json")
        if not os.path.exists(report_path):
            self.logger.info("No evaluation report for this seed; episodes follow the command-line configuration")
            return

        with open(report_path) as handle:
            meta = json.load(handle)["meta"]

        self._check_section("env", self.config.env, meta["environment"])
        self._check_section("reward", self.config.reward, meta["reward"])
        self._check_section("training", self.config.training, meta["caps"])

        applied = EpisodeSettings().build(self.config, meta["environment"], meta["reward"], meta["caps"]["max_disruptions_per_episode"], meta["caps"]["max_edits_per_round"])

        self.config.env      = applied.env
        self.config.reward   = applied.reward
        self.config.training = applied.training
        self.logger.info(f"Episodes follow the environment, reward and caps of {os.path.basename(report_path)}")

    def _build_environment(self):
        vroom.logger     = self.logger
        self.environment = Environment(self.config, logger=self.logger)

    def _load_model(self):
        self.model = Policy(self.config)
        self.model.to(self.model.device)
        PolicyCheckpoint().load(self.model, self.config.io.checkpoint_filename, self.run_dir, map_location=self.model.device)
        self.model.eval()

    def _make_agent(self, name):
        if name == "model":
            return ModelAgent(self.model, greedy=self.config.diagnostics.greedy)
        if name == "teacher":
            return TeacherAgent(RegretInsertionTeacher(self.config))
        if name == "insertion_only":
            return TeacherAgent(RegretInsertionTeacher(self.config, allow_removal=False))

        return FixedOperatorAgent(2)

    def _build_agents(self):
        diagnostics = self.config.diagnostics

        self.agents = []
        for name in diagnostics.agents:
            greedy = diagnostics.greedy if name == "model" else True
            self.agents.append({"agent_key": name, "agent": name, "greedy": greedy, "instance": self._make_agent(name)})

        if diagnostics.sampled_twin:
            self.agents.append({"agent_key": "model_sampled", "agent": "model", "greedy": False, "instance": ModelAgent(self.model, greedy=False)})

    def _index_meta(self):
        diagnostics = self.config.diagnostics

        return {
            "run_name"        : self.config.io.run_name,
            "seed"            : diagnostics.seed,
            "episodes"        : diagnostics.episodes,
            "greedy"          : diagnostics.greedy,
            "teacher_verdict" : diagnostics.teacher_verdict,
            "reward"          : asdict(self.config.reward),
            "environment"     : asdict(self.config.env),
            "caps"            : {"rounds": self.config.training.max_disruptions_per_episode, "edits": self.config.training.max_edits_per_round},
        }

    def _write_index(self):
        temporary = os.path.join(self.out_dir, "index.json.tmp")
        with open(temporary, "w") as handle:
            json.dump(self.index.build(), handle)

        os.replace(temporary, os.path.join(self.out_dir, "index.json"))

    def _record_all(self):
        diagnostics = self.config.diagnostics
        recorder    = EpisodeRecorder(self.config, self.environment, logger=self.logger)
        total       = diagnostics.episodes * len(self.agents)
        done        = 0

        self.index = DiagnosticsIndex(self._index_meta(), [entry["agent_key"] for entry in self.agents])

        for index in range(diagnostics.episodes):
            traces = {}

            for entry in self.agents:
                meta    = {"agent_key": entry["agent_key"], "agent": entry["agent"], "run_name": self.config.io.run_name if entry["agent"] == "model" else None, "greedy": entry["greedy"]}
                source  = {"kind": "evaluation", "run": self.config.io.run_name, "report_seed": diagnostics.seed, "index": index, "seed": diagnostics.seed + index}
                verdict = diagnostics.teacher_verdict if entry["agent_key"] == "model" else "none"

                trace                      = recorder.record(entry["instance"], meta, source, verdict_scope=verdict)
                traces[entry["agent_key"]] = trace

                self.files.write(os.path.join(self.out_dir, f"ep{index}_{entry['agent_key']}.json.gz"), trace)
                self.index.add(index, entry["agent_key"], trace, traces)
                self._write_index()

                done += 1
                self.logger.info(f"trace {done} of {total}: episode {index} {entry['agent_key']} reward {trace['summary']['total_reward']:.3f} decisions {trace['summary']['decisions']}")

    def _report(self):
        index = self.index.build()

        self.logger.kv_table({f"{agent_key} traces": f"{coverage['done']} of {coverage['total']}" for agent_key, coverage in index["coverage"].items()}, title="Diagnostics coverage")
        self.logger.info(f"Diagnostics index saved to {os.path.join(self.out_dir, 'index.json')}")

    def run(self):
        self._resolve_run()
        self._apply_report_meta()
        self._build_environment()
        self._load_model()
        self._build_agents()
        self._record_all()
        self._report()
        return self.index.build()
