"""Deterministic run findings F1 to F16: fixed rules over the evaluation report, the diagnostics index and the training curves."""

from web_errors import ApiError


class RunFindings:
    """Evaluates the run rules for one run and report seed.

    Attributes:
        runs    : RunBrowser giving reports, diagnostics indexes and curves.
        catalog : CheckpointCatalog giving the loadability verdict.
    """

    THRESHOLDS = {
        "insert_share_ratio"   : 0.5,
        "min_episodes"         : 30,
        "explained_variance"   : 0.2,
        "min_decisions"        : 30,
        "commit_work_left"     : 0.25,
        "profitable_left"      : 0.10,
        "wasted_share"         : 0.05,
        "holding_share"        : 0.5,
        "priority_spearman"    : 0.1,
        "chosen_probability"   : 0.5,
        "do_nothing_drift"     : 0.10,
    }

    def __init__(self, runs, catalog):
        self.runs    = runs
        self.catalog = catalog

    def _finding(self, identifier, status, title, text, metric=None, value=None, threshold=None, count=None, needs=None, evidence=None):
        return {"id": identifier, "status": status, "title": title, "metric": metric, "value": value, "threshold": threshold, "n": count, "text": text, "needs": needs, "evidence": evidence or []}

    def _no_data(self, identifier, title, needs):
        words = {"report": "an evaluation report", "traces": "recorded traces", "verdicts": "teacher verdicts", "curves": "training curves"}
        return self._finding(identifier, "NO DATA", title, f"Needs {words[needs]}.", needs=needs)

    def _link(self, label, target):
        return {"label": label, "hash": target}

    def _degenerate(self, name, report):
        agents = report["agents"]
        if "model" not in agents:
            return self._no_data("F1", "Policy identical to do_nothing", "report")

        model    = agents["model"]
        share    = model["operator_frequency"]["op2"]
        same     = model["identical_to"] == "do_nothing"
        evidence = [self._link("Agent table", f"#/diagnose/{name}/{report['meta']['seed']}")]

        if same or share >= 1.0:
            return self._finding("F1", "FAIL", "Policy identical to do_nothing", f"The greedy policy chose DO_NOTHING in {share:.0%} of decisions and its reward equals do_nothing in every one of {model['episodes']} episodes." if same else f"The greedy policy chose DO_NOTHING in {share:.0%} of decisions.", "model DO_NOTHING share", share, 1.0, model["episodes"], evidence=evidence)

        return self._finding("F1", "PASS", "Policy identical to do_nothing", f"The greedy policy chose DO_NOTHING in {share:.0%} of decisions.", "model DO_NOTHING share", share, 1.0, model["episodes"], evidence=evidence)

    def _separability(self, name, report):
        paired   = {agent: stats for agent, stats in report["paired"].items() if agent in ("teacher", "insertion_only")}
        evidence = [self._link("Paired scatter", f"#/diagnose/{name}/{report['meta']['seed']}")]
        if not paired:
            return self._no_data("F2", "Model against the baselines", "report")

        count = report["meta"]["episodes"]
        worse = [agent for agent, stats in paired.items() if stats["mean_diff"] < 0 and stats["ci95"][1] < 0]
        parts = [f"model minus {agent} {stats['mean_diff']:+.2f} [{stats['ci95'][0]:+.2f}, {stats['ci95'][1]:+.2f}]" for agent, stats in paired.items()]
        value = min(stats["mean_diff"] for stats in paired.values())

        if worse:
            return self._finding("F2", "FAIL", "Model against the baselines", f"The model is below {' and '.join(worse)} with an interval that excludes 0: " + "; ".join(parts) + ".", "paired mean difference", value, 0.0, count, evidence=evidence)

        return self._finding("F2", "PASS", "Model against the baselines", f"Not separable at n = {count}: " + "; ".join(parts) + ".", "paired mean difference", value, 0.0, count, evidence=evidence)

    def _under_insertion(self, name, report):
        agents = report["agents"]
        if "model" not in agents or "teacher" not in agents:
            return self._no_data("F3", "INSERT share against the teacher", "report")

        model   = agents["model"]["operator_frequency"]["op0"]
        teacher = agents["teacher"]["operator_frequency"]["op0"]
        limit   = self.THRESHOLDS["insert_share_ratio"] * teacher
        status  = "WARN" if model < limit else "PASS"

        return self._finding("F3", status, "INSERT share against the teacher", f"Model INSERT share {model:.0%}, teacher {teacher:.0%}.", "model INSERT share", model, limit, agents["model"]["episodes"], evidence=[self._link("Operator mix", f"#/diagnose/{name}/{report['meta']['seed']}")])

    def _sample_size(self, report):
        count  = report["meta"]["episodes"]
        status = "WARN" if count < self.THRESHOLDS["min_episodes"] else "PASS"
        return self._finding("F4", status, "Evaluation sample size", f"The report holds {count} episodes; intervals are wide below {self.THRESHOLDS['min_episodes']}.", "episodes", count, self.THRESHOLDS["min_episodes"], count)

    def _checkpoint(self, name, seed):
        run_dir = self.runs.layout.run_dir(name)
        status  = self.catalog.status(run_dir)
        report  = self.runs.layout.report_path(run_dir, seed) if seed is not None else None

        if not status["has_checkpoint"]:
            return self._finding("F5", "FAIL", "Checkpoint", "The run holds no checkpoint.")
        if not status["loadable"]:
            return self._finding("F5", "FAIL", "Checkpoint", f"The checkpoint does not load: {status['reason']}.")
        if report is not None and report.exists() and report.stat().st_mtime < status["mtime"]:
            return self._finding("F5", "WARN", "Checkpoint", "The evaluation report is older than the checkpoint.", evidence=[self._link("Evaluate", f"#/launch/evaluate?io.run_name={name}")])

        return self._finding("F5", "PASS", "Checkpoint", "The checkpoint loads on the current architecture.")

    def _model_digests(self, index):
        return [episode["agents"]["model"] for episode in index["episodes"] if "model" in episode["agents"]]

    def _critic(self, name, seed, index):
        critic = index["aggregates"]["critic"]
        if critic["n"] < self.THRESHOLDS["min_decisions"] or critic["explained_variance"] is None:
            return self._finding("F6", "NO DATA", "Critic on held-out decisions", f"Needs at least {self.THRESHOLDS['min_decisions']} model decisions; {critic['n']} recorded.", count=critic["n"], needs="traces")

        status = "WARN" if critic["explained_variance"] < self.THRESHOLDS["explained_variance"] else "PASS"
        return self._finding("F6", status, "Critic on held-out decisions", f"Explained variance {critic['explained_variance']:.2f}, bias {critic['bias']:+.2f}, mean absolute error {critic['mae']:.2f}.", "explained variance", critic["explained_variance"], self.THRESHOLDS["explained_variance"], critic["n"], evidence=[self._link("Critic calibration", f"#/diagnose/{name}/{seed}")])

    def _early_commits(self, name, seed, digests):
        rounds = sum(len(digest["edits_per_round"]) for digest in digests)
        if rounds == 0:
            return self._no_data("F7", "Commits with work left", "traces")

        share   = sum(digest["commits_with_work_left"] for digest in digests) / rounds
        status  = "WARN" if share > self.THRESHOLDS["commit_work_left"] else "PASS"
        text    = f"{share:.0%} of {rounds} rounds were committed with unassigned jobs and edits remaining."
        judged  = [digest for digest in digests if digest["profitable_left"] is not None]

        if judged:
            judged_rounds = sum(len(digest["edits_per_round"]) for digest in judged)
            left          = sum(digest["profitable_left"] for digest in judged)
            text         += f" The teacher table shows {left} profitable insertions left at {judged_rounds} round ends."

            if left / judged_rounds > self.THRESHOLDS["profitable_left"]:
                status = "WARN"

        return self._finding("F7", status, "Commits with work left", text, "share of rounds", share, self.THRESHOLDS["commit_work_left"], rounds, evidence=[self._link("Burst profile", f"#/diagnose/{name}/{seed}")])

    def _wasted(self, name, seed, digests):
        inserts = sum(digest["operator_counts"][0] for digest in digests)
        if inserts == 0:
            return self._finding("F8", "PASS", "Wasted edits", "The model made no INSERT.", "share of INSERTs", 0.0, self.THRESHOLDS["wasted_share"], 0)

        share  = sum(digest["wasted_edits"] for digest in digests) / inserts
        status = "WARN" if share > self.THRESHOLDS["wasted_share"] else "PASS"
        return self._finding("F8", status, "Wasted edits", f"{share:.0%} of {inserts} INSERTs left the plan unchanged and paid the edit cost.", "share of INSERTs", share, self.THRESHOLDS["wasted_share"], inserts, evidence=[self._link("Regret table", f"#/diagnose/{name}/{seed}")])

    def _holding(self, name, seed, digests):
        terms = {}
        for digest in digests:
            for term, value in digest["reward_by_term"].items():
                terms[term] = terms.get(term, 0.0) + value

        total  = sum(abs(value) for value in terms.values())
        share  = abs(terms["holding_reward"]) / total if total > 0 else 0.0
        status = "WARN" if share > self.THRESHOLDS["holding_share"] else "PASS"
        return self._finding("F9", status, "Holding share of the return", f"Holding is {share:.0%} of the summed absolute reward terms over {len(digests)} episodes.", "holding share", share, self.THRESHOLDS["holding_share"], len(digests), evidence=[self._link("Reward by term", f"#/diagnose/{name}/{seed}")])

    def _job_ranking(self, name, seed, index):
        ranking = index["aggregates"]["job_ranking"]
        if ranking["n"] < self.THRESHOLDS["min_decisions"] or ranking["spearman"] is None:
            return self._finding("F10", "NO DATA", "Job head against priority", f"Needs at least {self.THRESHOLDS['min_decisions']} ranked jobs; {ranking['n']} recorded.", count=ranking["n"], needs="traces")

        status = "WARN" if abs(ranking["spearman"]) < self.THRESHOLDS["priority_spearman"] else "PASS"
        return self._finding("F10", status, "Job head against priority", f"Spearman rho between job priority and job-head rank is {ranking['spearman']:+.2f}; a head that ranks by priority gives a clearly negative value.", "absolute Spearman rho", abs(ranking["spearman"]), self.THRESHOLDS["priority_spearman"], ranking["n"], evidence=[self._link("Job ranking", f"#/diagnose/{name}/{seed}")])

    def _abandonments(self, name, seed, digests):
        count  = sum(digest["abandonments"] for digest in digests)
        status = "WARN" if count > 0 else "PASS"
        return self._finding("F11", status, "Abandoned removals", f"{count} removals were not reinserted in their round.", "abandonments", count, 0, len(digests), evidence=[self._link("Regret table", f"#/diagnose/{name}/{seed}")])

    def _confidence(self, name, seed, index):
        rows = index["decisions"]
        if not rows:
            return self._no_data("F12", "Confidence of the chosen action", "traces")

        mean   = sum(row["chosen_joint_prob"] for row in rows) / len(rows)
        greedy = [episode["agents"]["model"]["total_reward"] for episode in index["episodes"] if "model" in episode["agents"] and "model_sampled" in episode["agents"]]
        sample = [episode["agents"]["model_sampled"]["total_reward"] for episode in index["episodes"] if "model" in episode["agents"] and "model_sampled" in episode["agents"]]
        status = "WARN" if mean < self.THRESHOLDS["chosen_probability"] else "PASS"
        text   = f"Mean joint probability of the chosen action is {mean:.2f}."

        if greedy:
            text += f" Sampled twin mean reward {sum(sample) / len(sample):.2f} against greedy {sum(greedy) / len(greedy):.2f}."
            if sum(sample) > sum(greedy):
                status = "WARN"

        return self._finding("F12", status, "Confidence of the chosen action", text, "mean chosen joint probability", mean, self.THRESHOLDS["chosen_probability"], len(rows), evidence=[self._link("Confidence", f"#/diagnose/{name}/{seed}")])

    def _tenth_means(self, values):
        width = max(len(values) // 10, 1)
        return sum(values[:width]) / width, sum(values[-width:]) / width

    def _drift(self, name, tags):
        series = tags.get("episode/operator_frequency/op2")
        if series is None or not series["values"]:
            return self._no_data("F13", "DO_NOTHING drift in training", "curves")

        first, last = self._tenth_means(series["values"])
        status      = "WARN" if last - first > self.THRESHOLDS["do_nothing_drift"] else "PASS"
        return self._finding("F13", status, "DO_NOTHING drift in training", f"DO_NOTHING share moved from {first:.0%} in the first tenth of training to {last:.0%} in the last.", "change of DO_NOTHING share", last - first, self.THRESHOLDS["do_nothing_drift"], len(series["values"]), evidence=[self._link("Training health", f"#/runs/{name}/health")])

    def _training_critic(self, name, tags):
        series = tags.get("batch/baseline_stats/explained_variance")
        if series is None or not series["values"]:
            return self._no_data("F14", "Critic during training", "curves")

        _, last = self._tenth_means(series["values"])
        status  = "WARN" if last < self.THRESHOLDS["explained_variance"] else "PASS"
        return self._finding("F14", status, "Critic during training", f"Explained variance over the last tenth of updates is {last:.2f}.", "explained variance", last, self.THRESHOLDS["explained_variance"], len(series["values"]), evidence=[self._link("Training health", f"#/runs/{name}/health")])

    def _anchor(self, name, tags):
        series = tags.get("batch/anchor_coefficient")
        if series is None or not series["values"]:
            return self._no_data("F15", "Anchor tether", "curves")

        last   = series["values"][-1]
        status = "WARN" if last > 0 else "PASS"
        return self._finding("F15", status, "Anchor tether", f"The anchor coefficient ended at {last:.3f}; while it is above 0 agreement with the pretrained policy is tethered.", "final anchor coefficient", last, 0.0, len(series["values"]), evidence=[self._link("Training health", f"#/runs/{name}/health")])

    def _numerics(self, name, tags):
        flags = {tag: series for tag, series in tags.items() if tag.endswith("/has_nan") or tag.endswith("/has_inf")}
        if not flags:
            return self._no_data("F16", "Non-finite gradients", "curves")

        bad    = sorted(tag for tag, series in flags.items() if any(value > 0 for value in series["values"]))
        status = "FAIL" if bad else "PASS"
        return self._finding("F16", status, "Non-finite gradients", f"{len(bad)} of {len(flags)} gradient flags fired" + (": " + ", ".join(bad) + "." if bad else "."), "flags fired", len(bad), 0, len(flags), evidence=[self._link("Curves", f"#/runs/{name}/curves")])

    def _report_rules(self, name, seed):
        run_dir = self.runs.layout.run_dir(name)
        if seed is None or not self.runs.layout.report_path(run_dir, seed).exists():
            return [self._no_data(identifier, title, "report") for identifier, title in (("F1", "Policy identical to do_nothing"), ("F2", "Model against the baselines"), ("F3", "INSERT share against the teacher"), ("F4", "Evaluation sample size"))]

        report = self.runs.evaluation(name, seed)
        return [self._degenerate(name, report), self._separability(name, report), self._under_insertion(name, report), self._sample_size(report)]

    def _trace_rules(self, name, seed):
        run_dir = self.runs.layout.run_dir(name)
        titles  = (("F6", "Critic on held-out decisions"), ("F7", "Commits with work left"), ("F8", "Wasted edits"), ("F9", "Holding share of the return"), ("F10", "Job head against priority"), ("F11", "Abandoned removals"), ("F12", "Confidence of the chosen action"))

        if seed is None or seed not in self.runs.layout.diagnostics_seeds(run_dir):
            return [self._no_data(identifier, title, "traces") for identifier, title in titles]

        index   = self.runs.diagnostics(name, seed)
        digests = self._model_digests(index)
        if not digests:
            return [self._no_data(identifier, title, "traces") for identifier, title in titles]

        return [
            self._critic(name, seed, index),
            self._early_commits(name, seed, digests),
            self._wasted(name, seed, digests),
            self._holding(name, seed, digests),
            self._job_ranking(name, seed, index),
            self._abandonments(name, seed, digests),
            self._confidence(name, seed, index),
        ]

    def _curve_rules(self, name):
        tags = self.runs.run_curves(name)["tags"]
        return [self._drift(name, tags), self._training_critic(name, tags), self._anchor(name, tags), self._numerics(name, tags)]

    def findings(self, name, seed=None):
        """Return the sixteen run findings in rule order.

        Args:
            name : Run name.
            seed : Report seed the report and trace rules read; None selects the newest report of the run, and leaves those rules at NO DATA when the run has none.

        Returns:
            List of {"id", "status": "FAIL", "WARN", "PASS" or "NO DATA", "title", "metric", "value", "threshold", "n", "text", "needs", "evidence": [{"label", "hash"}]}.

        Raises:
            ApiError : The run does not exist (404).
        """
        run_dir = self.runs.layout.run_dir(name)
        if run_dir is None:
            raise ApiError(404, f"unknown run '{name}'")

        if seed is None:
            seeds = self.runs.layout.report_seeds(run_dir)
            seed  = seeds[-1] if seeds else None

        report_rules = self._report_rules(name, seed)
        return report_rules + [self._checkpoint(name, seed)] + self._trace_rules(name, seed) + self._curve_rules(name)
