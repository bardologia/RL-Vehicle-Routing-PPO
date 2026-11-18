class EpisodeStep:
    def __init__(self, driver, round_index, edit_index, graph, mask_info, remaining):
        self.driver      = driver
        self.environment = driver.environment
        self.round_index = round_index
        self.edit_index  = edit_index
        self.graph       = graph
        self.mask_info   = mask_info
        self.remaining   = remaining
        self.action      = None

    def commit(self, action):
        self.action          = action
        old_state, new_state = self.environment.apply_action_to(self.environment.current_state, action)
        rewards, costs       = self.environment.step(old_state, new_state, action.operator)

        self.environment.current_state = new_state
        rewards["holding_reward"]      = self.driver.settle_holding()

        return old_state, new_state, rewards, costs


class RoundRecord:
    """What happened while one round opened: execution over the gap, then one disruption.

    Attributes:
        round_index      : Zero-based round this record opens.
        gap              : Seconds the world advanced before the round, 0 for round 0.
        holding          : Holding cost charged for the plan left at the previous commit, in reward points.
        executed         : Execution summary of the advance, {"served": [job_id], "failed": [job_id], "dropped": int}.
        event            : Disruption applied when the round opened, {"type", "count"}, or None when none was applied.
        plan_before      : Per assigned job before the advance, {job_id: {"vehicle_id", "priority", "kind"}}.
        onboard_before   : Bikes on board per vehicle before the advance, {vehicle_id: int}.
        onboard_advanced : Bikes on board per vehicle after the advance, {vehicle_id: int}.
        plan_advanced    : Vehicle of every assigned job after the advance and before the disruption, {job_id: vehicle_id}.
        pools            : Job and vehicle id sets at the stages "before", "advanced" and "disrupted".
    """

    def __init__(self, round_index):
        self.round_index      = round_index
        self.gap              = 0
        self.holding          = 0.0
        self.executed         = {"served": [], "failed": [], "dropped": 0}
        self.event            = None
        self.plan_before      = {}
        self.onboard_before   = {}
        self.onboard_advanced = {}
        self.plan_advanced    = {}
        self.pools            = {}

    def _capture_pools(self, stage, environment):
        self.pools[stage] = {"jobs": set(environment.jobs.ids), "vehicles": set(environment.vehicles.ids)}

    def capture_before(self, environment):
        """Record the pools, the planned jobs with their attributes and the onboard loads before the advance."""
        self._capture_pools("before", environment)
        self.onboard_before = {vehicle.id: int(vehicle.onboard) for vehicle in environment.vehicles}

        for route in environment.current_state.routes:
            for job_id in route.job_ids:
                if environment.jobs.contains(job_id):
                    job                      = environment.jobs.by_id(job_id)
                    self.plan_before[job_id] = {"vehicle_id": route.vehicle_id, "priority": int(job.priority), "kind": job.kind}

    def capture_advanced(self, environment):
        """Record the pools, onboard loads and job assignment after the advance and before the disruption."""
        self._capture_pools("advanced", environment)
        self.onboard_advanced = {vehicle.id: int(vehicle.onboard) for vehicle in environment.vehicles}
        self.plan_advanced    = {job_id: route.vehicle_id for route in environment.current_state.routes for job_id in route.job_ids}

    def capture_disrupted(self, environment):
        """Record the pools after the disruption."""
        self._capture_pools("disrupted", environment)


class EpisodeDriver:
    def __init__(self, environment, config):
        self.environment     = environment
        self.max_disruptions = config.training.max_disruptions_per_episode
        self.max_edits       = config.training.max_edits_per_round
        self.totals          = self._empty_totals()
        self.pending_holding = 0.0
        self.records         = []

    def _empty_totals(self):
        return {"served": 0, "failed": 0, "dropped": 0}

    def _accumulate(self, summary):
        self.totals["served"]  += len(summary["served"])
        self.totals["failed"]  += len(summary["failed"])
        self.totals["dropped"] += summary["dropped"]

    def _open_round(self):
        record     = RoundRecord(len(self.records))
        record.gap = self.environment.sample_disruption_gap()

        self.pending_holding = self.environment.holding_cost(self.environment.current_state, record.gap)
        record.holding       = self.pending_holding
        record.capture_before(self.environment)

        record.executed = self.environment.advance_execution(record.gap)
        self._accumulate(record.executed)
        record.capture_advanced(self.environment)

        record.event = self.environment.apply_disruption(self.environment.current_state)
        record.capture_disrupted(self.environment)

        self.records.append(record)

    def settle_holding(self):
        holding              = self.pending_holding
        self.pending_holding = 0.0
        return -holding

    def rounds(self):
        """Run the edit rounds on the environment as it stands, yielding one EpisodeStep per decision.

        The environment must already hold an episode opened by `Environment.sample_episode` or `Environment.open_episode`. `records` gains one RoundRecord per round, starting with the opening record of round 0.
        """
        self.totals          = self._empty_totals()
        self.pending_holding = 0.0
        self.records         = [self.environment.opening_record]

        for round_index in range(self.max_disruptions):
            if round_index > 0:
                self._open_round()

            for edit_index in range(self.max_edits):
                remaining        = self.max_edits - edit_index
                graph, mask_info = self.environment.observe(round_index=round_index, edits_remaining=remaining)
                step             = EpisodeStep(self, round_index, edit_index, graph, mask_info, remaining)

                yield step

                if step.action is not None and step.action.operator == 2:
                    break

    def episode(self, seed):
        self.environment.sample_episode(seed)

        yield from self.rounds()
