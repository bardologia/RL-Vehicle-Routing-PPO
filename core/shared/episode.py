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


class EpisodeDriver:
    def __init__(self, environment, config):
        self.environment     = environment
        self.max_disruptions = config.training.max_disruptions_per_episode
        self.max_edits       = config.training.max_edits_per_round
        self.totals          = self._empty_totals()
        self.pending_holding = 0.0

    def _empty_totals(self):
        return {"served": 0, "failed": 0, "dropped": 0}

    def _accumulate(self, summary):
        self.totals["served"]  += len(summary["served"])
        self.totals["failed"]  += len(summary["failed"])
        self.totals["dropped"] += summary["dropped"]

    def _open_round(self):
        gap = self.environment.sample_disruption_gap()

        self.pending_holding = self.environment.holding_cost(self.environment.current_state, gap)
        self._accumulate(self.environment.advance_execution(gap))
        self.environment.apply_disruption(self.environment.current_state)

    def settle_holding(self):
        holding              = self.pending_holding
        self.pending_holding = 0.0
        return -holding

    def episode(self, seed):
        self.environment.sample_episode(seed)
        self.totals          = self._empty_totals()
        self.pending_holding = 0.0

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
