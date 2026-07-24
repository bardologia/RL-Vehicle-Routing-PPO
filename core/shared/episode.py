class EpisodeStep:
    def __init__(self, environment, round_index, edit_index, graph, mask_info, remaining):
        self.environment = environment
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
        return old_state, new_state, rewards, costs


class EpisodeDriver:
    def __init__(self, environment, config):
        self.environment     = environment
        self.max_disruptions = config.training.max_disruptions_per_episode
        self.max_edits       = config.training.max_edits_per_round

    def episode(self, seed):
        self.environment.sample_episode(seed)

        for round_index in range(self.max_disruptions):
            if round_index > 0:
                gap = self.environment.sample_disruption_gap()
                self.environment.advance_execution(gap)
                self.environment.apply_disruption(self.environment.current_state)

            for edit_index in range(self.max_edits):
                graph, mask_info = self.environment.observe()
                step             = EpisodeStep(self.environment, round_index, edit_index, graph, mask_info, self.max_edits - edit_index)

                yield step

                if step.action is not None and step.action.operator == 2:
                    break
