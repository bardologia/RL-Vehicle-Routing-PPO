class EpisodeStep:
    def __init__(self, environment, index, graph, mask_info, remaining):
        self.environment = environment
        self.index       = index
        self.graph       = graph
        self.mask_info   = mask_info
        self.remaining   = remaining

    def commit(self, action):
        old_state, new_state = self.environment.apply_action_to(self.environment.current_state, action)
        rewards, costs       = self.environment.step(old_state, new_state, action.operator)

        self.environment.current_state = new_state
        return old_state, new_state, rewards, costs


class EpisodeDriver:
    def __init__(self, environment, config):
        self.environment = environment
        self.max_steps   = config.training.max_steps_per_episode

    def episode(self, seed):
        self.environment.sample_episode(seed)

        for step_index in range(self.max_steps):
            if step_index > 0:
                self.environment.advance_execution()
                self.environment.apply_random_event()

            graph, mask_info = self.environment.observe()
            yield EpisodeStep(self.environment, step_index, graph, mask_info, self.max_steps - step_index)
