from .state import RoutingState


class RewardCalculator:
    def _evaluate_cost(self, env, state: RoutingState):
        base_cost       = float(state.cost)
        num_unassigned  = state.num_unassigned
        active_vehicles = state.num_routes
        idle_vehicles   = len(env.vehicles) - active_vehicles

        unassigned_priority_sum = sum(
            env.jobs.by_id(job_id).priority
            for job_id in state.unassigned_ids
            if env.jobs.contains(job_id)
        )

        reward_config   = env.config.reward
        distance_cost   = reward_config.distance_weight * base_cost / 1000
        unassigned_cost = reward_config.unassigned_penalty_weight * num_unassigned
        idle_cost       = reward_config.idle_penalty_weight * idle_vehicles
        priority_cost   = reward_config.priority_penalty_weight * unassigned_priority_sum

        return distance_cost, unassigned_cost, idle_cost, priority_cost

    def evaluate_holding(self, env, state: RoutingState, gap_seconds: int) -> float:
        _, unassigned_cost, _, priority_cost = self._evaluate_cost(env, state)

        reference = max(float(env.config.reward.holding_reference_seconds), 1.0)
        return (unassigned_cost + priority_cost) * float(gap_seconds) / reference

    def _vehicle_by_job(self, state: RoutingState):
        return {stop.job_id: route.vehicle_id for route in state.routes for stop in route.stops if stop.kind in ("job", "pickup")}

    def _evaluate_disruption(self, old_state: RoutingState, new_state: RoutingState):
        old_vehicle_by_job = self._vehicle_by_job(old_state)
        new_vehicle_by_job = self._vehicle_by_job(new_state)

        return sum(1 for job_id, vehicle_id in old_vehicle_by_job.items() if new_vehicle_by_job.get(job_id) != vehicle_id)

    def evaluate(self, env, old_state, new_state, operator_index):
        old_distance_cost, old_unassigned_cost, old_idle_cost, old_priority_cost = self._evaluate_cost(env, old_state)
        new_distance_cost, new_unassigned_cost, new_idle_cost, new_priority_cost = self._evaluate_cost(env, new_state)

        distance_reward   = -(new_distance_cost   - old_distance_cost)
        unassigned_reward = -(new_unassigned_cost - old_unassigned_cost)
        idle_reward       = -(new_idle_cost       - old_idle_cost)
        priority_reward   = -(new_priority_cost   - old_priority_cost)

        reward_config  = env.config.reward
        operator_costs = {
            0: reward_config.add_job_cost,
            1: reward_config.remove_job_cost,
            2: reward_config.no_action_cost,
        }

        disruption    = self._evaluate_disruption(old_state, new_state)
        action_cost   = operator_costs[operator_index] + reward_config.disruption_cost * disruption
        action_reward = -action_cost

        costs = {
            "old_distance_cost"   : old_distance_cost,
            "old_unassigned_cost" : old_unassigned_cost,
            "old_idle_cost"       : old_idle_cost,
            "old_priority_cost"   : old_priority_cost,
            "new_distance_cost"   : new_distance_cost,
            "new_unassigned_cost" : new_unassigned_cost,
            "new_idle_cost"       : new_idle_cost,
            "new_priority_cost"   : new_priority_cost,
            "disruption"          : disruption,
            "action_cost"         : action_cost
        }

        rewards = {
            "distance_reward"   : distance_reward,
            "unassigned_reward" : unassigned_reward,
            "idle_reward"       : idle_reward,
            "priority_reward"   : priority_reward,
            "action_reward"     : action_reward
        }

        return rewards, costs
