from .services import vroom
from .state import RoutingState, Vehicle


class ActionHandler:
    def __init__(self, logger):
        self.logger    = logger
        self.operators = {
            0: self.insert_from_action,
            1: self.remove_from_action,
            2: self.keep_state,
        }

    def apply_job_insertion(self, env, state: RoutingState, vehicle_id: int, job_id: int) -> RoutingState:
        route           = state.route_of_vehicle(vehicle_id)
        current_job_ids = route.job_ids if route is not None else []

        candidate_ids  = current_job_ids + [job_id]
        candidate_jobs = [env.jobs.by_id(jid) for jid in candidate_ids if env.jobs.contains(jid)]
        vehicle        = env.vehicles.by_id(vehicle_id)

        new_route = self._solve_vehicle_route(env, candidate_jobs, vehicle)
        if new_route is None:
            self.logger.warning(f"Job insertion failed for vehicle_id={vehicle_id} job_id={job_id}")
            return state

        new_state = state.copy()
        new_state.replace_route(new_route)
        return new_state

    def apply_job_removal(self, env, state: RoutingState, vehicle_id: int, job_id: int) -> RoutingState:
        route = state.route_of_vehicle(vehicle_id)
        if route is None or job_id not in route.job_ids:
            return state

        remaining_ids  = [jid for jid in route.job_ids if jid != job_id]
        remaining_jobs = [env.jobs.by_id(jid) for jid in remaining_ids if env.jobs.contains(jid)]
        vehicle        = env.vehicles.by_id(vehicle_id)
        new_state      = state.copy()

        if not remaining_jobs and vehicle.onboard == 0:
            orphaned = new_state.remove_vehicles({vehicle_id})
            new_state.add_unassigned(orphaned)
            return new_state

        new_route = self._solve_vehicle_route(env, remaining_jobs, vehicle)
        if new_route is None:
            self.logger.warning(f"Job removal failed for vehicle_id={vehicle_id} job_id={job_id}")
            return state

        new_state.replace_route(new_route)
        new_state.add_unassigned({job_id})
        return new_state

    def _solve_vehicle_route(self, env, jobs, vehicle: Vehicle):
        solution = vroom.solve(jobs, [vehicle], depot=env.depot, clock=env.clock)
        if solution is None or not solution.routes:
            return None

        required = {job.id for job in jobs}
        if required & solution.unassigned_ids:
            return None

        return solution.routes[0]

    def insert_from_action(self, env, old_state, action):
        vehicle_id, job_id = self._action_ids(env, action)
        return self.apply_job_insertion(env, old_state, vehicle_id, job_id)

    def remove_from_action(self, env, old_state, action):
        vehicle_id, job_id = self._action_ids(env, action)
        return self.apply_job_removal(env, old_state, vehicle_id, job_id)

    def keep_state(self, env, old_state, action):
        return old_state

    def _action_ids(self, env, action):
        return env.vehicles[action.vehicle_index].id, env.jobs[action.job_index].id

    def apply(self, env, old_state, action):
        operator = action.operator
        if operator not in self.operators:
            raise ValueError(f"Unknown operator index: {operator}")

        return self.operators[operator](env, old_state, action)
