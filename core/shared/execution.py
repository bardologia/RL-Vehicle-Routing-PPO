from .services import vroom
from .state import RoutingState


class ExecutionAdvancer:
    def __init__(self, logger):
        self.logger = logger

    def _complete_due_stops(self, env, state, horizon, summary):
        remaining_by_vehicle = {}
        for route in state.routes:
            if not env.vehicles.contains(route.vehicle_id):
                continue

            remaining_by_vehicle[route.vehicle_id] = self._settle_route(env, route, horizon, summary)

        return remaining_by_vehicle

    def _settle_route(self, env, route, horizon, summary):
        vehicle   = env.vehicles.by_id(route.vehicle_id)
        completed = [stop for stop in route.stops if stop.completion <= horizon]

        planned_prev = vehicle.onboard
        for stop in completed:
            self._complete_stop(env, vehicle, stop, max(planned_prev - stop.load, 0), summary)
            planned_prev = stop.load

        if completed:
            vehicle.start = completed[-1].location

        self._carry_progress(vehicle, route, completed, env.clock, horizon)

        return [job_id for job_id in route.job_ids if env.jobs.contains(job_id)]

    def _complete_stop(self, env, vehicle, stop, planned_delivered, summary):
        if stop.kind == "delivery":
            dropped = min(planned_delivered, vehicle.onboard)
            summary["dropped"] += dropped
            vehicle.onboard    -= dropped
            return

        if stop.kind == "pickup":
            if env.execution_rng.random() < env.config.env.repossession_success_probability:
                vehicle.onboard += 1
                summary["served"].append(stop.job_id)
            else:
                summary["failed"].append(stop.job_id)
        else:
            summary["served"].append(stop.job_id)

        if env.jobs.contains(stop.job_id):
            env.jobs.remove({stop.job_id})

    def _carry_progress(self, vehicle, route, completed, clock, horizon):
        if len(completed) == len(route.stops):
            vehicle.progress_credit = 0
            return

        started_at              = completed[-1].completion if completed else vehicle.working_clock(clock)
        vehicle.progress_credit = max(int(horizon) - int(started_at), 0)

    def _resolve_remaining(self, env, state, remaining_by_vehicle):
        new_state = RoutingState(routes=[], unassigned_ids=set(state.unassigned_ids))
        for vehicle in env.vehicles:
            if vehicle.id not in remaining_by_vehicle:
                vehicle.progress_credit = 0

            self._resolve_vehicle(env, new_state, vehicle, remaining_by_vehicle.get(vehicle.id, []))

        return new_state

    def _resolve_vehicle(self, env, new_state, vehicle, remaining_ids):
        if not remaining_ids and vehicle.onboard == 0:
            vehicle.progress_credit = 0
            return

        if env.clock >= vehicle.time_window[1]:
            new_state.add_unassigned(set(remaining_ids))
            vehicle.progress_credit = 0
            return

        remaining_jobs = [env.jobs.by_id(job_id) for job_id in remaining_ids]
        solution       = vroom.solve(remaining_jobs, [vehicle], depot=env.depot, clock=env.clock)

        if solution is None:
            self.logger.warning(f"Execution re-solve failed for vehicle {vehicle.id}")
            new_state.add_unassigned(set(remaining_ids))
            vehicle.progress_credit = 0
            return

        if solution.routes:
            new_state.routes.append(solution.routes[0])
        new_state.add_unassigned(solution.unassigned_ids)

    def advance(self, env, state: RoutingState, gap_seconds: int):
        horizon = env.clock + gap_seconds
        summary = {"served": [], "failed": [], "dropped": 0}

        remaining_by_vehicle = self._complete_due_stops(env, state, horizon, summary)
        env.clock            = horizon

        vroom.clear_cache()

        new_state = self._resolve_remaining(env, state, remaining_by_vehicle)
        return new_state, summary
