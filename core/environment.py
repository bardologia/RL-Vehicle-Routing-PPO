import random
import numpy as np
from tools.auxiliary import add_jobs, add_vehicles, run_vroom, add_osrm_polylines
from core.graph import Graph
from core.mask import MaskContext
from core.state import State, StateHandler
import copy


class RouteHandler:
    @staticmethod
    def job_ids_from_steps(steps):
        return [
            int(step.get("job", step.get("id")))
            for step in (steps or [])
            if step.get("type") == "job" and step.get("job", step.get("id")) is not None
        ]

    @staticmethod
    def find_route_for_vehicle(routes, vehicle_id):
        for route in (routes or []):
            route_vehicle_raw = route.get("vehicle")
            if route_vehicle_raw is None:
                continue
            if int(route_vehicle_raw) == vehicle_id:
                return route
        return None
    
    @staticmethod
    def compute_route_costs(routes):
        total_cost     = sum(route.get("cost", 0) for route in (routes or []))
        total_duration = sum(route.get("duration", 0) for route in (routes or []))
        total_service  = sum(route.get("service", 0) for route in (routes or []))
        return total_cost, total_duration, total_service


class ActionHandler:
    @staticmethod
    def job_insertion(env, state_dict, vehicle_id, job_id):
        jobs_map = env.jobs_by_id
        vehicles_map = env.vehicles_by_id
        
        routes = state_dict.get("routes") or []

        target_route                = RouteHandler.find_route_for_vehicle(routes, vehicle_id)
        current_job_ids_for_vehicle = RouteHandler.job_ids_from_steps(target_route.get("steps")) if target_route else []

        original_vehicle = vehicles_map[vehicle_id]

        original_capacity_list  = list(original_vehicle.get("capacity") or [0])
        original_capacity_value = int(original_capacity_list[0]) if original_capacity_list else 0

        number_of_current_jobs  = len(current_job_ids_for_vehicle)

        if number_of_current_jobs >= original_capacity_value:
            new_capacity = original_capacity_value + 1
        else:
            new_capacity = original_capacity_value

        vehicle_for_test = copy.deepcopy(original_vehicle)
        
        if new_capacity != original_capacity_value:
            vehicle_for_test["capacity"] = [new_capacity]

        vehicles_payload_for_test = [vehicle_for_test]

        if not current_job_ids_for_vehicle:
            base_cost = 0
        else:
            base_cost = int(target_route.get("cost", 0)) if target_route else 0

        candidate_job_ids      = current_job_ids_for_vehicle + [job_id]
        candidate_jobs_payload = [jobs_map[jid] for jid in candidate_job_ids if jid in jobs_map]

        new_solution = run_vroom(candidate_jobs_payload, vehicles_payload_for_test)
        if not new_solution:
            return None
        routes = new_solution.get("routes") or []

        new_route = routes[0]
        new_cost_value = new_route.get("cost")

        delta_cost        = new_cost_value - base_cost
        best_new_solution = add_osrm_polylines(new_solution) 
        if best_new_solution is None:
            return None

        return {
            "vehicle"          : vehicle_for_test,
            "job"              : jobs_map[job_id],
            "delta_cost"       : int(delta_cost),
            "new_solution"     : best_new_solution,
        }

    @staticmethod
    def job_removal(env, state_dict, vehicle_id, job_id):
        jobs_map = env.jobs_by_id
        vehicles_map = env.vehicles_by_id
        
        routes_list = state_dict.get("routes") or []
        target_route   = RouteHandler.find_route_for_vehicle(routes_list, vehicle_id)
        if target_route is None:
            print("Error: target_route is None in job removal.")
            return None
        
        vehicle_object = vehicles_map.get(vehicle_id)

        steps_in_route   = target_route.get("steps") or []
        job_ids_in_route = RouteHandler.job_ids_from_steps(steps_in_route)

        if not job_ids_in_route or job_id not in job_ids_in_route:
            return None

        base_cost_raw   = target_route.get("cost")

        remaining_job_ids  = [jid for jid in job_ids_in_route if jid != job_id]
        removed_job_object = jobs_map.get(job_id)

        if not remaining_job_ids:
            delta_cost = base_cost_raw
            new_route = copy.deepcopy(target_route)

            if new_route is None:
                print("Error: new_route is None in job removal.")
                return None
            
            new_steps = [step for step in (steps_in_route or []) if step.get("type") != "job"]
            new_route["steps"]        = new_steps
            new_route["cost"]         = 0
            new_route["service"]      = 0
            new_route["duration"]     = 0
            new_route["waiting_time"] = 0

            if isinstance(new_route.get("amount"), list):
                new_route["amount"] = [0] * len(new_route["amount"])
            if isinstance(new_route.get("delivery"), list):
                new_route["delivery"] = [0] * len(new_route["delivery"])
            if isinstance(new_route.get("pickup"), list):
                new_route["pickup"] = [0] * len(new_route["pickup"])

            vehicle_solution = {"routes": [new_route]}
            vehicle_solution = add_osrm_polylines(vehicle_solution)

            return {
                "vehicle"          : vehicle_object,
                "delta"            : int(delta_cost),
                "job"              : removed_job_object,
                "vehicle_solution" : vehicle_solution,
            }

        remaining_jobs_payload = [jobs_map[jid] for jid in remaining_job_ids if jid in jobs_map]
        vehicle_payload        = [vehicle_object]
        vehicle_solution       = run_vroom(remaining_jobs_payload, vehicle_payload)

        if vehicle_solution is None:
            print("Error: vehicle_solution is None in job removal.")
            return None

        new_routes_for_vehicle = vehicle_solution.get("routes") or []

        new_route      = new_routes_for_vehicle[0]
        new_cost_raw   = new_route.get("cost")
        delta_cost     = base_cost_raw - new_cost_raw

        vehicle_solution = add_osrm_polylines(vehicle_solution)

        return {
            "vehicle"          : vehicle_object,
            "delta"            : int(delta_cost),
            "job"              : removed_job_object,
            "vehicle_solution" : vehicle_solution,
        }

    @staticmethod
    def merge_solution(original_state_dict, vehicle_solution):
        updated_state = copy.deepcopy(original_state_dict)
        vehicle_routes = vehicle_solution.get("routes") or []

        new_route         = copy.deepcopy(vehicle_routes[0])
        target_vehicle_id = int(new_route.get("vehicle"))

        previous_route_for_vehicle = None
        merged_routes = []

        has_jobs_in_new_route = any(step.get("type") == "job" for step in (new_route.get("steps") or []))

        for route in updated_state.get("routes", []):
            if int(route.get("vehicle")) == target_vehicle_id:
                previous_route_for_vehicle = route
                if has_jobs_in_new_route:
                    merged_routes.append(new_route)
            else:
                merged_routes.append(route)

        if previous_route_for_vehicle is None and has_jobs_in_new_route:
            merged_routes.append(new_route)

        updated_state["routes"] = merged_routes

        previous_job_ids = set(RouteHandler.job_ids_from_steps((previous_route_for_vehicle.get("steps") if previous_route_for_vehicle else [])))
        new_job_ids      = set(RouteHandler.job_ids_from_steps(new_route.get("steps")))
        removed_job_ids  = previous_job_ids - new_job_ids

        job_location_map = {}
        for step in (previous_route_for_vehicle.get("steps") if previous_route_for_vehicle else []) or []:
            if step.get("type") == "job" and "location" in step and step.get("job") is not None:
                job_location_map[int(step["job"])] = step["location"]

        for step in (new_route.get("steps") or []):
            if step.get("type") == "job" and "location" in step and step.get("job") is not None:
                jid = int(step["job"])
                if jid not in job_location_map:
                    job_location_map[jid] = step["location"]

        current_unassigned_entries = list(updated_state.get("unassigned") or [])
        current_unassigned_entries = [uns for uns in current_unassigned_entries if uns.get("id") is not None and int(uns.get("id")) not in new_job_ids]

        existing_unassigned_ids = {int(uns.get("id")) for uns in current_unassigned_entries if uns.get("id") is not None}

        for removed_job_id in sorted(removed_job_ids):
            if removed_job_id not in existing_unassigned_ids:
                current_unassigned_entries.append(
                    {
                        "id": int(removed_job_id),
                        "type": "job",
                        "location": job_location_map.get(int(removed_job_id)),
                    }
                )

        updated_state["unassigned"] = current_unassigned_entries
        
        routes = updated_state.get("routes") or []
        unassigned_entries = updated_state.get("unassigned") or []
        total_cost, total_duration, total_service = RouteHandler.compute_route_costs(routes)
        
        summary_data = dict(updated_state.get("summary") or {})
        summary_data["cost"] = int(total_cost)
        summary_data["routes"] = int(len(routes))
        summary_data["unassigned"] = int(len(unassigned_entries))
        summary_data["service"] = int(total_service)
        summary_data["duration"] = int(total_duration)
        updated_state["summary"] = summary_data

        return updated_state


class EventHandler:
    @staticmethod
    def apply_new_job(env_obj, unassigned_jobs, routes_list, num_items, config):
        env_cfg = config.env
        env_obj.jobs = add_jobs(env_obj.jobs, num_items, env_cfg.center, env_cfg.radius, env_cfg.outlier_probability, env_cfg.outlier_multiplier)
        new_jobs = env_obj.jobs[-num_items:]

        updated_unassigned = list(unassigned_jobs)
        for new_job in new_jobs:
            updated_unassigned.append({
                "id": new_job["id"],
                "location": new_job.get("location"),
                "type": "job",
                "description": new_job.get("description"),
            })

        env_obj.job_id_to_index, env_obj.jobs_by_id, env_obj.vehicle_id_to_index, env_obj.vehicles_by_id = StateHandler.rebuild_maps(env_obj.jobs, env_obj.vehicles)
        return routes_list, updated_unassigned

    @staticmethod
    def apply_new_vehicle(env_obj, unassigned_jobs, routes_list, num_items, config):
        env_cfg = config.env
        env_obj.vehicles = add_vehicles(env_obj.vehicles, num_items, env_cfg.center, env_cfg.radius, env_cfg.outlier_probability, env_cfg.outlier_multiplier)
        env_obj.job_id_to_index, env_obj.jobs_by_id, env_obj.vehicle_id_to_index, env_obj.vehicles_by_id = StateHandler.rebuild_maps(env_obj.jobs, env_obj.vehicles)
        return routes_list, unassigned_jobs

    @staticmethod
    def apply_remove_job(env_obj, unassigned_jobs, routes_list, num_items):
        if len(env_obj.jobs) == 0:
            return None

        num_items = min(num_items, len(env_obj.jobs))

        jobs_to_remove     = random.sample(env_obj.jobs, num_items)
        job_ids_to_remove  = {job.get("id") for job in jobs_to_remove}

        env_obj.jobs = [job for job in env_obj.jobs if job.get("id") not in job_ids_to_remove]
        env_obj.job_id_to_index, env_obj.jobs_by_id, env_obj.vehicle_id_to_index, env_obj.vehicles_by_id = StateHandler.rebuild_maps(env_obj.jobs, env_obj.vehicles)

        updated_unassigned = [u for u in unassigned_jobs if u.get("id") not in job_ids_to_remove]

        updated_routes = []
        for route in routes_list:
            steps = route.get("steps") or []

            kept_steps    = []
            delta_service = 0
            last_job_step = None
            end_step      = None

            for step in steps:
                stype = step.get("type")

                if stype == "job":
                    step_job_id = step.get("job", step.get("id"))
                    if step_job_id in job_ids_to_remove:
                        delta_service += step.get("service", 0)
                        continue
                    last_job_step = step
                    kept_steps.append(step)

                elif stype == "end":
                    end_step = step
                    kept_steps.append(step)

                else:
                    kept_steps.append(step)

            if delta_service:
                route["service"] = max(0, route.get("service", 0) - delta_service)

            route["steps"] = kept_steps

            if any(step.get("type") == "job" for step in kept_steps):
                if last_job_step is not None and end_step is not None:
                    end_step["location"] = last_job_step.get("location", end_step.get("location"))
                    end_step["arrival"]  = last_job_step.get("arrival",  end_step.get("arrival"))
                    end_step["duration"] = last_job_step.get("duration", end_step.get("duration"))
                    end_step["load"]     = last_job_step.get("load",     end_step.get("load"))

                updated_routes.append(route)

        return updated_routes, updated_unassigned

    @staticmethod
    def apply_remove_vehicle(env_obj, unassigned_jobs, routes_list, num_items):
        num_items = min(num_items, len(env_obj.vehicles) - 1)

        vehicles_to_remove    = random.sample(env_obj.vehicles, num_items)
        vehicle_ids_to_remove = {v.get("id") for v in vehicles_to_remove}

        env_obj.vehicles = [v for v in env_obj.vehicles if v.get("id") not in vehicle_ids_to_remove]
        env_obj.job_id_to_index, env_obj.jobs_by_id, env_obj.vehicle_id_to_index, env_obj.vehicles_by_id = StateHandler.rebuild_maps(env_obj.jobs, env_obj.vehicles)

        updated_routes = []
        updated_unassigned = list(unassigned_jobs)
        for route in routes_list:
            route_vehicle_id = route.get("vehicle")
            if route_vehicle_id in vehicle_ids_to_remove:
                for step in route.get("steps") or []:
                    if step.get("type") == "job":
                        removed_job_id_raw = step.get("job", step.get("id"))
                        if removed_job_id_raw is None:
                            continue
                        removed_job_id = int(removed_job_id_raw)

                        removed_job = env_obj.jobs_by_id.get(removed_job_id)
                        if removed_job is not None:
                            updated_unassigned.append({
                                "id": removed_job_id,
                                "location": removed_job.get("location"),
                                "type": removed_job.get("type", "job"),
                                "description": removed_job.get("description"),
                            })
            else:
                updated_routes.append(route)

        return updated_routes, updated_unassigned


class Environment:
    def __init__(self, config):
        self.config = config
        self.jobs     = []
        self.vehicles = []

        self.job_id_to_index     = {}
        self.vehicle_id_to_index = {}
        self.jobs_by_id          = {}
        self.vehicles_by_id      = {}

        self.current_state: State = None
        self.initial_state: State = None

        self.graph         = Graph(config.device)
        self.mask_context  = MaskContext()

        self.reset()

    def reset(self):
        env = self.config.env
        while True:
            n_jobs     = max(env.min_jobs,     int(np.random.normal(loc=env.mean_jobs,     scale=env.std_jobs)))
            n_vehicles = max(env.min_vehicles, int(np.random.normal(loc=env.mean_vehicles, scale=env.std_vehicles)))

            self.jobs     = add_jobs([], n_jobs, env.center, env.radius, env.outlier_probability, env.outlier_multiplier)
            self.vehicles = add_vehicles([], n_vehicles, env.center, env.radius, env.outlier_probability, env.outlier_multiplier)

            self.job_id_to_index, self.jobs_by_id, self.vehicle_id_to_index, self.vehicles_by_id = StateHandler.rebuild_maps(self.jobs, self.vehicles)

            vroom_dict = run_vroom(jobs_list=self.jobs, vehicles_list=self.vehicles)
            if vroom_dict is None:
                continue

            vroom_dict = add_osrm_polylines(vroom_dict)
            if vroom_dict is None:
                continue

            self.current_state = State()
            self.current_state.load_from_vroom(vroom_dict)
            self.initial_state = self.current_state.copy()
            break

    def load_from_dataset(self, dataset):
        self.jobs     = dataset.get('jobs')
        self.vehicles = dataset.get('vehicles')
        self.job_id_to_index, self.jobs_by_id, self.vehicle_id_to_index, self.vehicles_by_id = StateHandler.rebuild_maps(self.jobs, self.vehicles)
        vroom_dict = dataset.get('event_state')
        vroom_dict = add_osrm_polylines(vroom_dict)
        if vroom_dict is None:
            raise ValueError("Failed to add OSRM polylines in load_from_dataset")
        self.current_state = State()
        self.current_state.load_from_vroom(vroom_dict)

    def observe(self):
        graph = self.graph.build(self.jobs, self.vehicles, self.current_state.to_dict())
        
        mask_context = self.mask_context.build(
            self.current_state.routes,
            StateHandler.get_unassigned_job_ids(self.current_state),
            self.job_id_to_index,
            self.jobs_by_id,
            self.vehicle_id_to_index,
            self.vehicles
        )
        
        return graph, mask_context

    def generate_event(self):
        env = self.config.env
        event_type = random.choice(["new_job", "remove_job", "new_vehicle", "remove_vehicle"])

        if event_type == "new_job":
            num_items = random.randint(env.job_insert_min, env.job_insert_max)
            return event_type, num_items

        if event_type == "new_vehicle":
            num_items = random.randint(env.vehicle_insert_min, env.vehicle_insert_max)
            return event_type, num_items

        if event_type == "remove_job":
            total_jobs = len(self.jobs)
            max_items_possible = min(env.job_remove_max, total_jobs)
            min_items_possible = min(env.job_remove_min, max_items_possible)
            if min_items_possible > max_items_possible:
                min_items_possible = max_items_possible
            num_items = random.randint(min_items_possible, max_items_possible) if max_items_possible > 0 else 0
            return event_type, num_items

        if event_type == "remove_vehicle":
            total_vehicles = len(self.vehicles)
            max_items_possible = min(env.vehicle_remove_max, total_vehicles - 1)
            min_items_possible = min(env.vehicle_remove_min, max_items_possible)
            if min_items_possible > max_items_possible:
                min_items_possible = max_items_possible
            num_items = random.randint(min_items_possible, max_items_possible) if max_items_possible > 0 else 0
            return event_type, num_items

    def apply_event(self, state: State, event_type, num_items=1) -> State:
        event_state     = state.copy()
        unassigned_jobs = list(event_state.unassigned)
        routes_list     = list(event_state.routes)

        if event_type == "new_job":
            routes_list, unassigned_jobs = EventHandler.apply_new_job(self, unassigned_jobs, routes_list, num_items, self.config)
        
        elif event_type == "new_vehicle":
            routes_list, unassigned_jobs = EventHandler.apply_new_vehicle(self, unassigned_jobs, routes_list, num_items, self.config)
        
        elif event_type == "remove_job":
            routes_list, unassigned_jobs = EventHandler.apply_remove_job(self, unassigned_jobs, routes_list, num_items)
          
        elif event_type == "remove_vehicle":
            routes_list, unassigned_jobs = EventHandler.apply_remove_vehicle(self, unassigned_jobs, routes_list, num_items)
            
        event_state.routes     = routes_list
        event_state.unassigned = unassigned_jobs
        StateHandler.recompute_summary(event_state)
        self.current_state     = event_state
        return event_state

    def apply_action(self, action):
        operator      = action.operator
        vehicle_index = action.vehicle_index
        job_index     = action.job_index
        
        old_state      = self.current_state.copy()
        
        vehicle_object = self.vehicles[vehicle_index]
        vehicle_id     = int(vehicle_object["id"])
        
        job_object     = self.jobs[job_index]
        job_id         = int(job_object["id"])

        if operator == 0:
            insertion_result = ActionHandler.job_insertion(self, old_state.to_dict(), vehicle_id=vehicle_id, job_id=job_id)
            if insertion_result is None:
                return old_state, old_state
            vehicle_solution = insertion_result["new_solution"]
            merged_dict      = ActionHandler.merge_solution(old_state.to_dict(), vehicle_solution)
            merged_dict      = add_osrm_polylines(merged_dict)
            if merged_dict is None:
                return old_state, old_state
            new_state        = State()
            new_state.load_from_vroom(merged_dict)
            
        elif operator == 1:
            removal_result   = ActionHandler.job_removal(self, old_state.to_dict(), vehicle_id=vehicle_id, job_id=job_id)
            if removal_result is None:
                return old_state, old_state
            vehicle_solution = removal_result["vehicle_solution"]
            merged_dict      = ActionHandler.merge_solution(old_state.to_dict(), vehicle_solution)
            merged_dict      = add_osrm_polylines(merged_dict)
            if merged_dict is None:
                return old_state, old_state
            new_state        = State()
            new_state.load_from_vroom(merged_dict)

        elif operator == 2:
            new_state = old_state

        elif operator == 3:
            vroom_dict = run_vroom(jobs_list=self.jobs, vehicles_list=self.vehicles)
            if vroom_dict is None:
                return old_state, old_state
            vroom_dict = add_osrm_polylines(vroom_dict)
            if vroom_dict is None:
                return old_state, old_state
            new_state = State()
            new_state.load_from_vroom(vroom_dict)
        
        else:
            new_state = old_state
        
        return old_state, new_state

    def evaluate_cost(self, state: State):
        reward_cfg = self.config.reward
        base_cost       = float(state.cost)
        num_unassigned  = state.num_unassigned
        active_vehicles = state.num_routes
        idle_vehicles   = len(self.vehicles) - active_vehicles

        unassigned_priority_sum = 0.0
        for unassigned_job in state.unassigned:
            job = self.jobs_by_id.get(int(unassigned_job["id"]))
            if job and "priority" in job:
                unassigned_priority_sum += float(job["priority"])

        distance_cost   = reward_cfg.distance_weight * base_cost / 1000
        unassigned_cost = reward_cfg.unassigned_penalty_weight * num_unassigned
        idle_cost       = reward_cfg.idle_penalty_weight * idle_vehicles
        priority_cost   = reward_cfg.priority_penalty_weight * unassigned_priority_sum

        return distance_cost, unassigned_cost, idle_cost, priority_cost

    def step(self, old_state, new_state, operator_idx):
        old_distance_cost, old_unassigned_cost, old_idle_cost, old_priority_cost = self.evaluate_cost(old_state)

        old_costs = {
            "old_distance_cost": old_distance_cost,
            "old_unassigned_cost": old_unassigned_cost,
            "old_idle_cost": old_idle_cost,
            "old_priority_cost": old_priority_cost,
        }

        new_distance_cost, new_unassigned_cost, new_idle_cost, new_priority_cost = self.evaluate_cost(new_state)
        
        new_costs = {
            "new_distance_cost": new_distance_cost,
            "new_unassigned_cost": new_unassigned_cost,
            "new_idle_cost": new_idle_cost,
            "new_priority_cost": new_priority_cost
        }

        distance_reward   = -(new_distance_cost   - old_distance_cost)
        unassigned_reward = -(new_unassigned_cost - old_unassigned_cost)
        idle_reward       = -(new_idle_cost       - old_idle_cost)
        priority_reward   = -(new_priority_cost   - old_priority_cost)
        reward_cfg = self.config.reward
        
        action_penalties = {
            0: reward_cfg.add_job_penalty,
            1: reward_cfg.remove_job_penalty,
            2: reward_cfg.invalid_action_penalty,
            3: reward_cfg.reoptimize_penalty
        }
        action_reward = action_penalties.get(operator_idx, 0)

        rewards = {
            "distance_reward": distance_reward,
            "unassigned_reward": unassigned_reward,
            "idle_reward": idle_reward,
            "priority_reward": priority_reward,
            "action_reward": action_reward
        }

        return rewards, old_costs, new_costs
