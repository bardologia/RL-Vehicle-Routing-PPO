import math
import random
import threading

import requests


class ScenarioLab:
    OPERATOR_NAMES = {0: "INSERT", 1: "REMOVE", 2: "DO_NOTHING", 3: "REOPTIMIZE"}
    AGENTS         = ("model", "teacher", "insertion_only", "always_reoptimize", "do_nothing")

    def __init__(self, paths, logger):
        self.paths  = paths
        self.logger = logger
        self.lock   = threading.Lock()

        self._env    = None
        self._models = {}

    def health(self):
        from configuration import config

        status = {"osrm_url": config.service.osrm_url, "vroom_url": config.service.vroom_url}

        try:
            probe          = f"{config.service.osrm_url}/route/v1/driving/-46.63,-23.55;-46.62,-23.54?overview=false"
            status["osrm"] = requests.get(probe, timeout=2).status_code == 200
        except requests.RequestException:
            status["osrm"] = False

        try:
            status["vroom"] = requests.get(f"{config.service.vroom_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            status["vroom"] = False

        return status

    def checkpoints(self):
        if not self.paths.runs_dir.is_dir():
            return []

        entries = []
        for run_dir in sorted(self.paths.runs_dir.iterdir()):
            checkpoint = run_dir / "graph_ppo_policy.pt"
            if checkpoint.exists():
                entries.append({"run": run_dir.name, "mtime": checkpoint.stat().st_mtime})

        entries.sort(key=lambda entry: entry["mtime"], reverse=True)
        return entries

    def _ensure_env(self):
        if self._env is None:
            from configuration import config
            config.training.device = "cpu"

            from core.shared.environment import Environment
            self._env = Environment(config)

        return self._env

    def sample(self, num_jobs, num_vehicles, seed):
        import numpy as np
        from configuration import config
        from core.shared.environment import ScenarioSampler
        from core.shared.state import EntityPool

        random.seed(seed)
        np.random.seed(seed)

        sampler  = ScenarioSampler(config.env)
        jobs     = sampler.sample_jobs(EntityPool(), num_jobs)
        vehicles = sampler.sample_vehicles(EntityPool(), num_vehicles)

        return {
            "jobs"     : [job.to_dict() for job in jobs],
            "vehicles" : [vehicle.to_dict() for vehicle in vehicles],
        }

    def _decode_path(self, route):
        import polyline

        if route.geometry:
            return polyline.decode(route.geometry)
        if route.path_coords:
            return [list(point) for point in route.path_coords]
        return None

    def _render_state(self, state):
        routes = []
        for route in state.routes:
            routes.append({
                "vehicle_id" : route.vehicle_id,
                "cost"       : route.cost,
                "duration"   : route.duration,
                "distance"   : route.distance,
                "start"      : list(route.start) if route.start is not None else None,
                "end"        : list(route.end) if route.end is not None else None,
                "stops"      : [stop.to_dict() for stop in route.stops],
                "path"       : self._decode_path(route),
            })

        return {
            "routes"         : routes,
            "unassigned"     : sorted(state.unassigned_ids),
            "cost"           : state.cost,
            "duration"       : state.duration,
            "distance"       : state.distance,
            "num_routes"     : state.num_routes,
            "num_unassigned" : state.num_unassigned,
        }

    def solve(self, jobs, vehicles):
        with self.lock:
            self._ensure_env()

            from core.shared.services import vroom
            from core.shared.state import Job, Vehicle

            job_objects     = [Job.from_dict(job) for job in jobs]
            vehicle_objects = [Vehicle.from_dict(vehicle) for vehicle in vehicles]

            state = vroom.solve(job_objects, vehicle_objects)
            if state is None:
                return {"error": "VROOM returned no solution for this scenario"}

            return {"state": self._render_state(state)}

    def _build_agent(self, name, run_name):
        from configuration import config
        from core.inference.evaluation import FixedOperatorAgent, ModelAgent, TeacherAgent
        from core.training.pretraining import RegretInsertionTeacher

        if name == "model":
            return ModelAgent(self._model_for(run_name))
        if name == "teacher":
            return TeacherAgent(RegretInsertionTeacher(config))
        if name == "insertion_only":
            return TeacherAgent(RegretInsertionTeacher(config, reoptimize_margin=math.inf, allow_removal=False))
        if name == "always_reoptimize":
            return FixedOperatorAgent(3)
        if name == "do_nothing":
            return FixedOperatorAgent(2)

        raise ValueError(f"Unknown agent '{name}'")

    def _model_for(self, run_name):
        if not run_name:
            raise ValueError("agent 'model' needs a run_name with a checkpoint")

        from configuration import config
        from model.policy_model import Policy, PolicyCheckpoint

        run_dir    = self.paths.run_dir(run_name)
        checkpoint = run_dir / config.io.checkpoint_filename
        if not checkpoint.exists():
            raise ValueError(f"run '{run_name}' has no checkpoint")

        mtime  = checkpoint.stat().st_mtime
        cached = self._models.get(run_name)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        model = Policy(config)
        PolicyCheckpoint().load(model, config.io.checkpoint_filename, str(run_dir), map_location="cpu")
        model.eval()

        self._models[run_name] = (mtime, model)
        return model

    def _apply_events(self, env, probability):
        if probability <= 0 or random.random() >= probability:
            return []

        event_type, num_items = env.generate_event()
        if num_items <= 0:
            return []

        env.apply_event(env.current_state, event_type, num_items)
        return [{"type": event_type, "count": num_items}]

    def _render_action(self, env, action):
        entry = {
            "operator"      : action.operator,
            "operator_name" : self.OPERATOR_NAMES[action.operator],
        }

        if action.operator in (0, 1):
            entry["vehicle_id"] = env.vehicles[action.vehicle_index].id
            entry["job_id"]     = env.jobs[action.job_index].id

        return entry

    def _render_step(self, index, env, action, rewards, costs, events, cumulative_reward):
        step = {
            "index"             : index,
            "events"            : events,
            "action"            : action,
            "rewards"           : rewards,
            "costs"             : costs,
            "total_reward"      : sum(rewards.values()) if rewards else None,
            "cumulative_reward" : cumulative_reward,
            "state"             : self._render_state(env.current_state),
            "jobs"              : [job.to_dict() for job in env.jobs],
            "vehicles"          : [vehicle.to_dict() for vehicle in env.vehicles],
        }
        return step

    def run(self, payload):
        jobs              = payload.get("jobs") or []
        vehicles          = payload.get("vehicles") or []
        agent_name        = payload.get("agent", "model")
        run_name          = payload.get("run_name")
        max_steps         = int(payload.get("max_steps", 10))
        seed              = int(payload.get("seed", 0))
        event_probability = float(payload.get("event_probability", 0.0))

        if not jobs or not vehicles:
            return {"error": "scenario needs at least one job and one vehicle"}
        if agent_name not in self.AGENTS:
            return {"error": f"unknown agent '{agent_name}'"}
        if not 1 <= max_steps <= 50:
            return {"error": "max_steps must be between 1 and 50"}
        if not 0.0 <= event_probability <= 1.0:
            return {"error": "event_probability must be between 0 and 1"}

        with self.lock:
            env = self._ensure_env()

            import numpy as np
            import torch
            from core.shared.services import vroom
            from core.shared.state import EntityPool, Job, Vehicle

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            env.jobs     = EntityPool([Job.from_dict(job) for job in jobs])
            env.vehicles = EntityPool([Vehicle.from_dict(vehicle) for vehicle in vehicles])

            solution = vroom.solve(list(env.jobs), list(env.vehicles))
            if solution is None:
                return {"error": "VROOM returned no solution for this scenario"}

            env.current_state = solution
            env.initial_state = solution.copy()

            agent = self._build_agent(agent_name, run_name)

            steps             = [self._render_step(0, env, None, None, None, [], 0.0)]
            cumulative_reward = 0.0
            stopped_reason    = "max_steps_reached"

            for index in range(1, max_steps + 1):
                events = self._apply_events(env, event_probability) if index > 1 else []

                graph, mask_info = env.observe()
                action           = agent.act(env, graph, mask_info)

                old_state, new_state = env.apply_action(action)
                rewards, costs       = env.step(old_state, new_state, action.operator)

                env.current_state  = new_state
                cumulative_reward += sum(rewards.values())

                rendered = self._render_action(env, action)
                steps.append(self._render_step(index, env, rendered, rewards, costs, events, cumulative_reward))

                if action.operator == 2:
                    stopped_reason = "agent_do_nothing"
                    break

            initial = steps[0]["state"]
            final   = steps[-1]["state"]

            summary = {
                "agent"              : agent_name,
                "run_name"           : run_name if agent_name == "model" else None,
                "total_steps"        : len(steps) - 1,
                "stopped_reason"     : stopped_reason,
                "initial_cost"       : initial["cost"],
                "final_cost"         : final["cost"],
                "initial_unassigned" : initial["num_unassigned"],
                "final_unassigned"   : final["num_unassigned"],
                "cumulative_reward"  : cumulative_reward,
            }

            return {"steps": steps, "summary": summary}
