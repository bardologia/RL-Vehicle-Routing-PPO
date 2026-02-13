from typing import Dict, List
import copy


class State:
    def __init__(self):
        self._data = {
            "routes": [],
            "unassigned": [],
            "summary": {}
        }
    
    def load_from_vroom(self, vroom_dict: Dict) -> None:
        self._data = copy.deepcopy(vroom_dict)
    
    def to_dict(self) -> Dict:
        return self._data
    
    def copy(self) -> 'State':
        new_state = State()
        new_state._data = copy.deepcopy(self._data)
        return new_state
    
    @property
    def routes(self) -> List[Dict]:
        return self._data.get("routes", [])
    
    @routes.setter
    def routes(self, value: List[Dict]) -> None:
        self._data["routes"] = value
    
    @property
    def unassigned(self) -> List[Dict]:
        return self._data.get("unassigned", [])
    
    @unassigned.setter
    def unassigned(self, value: List[Dict]) -> None:
        self._data["unassigned"] = value
    
    @property
    def summary(self) -> Dict:
        return self._data.setdefault("summary", {})
    
    @summary.setter
    def summary(self, value: Dict) -> None:
        self._data["summary"] = value
    
    def update_summary(self, **kwargs) -> None:
        """Update summary fields safely."""
        summary = self.summary
        for key, value in kwargs.items():
            summary[key] = value
    
    @property
    def cost(self) -> int:
        return self.summary.get("cost", 0)
    
    @property
    def num_routes(self) -> int:
        return len(self.routes)
    
    @property
    def num_unassigned(self) -> int:
        return len(self.unassigned)
 
    def __repr__(self) -> str:
        return (
            f"State(routes={self.num_routes}, "
            f"unassigned={self.num_unassigned}, "
            f"cost={self.cost})"
        )


class StateHandler:
    @staticmethod
    def rebuild_maps(jobs, vehicles):
        job_id_to_index = {int(job["id"]): i for i, job in enumerate(jobs)}
        jobs_by_id = {int(job["id"]): job for job in jobs}
        vehicle_id_to_index = {int(v["id"]): i for i, v in enumerate(vehicles)}
        vehicles_by_id = {int(v["id"]): v for v in vehicles}
        
        return job_id_to_index, jobs_by_id, vehicle_id_to_index, vehicles_by_id
    
    @staticmethod
    def get_route_by_vehicle(state, vehicle_id: int):
        for route in state.routes:
            if int(route.get("vehicle")) == vehicle_id:
                return route
        return None
    
    @staticmethod
    def get_job_ids_in_route(route):
        job_ids = []
        for step in route.get("steps", []):
            if step.get("type") == "job":
                job_id = step.get("job") or step.get("id")
                if job_id is not None:
                    job_ids.append(int(job_id))
        return job_ids
    
    @staticmethod
    def get_unassigned_job_ids(state):
        return {
            int(u["id"]) 
            for u in state.unassigned 
            if u.get("id") is not None
        }
    
    @staticmethod
    def get_all_assigned_job_ids(state):
        job_ids = set()
        for route in state.routes:
            job_ids.update(StateHandler.get_job_ids_in_route(route))
        return job_ids
    
    @staticmethod
    def is_job_assigned(state, job_id: int) -> bool:
        return job_id in StateHandler.get_all_assigned_job_ids(state)
    
    @staticmethod
    def find_job_route(state, job_id: int):
        for route in state.routes:
            if job_id in StateHandler.get_job_ids_in_route(route):
                return route
        return None
    
    @staticmethod
    def recompute_summary(state) -> None:
        total_cost = 0
        total_duration = 0
        total_service = 0
        total_waiting_time = 0
        total_distance = 0
        
        for route in state.routes:
            total_cost += route.get("cost", 0)
            total_duration += route.get("duration", 0)
            total_service += route.get("service", 0)
            total_waiting_time += route.get("waiting_time", 0)
            total_distance += route.get("distance", 0)
        
        state.summary["cost"] = total_cost
        state.summary["routes"] = len(state.routes)
        state.summary["unassigned"] = len(state.unassigned)
        state.summary["service"] = total_service
        state.summary["duration"] = total_duration
        state.summary["waiting_time"] = total_waiting_time
        state.summary["distance"] = total_distance
    
