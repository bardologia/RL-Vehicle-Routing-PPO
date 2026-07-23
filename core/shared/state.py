import copy
import random
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Set, Tuple


@dataclass
class Job:
    id          : int
    location    : Tuple[float, float]
    service     : int = 300
    setup       : int = 0
    amount      : int = 1
    priority    : int = 1
    description : str = ""

    def vroom_payload(self) -> Dict:
        return {
            "id"          : int(self.id),
            "location"    : [float(self.location[0]), float(self.location[1])],
            "setup"       : int(self.setup),
            "service"     : int(self.service),
            "amount"      : [int(self.amount)],
            "priority"    : int(self.priority),
            "description" : self.description,
        }

    def to_dict(self) -> Dict:
        return {
            "id"          : int(self.id),
            "location"    : [float(self.location[0]), float(self.location[1])],
            "service"     : int(self.service),
            "setup"       : int(self.setup),
            "amount"      : int(self.amount),
            "priority"    : int(self.priority),
            "description" : self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Job":
        return cls(
            id          = int(data["id"]),
            location    = (float(data["location"][0]), float(data["location"][1])),
            service     = int(data["service"]),
            setup       = int(data["setup"]),
            amount      = int(data["amount"]),
            priority    = int(data["priority"]),
            description = data["description"],
        )


@dataclass
class Vehicle:
    id              : int
    start           : Tuple[float, float]
    capacity        : int = 1
    speed_factor    : float = 1.0
    time_window     : Tuple[int, int] = (8 * 3600, 20 * 3600)
    return_to_depot : bool = False
    description     : str = ""

    def vroom_payload(self) -> Dict:
        return {
            "id"              : int(self.id),
            "start"           : [float(self.start[0]), float(self.start[1])],
            "capacity"        : [int(self.capacity)],
            "time_window"     : [int(self.time_window[0]), int(self.time_window[1])],
            "speed_factor"    : float(self.speed_factor),
            "return_to_depot" : bool(self.return_to_depot),
            "description"     : self.description,
        }

    def to_dict(self) -> Dict:
        return {
            "id"              : int(self.id),
            "start"           : [float(self.start[0]), float(self.start[1])],
            "capacity"        : int(self.capacity),
            "speed_factor"    : float(self.speed_factor),
            "time_window"     : [int(self.time_window[0]), int(self.time_window[1])],
            "return_to_depot" : bool(self.return_to_depot),
            "description"     : self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Vehicle":
        return cls(
            id              = int(data["id"]),
            start           = (float(data["start"][0]), float(data["start"][1])),
            capacity        = int(data["capacity"]),
            speed_factor    = float(data["speed_factor"]),
            time_window     = (int(data["time_window"][0]), int(data["time_window"][1])),
            return_to_depot = bool(data["return_to_depot"]),
            description     = data["description"],
        )


@dataclass
class Stop:
    job_id   : int
    location : Tuple[float, float]
    arrival  : int = 0
    duration : int = 0
    service  : int = 0
    load     : int = 0

    def to_dict(self) -> Dict:
        return {
            "job_id"   : int(self.job_id),
            "location" : [float(self.location[0]), float(self.location[1])],
            "arrival"  : int(self.arrival),
            "duration" : int(self.duration),
            "service"  : int(self.service),
            "load"     : int(self.load),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Stop":
        return cls(
            job_id   = int(data["job_id"]),
            location = (float(data["location"][0]), float(data["location"][1])),
            arrival  = int(data["arrival"]),
            duration = int(data["duration"]),
            service  = int(data["service"]),
            load     = int(data["load"]),
        )


@dataclass
class Route:
    vehicle_id   : int
    stops        : List[Stop] = field(default_factory=list)
    start        : Optional[Tuple[float, float]] = None
    end          : Optional[Tuple[float, float]] = None
    cost         : int = 0
    duration     : int = 0
    service      : int = 0
    waiting_time : int = 0
    distance     : int = 0
    geometry     : Optional[str] = None
    path_coords  : Optional[List[Tuple[float, float]]] = None

    @property
    def job_ids(self) -> List[int]:
        return [stop.job_id for stop in self.stops]

    @property
    def locations(self) -> List[Tuple[float, float]]:
        points = []
        if self.start is not None:
            points.append(self.start)
        points.extend(stop.location for stop in self.stops)
        if self.end is not None:
            points.append(self.end)
        return points

    def remove_jobs(self, job_ids: Set[int]) -> None:
        removed_service = sum(stop.service for stop in self.stops if stop.job_id in job_ids)

        self.stops   = [stop for stop in self.stops if stop.job_id not in job_ids]
        self.service = max(0, self.service - removed_service)

        if self.stops:
            self.end = self.stops[-1].location

    def copy(self) -> "Route":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict:
        return {
            "vehicle_id"   : int(self.vehicle_id),
            "stops"        : [stop.to_dict() for stop in self.stops],
            "start"        : list(self.start) if self.start is not None else None,
            "end"          : list(self.end) if self.end is not None else None,
            "cost"         : int(self.cost),
            "duration"     : int(self.duration),
            "service"      : int(self.service),
            "waiting_time" : int(self.waiting_time),
            "distance"     : int(self.distance),
            "geometry"     : self.geometry,
            "path_coords"  : [list(point) for point in self.path_coords] if self.path_coords is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Route":
        return cls(
            vehicle_id   = int(data["vehicle_id"]),
            stops        = [Stop.from_dict(stop) for stop in data["stops"]],
            start        = tuple(data["start"]) if data["start"] is not None else None,
            end          = tuple(data["end"]) if data["end"] is not None else None,
            cost         = int(data["cost"]),
            duration     = int(data["duration"]),
            service      = int(data["service"]),
            waiting_time = int(data["waiting_time"]),
            distance     = int(data["distance"]),
            geometry     = data["geometry"],
            path_coords  = [tuple(point) for point in data["path_coords"]] if data["path_coords"] is not None else None,
        )

    @classmethod
    def from_vroom(cls, route: Dict) -> "Route":
        start = None
        end   = None
        stops = []

        for step in route.get("steps") or []:
            kind     = step.get("type")
            location = tuple(map(float, step["location"])) if step.get("location") else None

            if kind == "start":
                start = location
            elif kind == "end":
                end = location
            elif kind == "job" and location is not None:
                load = step.get("load") or [0]
                stops.append(
                    Stop(
                        job_id   = int(step.get("job", step.get("id"))),
                        location = location,
                        arrival  = int(step.get("arrival", 0)),
                        duration = int(step.get("duration", 0)),
                        service  = int(step.get("service", 0)),
                        load     = int(load[0]),
                    )
                )

        return cls(
            vehicle_id   = int(route["vehicle"]),
            stops        = stops,
            start        = start,
            end          = end,
            cost         = int(route.get("cost", 0)),
            duration     = int(route.get("duration", 0)),
            service      = int(route.get("service", 0)),
            waiting_time = int(route.get("waiting_time", 0)),
            distance     = int(route.get("distance", 0)),
            geometry     = route.get("geometry"),
        )


class RoutingState:
    def __init__(self, routes: Optional[List[Route]] = None, unassigned_ids: Optional[Set[int]] = None):
        self.routes         = list(routes or [])
        self.unassigned_ids = set(unassigned_ids or ())

    @property
    def num_routes(self) -> int:
        return len(self.routes)

    @property
    def num_unassigned(self) -> int:
        return len(self.unassigned_ids)

    @property
    def cost(self) -> int:
        return sum(route.cost for route in self.routes)

    @property
    def duration(self) -> int:
        return sum(route.duration for route in self.routes)

    @property
    def service(self) -> int:
        return sum(route.service for route in self.routes)

    @property
    def distance(self) -> int:
        return sum(route.distance for route in self.routes)

    @property
    def waiting_time(self) -> int:
        return sum(route.waiting_time for route in self.routes)

    @property
    def assigned_job_ids(self) -> Set[int]:
        return {stop.job_id for route in self.routes for stop in route.stops}

    @property
    def vehicle_ids_with_routes(self) -> Set[int]:
        return {route.vehicle_id for route in self.routes}

    def route_of_vehicle(self, vehicle_id: int) -> Optional[Route]:
        for route in self.routes:
            if route.vehicle_id == vehicle_id:
                return route
        return None

    def route_of_job(self, job_id: int) -> Optional[Route]:
        for route in self.routes:
            if job_id in route.job_ids:
                return route
        return None

    def add_unassigned(self, job_ids: Set[int]) -> None:
        self.unassigned_ids |= set(job_ids)

    def replace_route(self, new_route: Route) -> None:
        previous    = self.route_of_vehicle(new_route.vehicle_id)
        new_job_ids = set(new_route.job_ids)
        displaced   = set(previous.job_ids) - new_job_ids if previous is not None else set()

        self.routes = [route for route in self.routes if route.vehicle_id != new_route.vehicle_id]
        if new_route.stops:
            self.routes.append(new_route)

        self.unassigned_ids -= new_job_ids
        self.unassigned_ids |= displaced

    def remove_jobs(self, job_ids: Set[int]) -> None:
        for route in self.routes:
            route.remove_jobs(job_ids)

        self.routes          = [route for route in self.routes if route.stops]
        self.unassigned_ids -= set(job_ids)

    def remove_vehicles(self, vehicle_ids: Set[int]) -> Set[int]:
        orphaned = {job_id for route in self.routes if route.vehicle_id in vehicle_ids for job_id in route.job_ids}

        self.routes = [route for route in self.routes if route.vehicle_id not in vehicle_ids]
        return orphaned

    def copy(self) -> "RoutingState":
        return RoutingState(
            routes         = [route.copy() for route in self.routes],
            unassigned_ids = set(self.unassigned_ids),
        )

    def to_payload(self) -> Dict:
        return {
            "schema"     : "routing-state-v1",
            "routes"     : [route.to_dict() for route in self.routes],
            "unassigned" : sorted(self.unassigned_ids),
        }

    @classmethod
    def from_payload(cls, payload: Dict) -> "RoutingState":
        if payload.get("schema") != "routing-state-v1":
            raise ValueError(f"Unsupported state payload schema: {payload.get('schema')!r}")

        return cls(
            routes         = [Route.from_dict(route) for route in payload["routes"]],
            unassigned_ids = set(payload["unassigned"]),
        )

    @classmethod
    def from_vroom(cls, solution: Dict) -> "RoutingState":
        routes = [Route.from_vroom(route) for route in solution.get("routes") or []]

        unassigned_ids = {
            int(entry["id"])
            for entry in solution.get("unassigned") or []
            if entry.get("id") is not None
        }

        return cls(routes=routes, unassigned_ids=unassigned_ids)

    def __repr__(self) -> str:
        return f"RoutingState(routes={self.num_routes}, unassigned={self.num_unassigned}, cost={self.cost})"


class EntityPool:
    def __init__(self, items=None):
        self._items = list(items or [])
        self._reindex()

    @property
    def ids(self) -> List[int]:
        return [item.id for item in self._items]

    def contains(self, item_id: int) -> bool:
        return item_id in self._index_by_id

    def by_id(self, item_id: int):
        return self._items[self._index_by_id[item_id]]

    def index_of(self, item_id: int) -> Optional[int]:
        return self._index_by_id.get(item_id)

    def next_id(self) -> int:
        return max(self._index_by_id) + 1 if self._index_by_id else 0

    def add(self, items) -> None:
        self._items.extend(items)
        self._reindex()

    def remove(self, item_ids: Set[int]) -> None:
        self._items = [item for item in self._items if item.id not in item_ids]
        self._reindex()

    def _reindex(self) -> None:
        self._index_by_id = {item.id: index for index, item in enumerate(self._items)}

    def sample_ids(self, count: int) -> List[int]:
        return random.sample(self.ids, count)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __getitem__(self, index: int):
        return self._items[index]
