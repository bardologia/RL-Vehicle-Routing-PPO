import requests
import polyline as polyline_module
from typing import List, Optional, Tuple

from tools.config import config
from tools.auxiliary import retry_api_call
from core.state import Job, Route, RoutingState, Vehicle


class OsrmClient:
    def __init__(self, service_config):
        self.service = service_config
        self._distance_cache = {}

    @retry_api_call(max_retries=3, backoff_factor=0.5, timeout=20, verbose=False)
    def _fetch_distance_duration(self, lon_a, lat_a, lon_b, lat_b) -> Tuple[float, float]:
        url = (
            f"{self.service.osrm_url}/route/v1/driving/"
            f"{lon_a},{lat_a};{lon_b},{lat_b}"
            f"?overview=false&steps=false&alternatives=false&annotations=false"
        )

        response = self.service.http_session.get(url, timeout=20)
        response.raise_for_status()

        data  = response.json()
        route = (data.get("routes") or [{}])[0]
        return float(route.get("distance", 0.0)), float(route.get("duration", 0.0))

    @retry_api_call(max_retries=3, backoff_factor=0.5, timeout=20, verbose=False)
    def _fetch_geometry(self, points: List[Tuple[float, float]]) -> Optional[str]:
        coords = ";".join(f"{lon},{lat}" for lon, lat in points)
        url    = f"{self.service.osrm_url}/route/v1/driving/{coords}?overview=full&geometries=polyline"

        response = self.service.http_session.get(url, timeout=10)
        response.raise_for_status()

        routes = response.json().get("routes") or []
        return routes[0].get("geometry") if routes else None

    def distance_duration(self, lon_a, lat_a, lon_b, lat_b) -> Tuple[float, float]:
        key = (round(float(lon_a), 5), round(float(lat_a), 5), round(float(lon_b), 5), round(float(lat_b), 5))

        if key not in self._distance_cache:
            if len(self._distance_cache) > 100_000:
                self._distance_cache.clear()
            self._distance_cache[key] = self._fetch_distance_duration(*key)

        return self._distance_cache[key]

    def enrich_polylines(self, routes: List[Route]) -> None:
        for route in routes:
            if route.geometry is not None and route.path_coords is not None:
                continue

            points = route.locations
            if not points:
                continue

            geometry = self._fetch_geometry(points)
            if geometry:
                route.geometry    = geometry
                route.path_coords = polyline_module.decode(geometry)


class VroomClient:
    def __init__(self, service_config):
        self.service = service_config

    @retry_api_call(max_retries=3, backoff_factor=0.5, timeout=20, verbose=False)
    def _post(self, payload) -> Optional[dict]:
        response = requests.post(self.service.vroom_url, json=payload, timeout=20)
        if response.status_code != 200:
            print("VROOM error:", response.text)
            return None
        return response.json()

    def solve(self, jobs: List[Job], vehicles: List[Vehicle]) -> Optional[RoutingState]:
        payload = {
            "jobs"     : [job.vroom_payload() for job in jobs],
            "vehicles" : [vehicle.vroom_payload() for vehicle in vehicles],
            "options"  : self.service.options,
        }

        solution = self._post(payload)
        if solution is None:
            return None

        return RoutingState.from_vroom(solution)


osrm  = OsrmClient(config.service)
vroom = VroomClient(config.service)
