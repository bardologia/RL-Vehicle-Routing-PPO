import json
import time
from typing import List, Optional, Tuple

import polyline as polyline_module
import requests
from requests.exceptions import RequestException

from configuration import config
from tools.auxiliary import retry_api_call
from tools.logger import NullLogger
from .state import Job, Route, RoutingState, Vehicle


class OsrmClient:
    def __init__(self, service_config, logger=None):
        self.service = service_config
        self.logger  = logger or NullLogger()
        self._distance_cache = {}

    @retry_api_call(max_retries=3, backoff_factor=0.5)
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

    @retry_api_call(max_retries=3, backoff_factor=0.5)
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
    MAX_ATTEMPTS   = 3
    BACKOFF_FACTOR = 0.5

    def __init__(self, service_config, logger=None):
        self.service = service_config
        self.logger  = logger or NullLogger()
        self._solve_cache = {}

    def _vroom_error_code(self, response) -> Optional[int]:
        try:
            body = response.json()
        except ValueError:
            return None
        return body.get("code") if isinstance(body, dict) else None

    def _post(self, payload) -> Optional[dict]:
        for attempt in range(self.MAX_ATTEMPTS):
            transient = None

            try:
                response = requests.post(self.service.vroom_url, json=payload, timeout=20)
            except RequestException as error:
                transient = str(error)
            else:
                if response.status_code == 200:
                    return response.json()

                error_code = self._vroom_error_code(response)
                if error_code is not None:
                    self.logger.warning(f"VROOM could not solve the request (code {error_code}): {response.text[:160]}")
                    return None
                if response.status_code < 500:
                    self.logger.warning(f"VROOM rejected the request ({response.status_code}): {response.text[:160]}")
                    return None

                transient = f"{response.status_code}: {response.text[:160]}"

            if attempt + 1 < self.MAX_ATTEMPTS:
                time.sleep(self.BACKOFF_FACTOR * (2 ** attempt))

        self.logger.warning(f"VROOM unavailable after {self.MAX_ATTEMPTS} attempts ({transient}); treating as no solution")
        return None

    def solve(self, jobs: List[Job], vehicles: List[Vehicle], depot=None, clock=None) -> Optional[RoutingState]:
        job_payloads = []
        shipments    = []

        for job in jobs:
            if job.kind == "repossession":
                if depot is None:
                    raise ValueError(f"Repossession job {job.id} requires a depot")
                shipments.append(job.shipment_payload(depot, config.env.depot_service))
            else:
                job_payloads.append(job.vroom_payload())

        for vehicle in vehicles:
            if vehicle.onboard > 0:
                if depot is None:
                    raise ValueError(f"Vehicle {vehicle.id} carries onboard load and requires a depot")
                job_payloads.append(vehicle.drop_payload(depot, config.env.depot_service))

        payload = {
            "jobs"     : job_payloads,
            "vehicles" : [vehicle.vroom_payload(clock) for vehicle in vehicles],
            "options"  : self.service.options,
        }
        if shipments:
            payload["shipments"] = shipments

        key = json.dumps(payload, sort_keys=True)
        if key not in self._solve_cache:
            if len(self._solve_cache) > 50_000:
                self._solve_cache.clear()

            solution = self._post(payload)
            if solution is None:
                return None

            self._solve_cache[key] = RoutingState.from_vroom(solution)

        return self._solve_cache[key].copy()


osrm  = OsrmClient(config.service)
vroom = VroomClient(config.service)
