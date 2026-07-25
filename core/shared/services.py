import json
import time
from typing import List, Optional

import requests
from requests.exceptions import RequestException

from configuration import config
from tools.logger import NullLogger
from .state import Job, RoutingState, Vehicle


class VroomClient:
    MAX_ATTEMPTS    = 3
    BACKOFF_FACTOR  = 0.5
    PERMANENT_CODES = (1, 2, 4)
    ROUTING_CODE    = 3

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
        transient    = None
        routing_fail = None

        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = requests.post(self.service.vroom_url, json=payload, timeout=20)
            except RequestException as error:
                transient = str(error)
            else:
                if response.status_code == 200:
                    return response.json()

                error_code = self._vroom_error_code(response)
                if error_code in self.PERMANENT_CODES:
                    self.logger.warning(f"VROOM could not solve the request (code {error_code}): {response.text[:160]}")
                    return None
                if error_code == self.ROUTING_CODE:
                    routing_fail = response.text[:160]
                elif response.status_code < 500:
                    self.logger.warning(f"VROOM rejected the request ({response.status_code}): {response.text[:160]}")
                    return None
                else:
                    transient = f"{response.status_code}: {response.text[:160]}"

            if attempt + 1 < self.MAX_ATTEMPTS:
                time.sleep(self.BACKOFF_FACTOR * (2 ** attempt))

        if routing_fail is not None:
            self.logger.warning(f"VROOM could not route the request after {self.MAX_ATTEMPTS} attempts: {routing_fail}")
            return None

        raise RuntimeError(f"VROOM unavailable after {self.MAX_ATTEMPTS} attempts ({transient})")

    def _cache_key(self, payload) -> str:
        canonical = {
            "jobs"      : sorted(payload["jobs"], key=lambda job: job["id"]),
            "vehicles"  : sorted(payload["vehicles"], key=lambda vehicle: vehicle["id"]),
            "shipments" : sorted(payload.get("shipments", []), key=lambda shipment: shipment["pickup"]["id"]),
            "options"   : payload["options"],
        }

        return json.dumps(canonical, sort_keys=True)

    def clear_cache(self) -> None:
        self._solve_cache.clear()

    def solve(self, jobs: List[Job], vehicles: List[Vehicle], depot=None, clock=None, options=None) -> Optional[RoutingState]:
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
            "options"  : options or self.service.options,
        }
        if shipments:
            payload["shipments"] = shipments

        key = self._cache_key(payload)
        if key not in self._solve_cache:
            if len(self._solve_cache) >= self.service.solve_cache_limit:
                self._solve_cache.clear()

            solution = self._post(payload)
            if solution is None:
                return None

            self._solve_cache[key] = RoutingState.from_vroom(solution)

        return self._solve_cache[key].copy()


vroom = VroomClient(config.service)
