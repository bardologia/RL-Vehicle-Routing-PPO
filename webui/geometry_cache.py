"""Road geometry of stop sequences, fetched from OSRM once per sequence."""

import threading

import polyline
import requests

from web_errors import ApiError


class GeometryCache:
    """Returns the road path and distance of an ordered list of points.

    Attributes:
        osrm_url : Base URL of the OSRM server.
        session  : Object with a `get(url, timeout=...)` method; `requests` by default.
    """

    def __init__(self, osrm_url, session=None):
        self.osrm_url = osrm_url.rstrip("/")
        self.session  = session or requests
        self._cache   = {}
        self._lock    = threading.Lock()

    def _fetch(self, key):
        coordinates = ";".join(f"{lon},{lat}" for lon, lat in key)
        url         = f"{self.osrm_url}/route/v1/driving/{coordinates}?overview=full&geometries=polyline&steps=false"

        try:
            response = self.session.get(url, timeout=10)
        except requests.RequestException as error:
            raise ApiError(503, f"OSRM unavailable: {error}")

        if response.status_code != 200:
            raise ApiError(503, f"OSRM answered {response.status_code} for a route of {len(key)} points")

        route = response.json()["routes"][0]
        return {"path": [[lat, lon] for lat, lon in polyline.decode(route["geometry"])], "distance_m": int(round(route["distance"]))}

    def route(self, points):
        """Return the road geometry through a list of points.

        Args:
            points : Ordered [lon, lat] pairs; a single point yields a one-point path of length 0.

        Returns:
            {"path": [[lat, lon]], "distance_m": int metres}.

        Raises:
            ApiError : OSRM is unreachable or rejects the request (503).
        """
        key = tuple((round(float(point[0]), 6), round(float(point[1]), 6)) for point in points)

        if len(key) < 2:
            return {"path": [[lat, lon] for lon, lat in key], "distance_m": 0}

        with self._lock:
            cached = self._cache.get(key)

        if cached is None:
            cached = self._fetch(key)
            with self._lock:
                self._cache[key] = cached

        return cached
