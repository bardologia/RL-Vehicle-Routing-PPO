"""Building footprints for the map, read from Overpass once per block of tiles and cached on disk as GeoJSON."""

import json
import math
import re
import threading
import time

import requests

from web_errors import ApiError


class BuildingTiles:
    """Serves zoom-16 GeoJSON tiles of building polygons with heights.

    A cache miss fetches the whole parent zoom-14 tile in one Overpass query, splits the buildings by centroid into its sixteen zoom-16 tiles and writes all of them, so a block costs one request.

    Attributes:
        directory : Cache directory; tiles are stored as `16/<x>/<y>.json`.
        endpoints : Overpass interpreter URLs, tried in order.
        session   : Object with a `post(url, data=..., headers=..., timeout=...)` method; `requests` by default.
        clock     : Callable returning the time in epoch seconds; `time.time` by default.

    A block whose fetch failed answers 503 from memory for RETRY_AFTER seconds before Overpass is asked again, so the tiles of one view do not queue up behind repeated timeouts.
    """

    ENDPOINTS    = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
    ZOOM         = 16
    PARENT_ZOOM  = 14
    TIMEOUT      = 90
    RETRY_AFTER  = 60
    USER_AGENT   = "routing-console/1.0"
    LEVEL_HEIGHT = 3.2
    DEFAULT      = 8.0
    NUMBER       = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")

    def __init__(self, directory, endpoints=None, session=None, clock=None):
        self.directory = directory
        self.endpoints = tuple(endpoints or self.ENDPOINTS)
        self.session   = session or requests
        self.clock     = clock or time.time

        self.state_lock    = threading.Lock()
        self.block_locks   = {}
        self.failed_blocks = {}
        self.pending       = 0
        self.last_error    = None
        self.last_error_at = None

    def _tile_path(self, x, y):
        return self.directory / str(self.ZOOM) / str(x) / f"{y}.json"

    def _tile_of(self, lon, lat, zoom):
        scale = 2 ** zoom
        x     = int((lon + 180.0) / 360.0 * scale)
        y     = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * scale)
        return x, y

    def _bounds(self, x, y, zoom):
        scale = 2 ** zoom
        west  = x / scale * 360.0 - 180.0
        east  = (x + 1) / scale * 360.0 - 180.0
        north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / scale))))
        south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / scale))))
        return south, west, north, east

    def _query(self, south, west, north, east):
        box = f"{south:.6f},{west:.6f},{north:.6f},{east:.6f}"
        return f'[out:json][timeout:{self.TIMEOUT}];(way["building"]({box});relation["building"]["type"="multipolygon"]({box}););out geom;'

    def _fetch(self, query):
        failures = []

        for endpoint in self.endpoints:
            try:
                response = self.session.post(endpoint, data={"data": query}, headers={"User-Agent": self.USER_AGENT}, timeout=self.TIMEOUT)
            except requests.RequestException as error:
                failures.append(f"{endpoint}: {type(error).__name__}")
                continue

            if response.status_code == 200:
                return response.json()

            failures.append(f"{endpoint}: HTTP {response.status_code}")

        raise ApiError(503, "building data unavailable (" + "; ".join(failures) + ")")

    def _number(self, tags, key):
        match = self.NUMBER.match(str(tags[key])) if key in tags else None
        return float(match.group(1)) if match else None

    def _heights(self, tags):
        height = self._number(tags, "height")
        levels = self._number(tags, "building:levels")
        base   = self._number(tags, "min_height")
        floor  = self._number(tags, "building:min_level")

        if height is None:
            height = levels * self.LEVEL_HEIGHT if levels is not None else self.DEFAULT
        if base is None:
            base = floor * self.LEVEL_HEIGHT if floor is not None else 0.0

        return {"height": height, "minHeight": base}

    def _ring(self, geometry):
        ring = [[float(point["lon"]), float(point["lat"])] for point in geometry]
        if len(ring) < 3:
            return None
        if ring[0] != ring[-1]:
            ring.append(list(ring[0]))

        return ring if len(ring) >= 4 else None

    def _rings(self, element):
        if element["type"] == "way":
            geometries = [element.get("geometry") or []]
        else:
            geometries = [member.get("geometry") or [] for member in element.get("members") or [] if member.get("type") == "way" and member.get("role") == "outer"]

        return [ring for ring in (self._ring(geometry) for geometry in geometries) if ring is not None]

    def features(self, payload):
        """Convert an Overpass `out geom` response into GeoJSON building features.

        Args:
            payload : Parsed Overpass JSON with way and multipolygon relation elements.

        Returns:
            List of GeoJSON Feature dicts, one Polygon per outer ring, with `properties` {"height", "minHeight"} in metres.
        """
        features = []
        for element in payload["elements"]:
            properties = self._heights(element.get("tags") or {})

            for ring in self._rings(element):
                features.append({"type": "Feature", "properties": properties, "geometry": {"type": "Polygon", "coordinates": [ring]}})

        return features

    def split(self, features, parent_x, parent_y):
        """Assign features to the sixteen zoom-16 tiles of a zoom-14 tile by ring centroid.

        Returns:
            {(x, y): [feature]} with an entry for every one of the sixteen tiles; features whose centroid lies outside the block are left out.
        """
        factor = 2 ** (self.ZOOM - self.PARENT_ZOOM)
        tiles  = {(parent_x * factor + dx, parent_y * factor + dy): [] for dx in range(factor) for dy in range(factor)}

        for feature in features:
            ring = feature["geometry"]["coordinates"][0][:-1]
            lon  = sum(point[0] for point in ring) / len(ring)
            lat  = sum(point[1] for point in ring) / len(ring)
            tile = self._tile_of(lon, lat, self.ZOOM)

            if tile in tiles:
                tiles[tile].append(feature)

        return tiles

    def _write_block(self, tiles):
        for (x, y), features in tiles.items():
            path = self._tile_path(x, y)
            path.parent.mkdir(parents=True, exist_ok=True)

            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
            temporary.replace(path)

    def _fill_block(self, x, y):
        factor   = 2 ** (self.ZOOM - self.PARENT_ZOOM)
        parent_x = x // factor
        parent_y = y // factor

        with self.state_lock:
            block_lock    = self.block_locks.setdefault((parent_x, parent_y), threading.Lock())
            self.pending += 1

        try:
            with block_lock:
                if self._tile_path(x, y).exists():
                    return

                failed = self.failed_blocks.get((parent_x, parent_y))
                if failed is not None and self.clock() - failed[0] < self.RETRY_AFTER:
                    raise ApiError(503, failed[1])

                try:
                    payload = self._fetch(self._query(*self._bounds(parent_x, parent_y, self.PARENT_ZOOM)))
                except ApiError as error:
                    with self.state_lock:
                        self.last_error    = error.message
                        self.last_error_at = self.clock()

                    self.failed_blocks[(parent_x, parent_y)] = (self.clock(), error.message)
                    raise

                self.failed_blocks.pop((parent_x, parent_y), None)

                self._write_block(self.split(self.features(payload), parent_x, parent_y))
        finally:
            with self.state_lock:
                self.pending -= 1

    def tile(self, zoom, x, y):
        """Return the raw GeoJSON bytes of one building tile.

        Args:
            zoom : Tile zoom; only 16 is served.
            x    : Tile column.
            y    : Tile row.

        Returns:
            UTF-8 encoded GeoJSON FeatureCollection.

        Raises:
            ApiError : Another zoom or an out-of-range tile (400), or no Overpass endpoint answered (503); nothing is cached on failure.
        """
        if zoom != self.ZOOM or not 0 <= x < 2 ** self.ZOOM or not 0 <= y < 2 ** self.ZOOM:
            raise ApiError(400, f"building tiles are served at zoom {self.ZOOM} only")

        path = self._tile_path(x, y)
        if not path.exists():
            self._fill_block(x, y)

        return path.read_bytes()

    def status(self):
        """Return {"cached_tiles", "pending", "last_error", "last_error_at"}."""
        root = self.directory / str(self.ZOOM)

        with self.state_lock:
            return {
                "cached_tiles"  : sum(1 for _ in root.glob("*/*.json")) if root.is_dir() else 0,
                "pending"       : self.pending,
                "last_error"    : self.last_error,
                "last_error_at" : self.last_error_at,
            }
