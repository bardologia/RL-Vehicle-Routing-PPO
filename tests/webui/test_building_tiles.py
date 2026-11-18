import json

import pytest
import requests

from building_tiles import BuildingTiles
from web_errors import ApiError


X, Y = 23280, 37160


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self.payload     = payload

    def json(self):
        return self.payload


class FakeOverpass:
    def __init__(self, payload, status_code=200, fail=False):
        self.payload     = payload
        self.status_code = status_code
        self.fail        = fail
        self.calls       = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.calls.append({"url": url, "data": data, "headers": headers, "timeout": timeout})
        if self.fail:
            raise requests.ConnectionError("unreachable")

        return FakeResponse(self.status_code, self.payload)


def square(lon, lat, size=0.0002):
    return [{"lon": lon, "lat": lat}, {"lon": lon + size, "lat": lat}, {"lon": lon + size, "lat": lat + size}, {"lon": lon, "lat": lat + size}, {"lon": lon, "lat": lat}]


def tiles(paths, session, endpoints=None):
    return BuildingTiles(paths.buildings_dir, endpoints, session)


def payload_for(service):
    south, west, north, east = service._bounds(X, Y, 16)
    inside                   = ((west + east) / 2, (south + north) / 2)
    neighbour                = (inside[0] + (east - west), inside[1])
    far                      = (inside[0] + 40 * (east - west), inside[1])

    return {"elements": [
        {"type": "way", "id": 1, "geometry": square(*inside), "tags": {"building": "yes", "height": "12.5 m", "min_height": "3"}},
        {"type": "way", "id": 2, "geometry": square(*neighbour), "tags": {"building": "yes", "building:levels": "5", "building:min_level": "1"}},
        {"type": "way", "id": 3, "geometry": square(*far), "tags": {"building": "yes"}},
        {"type": "way", "id": 4, "geometry": square(*inside)[:2], "tags": {"building": "yes"}},
        {"type": "relation", "id": 5, "tags": {"building": "yes", "type": "multipolygon", "height": "tall"}, "members": [
            {"type": "way", "role": "outer", "geometry": square(*inside)[:4]},
            {"type": "way", "role": "inner", "geometry": square(*inside)},
        ]},
    ]}


def test_overpass_elements_become_polygons_with_heights(paths):
    service  = tiles(paths, FakeOverpass({}))
    features = service.features(payload_for(service))

    assert [feature["properties"] for feature in features] == [
        {"height": 12.5, "minHeight": 3.0},
        {"height": 16.0, "minHeight": 3.2},
        {"height": 8.0, "minHeight": 0.0},
        {"height": 8.0, "minHeight": 0.0},
    ]
    assert all(feature["geometry"]["type"] == "Polygon" for feature in features)
    assert all(feature["geometry"]["coordinates"][0][0] == feature["geometry"]["coordinates"][0][-1] for feature in features)


def test_tile_fetches_the_parent_block_once_and_splits_by_centroid(paths):
    session = FakeOverpass({})
    service = tiles(paths, session)
    session.payload = payload_for(service)

    tile      = json.loads(service.tile(16, X, Y))
    neighbour = json.loads(service.tile(16, X + 1, Y))
    empty     = json.loads(service.tile(16, X + 2, Y + 1))

    assert len(session.calls) == 1
    assert session.calls[0]["headers"] == {"User-Agent": "routing-console/1.0"}
    assert 'way["building"]' in session.calls[0]["data"]["data"]
    assert tile["type"] == "FeatureCollection"
    assert len(tile["features"]) == 2
    assert len(neighbour["features"]) == 1
    assert empty["features"] == []
    assert service.status()["cached_tiles"] == 16
    assert service.status()["pending"] == 0


def test_failure_answers_503_caches_nothing_and_reports_the_error(paths):
    session = FakeOverpass({}, fail=True)
    service = tiles(paths, session, ["http://one.invalid/api", "http://two.invalid/api"])

    with pytest.raises(ApiError) as raised:
        service.tile(16, X, Y)

    status = service.status()

    assert raised.value.status == 503
    assert [call["url"] for call in session.calls] == ["http://one.invalid/api", "http://two.invalid/api"]
    assert status["cached_tiles"] == 0
    assert "ConnectionError" in status["last_error"]
    assert status["last_error_at"] is not None
    assert status["pending"] == 0


def test_second_endpoint_is_used_when_the_first_refuses(paths):
    session = FakeOverpass({"elements": []}, status_code=200)
    service = tiles(paths, session, ["http://one.invalid/api", "http://two.invalid/api"])

    answers       = iter([FakeResponse(429, {}), FakeResponse(200, {"elements": []})])
    original_post = session.post
    session.post  = lambda url, **kwargs: (original_post(url, **kwargs), next(answers))[1]

    assert json.loads(service.tile(16, X, Y))["features"] == []
    assert len(session.calls) == 2


def test_other_zooms_answer_400(paths):
    service = tiles(paths, FakeOverpass({}))

    for arguments in ((15, 1, 1), (16, -1, 0), (16, 2 ** 16, 0)):
        with pytest.raises(ApiError) as raised:
            service.tile(*arguments)
        assert raised.value.status == 400
