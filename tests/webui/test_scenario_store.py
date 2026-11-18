import pytest

from scenario_store import ScenarioStore
from web_errors import ApiError
from tests.conftest import make_jobs, make_vehicles


def body():
    return {
        "jobs"       : [job.to_dict() for job in make_jobs(3)],
        "vehicles"   : [vehicle.to_dict() for vehicle in make_vehicles(2)],
        "depot"      : [-46.63, -23.55],
        "assignment" : {0: [0, 1]},
        "template"   : "cheap_detour",
    }


def test_put_get_list_delete_round_trip(paths):
    store = ScenarioStore(paths.scenarios_dir)

    assert store.list() == []
    assert store.put("first_case", body()) == {"ok": True}

    saved = store.get("first_case")
    assert saved["name"] == "first_case"
    assert saved["assignment"] == {"0": [0, 1]}
    assert saved["template"] == "cheap_detour"
    assert len(saved["jobs"]) == 3

    listed = store.list()
    assert [(entry["name"], entry["jobs"], entry["vehicles"]) for entry in listed] == [("first_case", 3, 2)]

    assert store.delete("first_case") == {"ok": True}
    assert store.list() == []


def test_put_replaces_an_existing_scenario(paths):
    store = ScenarioStore(paths.scenarios_dir)
    store.put("case", body())

    smaller = body()
    smaller["jobs"]       = smaller["jobs"][:1]
    smaller["assignment"] = None
    store.put("case", smaller)

    assert len(store.get("case")["jobs"]) == 1
    assert store.get("case")["assignment"] is None


def test_bad_names_and_bodies_answer_400(paths):
    store = ScenarioStore(paths.scenarios_dir)

    for name in ("../escape", "has space", "", "x" * 49):
        with pytest.raises(ApiError) as raised:
            store.put(name, body())
        assert raised.value.status == 400

    broken = body()
    broken["jobs"][0].pop("priority")
    with pytest.raises(ApiError) as raised:
        store.put("case", broken)
    assert raised.value.status == 400


def test_unknown_scenario_answers_404(paths):
    store = ScenarioStore(paths.scenarios_dir)

    for call in (store.get, store.delete):
        with pytest.raises(ApiError) as raised:
            call("missing")
        assert raised.value.status == 404
