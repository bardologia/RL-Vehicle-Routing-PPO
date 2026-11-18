import pytest

from configuration import Config
from core.inference import EpisodeLedger, EpisodeSettings, PlanMetrics
from core.shared import EntityPool, Job, Route, RoutingState, Stop, Vehicle


def metrics():
    return PlanMetrics(Config().reward)


def build_plan():
    jobs = EntityPool([
        Job(id=0, location=(0.0, 1.0), priority=3),
        Job(id=1, location=(1.0, 0.0), kind="repossession", amount=1, priority=5),
        Job(id=2, location=(1.0, 1.0), priority=2),
        Job(id=3, location=(5.0, 5.0), priority=4),
    ])
    vehicles = EntityPool([
        Vehicle(id=0, start=(0.0, 0.0), capacity=4, onboard=1),
        Vehicle(id=1, start=(0.0, 1.0001), capacity=3),
        Vehicle(id=2, start=(9.0, 9.0), capacity=3),
    ])

    first = Route(
        vehicle_id = 0,
        start      = (0.0, 0.0),
        cost       = 3000,
        service    = 900,
        stops      = [
            Stop(job_id=1, location=(1.0, 0.0), kind="pickup", arrival=30000, duration=1000, service=300, load=2),
            Stop(job_id=2, location=(1.0, 1.0), kind="job", arrival=32000, duration=2500, service=600, load=2),
            Stop(job_id=20_000_000, location=(0.5, 0.5), kind="delivery", arrival=33000, duration=3000, service=120, load=0),
        ],
    )
    second = Route(
        vehicle_id = 1,
        start      = (0.0, 1.0001),
        cost       = 1000,
        service    = 300,
        stops      = [Stop(job_id=0, location=(2.0, 0.2), kind="job", arrival=30000, duration=1000, service=300, load=0)],
    )

    return jobs, vehicles, RoutingState(routes=[first, second], unassigned_ids={3})


def test_plan_kpis_follow_the_reward_formulas():
    jobs, vehicles, state = build_plan()
    ledger                = EpisodeLedger()
    ledger.first_seen     = {3: 28800}
    ledger.served         = 2

    plan = metrics().plan(jobs, vehicles, state, 30600, ledger)
    kpis = plan["kpis"]

    assert kpis["drive_s"] == 4000
    assert kpis["unassigned"] == 1
    assert kpis["unassigned_priority"] == 4
    assert kpis["oldest_unassigned_s"] == 1800
    assert kpis["idle_vehicles"] == 1
    assert kpis["standing_cost"] == pytest.approx(1.5 * 4.0 + 1.0 + 0.5 + 0.5 * 4)
    assert kpis["holding_rate"] == pytest.approx(1.0 + 2.0)
    assert kpis["drive_per_served_s"] == 2000
    assert kpis["load_use"] == pytest.approx(2 / 7)
    assert plan["idle_vehicle_ids"] == [2]
    assert plan["unassigned"] == [{"job_id": 3, "priority": 4, "kind": "support", "waiting_s": 1800}]


def test_route_kpis_cover_load_slack_and_shares():
    jobs, vehicles, state = build_plan()
    ledger                = EpisodeLedger()
    ledger.previous_drive = {0: 2500}

    plan  = metrics().plan(jobs, vehicles, state, 30600, ledger)
    route = plan["routes"][0]["kpis"]

    assert route["state"] == "ACTIVE"
    assert (route["stops"], route["stops_support"], route["stops_repossession"], route["depot_drops"]) == (2, 1, 1, 1)
    assert route["load_profile"] == [1, 2, 2, 0]
    assert route["peak_load"] == 2
    assert route["end_clock"] == 33120
    assert route["slack_s"] == 72000 - 33120
    assert route["drive_share"] == pytest.approx(0.75)
    assert route["priority_sum"] == 7
    assert route["reward_weight"] == pytest.approx(4.5)
    assert route["drive_delta_s"] == 500
    assert [stop["leg_s"] for stop in plan["routes"][0]["stops"]] == [1000, 1500, 500]
    assert plan["idle"][0]["kpis"]["state"] == "IDLE"


def test_crossings_count_proper_intersections_between_routes_only():
    jobs, vehicles, state = build_plan()
    plan                  = metrics().plan(jobs, vehicles, state, 30600, EpisodeLedger())

    assert plan["kpis"]["crossings"] == 2
    assert [route["kpis"]["crossings"] for route in plan["routes"]] == [2, 2]

    total, per_route = metrics().crossings([[(0, 0), (1, 1)], [(1, 1), (2, 0)], [(0, 1), (0.4, 0.6)]])
    assert total == 0
    assert per_route == [0, 0, 0]


def test_shift_over_state_and_empty_plan():
    jobs, vehicles, _ = build_plan()
    plan              = metrics().plan(jobs, vehicles, RoutingState(), 72000, EpisodeLedger())

    assert plan["routes"] == []
    assert {entry["kpis"]["state"] for entry in plan["idle"]} == {"SHIFT OVER"}
    assert plan["kpis"]["drive_per_served_s"] is None
    assert plan["kpis"]["load_use"] == 0.0


def test_add_geometry_fills_distance_and_detour():
    jobs, vehicles, state = build_plan()
    tool                  = metrics()
    plan                  = tool.plan(jobs, vehicles, state, 30600, EpisodeLedger())

    def road(points):
        return {"path": [[point[1], point[0]] for point in points], "distance_m": 2 * tool.straight_length([tuple(point) for point in points])}

    tool.add_geometry(plan, road)

    assert all(route["kpis"]["detour"] == pytest.approx(2.0, rel=1e-3) for route in plan["routes"])
    assert plan["kpis"]["distance_m"] == sum(route["distance_m"] for route in plan["routes"])
    assert len(plan["routes"][0]["path"]) == 4


def test_break_even_matches_the_documented_thresholds():
    tool = metrics()

    assert tool.break_even(1, 0, False) == pytest.approx(933.33, abs=0.01)
    assert tool.break_even(5, 0, False) == pytest.approx(2266.67, abs=0.01)
    assert tool.break_even(1, 0, True) - tool.break_even(1, 0, False) == pytest.approx(0.5 / 1.5 * 1000)
    assert tool.break_even(1, 1, False) == pytest.approx((1.0 + 0.5 - 0.1 - 0.3) / 1.5 * 1000)


def test_episode_settings_copy_and_reject_unknown_fields():
    base    = Config()
    applied = EpisodeSettings().build(base, {"min_jobs": 10, "center": [-46.0, -23.0]}, {"distance_weight": 2.0}, 3, 4)

    assert applied.env.min_jobs == 10
    assert applied.env.center == (-46.0, -23.0)
    assert applied.reward.distance_weight == 2.0
    assert applied.training.max_disruptions_per_episode == 3
    assert applied.training.max_edits_per_round == 4
    assert base.env.min_jobs == 4

    with pytest.raises(ValueError):
        EpisodeSettings().build(base, {"no_such_field": 1})
