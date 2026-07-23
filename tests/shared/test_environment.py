import pytest

import core.shared.environment as environment_module

from core.shared import Environment, RoutingState
from model.policy_model import Action
from tests.conftest import make_jobs, make_route, make_vehicles


def test_reset_produces_consistent_state(environment):
    assert len(environment.jobs) >= environment.config.env.min_jobs
    assert len(environment.vehicles) >= environment.config.env.min_vehicles
    assert environment.current_state.num_routes >= 1
    assert environment.current_state.assigned_job_ids <= set(environment.jobs.ids)


def test_reset_stores_independent_initial_state(environment):
    assert environment.initial_state is not environment.current_state
    assert environment.initial_state.to_payload() == environment.current_state.to_payload()


def test_reset_raises_when_solver_never_returns_solution(cpu_config, seeded, monkeypatch):
    class DeadVroom:
        def solve(self, jobs, vehicles):
            return None

    monkeypatch.setattr("core.shared.environment.vroom", DeadVroom())
    cpu_config.env.reset_max_attempts = 4

    with pytest.raises(RuntimeError):
        Environment(cpu_config)


def test_apply_event_rejects_unknown_type(environment):
    with pytest.raises(ValueError):
        environment.apply_event(environment.current_state, "teleport")


def test_new_job_event_grows_pool_and_unassigned(environment):
    before_jobs       = len(environment.jobs)
    before_unassigned = environment.current_state.num_unassigned

    state = environment.apply_event(environment.current_state, "new_job", 2)

    assert len(environment.jobs) == before_jobs + 2
    assert state.num_unassigned == before_unassigned + 2
    assert environment.current_state is state


def test_remove_job_event_purges_pool_routes_and_unassigned(environment):
    state      = environment.current_state
    before_ids = set(environment.jobs.ids)

    new_state = environment.apply_event(state, "remove_job", 2)

    removed = before_ids - set(environment.jobs.ids)

    assert len(removed) == 2
    assert removed.isdisjoint(new_state.assigned_job_ids)
    assert removed.isdisjoint(new_state.unassigned_ids)


def test_remove_job_event_on_empty_pool_returns_copy(environment):
    environment.jobs.remove(set(environment.jobs.ids))

    state     = environment.current_state
    new_state = environment.apply_event(state, "remove_job", 2)

    assert new_state is not state
    assert len(environment.jobs) == 0


def test_remove_vehicle_event_orphans_jobs_to_unassigned(environment):
    state      = environment.current_state
    loaded_ids = state.vehicle_ids_with_routes

    new_state = environment.apply_event(state, "remove_vehicle", 1)

    assert len(environment.vehicles) >= 1
    removed_vehicles = loaded_ids - set(environment.vehicles.ids)
    for vehicle_id in removed_vehicles:
        assert new_state.route_of_vehicle(vehicle_id) is None


def test_remove_vehicle_event_keeps_at_least_one_vehicle(cpu_config, seeded, fake_vroom):
    environment = Environment(cpu_config)
    environment.vehicles.remove(set(environment.vehicles.ids[1:]))

    state     = environment.current_state
    new_state = environment.apply_event(state, "remove_vehicle", 1)

    assert len(environment.vehicles) == 1
    assert new_state is not state


def test_new_vehicle_event_grows_fleet_only(environment):
    before_vehicles = len(environment.vehicles)
    before_payload  = environment.current_state.to_payload()

    state = environment.apply_event(environment.current_state, "new_vehicle", 1)

    assert len(environment.vehicles) == before_vehicles + 1
    assert state.to_payload() == before_payload


def test_generate_event_new_job_count_in_config_range(environment, monkeypatch):
    monkeypatch.setattr(environment_module.random, "choice", lambda seq: "new_job")

    event_type, num_items = environment.generate_event()

    assert event_type == "new_job"
    assert environment.config.env.job_insert_min <= num_items <= environment.config.env.job_insert_max


def test_generate_event_new_vehicle_count_in_config_range(environment, monkeypatch):
    monkeypatch.setattr(environment_module.random, "choice", lambda seq: "new_vehicle")

    event_type, num_items = environment.generate_event()

    assert event_type == "new_vehicle"
    assert environment.config.env.vehicle_insert_min <= num_items <= environment.config.env.vehicle_insert_max


def test_generate_event_remove_job_capped_by_pool(environment, monkeypatch):
    monkeypatch.setattr(environment_module.random, "choice", lambda seq: "remove_job")

    event_type, num_items = environment.generate_event()

    assert event_type == "remove_job"
    assert 0 <= num_items <= min(environment.config.env.job_remove_max, len(environment.jobs))


def test_generate_event_remove_vehicle_capped_by_fleet(environment, monkeypatch):
    monkeypatch.setattr(environment_module.random, "choice", lambda seq: "remove_vehicle")

    event_type, num_items = environment.generate_event()

    assert event_type == "remove_vehicle"
    assert 0 <= num_items <= max(0, len(environment.vehicles) - 1)


def load_insertion_scenario(environment, num_jobs=3, loaded_stops=1):
    jobs     = make_jobs(num_jobs)
    vehicles = make_vehicles(2)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:loaded_stops], cost=100 * loaded_stops)],
        unassigned_ids = {job.id for job in jobs[loaded_stops:]},
    )

    environment.load_from_dataset({
        "jobs"     : [job.to_dict() for job in jobs],
        "vehicles" : [vehicle.to_dict() for vehicle in vehicles],
        "state"    : state.to_payload(),
    })

    return jobs, vehicles


def test_insertion_action_assigns_unassigned_job(environment, fake_vroom):
    jobs, _ = load_insertion_scenario(environment)

    new_job_id = jobs[1].id
    action     = Action(operator=0, vehicle_index=0, job_index=environment.jobs.index_of(new_job_id))

    old_state, new_state = environment.apply_action(action)

    assert new_job_id in new_state.assigned_job_ids
    assert new_job_id not in new_state.unassigned_ids
    assert new_job_id in old_state.unassigned_ids


def test_insertion_beyond_capacity_raises(environment, fake_vroom):
    jobs, vehicles = load_insertion_scenario(environment, num_jobs=4, loaded_stops=2)

    assert vehicles[0].capacity == 2

    action = Action(operator=0, vehicle_index=0, job_index=environment.jobs.index_of(jobs[2].id))

    with pytest.raises(ValueError):
        environment.apply_action(action)


def test_removal_action_moves_job_to_unassigned(environment):
    state  = environment.current_state
    route  = state.routes[0]
    job_id = route.job_ids[0]
    action = Action(
        operator      = 1,
        vehicle_index = environment.vehicles.index_of(route.vehicle_id),
        job_index     = environment.jobs.index_of(job_id),
    )

    _, new_state = environment.apply_action(action)

    assert job_id in new_state.unassigned_ids
    assert new_state.route_of_job(job_id) is None


def test_removal_action_no_op_when_job_not_in_route(environment):
    environment.apply_event(environment.current_state, "new_job", 1)

    state      = environment.current_state
    route      = state.routes[0]
    outside_id = environment.jobs.ids[-1]
    action     = Action(
        operator      = 1,
        vehicle_index = environment.vehicles.index_of(route.vehicle_id),
        job_index     = environment.jobs.index_of(outside_id),
    )

    old_state, new_state = environment.apply_action(action)

    assert new_state is old_state
    assert outside_id in new_state.unassigned_ids


def test_do_nothing_action_returns_same_state(environment):
    action = Action(operator=2, vehicle_index=0, job_index=0)

    old_state, new_state = environment.apply_action(action)

    assert new_state is old_state


def test_apply_action_rejects_unknown_operator(environment):
    for operator in (3, 9):
        with pytest.raises(ValueError):
            environment.apply_action(Action(operator=operator, vehicle_index=0, job_index=0))


def test_evaluate_cost_matches_reward_config(environment):
    state  = environment.current_state
    reward = environment.config.reward

    distance_cost, unassigned_cost, idle_cost, priority_cost = environment.evaluate_cost(state)

    expected_priority = sum(
        environment.jobs.by_id(job_id).priority
        for job_id in state.unassigned_ids
        if environment.jobs.contains(job_id)
    )

    assert distance_cost == pytest.approx(reward.distance_weight * state.cost / 1000)
    assert unassigned_cost == pytest.approx(reward.unassigned_penalty_weight * state.num_unassigned)
    assert idle_cost == pytest.approx(reward.idle_penalty_weight * (len(environment.vehicles) - state.num_routes))
    assert priority_cost == pytest.approx(reward.priority_penalty_weight * expected_priority)


def test_step_rewards_are_negated_cost_deltas(environment):
    old_state = environment.current_state
    new_state = environment.event_handler.apply_new_job(environment, old_state, 2)

    rewards, costs = environment.step(old_state, new_state, operator_index=0)

    assert rewards["unassigned_reward"] == pytest.approx(-(costs["new_unassigned_cost"] - costs["old_unassigned_cost"]))
    assert rewards["distance_reward"] == pytest.approx(-(costs["new_distance_cost"] - costs["old_distance_cost"]))
    assert rewards["action_reward"] == pytest.approx(-environment.config.reward.add_job_cost)


def test_step_action_reward_subtracts_operator_cost(environment):
    old_state = environment.current_state
    new_state = old_state.copy()

    operator_costs = {
        0: environment.config.reward.add_job_cost,
        1: environment.config.reward.remove_job_cost,
        2: environment.config.reward.no_action_cost,
    }

    for operator_index, expected in operator_costs.items():
        rewards, costs = environment.step(old_state, new_state, operator_index)
        assert costs["disruption"] == 0
        assert rewards["action_reward"] == pytest.approx(-expected)


def test_step_disruption_counts_reassigned_and_dropped_jobs(environment):
    old_state = environment.current_state
    reward    = environment.config.reward

    moved_route = old_state.routes[0]
    moved_job   = moved_route.job_ids[0]

    new_state = old_state.copy()
    new_state.remove_jobs({moved_job})
    new_state.add_unassigned({moved_job})

    rewards, costs = environment.step(old_state, new_state, operator_index=1)

    assert costs["disruption"] == 1
    assert costs["action_cost"] == pytest.approx(reward.remove_job_cost + reward.disruption_cost)
    assert rewards["action_reward"] == pytest.approx(-costs["action_cost"])


def test_insert_then_remove_cycle_has_negative_total_reward(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:1], cost=100)], unassigned_ids={jobs[1].id})

    environment.load_from_dataset({
        "jobs"     : [job.to_dict() for job in jobs],
        "vehicles" : [vehicle.to_dict() for vehicle in vehicles],
        "state"    : state.to_payload(),
    })

    loaded = environment.current_state

    inserted = environment.action_handler.apply_job_insertion(environment, loaded, vehicles[0].id, jobs[1].id)
    insert_rewards, _ = environment.step(loaded, inserted, operator_index=0)

    removed = environment.action_handler.apply_job_removal(environment, inserted, vehicles[0].id, jobs[1].id)
    remove_rewards, _ = environment.step(inserted, removed, operator_index=1)

    reward      = environment.config.reward
    cycle_total = sum(insert_rewards.values()) + sum(remove_rewards.values())

    assert cycle_total < 0.0
    assert cycle_total == pytest.approx(-(reward.add_job_cost + reward.remove_job_cost + reward.disruption_cost))


def test_load_from_dataset_round_trips(environment):
    item = {
        "state"    : environment.current_state.to_payload(),
        "jobs"     : [job.to_dict() for job in environment.jobs],
        "vehicles" : [vehicle.to_dict() for vehicle in environment.vehicles],
    }

    environment.apply_event(environment.current_state, "new_job", 2)
    environment.load_from_dataset(item)

    assert [job.to_dict() for job in environment.jobs] == item["jobs"]
    assert [vehicle.to_dict() for vehicle in environment.vehicles] == item["vehicles"]
    assert environment.current_state.to_payload() == item["state"]
