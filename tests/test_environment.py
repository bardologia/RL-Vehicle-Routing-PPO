import pytest

from core.shared import Action


def test_reset_produces_consistent_state(environment):
    assert len(environment.jobs) >= environment.config.env.min_jobs
    assert len(environment.vehicles) >= environment.config.env.min_vehicles
    assert environment.current_state.num_routes >= 1
    assert environment.current_state.assigned_job_ids <= set(environment.jobs.ids)


def test_new_job_event_grows_pool_and_unassigned(environment):
    before_jobs       = len(environment.jobs)
    before_unassigned = environment.current_state.num_unassigned

    state = environment.apply_event(environment.current_state, "new_job", 2)

    assert len(environment.jobs) == before_jobs + 2
    assert state.num_unassigned == before_unassigned + 2
    assert environment.current_state is state


def test_remove_job_event_purges_pool_routes_and_unassigned(environment):
    state       = environment.current_state
    before_ids  = set(environment.jobs.ids)

    new_state = environment.apply_event(state, "remove_job", 2)

    removed = before_ids - set(environment.jobs.ids)

    assert len(removed) == 2
    assert removed.isdisjoint(new_state.assigned_job_ids)
    assert removed.isdisjoint(new_state.unassigned_ids)


def test_remove_vehicle_event_orphans_jobs_to_unassigned(environment):
    state      = environment.current_state
    loaded_ids = state.vehicle_ids_with_routes

    new_state = environment.apply_event(state, "remove_vehicle", 1)

    assert len(environment.vehicles) >= 1
    removed_vehicles = loaded_ids - set(environment.vehicles.ids)
    for vehicle_id in removed_vehicles:
        assert new_state.route_of_vehicle(vehicle_id) is None


def test_new_vehicle_event_grows_fleet_only(environment):
    before_vehicles = len(environment.vehicles)
    before_payload  = environment.current_state.to_payload()

    state = environment.apply_event(environment.current_state, "new_vehicle", 1)

    assert len(environment.vehicles) == before_vehicles + 1
    assert state.to_payload() == before_payload


def test_insertion_action_assigns_unassigned_job(environment, fake_vroom):
    state = environment.apply_event(environment.current_state, "new_job", 1)

    new_job_id    = environment.jobs.ids[-1]
    job_index     = environment.jobs.index_of(new_job_id)
    vehicle_index = 0

    action = Action(operator=0, vehicle_index=vehicle_index, job_index=job_index)
    old_state, new_state = environment.apply_action(action)

    assert new_job_id in new_state.assigned_job_ids
    assert new_job_id not in new_state.unassigned_ids
    assert new_job_id in old_state.unassigned_ids


def test_removal_action_moves_job_to_unassigned(environment):
    state      = environment.current_state
    route      = state.routes[0]
    job_id     = route.job_ids[0]
    action     = Action(
        operator      = 1,
        vehicle_index = environment.vehicles.index_of(route.vehicle_id),
        job_index     = environment.jobs.index_of(job_id),
    )

    _, new_state = environment.apply_action(action)

    assert job_id in new_state.unassigned_ids
    assert new_state.route_of_job(job_id) is None


def test_do_nothing_action_returns_same_state(environment):
    action = Action(operator=2, vehicle_index=0, job_index=0)

    old_state, new_state = environment.apply_action(action)

    assert new_state is old_state


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

    rewards, costs = environment.step(old_state, new_state, operator_idx=0)

    assert rewards["unassigned_reward"] == pytest.approx(-(costs["new_unassigned_cost"] - costs["old_unassigned_cost"]))
    assert rewards["distance_reward"] == pytest.approx(-(costs["new_distance_cost"] - costs["old_distance_cost"]))
    assert rewards["action_reward"] == environment.config.reward.add_job_penalty


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
