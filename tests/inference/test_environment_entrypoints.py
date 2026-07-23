import torch

from model.policy_model import Action


def test_observe_matches_explicit_state_entry_points(environment):
    graph, mask_info = environment.observe()

    assert environment.mask_info_for(environment.current_state) == mask_info
    assert torch.equal(environment.graph_for(environment.current_state)["job"].x, graph["job"].x)


def test_graph_for_reflects_the_given_state_not_current(environment):
    baseline_graph = environment.graph_for(environment.current_state)

    grown_state = environment.apply_event(environment.current_state, "new_job", 3)
    grown_graph = environment.graph_for(grown_state)

    assert grown_graph["job"].x.shape[0] == baseline_graph["job"].x.shape[0] + 3


def test_apply_action_to_operates_on_given_state_without_touching_current(environment):
    state  = environment.current_state
    route  = state.routes[0]
    job_id = route.job_ids[0]
    action = Action(
        operator      = 1,
        vehicle_index = environment.vehicles.index_of(route.vehicle_id),
        job_index     = environment.jobs.index_of(job_id),
    )

    baseline             = environment.current_state
    old_state, new_state = environment.apply_action_to(state, action)

    assert job_id in new_state.unassigned_ids
    assert job_id in old_state.assigned_job_ids
    assert environment.current_state is baseline


def test_apply_action_delegates_to_apply_action_to(environment):
    action = Action(operator=2, vehicle_index=0, job_index=0)

    direct_old, direct_new       = environment.apply_action_to(environment.current_state, action)
    delegated_old, delegated_new = environment.apply_action(action)

    assert delegated_new is delegated_old
    assert direct_new is direct_old


def test_evaluate_cost_does_not_mutate_state(environment):
    state          = environment.current_state
    payload_before = state.to_payload()

    environment.evaluate_cost(state)

    assert state.to_payload() == payload_before


def test_step_reads_two_states_without_mutating_them(environment):
    old_state = environment.current_state
    new_state = environment.apply_action_to(old_state, Action(operator=1, vehicle_index=0, job_index=0))[1]

    old_payload = old_state.to_payload()
    new_payload = new_state.to_payload()

    environment.step(old_state, new_state, operator_index=1)

    assert old_state.to_payload() == old_payload
    assert new_state.to_payload() == new_payload
