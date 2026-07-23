import pytest

from core.inference import InferenceResult, InferenceStep
from core.shared import RoutingState
from model.policy_model import Action
from tests.conftest import make_jobs, make_route, make_vehicles


def state_with_cost(cost, unassigned=()):
    vehicles = make_vehicles(1)
    jobs     = make_jobs(2)
    return RoutingState(routes=[make_route(vehicles[0], jobs, cost=cost)], unassigned_ids=set(unassigned))


def test_inference_step_to_dict_without_action():
    step = InferenceStep(step_number=0, state=state_with_cost(500), cost=500, num_routes=1, num_unassigned=0)

    payload = step.to_dict()

    assert payload["step_number"] == 0
    assert payload["operator"] is None
    assert payload["vehicle_index"] is None
    assert payload["cost"] == 500


def test_inference_step_to_dict_with_action():
    step = InferenceStep(
        step_number    = 2,
        state          = state_with_cost(400),
        action         = Action(operator=1, vehicle_index=3, job_index=5),
        cost           = 400,
        num_routes     = 1,
        num_unassigned = 0,
    )

    payload = step.to_dict()

    assert payload["operator"] == 1
    assert payload["vehicle_index"] == 3
    assert payload["job_index"] == 5


def test_cost_improvement_and_percentage():
    result = InferenceResult(
        initial_state = state_with_cost(1000),
        final_state   = state_with_cost(600),
        total_steps   = 3,
    )

    assert result.get_initial_cost() == 1000
    assert result.get_final_cost() == 600
    assert result.get_cost_improvement() == 400
    assert result.get_cost_improvement_percentage() == pytest.approx(40.0)


def test_cost_improvement_percentage_guards_zero_initial():
    result = InferenceResult(initial_state=state_with_cost(0), final_state=state_with_cost(0))

    assert result.get_cost_improvement_percentage() == 0.0


def test_summary_exposes_full_key_set():
    result = InferenceResult(
        initial_state  = state_with_cost(1000, unassigned=(7,)),
        final_state    = state_with_cost(600),
        stopped_reason = "max_steps_reached",
        total_steps    = 5,
    )

    summary = result.summary()

    assert set(summary.keys()) == {
        "total_steps",
        "stopped_reason",
        "initial_cost",
        "final_cost",
        "cost_improvement",
        "cost_improvement_pct",
        "initial_routes",
        "final_routes",
        "initial_unassigned",
        "final_unassigned",
    }
    assert summary["initial_unassigned"] == 1
    assert summary["stopped_reason"] == "max_steps_reached"


def test_summary_defaults_when_states_absent():
    result = InferenceResult()

    summary = result.summary()

    assert summary["initial_cost"] == 0
    assert summary["final_cost"] == 0
    assert summary["cost_improvement"] == 0


def test_get_trajectory_maps_every_step():
    steps = [
        InferenceStep(step_number=0, state=state_with_cost(1000), cost=1000),
        InferenceStep(step_number=1, state=state_with_cost(800), action=Action(0, 0, 0), cost=800),
    ]
    result = InferenceResult(steps=steps)

    trajectory = result.get_trajectory()

    assert len(trajectory) == 2
    assert trajectory[0]["step_number"] == 0
    assert trajectory[1]["operator"] == 0
