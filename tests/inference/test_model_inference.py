from core.inference import InferenceResult, ModelInference
from core.shared import Environment
from model.policy_model import Action, Policy
from tests.conftest import ScriptedPolicy


def test_model_inference_runs_without_mutating_environment_state(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    model         = Policy(cpu_config)
    baseline      = environment.current_state
    initial_state = environment.current_state.copy()

    inference = ModelInference(model, environment, max_steps=5, device="cpu", verbose=False)
    result    = inference.run(initial_state)

    assert isinstance(result, InferenceResult)
    assert result.total_steps >= 1
    assert result.steps[0].step_number == 0
    assert environment.current_state is baseline
    assert {"total_steps", "stopped_reason", "initial_cost", "final_cost"} <= set(result.summary().keys())


def test_model_inference_trajectory_tracks_steps(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    model         = Policy(cpu_config)
    initial_state = environment.current_state.copy()

    inference  = ModelInference(model, environment, max_steps=4, device="cpu", verbose=False)
    result     = inference.run(initial_state)
    trajectory = result.get_trajectory()

    assert trajectory[0]["step_number"] == 0
    assert len(trajectory) == len(result.steps)


def test_do_nothing_operator_stops_the_loop(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    initial_state = environment.current_state.copy()

    scripted  = ScriptedPolicy([Action(3, 0, 0), Action(3, 0, 0), Action(2, 0, 0), Action(3, 0, 0)])
    inference = ModelInference(scripted, environment, max_steps=10, device="cpu", verbose=False)

    result = inference.run(initial_state)

    assert result.stopped_reason == "model_do_nothing"
    assert result.total_steps == 3
    assert len(result.steps) == 3


def test_max_steps_cap_stops_the_loop(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    initial_state = environment.current_state.copy()

    scripted  = ScriptedPolicy([Action(3, 0, 0)])
    inference = ModelInference(scripted, environment, max_steps=4, device="cpu", verbose=False)

    result = inference.run(initial_state)

    assert result.stopped_reason == "max_steps_reached"
    assert result.total_steps == 4
    assert len(result.steps) == 5


def test_first_do_nothing_stops_before_any_apply(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    initial_state = environment.current_state.copy()

    scripted  = ScriptedPolicy([Action(2, 0, 0)])
    inference = ModelInference(scripted, environment, max_steps=5, device="cpu", verbose=False)

    result = inference.run(initial_state)

    assert result.stopped_reason == "model_do_nothing"
    assert result.total_steps == 1
    assert len(result.steps) == 1


def test_applied_steps_record_action_info_and_costs(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    initial_state = environment.current_state.copy()

    scripted  = ScriptedPolicy([Action(3, 0, 0), Action(2, 0, 0)])
    inference = ModelInference(scripted, environment, max_steps=5, device="cpu", verbose=False)

    result = inference.run(initial_state)
    applied = result.steps[1]

    assert applied.action.operator == 3
    assert applied.action_info["operator_name"] == "REOPTIMIZE"
    assert applied.reward_info is not None
    assert applied.cost == applied.state.cost


def test_summary_cost_accounting_matches_states(cpu_config, seeded, fake_vroom):
    environment   = Environment(cpu_config)
    initial_state = environment.current_state.copy()

    scripted  = ScriptedPolicy([Action(3, 0, 0), Action(2, 0, 0)])
    inference = ModelInference(scripted, environment, max_steps=5, device="cpu", verbose=False)

    result  = inference.run(initial_state)
    summary = result.summary()

    assert summary["initial_cost"] == initial_state.cost
    assert summary["final_cost"] == result.final_state.cost
    assert summary["cost_improvement"] == result.get_initial_cost() - result.get_final_cost()
