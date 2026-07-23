from core.inference import InferenceResult, ModelInference
from core.shared import Environment
from model.policy_model import Policy


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
