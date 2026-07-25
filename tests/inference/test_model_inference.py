from core.inference import InferenceResult, ModelInference
from core.shared import Environment
from model.policy_model import Action, Policy
from tests.conftest import ScriptedPolicy


def build_inference(cpu_config, policy, disruptions=2, edits=1):
    cpu_config.training.max_disruptions_per_episode = disruptions
    cpu_config.training.max_edits_per_round         = edits

    return ModelInference(policy, Environment(cpu_config), cpu_config, device="cpu")


def test_inference_walks_the_disruption_rounds(cpu_config, seeded, fake_vroom):
    inference = build_inference(cpu_config, Policy(cpu_config), disruptions=3, edits=1)

    result = inference.run(seed=5)

    assert isinstance(result, InferenceResult)
    assert result.stopped_reason == "episode_complete"
    assert [step.round_index for step in result.steps] == [0, 1, 2]
    assert result.total_steps == 3


def test_inference_ends_a_round_on_do_nothing(cpu_config, seeded, fake_vroom):
    scripted  = ScriptedPolicy([Action(1, 0, 0), Action(2, 0, 0)])
    inference = build_inference(cpu_config, scripted, disruptions=1, edits=5)

    result = inference.run(seed=5)

    assert result.total_steps == 2
    assert [step.action.operator for step in result.steps] == [1, 2]


def test_applied_steps_record_action_info_and_costs(cpu_config, seeded, fake_vroom):
    scripted  = ScriptedPolicy([Action(1, 0, 0), Action(2, 0, 0)])
    inference = build_inference(cpu_config, scripted, disruptions=1, edits=5)

    result  = inference.run(seed=5)
    applied = result.steps[0]

    assert applied.action.operator == 1
    assert applied.action_info["operator_name"] == "REMOVE"
    assert applied.reward_info is not None
    assert applied.cost == applied.state.cost


def test_summary_cost_accounting_matches_states(cpu_config, seeded, fake_vroom):
    scripted  = ScriptedPolicy([Action(2, 0, 0)])
    inference = build_inference(cpu_config, scripted, disruptions=2, edits=2)

    result  = inference.run(seed=5)
    summary = result.summary()

    assert summary["initial_cost"] == result.initial_state.cost
    assert summary["final_cost"] == result.final_state.cost
    assert summary["cost_improvement"] == result.get_initial_cost() - result.get_final_cost()


def test_summary_carries_execution_totals(cpu_config, seeded, fake_vroom):
    scripted  = ScriptedPolicy([Action(2, 0, 0)])
    inference = build_inference(cpu_config, scripted, disruptions=3, edits=1)

    summary = inference.run(seed=5).summary()

    assert {"served", "failed", "dropped"} <= set(summary)
    assert summary["served"] >= 0
