from core.shared import Environment, EpisodeDriver
from model.policy_model import Action


def test_driver_yields_one_step_per_round_when_edit_cap_is_one(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 4
    cpu_config.training.max_edits_per_round         = 1

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    steps  = list(driver.episode(0))

    assert [step.round_index for step in steps] == [0, 1, 2, 3]
    assert all(step.edit_index == 0 for step in steps)
    assert all(step.remaining == 1 for step in steps)


def test_driver_bursts_up_to_edit_cap_per_round(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 3

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    steps  = list(driver.episode(0))

    assert [(step.round_index, step.edit_index) for step in steps] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert [step.remaining for step in steps] == [3, 2, 1, 3, 2, 1]


def test_committing_advance_ends_the_round_early(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 5

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    rounds = []

    for step in driver.episode(0):
        rounds.append((step.round_index, step.edit_index))
        step.commit(Action(operator=2, vehicle_index=0, job_index=0))

    assert rounds == [(0, 0), (1, 0)]


def test_driver_commit_applies_action_and_advances_current_state(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 1
    cpu_config.training.max_edits_per_round         = 1

    environment = Environment(cpu_config)
    driver      = EpisodeDriver(environment, cpu_config)

    step                                 = next(driver.episode(0))
    old_state, new_state, rewards, costs = step.commit(Action(operator=2, vehicle_index=0, job_index=0))

    assert environment.current_state is new_state
    assert step.action.operator == 2
    assert "action_reward" in rewards
    assert "action_cost" in costs


def test_driver_episode_is_deterministic_per_seed(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 3

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)

    first  = next(driver.episode(42)).graph["job"].x
    second = next(driver.episode(42)).graph["job"].x

    assert first.shape == second.shape
    assert (first == second).all()
