from core.shared import Environment, EpisodeDriver
from model.policy_model import Action


def test_driver_yields_one_step_per_max_step(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 4

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    steps  = list(driver.episode(0))

    assert [step.index for step in steps] == [0, 1, 2, 3]
    assert [step.remaining for step in steps] == [4, 3, 2, 1]


def test_driver_commit_applies_action_and_advances_current_state(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 1

    environment = Environment(cpu_config)
    driver      = EpisodeDriver(environment, cpu_config)

    step                                 = next(driver.episode(0))
    old_state, new_state, rewards, costs = step.commit(Action(operator=2, vehicle_index=0, job_index=0))

    assert environment.current_state is new_state
    assert "action_reward" in rewards
    assert "action_cost" in costs


def test_driver_episode_is_deterministic_per_seed(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 3

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)

    first  = next(driver.episode(42)).graph["job"].x
    second = next(driver.episode(42)).graph["job"].x

    assert first.shape == second.shape
    assert (first == second).all()
