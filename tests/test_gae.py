import pytest
import torch

from core.training import PPO


@pytest.fixture
def ppo(cpu_config, seeded):
    cpu_config.ppo.gamma      = 0.5
    cpu_config.ppo.gae_lambda = 0.5
    return PPO(optimizer=None, config=cpu_config)


def test_gae_matches_hand_computed_values(ppo):
    rewards    = torch.tensor([1.0, 1.0])
    values     = torch.tensor([0.5, 0.25])
    dones      = torch.tensor([0.0, 1.0])
    bootstraps = torch.tensor([0.0, 2.0])

    advantages, returns = ppo.gae(rewards, values, dones, bootstraps)

    expected_last  = 1.0 + 0.5 * 2.0 - 0.25
    expected_first = (1.0 + 0.5 * 0.25 - 0.5) + 0.5 * 0.5 * expected_last

    assert advantages[1].item() == pytest.approx(expected_last)
    assert advantages[0].item() == pytest.approx(expected_first)
    assert torch.allclose(returns, advantages + values)


def test_gae_uses_bootstrap_value_at_truncation(ppo):
    rewards = torch.tensor([1.0])
    values  = torch.tensor([0.0])
    dones   = torch.tensor([1.0])

    adv_zero, _ = ppo.gae(rewards, values, dones, torch.tensor([0.0]))
    adv_boot, _ = ppo.gae(rewards, values, dones, torch.tensor([4.0]))

    assert adv_boot.item() == pytest.approx(adv_zero.item() + 0.5 * 4.0)


def test_gae_does_not_leak_across_episode_boundaries(ppo):
    rewards    = torch.tensor([1.0, 1.0, 5.0, 5.0])
    values     = torch.tensor([0.0, 0.0, 0.0, 0.0])
    dones      = torch.tensor([0.0, 1.0, 0.0, 1.0])
    bootstraps = torch.tensor([0.0, 0.0, 0.0, 0.0])

    advantages, _ = ppo.gae(rewards, values, dones, bootstraps)

    rewards_alt         = rewards.clone()
    rewards_alt[2:]     = 100.0
    advantages_alt, _   = ppo.gae(rewards_alt, values, dones, bootstraps)

    assert torch.allclose(advantages[:2], advantages_alt[:2])


def test_gae_flows_within_episode(ppo):
    rewards    = torch.tensor([0.0, 0.0, 8.0])
    values     = torch.tensor([0.0, 0.0, 0.0])
    dones      = torch.tensor([0.0, 0.0, 1.0])
    bootstraps = torch.tensor([0.0, 0.0, 0.0])

    advantages, _ = ppo.gae(rewards, values, dones, bootstraps)

    assert advantages[2].item() == pytest.approx(8.0)
    assert advantages[1].item() > 0.0
    assert advantages[0].item() > 0.0
    assert advantages[0].item() < advantages[1].item() < advantages[2].item()
