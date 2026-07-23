import torch

from core.shared import ActionMasker
from core.training import ActionDistribution


MASK_INFO = {
    "unassigned_job_indices"         : [2, 5, 11],
    "vehicles_with_jobs_indices"     : [0, 2, 4],
    "vehicle_to_job_indices"         : {0: [1, 3], 1: [], 2: [7, 8, 9], 3: [], 4: [0], 5: []},
    "vehicles_with_capacity_indices" : [1, 3, 5],
}


def build_distribution(cpu_config):
    masker = ActionMasker(cpu_config)
    return masker, ActionDistribution(cpu_config, masker)


def test_categorical_kl_of_identical_logits_is_zero():
    logits = torch.randn(6)

    kl = ActionDistribution.categorical_kl(logits, logits)

    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_categorical_kl_is_positive_for_different_logits():
    torch.manual_seed(0)

    kl = ActionDistribution.categorical_kl(torch.randn(6), torch.randn(6))

    assert kl.item() > 0.0


def test_entropy_is_maximal_for_uniform_logits():
    uniform = ActionDistribution._entropy(torch.zeros(4))
    peaked  = ActionDistribution._entropy(torch.tensor([10.0, 0.0, 0.0, 0.0]))

    assert uniform.item() > peaked.item()


def test_masked_action_logits_none_info_returns_clones(cpu_config):
    _, distribution = build_distribution(cpu_config)

    vehicle_logits = torch.randn(4, 6)
    job_logits     = torch.randn(4, 6, 20)

    vehicle_masked, job_masked = distribution.masked_action_logits(vehicle_logits, job_logits, None)

    assert torch.equal(vehicle_masked, vehicle_logits)
    assert torch.equal(job_masked, job_logits)


def test_masked_action_logits_matches_per_row_masking(cpu_config, seeded):
    masker, distribution = build_distribution(cpu_config)
    O, V, J              = 4, 6, 20

    vehicle_logits = torch.randn(O, V)
    job_logits     = torch.randn(O, V, J)

    vehicle_masked, job_masked = distribution.masked_action_logits(vehicle_logits, job_logits, MASK_INFO)

    for operator_index in range(O):
        expected_vehicle = masker.mask_vehicle(vehicle_logits[operator_index], MASK_INFO, operator_index)
        assert torch.equal(vehicle_masked[operator_index], expected_vehicle)

        for vehicle_index in range(V):
            expected_job = masker.mask_job(job_logits[operator_index, vehicle_index], MASK_INFO, operator_index, vehicle_index)
            assert torch.equal(job_masked[operator_index, vehicle_index], expected_job)


def test_compute_returns_zero_kl_for_unchanged_policy(cpu_config, seeded):
    masker, distribution = build_distribution(cpu_config)

    operator_logits = torch.randn(4)
    vehicle_logits  = torch.randn(4, 6)
    job_logits      = torch.randn(4, 6, 20)

    entropy, kl = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits, vehicle_logits, job_logits, MASK_INFO)

    assert abs(kl["total_kl"]) < 1e-5
    assert entropy["total_entropy"].item() > 0.0


def test_compute_mean_kl_is_total_over_three(cpu_config, seeded):
    _, distribution = build_distribution(cpu_config)

    operator_logits = torch.randn(4)
    vehicle_logits  = torch.randn(4, 6)
    job_logits      = torch.randn(4, 6, 20)

    _, kl = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits + torch.randn(4), vehicle_logits + torch.randn(4, 6), job_logits + torch.randn(4, 6, 20), MASK_INFO)

    assert kl["mean_kl"] == kl["total_kl"] / 3


def test_compute_kl_grows_with_policy_change(cpu_config, seeded):
    masker, distribution = build_distribution(cpu_config)

    operator_logits = torch.randn(4)
    vehicle_logits  = torch.randn(4, 6)
    job_logits      = torch.randn(4, 6, 20)

    _, kl_small = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits + 0.01 * torch.randn(4), vehicle_logits + 0.01 * torch.randn(4, 6), job_logits + 0.01 * torch.randn(4, 6, 20), MASK_INFO)
    _, kl_large = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits + torch.randn(4), vehicle_logits + torch.randn(4, 6), job_logits + torch.randn(4, 6, 20), MASK_INFO)

    assert kl_small["total_kl"] > 0.0
    assert kl_large["total_kl"] > kl_small["total_kl"]


def test_compute_entropy_ignores_masked_entries(cpu_config):
    masker, distribution = build_distribution(cpu_config)

    operator_logits = torch.zeros(4)
    vehicle_logits  = torch.zeros(4, 6)
    job_logits      = torch.zeros(4, 6, 20)

    entropy, _        = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits, vehicle_logits, job_logits, None)
    entropy_masked, _ = distribution.compute(operator_logits, vehicle_logits, job_logits, operator_logits, vehicle_logits, job_logits, MASK_INFO)

    assert entropy_masked["total_entropy"].item() < entropy["total_entropy"].item()
