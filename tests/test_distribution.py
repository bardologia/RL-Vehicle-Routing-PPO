import torch

from core.shared import PPOMasking
from core.training import PPODistribution


MASK_INFO = {
    "unassigned_job_indices"     : [2, 5, 11],
    "vehicles_with_jobs_indices" : [0, 2, 4],
    "vehicle_to_job_indices"     : {0: [1, 3], 1: [], 2: [7, 8, 9], 3: [], 4: [0], 5: []},
}


def build_distribution(cpu_config):
    masking = PPOMasking(cpu_config)
    return masking, PPODistribution(cpu_config, masking)


def test_categorical_kl_of_identical_logits_is_zero():
    logits = torch.randn(6)

    kl = PPODistribution.categorical_kl(logits, logits)

    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_categorical_kl_is_positive_for_different_logits():
    torch.manual_seed(0)

    kl = PPODistribution.categorical_kl(torch.randn(6), torch.randn(6))

    assert kl.item() > 0.0


def test_masked_action_logits_matches_per_row_masking(cpu_config, seeded):
    masking, distribution = build_distribution(cpu_config)
    O, V, J               = 4, 6, 20

    veh_logits = torch.randn(O, V)
    job_logits = torch.randn(O, V, J)

    veh_masked, job_masked = distribution.masked_action_logits(veh_logits, job_logits, MASK_INFO)

    for op_index in range(O):
        expected_veh = masking.mask_vehicle(veh_logits[op_index], MASK_INFO, op_index)
        assert torch.equal(veh_masked[op_index], expected_veh)

        for veh_index in range(V):
            expected_job = masking.mask_job(job_logits[op_index, veh_index], MASK_INFO, op_index, veh_index)
            assert torch.equal(job_masked[op_index, veh_index], expected_job)


def test_compute_returns_zero_kl_for_unchanged_policy(cpu_config, seeded):
    masking, distribution = build_distribution(cpu_config)

    op_logits  = torch.randn(4)
    veh_logits = torch.randn(4, 6)
    job_logits = torch.randn(4, 6, 20)

    entropy, kl = distribution.compute(op_logits, veh_logits, job_logits, op_logits, veh_logits, job_logits, MASK_INFO)

    assert abs(kl["total_kl"]) < 1e-5
    assert entropy["total_entropy"].item() > 0.0


def test_compute_kl_grows_with_policy_change(cpu_config, seeded):
    masking, distribution = build_distribution(cpu_config)

    op_logits  = torch.randn(4)
    veh_logits = torch.randn(4, 6)
    job_logits = torch.randn(4, 6, 20)

    _, kl_small = distribution.compute(op_logits, veh_logits, job_logits, op_logits + 0.01 * torch.randn(4), veh_logits + 0.01 * torch.randn(4, 6), job_logits + 0.01 * torch.randn(4, 6, 20), MASK_INFO)
    _, kl_large = distribution.compute(op_logits, veh_logits, job_logits, op_logits + torch.randn(4), veh_logits + torch.randn(4, 6), job_logits + torch.randn(4, 6, 20), MASK_INFO)

    assert kl_small["total_kl"] > 0.0
    assert kl_large["total_kl"] > kl_small["total_kl"]


def test_compute_entropy_ignores_masked_entries(cpu_config):
    masking, distribution = build_distribution(cpu_config)

    op_logits  = torch.zeros(4)
    veh_logits = torch.zeros(4, 6)
    job_logits = torch.zeros(4, 6, 20)

    entropy, _ = distribution.compute(op_logits, veh_logits, job_logits, op_logits, veh_logits, job_logits, None)
    entropy_masked, _ = distribution.compute(op_logits, veh_logits, job_logits, op_logits, veh_logits, job_logits, MASK_INFO)

    assert entropy_masked["total_entropy"].item() < entropy["total_entropy"].item()
