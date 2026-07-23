import torch

from core.shared import ActionMaskBuilder, ActionMasker, EntityPool, RoutingState
from tests.conftest import make_jobs, make_route, make_vehicles


def build_context():
    jobs     = make_jobs(5)
    vehicles = make_vehicles(3)
    state    = RoutingState(
        routes         = [make_route(vehicles[0], jobs[:2])],
        unassigned_ids = {jobs[2].id, jobs[3].id},
    )
    return EntityPool(jobs), EntityPool(vehicles), state


def test_mask_context_reports_eligible_unassigned_indices():
    jobs, vehicles, state = build_context()

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["unassigned_job_indices"] == [2, 3]


def test_mask_context_excludes_unassigned_ids_missing_from_pool():
    jobs, vehicles, state = build_context()
    state.unassigned_ids.add(999)

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["unassigned_job_indices"] == [2, 3]


def test_mask_context_maps_vehicles_to_their_route_jobs():
    jobs, vehicles, state = build_context()

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["vehicle_to_job_indices"][0] == [0, 1]
    assert info["vehicle_to_job_indices"][1] == []
    assert info["vehicles_with_jobs_indices"] == [0]


def test_mask_operator_blocks_insert_without_unassigned(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[1] == 0.0


def test_mask_operator_blocks_remove_without_loaded_vehicles(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [], "vehicle_to_job_indices": {}}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == 0.0
    assert masked[1] == cpu_config.training.large_negative_value


def test_mask_vehicle_restricts_remove_to_loaded_vehicles(cpu_config):
    masker         = ActionMasker(cpu_config)
    vehicle_logits = torch.zeros(3)
    info           = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [1], "vehicle_to_job_indices": {1: [0]}}

    masked = masker.mask_vehicle(vehicle_logits, info, selected_operator_index=1)

    assert masked[1] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[2] == cpu_config.training.large_negative_value


def test_mask_job_restricts_remove_to_selected_vehicle_jobs(cpu_config):
    masker     = ActionMasker(cpu_config)
    job_logits = torch.zeros(5)
    info       = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1, 3]}}

    masked = masker.mask_job(job_logits, info, selected_operator_index=1, selected_vehicle_index=0)

    assert masked[1] == 0.0
    assert masked[3] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[2] == cpu_config.training.large_negative_value


def test_mask_job_restricts_insert_to_unassigned(cpu_config):
    masker     = ActionMasker(cpu_config)
    job_logits = torch.zeros(4)
    info       = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [], "vehicle_to_job_indices": {}}

    masked = masker.mask_job(job_logits, info, selected_operator_index=0, selected_vehicle_index=0)

    assert masked[2] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value
