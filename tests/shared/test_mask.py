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


def test_mask_context_excludes_ids_that_are_also_assigned():
    jobs, vehicles, state = build_context()
    state.unassigned_ids.add(jobs[0].id)

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["unassigned_job_indices"] == [2, 3]


def test_mask_context_maps_vehicles_to_their_route_jobs():
    jobs, vehicles, state = build_context()

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["vehicle_to_job_indices"][0] == [0, 1]
    assert info["vehicle_to_job_indices"][1] == []
    assert info["vehicles_with_jobs_indices"] == [0]


def test_mask_context_reports_vehicles_with_spare_capacity():
    jobs, vehicles, state = build_context()

    info = ActionMaskBuilder().build(jobs, vehicles, state)

    assert info["vehicles_with_capacity_indices"] == [1, 2]


def test_mask_context_counts_route_load_against_capacity():
    jobs     = make_jobs(3)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:1])], unassigned_ids={jobs[1].id, jobs[2].id})

    info = ActionMaskBuilder().build(EntityPool(jobs), EntityPool(vehicles), state)

    assert info["vehicles_with_capacity_indices"] == [0, 1]


def test_mask_context_skips_route_of_vehicle_absent_from_pool():
    jobs     = make_jobs(3)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(make_vehicles(1, first_id=99)[0], jobs)])

    info = ActionMaskBuilder().build(EntityPool(jobs), EntityPool(vehicles), state)

    assert info["vehicles_with_jobs_indices"] == []
    assert info["vehicle_to_job_indices"] == {0: []}


def test_mask_operator_none_info_returns_clone_unchanged():
    masker  = _masker()
    logits  = torch.arange(4, dtype=torch.float32)

    masked = masker.mask_operator(logits, None)

    assert torch.equal(masked, logits)
    assert masked is not logits


def test_mask_operator_blocks_insert_without_unassigned(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}, "vehicles_with_capacity_indices": [0]}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[1] == 0.0


def test_mask_operator_blocks_insert_without_capacity(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}, "vehicles_with_capacity_indices": []}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[1] == 0.0


def test_mask_operator_blocks_remove_without_loaded_vehicles(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [], "vehicle_to_job_indices": {}, "vehicles_with_capacity_indices": [0]}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == 0.0
    assert masked[1] == cpu_config.training.large_negative_value


def test_mask_operator_blocks_both_when_no_actions(cpu_config):
    masker          = ActionMasker(cpu_config)
    operator_logits = torch.zeros(4)
    info            = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [], "vehicle_to_job_indices": {}, "vehicles_with_capacity_indices": []}

    masked = masker.mask_operator(operator_logits, info)

    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[1] == cpu_config.training.large_negative_value
    assert masked[2] == 0.0
    assert masked[3] == 0.0


def test_mask_vehicle_none_info_returns_clone_unchanged():
    masker = _masker()
    logits = torch.arange(3, dtype=torch.float32)

    masked = masker.mask_vehicle(logits, None, selected_operator_index=1)

    assert torch.equal(masked, logits)


def test_mask_vehicle_restricts_remove_to_loaded_vehicles(cpu_config):
    masker         = ActionMasker(cpu_config)
    vehicle_logits = torch.zeros(3)
    info           = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [1], "vehicle_to_job_indices": {1: [0]}, "vehicles_with_capacity_indices": [0, 2]}

    masked = masker.mask_vehicle(vehicle_logits, info, selected_operator_index=1)

    assert masked[1] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[2] == cpu_config.training.large_negative_value


def test_mask_vehicle_restricts_insert_to_spare_capacity(cpu_config):
    masker         = ActionMasker(cpu_config)
    vehicle_logits = torch.zeros(3)
    info           = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}, "vehicles_with_capacity_indices": [1, 2]}

    masked = masker.mask_vehicle(vehicle_logits, info, selected_operator_index=0)

    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[1] == 0.0
    assert masked[2] == 0.0


def test_mask_vehicle_no_op_operator_collapses_to_first_index(cpu_config):
    masker         = ActionMasker(cpu_config)
    vehicle_logits = torch.zeros(3)
    info           = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}, "vehicles_with_capacity_indices": [1]}

    masked = masker.mask_vehicle(vehicle_logits, info, selected_operator_index=2)

    assert masked[0] == 0.0
    assert masked[1] == cpu_config.training.large_negative_value
    assert masked[2] == cpu_config.training.large_negative_value


def test_mask_job_none_info_returns_clone_unchanged():
    masker = _masker()
    logits = torch.arange(4, dtype=torch.float32)

    masked = masker.mask_job(logits, None, selected_operator_index=0, selected_vehicle_index=0)

    assert torch.equal(masked, logits)


def test_mask_job_restricts_remove_to_selected_vehicle_jobs(cpu_config):
    masker     = ActionMasker(cpu_config)
    job_logits = torch.zeros(5)
    info       = {"unassigned_job_indices": [], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1, 3]}, "vehicles_with_capacity_indices": [1]}

    masked = masker.mask_job(job_logits, info, selected_operator_index=1, selected_vehicle_index=0)

    assert masked[1] == 0.0
    assert masked[3] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value
    assert masked[2] == cpu_config.training.large_negative_value


def test_mask_job_restricts_insert_to_unassigned(cpu_config):
    masker     = ActionMasker(cpu_config)
    job_logits = torch.zeros(4)
    info       = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [], "vehicle_to_job_indices": {}, "vehicles_with_capacity_indices": [0]}

    masked = masker.mask_job(job_logits, info, selected_operator_index=0, selected_vehicle_index=0)

    assert masked[2] == 0.0
    assert masked[0] == cpu_config.training.large_negative_value


def test_mask_job_reoptimize_collapses_to_first_index(cpu_config):
    masker     = ActionMasker(cpu_config)
    job_logits = torch.zeros(4)
    info       = {"unassigned_job_indices": [2], "vehicles_with_jobs_indices": [0], "vehicle_to_job_indices": {0: [1]}, "vehicles_with_capacity_indices": [0]}

    masked = masker.mask_job(job_logits, info, selected_operator_index=3, selected_vehicle_index=0)

    assert masked[0] == 0.0
    assert masked[1] == cpu_config.training.large_negative_value


def _masker():
    from configuration import Config
    config = Config()
    config.training.device = "cpu"
    return ActionMasker(config)
