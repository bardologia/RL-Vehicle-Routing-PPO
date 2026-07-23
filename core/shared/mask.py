import torch


class ActionMaskBuilder:
    def build(self, jobs, vehicles, state):
        assigned_ids = state.assigned_job_ids
        eligible_ids = {job_id for job_id in state.unassigned_ids if jobs.contains(job_id) and job_id not in assigned_ids}

        insert_indices = sorted(jobs.index_of(job_id) for job_id in eligible_ids)

        vehicle_to_job_indices = {vehicle_index: [] for vehicle_index in range(len(vehicles))}
        for route in state.routes:
            vehicle_index = vehicles.index_of(route.vehicle_id)
            if vehicle_index is None:
                continue

            for job_id in route.job_ids:
                job_index = jobs.index_of(job_id)
                if job_index is not None:
                    vehicle_to_job_indices[vehicle_index].append(job_index)

        vehicles_with_jobs = [vehicle_index for vehicle_index, job_indices in vehicle_to_job_indices.items() if job_indices]

        load_by_vehicle_id     = {route.vehicle_id: len(route.stops) for route in state.routes}
        vehicles_with_capacity = [vehicle_index for vehicle_index, vehicle in enumerate(vehicles) if load_by_vehicle_id.get(vehicle.id, 0) < vehicle.capacity]

        return {
            "unassigned_job_indices"         : insert_indices,
            "vehicles_with_jobs_indices"     : vehicles_with_jobs,
            "vehicle_to_job_indices"         : vehicle_to_job_indices,
            "vehicles_with_capacity_indices" : vehicles_with_capacity,
        }


class ActionMasker:
    def __init__(self, config):
        self.large_negative_value = config.training.large_negative_value

    def mask_operator(self, operator_logits, mask_info):
        masked_operator_logits = operator_logits.clone()

        if mask_info is None:
            return masked_operator_logits

        unassigned_job_indices         = mask_info["unassigned_job_indices"]
        vehicles_with_jobs_indices     = mask_info["vehicles_with_jobs_indices"]
        vehicles_with_capacity_indices = mask_info["vehicles_with_capacity_indices"]

        if (len(unassigned_job_indices) == 0 or len(vehicles_with_capacity_indices) == 0) and masked_operator_logits.numel() > 0:
            masked_operator_logits[0] = self.large_negative_value

        if len(vehicles_with_jobs_indices) == 0 and masked_operator_logits.numel() > 1:
            masked_operator_logits[1] = self.large_negative_value

        return masked_operator_logits

    def mask_vehicle(self, vehicle_logits, mask_info, selected_operator_index):
        masked_vehicle_logits = vehicle_logits.clone()

        if mask_info is None:
            return masked_vehicle_logits

        vehicles_with_jobs_indices     = mask_info["vehicles_with_jobs_indices"]
        vehicles_with_capacity_indices = mask_info["vehicles_with_capacity_indices"]

        if selected_operator_index == 0 and len(vehicles_with_capacity_indices) > 0:
            invalid_vehicle_mask = torch.ones_like(masked_vehicle_logits, dtype=torch.bool)
            invalid_vehicle_mask[vehicles_with_capacity_indices] = False
            masked_vehicle_logits[invalid_vehicle_mask] = self.large_negative_value

        elif selected_operator_index == 1 and len(vehicles_with_jobs_indices) > 0:  # REMOVE operator
            invalid_vehicle_mask = torch.ones_like(masked_vehicle_logits, dtype=torch.bool)
            invalid_vehicle_mask[vehicles_with_jobs_indices] = False
            masked_vehicle_logits[invalid_vehicle_mask] = self.large_negative_value

        elif selected_operator_index in [2, 3]:
            if masked_vehicle_logits.numel() > 1:
                masked_vehicle_logits[1:] = self.large_negative_value

        return masked_vehicle_logits

    def mask_job(self, job_logits, mask_info, selected_operator_index, selected_vehicle_index):

        masked_job_logits = job_logits.clone()

        if mask_info is None:
            return masked_job_logits

        unassigned_job_indices = mask_info["unassigned_job_indices"]
        vehicle_to_job_indices = mask_info["vehicle_to_job_indices"]

        if selected_operator_index == 0:
            if len(unassigned_job_indices) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[unassigned_job_indices] = False
                masked_job_logits[invalid_job_mask] = self.large_negative_value

        elif selected_operator_index == 1 and selected_vehicle_index is not None:  # REMOVE operator
            valid_jobs_for_vehicle = vehicle_to_job_indices[int(selected_vehicle_index)]
            if len(valid_jobs_for_vehicle) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[valid_jobs_for_vehicle] = False
                masked_job_logits[invalid_job_mask] = self.large_negative_value

        elif selected_operator_index in [2, 3]:
            if masked_job_logits.numel() > 1:
                masked_job_logits[1:] = self.large_negative_value

        return masked_job_logits

