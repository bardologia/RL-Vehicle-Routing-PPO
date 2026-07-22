import torch


class MaskContext:
    def build(self, environment):
        state    = environment.current_state
        jobs     = environment.jobs
        vehicles = environment.vehicles

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

        return {
            "unassigned_job_indices"     : insert_indices,
            "vehicles_with_jobs_indices" : vehicles_with_jobs,
            "vehicle_to_job_indices"     : vehicle_to_job_indices,
        }


class PPOMasking:
    def __init__(self, config):
        self.large_negative_value = config.training.large_negative_value

    def mask_operator(self, op_logits, mask_info):
        masked_op_logits = op_logits.clone()

        if mask_info is None:
            return masked_op_logits

        unassigned_job_indices = mask_info.get("unassigned_job_indices", [])
        vehicles_with_jobs_indices = mask_info.get("vehicles_with_jobs_indices", [])

        if len(unassigned_job_indices) == 0 and masked_op_logits.numel() > 0:
            masked_op_logits[0] = self.large_negative_value

        if len(vehicles_with_jobs_indices) == 0 and masked_op_logits.numel() > 1:
            masked_op_logits[1] = self.large_negative_value

        return masked_op_logits

    def mask_vehicle(self, veh_logits, mask_info, selected_op_idx):
        masked_veh_logits = veh_logits.clone()

        if mask_info is None:
            return masked_veh_logits

        vehicles_with_jobs_indices = mask_info.get("vehicles_with_jobs_indices", [])

        if selected_op_idx == 1 and len(vehicles_with_jobs_indices) > 0:  # REMOVE operator
            invalid_vehicle_mask = torch.ones_like(masked_veh_logits, dtype=torch.bool)
            invalid_vehicle_mask[vehicles_with_jobs_indices] = False
            masked_veh_logits[invalid_vehicle_mask] = self.large_negative_value
        
        elif selected_op_idx in [2, 3]: 
            if masked_veh_logits.numel() > 1:
                masked_veh_logits[1:] = self.large_negative_value

        return masked_veh_logits

    def mask_job(self, job_logits, mask_info, selected_op_idx, selected_veh_idx):

        masked_job_logits = job_logits.clone()

        if mask_info is None:
            return masked_job_logits

        unassigned_job_indices = mask_info.get("unassigned_job_indices", [])
        vehicle_to_job_indices = mask_info.get("vehicle_to_job_indices", {})

        if selected_op_idx == 0:  
            if len(unassigned_job_indices) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[unassigned_job_indices] = False
                masked_job_logits[invalid_job_mask] = self.large_negative_value

        elif selected_op_idx == 1 and selected_veh_idx is not None:  # REMOVE operator
            valid_jobs_for_vehicle = vehicle_to_job_indices.get(int(selected_veh_idx), [])
            if len(valid_jobs_for_vehicle) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[valid_jobs_for_vehicle] = False
                masked_job_logits[invalid_job_mask] = self.large_negative_value
        
        elif selected_op_idx in [2, 3]: 
            if masked_job_logits.numel() > 1:
                masked_job_logits[1:] = self.large_negative_value

        return masked_job_logits

