import torch


class MaskContext:
    def __init__(self):
        self.job_ids_in_routes          = set()
        self.eligible_unassigned_ids    = set()
        self.valid_insert_job_indices   = []
        self.vehicles_with_jobs_indices = []
        self.vehicle_to_job_indices     = {}

    def _reset(self):
        self.job_ids_in_routes.clear()
        self.eligible_unassigned_ids.clear()
        self.valid_insert_job_indices.clear()
        self.vehicles_with_jobs_indices.clear()
        self.vehicle_to_job_indices.clear()

    def _job_ids_in_routes(self, current_routes):
        job_ids = set()
        # import here to avoid circular import at module import time
        from core.environment import RouteHandler

        for route in current_routes:
            for step_job_id in RouteHandler.job_ids_from_steps(route.get("steps")):
                job_ids.add(step_job_id)

        self.job_ids_in_routes = job_ids

    def _eligible_unassigned_ids(self, unassigned_ids, jobs_by_id):
        eligible = set()
        for job_id in unassigned_ids:
            if job_id in jobs_by_id and job_id not in self.job_ids_in_routes:
                eligible.add(job_id)
        
        self.eligible_unassigned_ids = eligible

    def _valid_insert_job_indices(self, job_id_to_index):
        self.valid_insert_job_indices = [
            job_id_to_index[job_id]
            for job_id in self.eligible_unassigned_ids
            if job_id in job_id_to_index
        ]

    def _vehicle_to_job_indices(self, current_routes, vehicle_id_to_index, job_id_to_index, vehicles):
        vehicle_to_job_indices = {v_idx: [] for v_idx in range(len(vehicles))}

        for route in current_routes:
            route_vehicle_raw = route.get("vehicle")
            if route_vehicle_raw is None:
                continue
            route_vehicle_id = int(route_vehicle_raw)

            vehicle_index = vehicle_id_to_index.get(route_vehicle_id)
            if vehicle_index is None:
                continue

            # import here to avoid circular import at module import time
            from core.environment import RouteHandler

            for step_job_id in RouteHandler.job_ids_from_steps(route.get("steps")):
                job_index = job_id_to_index.get(step_job_id)
                if job_index is not None:
                    vehicle_to_job_indices[vehicle_index].append(job_index)

        self.vehicle_to_job_indices = vehicle_to_job_indices
    
    def _vehicles_with_jobs_indices(self):
        self.vehicles_with_jobs_indices = [
            v_idx for v_idx, job_indices in self.vehicle_to_job_indices.items() if job_indices
        ]

    def build(self, current_routes, unassigned_job_ids, job_id_to_index, jobs_by_id, vehicle_id_to_index, vehicles):
        self._reset()
        unassigned_ids = set(unassigned_job_ids)

        self._job_ids_in_routes(current_routes)
        self._eligible_unassigned_ids(unassigned_ids, jobs_by_id)
        self._valid_insert_job_indices(job_id_to_index)
        self._vehicle_to_job_indices(current_routes, vehicle_id_to_index, job_id_to_index, vehicles)
        self._vehicles_with_jobs_indices()

        return {
            "unassigned_job_indices"     : self.valid_insert_job_indices,
            "vehicles_with_jobs_indices" : self.vehicles_with_jobs_indices,
            "vehicle_to_job_indices"     : self.vehicle_to_job_indices,
        }


class PPOMasking:
    @staticmethod
    def mask_operator(operator_logits, mask_info, large_negative_value):
        masked_operator_logits = operator_logits.clone()

        if mask_info is None:
            return masked_operator_logits

        unassigned_job_indices = mask_info.get("unassigned_job_indices", [])
        vehicles_with_jobs_indices = mask_info.get("vehicles_with_jobs_indices", [])

        if len(unassigned_job_indices) == 0 and masked_operator_logits.numel() > 0:
            masked_operator_logits[0] = large_negative_value

        if len(vehicles_with_jobs_indices) == 0 and masked_operator_logits.numel() > 1:
            masked_operator_logits[1] = large_negative_value

        return masked_operator_logits

    @staticmethod
    def mask_vehicle(vehicle_logits, mask_info, selected_operator_index, large_negative_value):
        masked_vehicle_logits = vehicle_logits.clone()

        if mask_info is None:
            return masked_vehicle_logits

        vehicles_with_jobs_indices = mask_info.get("vehicles_with_jobs_indices", [])

        if selected_operator_index == 1 and len(vehicles_with_jobs_indices) > 0:
            invalid_vehicle_mask = torch.ones_like(masked_vehicle_logits, dtype=torch.bool)
            invalid_vehicle_mask[vehicles_with_jobs_indices] = False
            masked_vehicle_logits[invalid_vehicle_mask] = large_negative_value

        return masked_vehicle_logits

    @staticmethod
    def mask_job(job_logits, mask_info, selected_operator_index, selected_vehicle_index, large_negative_value):
        masked_job_logits = job_logits.clone()

        if mask_info is None:
            return masked_job_logits

        unassigned_job_indices = mask_info.get("unassigned_job_indices", [])
        vehicle_to_job_indices = mask_info.get("vehicle_to_job_indices", {})

        if selected_operator_index == 0:
            if len(unassigned_job_indices) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[unassigned_job_indices] = False
                masked_job_logits[invalid_job_mask] = large_negative_value

        elif selected_operator_index == 1 and selected_vehicle_index is not None:
            valid_jobs_for_vehicle = vehicle_to_job_indices.get(int(selected_vehicle_index), [])
            if len(valid_jobs_for_vehicle) > 0:
                invalid_job_mask = torch.ones_like(masked_job_logits, dtype=torch.bool)
                invalid_job_mask[valid_jobs_for_vehicle] = False
                masked_job_logits[invalid_job_mask] = large_negative_value

        return masked_job_logits

