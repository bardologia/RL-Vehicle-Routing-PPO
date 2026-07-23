class ScenarioTemplates:
    def _job(self, job_id, lon, lat, priority):
        return {
            "id"          : job_id,
            "location"    : [lon, lat],
            "setup"       : 0,
            "service"     : 300,
            "amount"      : 1,
            "priority"    : priority,
            "description" : f"Job {job_id}",
        }

    def _vehicle(self, vehicle_id, lon, lat, capacity):
        return {
            "id"              : vehicle_id,
            "start"           : [lon, lat],
            "capacity"        : capacity,
            "time_window"     : [28800, 72000],
            "speed_factor"    : 1.0,
            "return_to_depot" : False,
            "description"     : f"Vehicle {vehicle_id}",
        }

    def cheap_detour(self):
        return {
            "key"         : "cheap_detour",
            "title"       : "Cheap detour insert",
            "description" : "One vehicle drives a short route through Pinheiros. A new job sits directly between its two stops, so serving it costs almost nothing extra.",
            "expected"    : "INSERT the middle job into the existing route with a small distance penalty.",
            "jobs"        : [
                self._job(0, -46.685, -23.560, 3),
                self._job(1, -46.665, -23.555, 3),
                self._job(2, -46.675, -23.5575, 3),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.565, 4)],
            "assignment" : {0: [0, 1]},
        }

    def far_low_priority(self):
        return {
            "key"         : "far_low_priority",
            "title"       : "Far low-priority job",
            "description" : "The same short Pinheiros route, but the new job is priority 1 and sits on the far east side of the city. The detour costs more than the job is worth.",
            "expected"    : "DO_NOTHING: a rational agent refuses the insertion and leaves the job unassigned.",
            "jobs"        : [
                self._job(0, -46.685, -23.560, 3),
                self._job(1, -46.665, -23.555, 3),
                self._job(2, -46.440, -23.500, 1),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.565, 4)],
            "assignment" : {0: [0, 1]},
        }

    def priority_rescue(self):
        return {
            "key"         : "priority_rescue",
            "title"       : "Priority overrides distance",
            "description" : "Same route, but the distant job is priority 5. The detour is identical to a refusal case at low priority; the value of the job flips the decision. Lower its priority in the editor and rerun to see the refusal.",
            "expected"    : "INSERT the priority-5 job despite the long detour.",
            "jobs"        : [
                self._job(0, -46.685, -23.560, 3),
                self._job(1, -46.665, -23.555, 3),
                self._job(2, -46.600, -23.550, 5),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.565, 4)],
            "assignment" : {0: [0, 1]},
        }

    def wrong_vehicle(self):
        return {
            "key"         : "wrong_vehicle",
            "title"       : "Job on the wrong vehicle",
            "description" : "Vehicle 0 works Santana in the north but carries one job in Mooca, right where vehicle 1 is working with spare capacity. Removing it costs unassigned and priority penalties for one step, which pays off when it lands on the right vehicle.",
            "expected"    : "REMOVE the Mooca job from vehicle 0, then INSERT it into vehicle 1.",
            "jobs"        : [
                self._job(0, -46.635, -23.485, 3),
                self._job(1, -46.625, -23.492, 3),
                self._job(2, -46.600, -23.552, 2),
                self._job(3, -46.605, -23.548, 3),
                self._job(4, -46.595, -23.556, 3),
            ],
            "vehicles" : [
                self._vehicle(0, -46.630, -23.490, 4),
                self._vehicle(1, -46.598, -23.550, 4),
            ],
            "assignment" : {0: [0, 1, 2], 1: [3, 4]},
        }

    def capacity_swap(self):
        return {
            "key"         : "capacity_swap",
            "title"       : "Full vehicle, better job waiting",
            "description" : "The only vehicle is at capacity with low-priority jobs while a priority-5 job waits nearby. The only way to serve it is to drop a cheap job first.",
            "expected"    : "REMOVE a priority-1 job, then INSERT the priority-5 job into the freed slot.",
            "jobs"        : [
                self._job(0, -46.690, -23.560, 1),
                self._job(1, -46.680, -23.555, 2),
                self._job(2, -46.670, -23.565, 1),
                self._job(3, -46.678, -23.558, 5),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.562, 3)],
            "assignment" : {0: [0, 1, 2]},
        }

    def abandon_outlier(self):
        return {
            "key"         : "abandon_outlier",
            "title"       : "Abandon the outlier",
            "description" : "A city vehicle is stuck with one job out in Jundiai, an hour away. No other vehicle can take it. Dropping it entirely saves more driving than the unassigned penalties cost.",
            "expected"    : "REMOVE the Jundiai job and leave it unassigned.",
            "jobs"        : [
                self._job(0, -46.690, -23.560, 3),
                self._job(1, -46.675, -23.552, 3),
                self._job(2, -46.880, -23.200, 1),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.565, 4)],
            "assignment" : {0: [0, 1, 2]},
        }

    def wake_idle(self):
        return {
            "key"         : "wake_idle",
            "title"       : "Wake the idle vehicle",
            "description" : "Vehicle 0 is full in Mooca. Vehicle 1 sits idle in Santo Amaro with two unassigned jobs next to it. Activating an idle vehicle earns the idle bonus on top of the assignments.",
            "expected"    : "INSERT the nearby jobs into idle vehicle 1, one per step.",
            "jobs"        : [
                self._job(0, -46.600, -23.550, 3),
                self._job(1, -46.595, -23.555, 3),
                self._job(2, -46.605, -23.545, 3),
                self._job(3, -46.705, -23.650, 4),
                self._job(4, -46.710, -23.645, 3),
            ],
            "vehicles" : [
                self._vehicle(0, -46.598, -23.552, 3),
                self._vehicle(1, -46.708, -23.648, 4),
            ],
            "assignment" : {0: [0, 1, 2]},
        }

    def tangled_routes(self):
        return {
            "key"         : "tangled_routes",
            "title"       : "Tangled routes",
            "description" : "Vehicle 0 starts in Santana but serves Santo Amaro; vehicle 1 does the opposite. Both cross the whole city. Fixing this one job at a time takes too many steps; one global re-solve untangles it at the price of disrupting every job.",
            "expected"    : "REOPTIMIZE: the wholesale swap beats surgical remove-insert pairs.",
            "jobs"        : [
                self._job(0, -46.705, -23.650, 3),
                self._job(1, -46.712, -23.643, 3),
                self._job(2, -46.700, -23.657, 3),
                self._job(3, -46.632, -23.487, 3),
                self._job(4, -46.626, -23.493, 3),
                self._job(5, -46.638, -23.482, 3),
            ],
            "vehicles" : [
                self._vehicle(0, -46.630, -23.490, 4),
                self._vehicle(1, -46.706, -23.648, 4),
            ],
            "assignment" : {0: [0, 1, 2], 1: [3, 4, 5]},
        }

    def all_settled(self):
        return {
            "key"         : "all_settled",
            "title"       : "Nothing to do",
            "description" : "Two compact clusters, each served by its own vehicle, nothing unassigned. Any action costs more than it gains.",
            "expected"    : "DO_NOTHING immediately.",
            "jobs"        : [
                self._job(0, -46.690, -23.558, 3),
                self._job(1, -46.682, -23.562, 3),
                self._job(2, -46.602, -23.550, 3),
                self._job(3, -46.596, -23.554, 3),
            ],
            "vehicles" : [
                self._vehicle(0, -46.695, -23.560, 4),
                self._vehicle(1, -46.600, -23.552, 4),
            ],
            "assignment" : {0: [0, 1], 1: [2, 3]},
        }

    def catalog(self):
        return [
            self.cheap_detour(),
            self.far_low_priority(),
            self.priority_rescue(),
            self.wrong_vehicle(),
            self.capacity_swap(),
            self.abandon_outlier(),
            self.wake_idle(),
            self.tangled_routes(),
            self.all_settled(),
        ]
