class ScenarioTemplates:
    def _job(self, job_id, lon, lat, priority, kind="support"):
        return {
            "id"          : job_id,
            "location"    : [lon, lat],
            "kind"        : kind,
            "setup"       : 0,
            "service"     : 600 if kind == "support" else 300,
            "amount"      : 1 if kind == "repossession" else 0,
            "priority"    : priority,
            "description" : f"Job {job_id}",
        }

    def _vehicle(self, vehicle_id, lon, lat, capacity):
        return {
            "id"              : vehicle_id,
            "start"           : [lon, lat],
            "capacity"        : capacity,
            "onboard"         : 0,
            "time_window"     : [28800, 72000],
            "speed_factor"    : 1.0,
            "return_to_depot" : False,
            "description"     : f"Vehicle {vehicle_id}",
        }

    def cheap_detour(self):
        return {
            "key"         : "cheap_detour",
            "title"       : "Cheap detour insert",
            "description" : "One vehicle drives a short support route through Pinheiros. A new client-support call sits directly between its two stops, so serving it costs almost nothing extra.",
            "expected"    : "INSERT the middle job into the existing route with a small distance penalty.",
            "depot"       : [-46.640, -23.545],
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
            "description" : "The same short Pinheiros route, but the new support call is priority 1 and sits on the far east side of the city. The detour costs more than the job is worth.",
            "expected"    : "DO_NOTHING: a rational agent refuses the insertion and leaves the job unassigned.",
            "depot"       : [-46.640, -23.545],
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
            "description" : "Same route, but the distant call is priority 5. The detour is identical to a refusal case at low priority; the value of the job flips the decision. Lower its priority in the editor and rerun to see the refusal.",
            "expected"    : "INSERT the priority-5 job despite the long detour.",
            "depot"       : [-46.640, -23.545],
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
            "description" : "Vehicle 0 works Santana in the north but carries one support call in Mooca, right where vehicle 1 is working. Removing it costs unassigned and priority penalties for one step, which pays off when it lands on the right vehicle.",
            "expected"    : "REMOVE the Mooca job from vehicle 0, then INSERT it into vehicle 1.",
            "depot"       : [-46.625, -23.535],
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

    def multi_trip(self):
        return {
            "key"         : "multi_trip",
            "title"       : "Multi-trip repossessions",
            "description" : "One truck with capacity 2 faces four motorcycle repossessions around Pinheiros. The planner must interleave depot drops: collect two, unload at the central, collect two more. Watch the route pass through the depot mid-plan and the onboard load reset as drops execute.",
            "expected"    : "INSERT all four repossessions; the route includes depot deliveries between pickups.",
            "depot"       : [-46.640, -23.545],
            "jobs"        : [
                self._job(0, -46.690, -23.560, 3, kind="repossession"),
                self._job(1, -46.680, -23.552, 3, kind="repossession"),
                self._job(2, -46.670, -23.565, 4, kind="repossession"),
                self._job(3, -46.660, -23.548, 3, kind="repossession"),
            ],
            "vehicles"   : [self._vehicle(0, -46.700, -23.562, 2)],
            "assignment" : {0: []},
        }

    def abandon_outlier(self):
        return {
            "key"         : "abandon_outlier",
            "title"       : "Abandon the outlier",
            "description" : "A city vehicle is stuck with one support call out in Jundiai, an hour away. No other vehicle can take it. Dropping it entirely saves more driving than the unassigned penalties cost.",
            "expected"    : "REMOVE the Jundiai job and leave it unassigned.",
            "depot"       : [-46.640, -23.545],
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
            "description" : "Vehicle 0 is busy in Mooca. Vehicle 1 sits idle in Santo Amaro with two unassigned support calls next to it. Activating an idle vehicle earns the idle bonus on top of the assignments.",
            "expected"    : "INSERT the nearby jobs into idle vehicle 1, one per step.",
            "depot"       : [-46.625, -23.560],
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
            "description" : "Vehicle 0 starts in Santana but serves a Santo Amaro call; vehicle 1 does the opposite. Both cross the whole city for one job. There is no global re-solve operator, so untangling means migrating the jobs one at a time.",
            "expected"    : "REMOVE and INSERT pairs swap the jobs onto the vehicles in their own neighbourhoods.",
            "depot"       : [-46.660, -23.570],
            "jobs"        : [
                self._job(0, -46.705, -23.650, 3),
                self._job(1, -46.632, -23.487, 3),
            ],
            "vehicles" : [
                self._vehicle(0, -46.630, -23.490, 4),
                self._vehicle(1, -46.706, -23.648, 4),
            ],
            "assignment" : {0: [0], 1: [1]},
        }

    def all_settled(self):
        return {
            "key"         : "all_settled",
            "title"       : "Nothing to do",
            "description" : "Two compact support clusters, each served by its own vehicle, nothing unassigned. Any action costs more than it gains; with time moving, the right play is to let the routes execute.",
            "expected"    : "DO_NOTHING while the vehicles work through their stops.",
            "depot"       : [-46.640, -23.550],
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
            self.multi_trip(),
            self.abandon_outlier(),
            self.wake_idle(),
            self.tangled_routes(),
            self.all_settled(),
        ]
