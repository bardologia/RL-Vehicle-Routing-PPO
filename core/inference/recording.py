"""Episode recording for diagnostics: settings, plan KPIs, policy and feature readers, teacher verdicts and the recorder.

Everything here observes the trained MDP through `EpisodeDriver` without changing it: the recorder runs the same rounds evaluation runs and writes one trace per agent in the shape the web console reads.
"""

import copy
import gzip
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from tools.auxiliary import haversine_distance
from tools.logger import NullLogger
from core.shared import EpisodeDriver, services
from core.shared.state import EntityPool, Job, RoutingState, Vehicle
from core.training.ppo import ActionDistribution
from core.training.pretraining import RegretInsertionTeacher
from model.policy_model import Action


OPERATOR_NAMES = {0: "INSERT", 1: "REMOVE", 2: "DO_NOTHING"}
REWARD_TERMS   = ("distance_reward", "unassigned_reward", "idle_reward", "priority_reward", "action_reward", "holding_reward")
PLAN_TERMS     = ("distance_reward", "unassigned_reward", "idle_reward", "priority_reward")


class EpisodeSettings:
    """Builds the configuration one recorded episode runs under."""

    def _apply_section(self, section, values):
        known = {spec.name for spec in fields(section) if spec.init}

        for name, value in values.items():
            if name not in known:
                raise ValueError(f"Unknown field {name!r} for {type(section).__name__}")

            setattr(section, name, tuple(value) if isinstance(value, list) else value)

    def build(self, base_config, environment=None, reward=None, rounds=None, edits=None):
        """Return a deep copy of a configuration with episode overrides applied.

        Args:
            base_config : Config to copy; it is left untouched.
            environment : Mapping of EnvironmentConfig fields, as written in an evaluation report meta, or None.
            reward      : Mapping of RewardConfig fields, or None.
            rounds      : Disruption rounds per episode, or None to keep the base value.
            edits       : Edit cap per round, or None to keep the base value.

        Returns:
            The new Config.
        """
        config = copy.deepcopy(base_config)

        if environment is not None:
            self._apply_section(config.env, environment)
        if reward is not None:
            self._apply_section(config.reward, reward)
        if rounds is not None:
            config.training.max_disruptions_per_episode = int(rounds)
        if edits is not None:
            config.training.max_edits_per_round = int(edits)

        return config


class EpisodeLedger:
    """Running totals of one recorded episode that plan KPIs depend on.

    Attributes:
        first_seen     : Clock at which each job id first appeared in the pool, in seconds of day.
        served         : Jobs served so far.
        failed         : Pickups failed so far.
        dropped        : Bikes dropped at the depot so far.
        returns        : Cumulative reward per term, keyed by the six reward term names.
        last_executed  : Per vehicle, jobs served and failed in the latest advance, {vehicle_id: {"served", "failed"}}.
        previous_drive : Drive seconds per vehicle in the previous frame, {vehicle_id: seconds}; None until a first frame was rendered.
    """

    def __init__(self):
        self.first_seen     = {}
        self.served         = 0
        self.failed         = 0
        self.dropped        = 0
        self.returns        = {term: 0.0 for term in REWARD_TERMS}
        self.last_executed  = {}
        self.previous_drive = None

    def see_jobs(self, jobs, clock):
        """Stamp the clock on every job id not seen before."""
        for job in jobs:
            self.first_seen.setdefault(job.id, int(clock))

    def execute(self, served, failed, dropped):
        """Add one advance: served and failed entries carry "vehicle_id", dropped entries carry "count"."""
        self.served  += len(served)
        self.failed  += len(failed)
        self.dropped += sum(entry["count"] for entry in dropped)

        self.last_executed = {}
        for key, entries in (("served", served), ("failed", failed)):
            for entry in entries:
                self.last_executed.setdefault(entry["vehicle_id"], {"served": 0, "failed": 0})[key] += 1

    def reward(self, rewards):
        """Add one decision's reward terms to the cumulative returns."""
        for term in REWARD_TERMS:
            self.returns[term] += float(rewards[term])

    def drive_delta(self, vehicle_id, drive_s):
        """Return the change of a vehicle's drive seconds against the previous frame; 0 on the first frame, which has no predecessor."""
        if self.previous_drive is None:
            return 0

        return int(drive_s) - int(self.previous_drive.get(vehicle_id, 0))

    def remember_drive(self, state):
        """Keep the drive seconds per vehicle of the frame just rendered."""
        self.previous_drive = {route.vehicle_id: int(route.cost) for route in state.routes}


class PlanMetrics:
    """Per-plan and per-route KPIs computed from a RoutingState and the pools it refers to.

    Attributes:
        reward : RewardConfig whose weights price the plan.
    """

    def __init__(self, reward_config):
        self.reward = reward_config

    def _orientation(self, a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def _segments_cross(self, first, second):
        a, b = first
        c, d = second

        d1 = self._orientation(c, d, a)
        d2 = self._orientation(c, d, b)
        d3 = self._orientation(a, b, c)
        d4 = self._orientation(a, b, d)

        return d1 * d2 < 0 and d3 * d4 < 0

    def route_points(self, route):
        """Return the straight-leg polyline of a route: its start followed by every stop, as (lon, lat) tuples."""
        points = [tuple(route.start)] if route.start is not None else []
        points.extend(tuple(stop.location) for stop in route.stops)
        return points

    def crossings(self, polylines):
        """Count proper intersections between straight legs of different routes.

        Args:
            polylines : One list of (lon, lat) points per route.

        Returns:
            Tuple of the total number of crossings and the list of crossings each route takes part in.
        """
        segments  = [list(zip(points, points[1:])) for points in polylines]
        per_route = [0] * len(polylines)
        total     = 0

        for first in range(len(segments)):
            for second in range(first + 1, len(segments)):
                count = sum(1 for one in segments[first] for other in segments[second] if self._segments_cross(one, other))

                per_route[first]  += count
                per_route[second] += count
                total             += count

        return total, per_route

    def straight_length(self, points):
        """Return the summed haversine length of a polyline of (lon, lat) points, in metres."""
        return sum(haversine_distance(a[0], a[1], b[0], b[1])[0] for a, b in zip(points, points[1:]))

    def break_even(self, priority, disruption, wakes_idle):
        """Return the added drive seconds at which inserting a job stops paying, per the reward weights.

        Args:
            priority   : Priority of the inserted job, 1 to 5.
            disruption : Stops whose vehicle changes because of the insertion.
            wakes_idle : Whether the receiving vehicle had no route before.

        Returns:
            Threshold in seconds; the insertion reward is non-negative when the added drive time is at most this.
        """
        reward = self.reward
        gain   = reward.unassigned_penalty_weight + reward.priority_penalty_weight * priority - reward.add_job_cost - reward.disruption_cost * disruption

        if wakes_idle:
            gain += reward.idle_penalty_weight

        return gain / reward.distance_weight * 1000.0

    def _vehicle_state(self, vehicle, clock, active):
        if clock >= vehicle.time_window[1]:
            return "SHIFT OVER"

        return "ACTIVE" if active else "IDLE"

    def _stops(self, route):
        stops    = []
        previous = 0

        for stop in route.stops:
            stops.append({
                "job_id"     : int(stop.job_id),
                "kind"       : stop.kind,
                "location"   : [float(stop.location[0]), float(stop.location[1])],
                "arrival"    : int(stop.arrival),
                "completion" : int(stop.completion),
                "leg_s"      : int(stop.duration) - previous,
                "load"       : int(stop.load),
            })
            previous = int(stop.duration)

        return stops

    def _route_kpis(self, route, vehicle, jobs, clock, plan_drive, crossings, ledger):
        visits       = [stop for stop in route.stops if stop.kind in ("job", "pickup")]
        loads        = [int(vehicle.onboard)] + [int(stop.load) for stop in route.stops]
        end_clock    = int(route.stops[-1].completion) if route.stops else None
        executed     = ledger.last_executed.get(vehicle.id, {"served": 0, "failed": 0})
        priority_sum = sum(int(jobs.by_id(stop.job_id).priority) for stop in visits if jobs.contains(stop.job_id))

        return {
            "state"              : self._vehicle_state(vehicle, clock, True),
            "stops"              : len(visits),
            "stops_support"      : sum(1 for stop in visits if stop.kind == "job"),
            "stops_repossession" : sum(1 for stop in visits if stop.kind == "pickup"),
            "depot_drops"        : sum(1 for stop in route.stops if stop.kind == "delivery"),
            "drive_s"            : int(route.cost),
            "drive_share"        : float(route.cost) / plan_drive if plan_drive > 0 else 0.0,
            "distance_m"         : None,
            "service_s"          : int(route.service),
            "end_clock"          : end_clock,
            "slack_s"            : int(vehicle.time_window[1]) - end_clock if end_clock is not None else None,
            "peak_load"          : max(loads),
            "capacity"           : int(vehicle.capacity),
            "load_profile"       : loads,
            "onboard"            : int(vehicle.onboard),
            "priority_sum"       : priority_sum,
            "detour"             : None,
            "crossings"          : crossings,
            "reward_weight"      : self.reward.distance_weight * float(route.cost) / 1000.0,
            "served_last"        : executed["served"],
            "failed_last"        : executed["failed"],
            "drive_delta_s"      : ledger.drive_delta(vehicle.id, route.cost),
        }

    def _idle_kpis(self, vehicle, clock, ledger):
        executed = ledger.last_executed.get(vehicle.id, {"served": 0, "failed": 0})

        return {
            "state"              : self._vehicle_state(vehicle, clock, False),
            "stops"              : 0,
            "stops_support"      : 0,
            "stops_repossession" : 0,
            "depot_drops"        : 0,
            "drive_s"            : 0,
            "drive_share"        : 0.0,
            "distance_m"         : 0,
            "service_s"          : 0,
            "end_clock"          : None,
            "slack_s"            : None,
            "peak_load"          : int(vehicle.onboard),
            "capacity"           : int(vehicle.capacity),
            "load_profile"       : [int(vehicle.onboard)],
            "onboard"            : int(vehicle.onboard),
            "priority_sum"       : 0,
            "detour"             : None,
            "crossings"          : 0,
            "reward_weight"      : 0.0,
            "served_last"        : executed["served"],
            "failed_last"        : executed["failed"],
            "drive_delta_s"      : ledger.drive_delta(vehicle.id, 0),
        }

    def _plan_kpis(self, jobs, vehicles, state, clock, routes, crossings, ledger):
        reward     = self.reward
        unassigned = [jobs.by_id(job_id) for job_id in sorted(state.unassigned_ids) if jobs.contains(job_id)]
        priority   = sum(int(job.priority) for job in unassigned)
        idle       = len(vehicles) - state.num_routes
        drive      = int(state.cost)
        waits      = [int(clock) - ledger.first_seen.get(job.id, int(clock)) for job in unassigned]
        capacity   = sum(route["kpis"]["capacity"] for route in routes)
        returns    = ledger.returns

        return {
            "drive_s"             : drive,
            "distance_m"          : None,
            "service_s"           : int(state.service),
            "unassigned"          : len(unassigned),
            "unassigned_priority" : priority,
            "oldest_unassigned_s" : max(waits) if waits else 0,
            "idle_vehicles"       : idle,
            "standing_cost"       : reward.distance_weight * drive / 1000.0 + reward.unassigned_penalty_weight * len(unassigned) + reward.idle_penalty_weight * idle + reward.priority_penalty_weight * priority,
            "holding_rate"        : (reward.unassigned_penalty_weight * len(unassigned) + reward.priority_penalty_weight * priority) * 3600.0 / max(float(reward.holding_reference_seconds), 1.0),
            "served"              : ledger.served,
            "failed"              : ledger.failed,
            "dropped"             : ledger.dropped,
            "drive_per_served_s"  : drive / ledger.served if ledger.served > 0 else None,
            "load_use"            : sum(route["kpis"]["peak_load"] for route in routes) / capacity if capacity > 0 else 0.0,
            "crossings"           : crossings,
            "return"              : sum(returns.values()),
            "return_holding"      : returns["holding_reward"],
            "return_action"       : returns["action_reward"],
            "return_plan"         : sum(returns[term] for term in PLAN_TERMS),
        }

    def plan(self, jobs, vehicles, state, clock, ledger):
        """Render a plan with its KPIs.

        Args:
            jobs     : EntityPool of jobs the state refers to.
            vehicles : EntityPool of vehicles.
            state    : RoutingState to render.
            clock    : Current clock in seconds of day.
            ledger   : EpisodeLedger with the running totals of the episode.

        Returns:
            Plan dict with "routes", "idle", "idle_vehicle_ids", "unassigned", "vehicles" and "kpis". Geometry fields ("path", "distance_m", "detour") are None until `add_geometry` fills them.
        """
        active = [route for route in state.routes if vehicles.contains(route.vehicle_id)]
        active.sort(key=lambda route: route.vehicle_id)

        total_crossings, per_route = self.crossings([self.route_points(route) for route in active])
        plan_drive                 = float(state.cost)

        routes = []
        for route, crossings in zip(active, per_route):
            vehicle = vehicles.by_id(route.vehicle_id)
            routes.append({
                "vehicle_id" : int(route.vehicle_id),
                "start"      : [float(route.start[0]), float(route.start[1])] if route.start is not None else [float(vehicle.start[0]), float(vehicle.start[1])],
                "drive_s"    : int(route.cost),
                "service_s"  : int(route.service),
                "distance_m" : None,
                "path"       : None,
                "stops"      : self._stops(route),
                "kpis"       : self._route_kpis(route, vehicle, jobs, clock, plan_drive, crossings, ledger),
            })

        busy = {route["vehicle_id"] for route in routes}
        idle = [{"vehicle_id": int(vehicle.id), "kpis": self._idle_kpis(vehicle, clock, ledger)} for vehicle in vehicles if vehicle.id not in busy]

        unassigned = [
            {"job_id": int(job_id), "priority": int(jobs.by_id(job_id).priority), "kind": jobs.by_id(job_id).kind, "waiting_s": int(clock) - ledger.first_seen.get(job_id, int(clock))}
            for job_id in sorted(state.unassigned_ids) if jobs.contains(job_id)
        ]

        return {
            "routes"           : routes,
            "idle"             : idle,
            "idle_vehicle_ids" : [entry["vehicle_id"] for entry in idle],
            "unassigned"       : unassigned,
            "vehicles"         : {str(vehicle.id): {"start": [float(vehicle.start[0]), float(vehicle.start[1])], "onboard": int(vehicle.onboard), "progress_credit": int(vehicle.progress_credit)} for vehicle in vehicles},
            "kpis"             : self._plan_kpis(jobs, vehicles, state, clock, routes, total_crossings, ledger),
        }

    def add_geometry(self, plan, geometry):
        """Fill the road path, distance and detour of every route of a plan, in place.

        Args:
            plan     : Plan dict produced by `plan`.
            geometry : Callable taking a list of [lon, lat] points and returning {"path": [[lat, lon]], "distance_m": int}.

        Returns:
            The same plan dict.
        """
        total = 0

        for route in plan["routes"]:
            points   = [route["start"]] + [stop["location"] for stop in route["stops"]]
            road     = geometry(points)
            straight = self.straight_length([tuple(point) for point in points])

            route["path"]               = road["path"]
            route["distance_m"]         = int(road["distance_m"])
            route["kpis"]["distance_m"] = int(road["distance_m"])
            route["kpis"]["detour"]     = road["distance_m"] / straight if straight > 0 else None

            total += int(road["distance_m"])

        plan["kpis"]["distance_m"] = total
        return plan


class ActionReader:
    """Translates pool-index actions into id-based dicts."""

    def describe(self, environment, action):
        """Return {"operator", "operator_name", "vehicle_id", "job_id"} for an Action; ids are None for DO_NOTHING."""
        entry = {"operator": int(action.operator), "operator_name": OPERATOR_NAMES[action.operator], "vehicle_id": None, "job_id": None}

        if action.operator in (0, 1):
            entry["vehicle_id"] = int(environment.vehicles[action.vehicle_index].id)
            entry["job_id"]     = int(environment.jobs[action.job_index].id)

        return entry

    def mask(self, environment, mask_info):
        """Return the action mask with ids: {"insert_job_ids", "remove_vehicle_ids", "vehicle_to_job_ids"}."""
        jobs     = environment.jobs
        vehicles = environment.vehicles

        return {
            "insert_job_ids"     : [int(jobs[index].id) for index in mask_info["unassigned_job_indices"]],
            "remove_vehicle_ids" : [int(vehicles[index].id) for index in mask_info["vehicles_with_jobs_indices"]],
            "vehicle_to_job_ids" : {str(vehicles[int(vehicle_index)].id): [int(jobs[index].id) for index in job_indices] for vehicle_index, job_indices in mask_info["vehicle_to_job_indices"].items()},
        }


class PolicyReader:
    """Turns the raw output of `Policy.select_action` into masked probabilities over the three heads.

    Attributes:
        masker       : ActionMasker of the policy, used to rebuild the masked vehicle and job logits.
        distribution : ActionDistribution used for the normalised entropies.
        threshold    : Logit value below which an option counts as masked.
    """

    def __init__(self, config, masker):
        self.masker       = masker
        self.distribution = ActionDistribution(config, masker)
        self.threshold    = config.training.large_negative_value / 2

    def _probabilities(self, masked_logits):
        probabilities = torch.softmax(masked_logits.float(), dim=-1).tolist()
        allowed       = (masked_logits > self.threshold).tolist()
        return [probability if keep else None for probability, keep in zip(probabilities, allowed)]

    def _head_tables(self, output, mask_info, operator_masked):
        vehicle_logits = output["old_vehicle_logits"].cpu()
        job_logits     = output["old_job_logits"].cpu()
        num_vehicles   = vehicle_logits.size(1)
        num_jobs       = job_logits.size(2)

        vehicle_probs = []
        job_probs     = []

        for operator in (0, 1):
            if operator_masked[operator]:
                vehicle_probs.append([None] * num_vehicles)
                job_probs.append([[None] * num_jobs for _ in range(num_vehicles)])
                continue

            vehicle_row = self._probabilities(self.masker.mask_vehicle(vehicle_logits[operator], mask_info, operator))
            job_rows    = []

            for vehicle_index in range(num_vehicles):
                if vehicle_row[vehicle_index] is None or num_jobs == 0:
                    job_rows.append([None] * num_jobs)
                else:
                    job_rows.append(self._probabilities(self.masker.mask_job(job_logits[operator, vehicle_index], mask_info, operator, vehicle_index)))

            vehicle_probs.append(vehicle_row)
            job_probs.append(job_rows)

        return vehicle_probs, job_probs

    def _joint(self, operator_probs, vehicle_probs, job_probs):
        entries = [(operator_probs[2], 2, None, None)]

        for operator in (0, 1):
            for vehicle_index, vehicle_probability in enumerate(vehicle_probs[operator]):
                if vehicle_probability is None:
                    continue

                for job_index, job_probability in enumerate(job_probs[operator][vehicle_index]):
                    if job_probability is not None:
                        entries.append((operator_probs[operator] * vehicle_probability * job_probability, operator, vehicle_index, job_index))

        return entries

    def _argmax(self, row):
        best = None
        for index, value in enumerate(row):
            if value is not None and (best is None or value > row[best]):
                best = index
        return best

    def _greedy(self, operator_probs, vehicle_probs, job_probs):
        operator = self._argmax(operator_probs)
        if operator == 2:
            return (2, None, None)

        vehicle_index = self._argmax(vehicle_probs[operator])
        return (operator, vehicle_index, self._argmax(job_probs[operator][vehicle_index]))

    def _ids(self, choice, vehicle_ids, job_ids):
        operator, vehicle_index, job_index = choice
        return {
            "operator"   : int(operator),
            "vehicle_id" : None if vehicle_index is None else vehicle_ids[vehicle_index],
            "job_id"     : None if job_index is None else job_ids[job_index],
        }

    def _entropy(self, output, mask_info):
        entropy, _ = self.distribution.compute(
            old_operator_logits = output["old_operator_logits"].cpu(),
            old_vehicle_logits  = output["old_vehicle_logits"].cpu(),
            old_job_logits      = output["old_job_logits"].cpu(),
            new_operator_logits = output["old_operator_logits"].cpu(),
            new_vehicle_logits  = output["old_vehicle_logits"].cpu(),
            new_job_logits      = output["old_job_logits"].cpu(),
            mask_info           = mask_info,
        )

        return {"operator": float(entropy["operator_entropy"]), "vehicle": float(entropy["vehicle_entropy_exp"]), "job": float(entropy["job_entropy_exp"])}

    def read(self, output, mask_info, environment, action):
        """Build the policy block of a decision.

        Args:
            output      : Dict returned by `Policy.select_action` for this decision.
            mask_info   : Action mask the decision was taken under.
            environment : Environment whose pools give the vehicle and job ids of the logit axes.
            action      : Action that was taken.

        Returns:
            Dict with "value", "operator" ({"raw_probs", "probs", "masked"}), "vehicle_ids", "job_ids", "vehicle_probs" (2 x V, INSERT then REMOVE), "job_probs" (2 x V x J), "entropy", "greedy_action", "joint_argmax_action", "chosen_joint_prob" and "chosen_rank". Masked entries are None.
        """
        operator_logits = output["old_operator_logits"].cpu().float()
        masked_logits   = output["masked_operator_logits"].cpu().float()
        operator_probs  = torch.softmax(masked_logits, dim=-1).tolist()
        operator_masked = (masked_logits <= self.threshold).tolist()

        vehicle_ids = [int(vehicle.id) for vehicle in environment.vehicles]
        job_ids     = [int(job.id) for job in environment.jobs]

        vehicle_probs, job_probs = self._head_tables(output, mask_info, operator_masked)

        joint  = self._joint(operator_probs, vehicle_probs, job_probs)
        chosen = (action.operator, None, None) if action.operator == 2 else (action.operator, action.vehicle_index, action.job_index)

        chosen_probability = next(entry[0] for entry in joint if entry[1:] == chosen)
        best               = max(joint, key=lambda entry: entry[0])

        return {
            "value"               : float(output["state_value"]),
            "operator"            : {"raw_probs": torch.softmax(operator_logits, dim=-1).tolist(), "probs": operator_probs, "masked": operator_masked},
            "vehicle_ids"         : vehicle_ids,
            "job_ids"             : job_ids,
            "vehicle_probs"       : vehicle_probs,
            "job_probs"           : job_probs,
            "entropy"             : self._entropy(output, mask_info),
            "greedy_action"       : self._ids(self._greedy(operator_probs, vehicle_probs, job_probs), vehicle_ids, job_ids),
            "joint_argmax_action" : self._ids(best[1:], vehicle_ids, job_ids),
            "chosen_joint_prob"   : float(chosen_probability),
            "chosen_rank"         : 1 + sum(1 for entry in joint if entry[0] > chosen_probability),
        }


class FeatureReader:
    """Reads the model inputs of one observation with their names and raw values."""

    JOB_NAMES     = ["lon", "lat", "priority", "service_s", "setup_s", "is_unassigned", "has_vehicle", "is_repossession", "depot_distance_m"]
    VEHICLE_NAMES = ["lon", "lat", "speed_factor", "shift_left_s", "capacity", "onboard", "planned_jobs", "planned_load", "depot_distance_m"]
    CONTEXT_NAMES = ["jobs", "unassigned", "vehicles", "idle_vehicles", "drive_s", "clock_s", "round_index", "edits_remaining"]

    def _job_raw(self, node):
        metadata = node["metadata"]
        return [
            node["longitude"], node["latitude"], metadata["priority"], metadata["service"], metadata["setup"],
            int(metadata["is_unassigned"]), int(metadata["assigned_vehicle_id"] is not None), int(metadata["kind"] == "repossession"), metadata["depot_distance"],
        ]

    def _vehicle_raw(self, node, clock):
        metadata = node["metadata"]
        return [
            node["longitude"], node["latitude"], metadata["speed_factor"], max(metadata["time_window"][1] - clock, 0), metadata["capacity"],
            metadata["onboard"], metadata["planned_jobs"], metadata["onboard"] + metadata["planned_pickups"], metadata["depot_distance"],
        ]

    def read(self, environment, graph):
        """Return the feature block of the observation just built by `environment.observe`.

        Args:
            environment : Environment whose graph builder still holds the nodes of this observation.
            graph       : HeteroData returned by that observe call; job x has shape (J, 9), vehicle x (V, 9), context (1, 8).

        Returns:
            Dict with "context" ({"names", "raw", "normalised"}), "job" and "vehicle" ({"names", "rows": {id: {"raw", "normalised"}}}).
        """
        builder = environment.graph
        state   = builder.state

        job_nodes     = [builder.nodes[index] for index in builder.nodes_by_type["job"]]
        vehicle_nodes = [builder.nodes[index] for index in builder.nodes_by_type["vehicle"]]
        job_x         = graph["job"].x.cpu().tolist()
        vehicle_x     = graph["vehicle"].x.cpu().tolist()

        context_raw = [
            len(job_nodes), len(state.unassigned_ids), len(vehicle_nodes), max(len(vehicle_nodes) - state.num_routes, 0),
            int(state.cost), int(builder.clock), int(builder.round_index), int(builder.edits_remaining),
        ]

        return {
            "context" : {"names": self.CONTEXT_NAMES, "raw": context_raw, "normalised": graph.context.cpu().tolist()[0]},
            "job"     : {"names": self.JOB_NAMES, "rows": {str(node["metadata"]["job_id"]): {"raw": self._job_raw(node), "normalised": row} for node, row in zip(job_nodes, job_x)}},
            "vehicle" : {"names": self.VEHICLE_NAMES, "rows": {str(node["metadata"]["vehicle_id"]): {"raw": self._vehicle_raw(node, builder.clock), "normalised": row} for node, row in zip(vehicle_nodes, vehicle_x)}},
        }


class VerdictBuilder:
    """Scores a decision against everything the regret-insertion teacher considers at the same state.

    Attributes:
        config  : Config the episode runs under.
        metrics : PlanMetrics used for the break-even thresholds.
    """

    SCOPES            = ("insertions", "full")
    PROFITABLE_MARGIN = 0.05

    def __init__(self, config, metrics):
        self.config  = config
        self.metrics = metrics
        self.actions = ActionReader()

    def _insertions(self, environment, state, explanation):
        entries = []

        for job_id in sorted(explanation["insertions"]):
            for _, vehicle_id in sorted(explanation["insertions"][job_id], key=lambda entry: entry[1]):
                new_state  = environment.action_handler.apply_job_insertion(environment, state, vehicle_id, job_id)
                rewards, _ = environment.step(state, new_state, 0)

                entries.append({
                    "job_id"        : int(job_id),
                    "vehicle_id"    : int(vehicle_id),
                    "reward"        : float(sum(rewards.values())),
                    "terms"         : {term: float(value) for term, value in rewards.items()},
                    "added_drive_s" : int(new_state.cost) - int(state.cost),
                })

        return entries

    def _eligible(self, environment, state):
        assigned = state.assigned_job_ids
        return sorted(job_id for job_id in state.unassigned_ids if environment.jobs.contains(job_id) and job_id not in assigned)

    def _best_by_job(self, insertions):
        best = {}
        for entry in insertions:
            if entry["job_id"] not in best or entry["reward"] > best[entry["job_id"]]["reward"]:
                best[entry["job_id"]] = entry
        return best

    def _break_even(self, environment, state, eligible, best_by_job):
        thresholds = {}
        for job_id in eligible:
            best       = best_by_job.get(job_id)
            wakes_idle = best is not None and state.route_of_vehicle(best["vehicle_id"]) is None

            thresholds[str(job_id)] = self.metrics.break_even(environment.jobs.by_id(job_id).priority, 0, wakes_idle)

        return thresholds

    def _match(self, taken, teacher):
        same_operator = taken["operator"] == teacher["operator"]
        comparable    = same_operator and taken["operator"] != 2

        return {
            "operator" : same_operator,
            "vehicle"  : taken["vehicle_id"] == teacher["vehicle_id"] if comparable else None,
            "job"      : taken["job_id"] == teacher["job_id"] if comparable else None,
        }

    def build(self, environment, state, remaining, scope, taken_action, taken_rewards):
        """Compute the teacher verdict for one decision.

        Args:
            environment   : Environment holding the pools, depot and clock of the decision; it is not modified.
            state         : RoutingState the decision was taken on.
            remaining     : Edits left in the round including this one.
            scope         : "insertions" for the insertion table only, "full" to add the removal scan with continuation values.
            taken_action  : Id-based action dict of the action that was taken.
            taken_rewards : Six reward terms the taken action realised.

        Returns:
            Verdict dict: "scope", "baseline", "insertions", "infeasible", "best", "plan", "removals", "teacher_action", "match", "taken_reward", "one_step_regret", "profitable_left" and "break_even".
        """
        if scope not in self.SCOPES:
            raise ValueError(f"Unknown verdict scope {scope!r}; expected one of {self.SCOPES}")

        teacher     = RegretInsertionTeacher(self.config, allow_removal=scope == "full")
        explanation = teacher.explain(environment, state, remaining)
        baseline    = float(explanation["baseline"])

        insertions  = self._insertions(environment, state, explanation)
        eligible    = self._eligible(environment, state)
        feasible    = {(entry["job_id"], entry["vehicle_id"]) for entry in insertions}
        best_by_job = self._best_by_job(insertions)

        best = explanation["best_insertion"]
        if best is not None:
            best = {"job_id": int(best["job_id"]), "vehicle_id": int(best["vehicle_id"]), "reward": float(best["reward"]), "regret": None if math.isinf(best["regret"]) else float(best["regret"])}

        removals = None
        if explanation["removals"] is not None:
            removals = [{"vehicle_id": int(entry["vehicle_id"]), "job_id": int(entry["job_id"]), "reward": float(entry["reward"]), "continuation": float(entry["continuation"]), "value": float(entry["value"])} for entry in explanation["removals"]]

        teacher_action = self.actions.describe(environment, explanation["action"])
        plan_action    = self.actions.describe(environment, explanation["plan"]["action"] or Action(operator=2, vehicle_index=0, job_index=0))
        taken_reward   = float(sum(value for term, value in taken_rewards.items() if term != "holding_reward"))
        candidates     = [baseline] + [entry["reward"] for entry in insertions] + [entry["reward"] for entry in removals or []]
        profitable     = [entry["reward"] for entry in best_by_job.values() if entry["reward"] > baseline + self.PROFITABLE_MARGIN]

        return {
            "scope"           : scope,
            "baseline"        : baseline,
            "insertions"      : insertions,
            "infeasible"      : [[int(job_id), int(vehicle.id)] for job_id in eligible for vehicle in environment.vehicles if (job_id, vehicle.id) not in feasible],
            "best"            : best,
            "plan"            : {"value": float(explanation["plan"]["value"]), "action": plan_action},
            "removals"        : removals,
            "teacher_action"  : teacher_action,
            "match"           : self._match(taken_action, teacher_action),
            "taken_reward"    : taken_reward,
            "one_step_regret" : max(candidates) - taken_reward,
            "profitable_left" : {"count": len(profitable), "reward_sum": float(sum(profitable))},
            "break_even"      : self._break_even(environment, state, eligible, best_by_job),
        }


class ScenarioPlanner:
    """Builds the starting plan of a hand-made scenario, pinned to an assignment or solved freely."""

    def _pinned_state(self, jobs, vehicles, assignment, depot, clock):
        routes   = []
        assigned = set()

        for vehicle_key in sorted(assignment, key=int):
            vehicle_id = int(vehicle_key)
            if not vehicles.contains(vehicle_id):
                raise ValueError(f"assignment references unknown vehicle {vehicle_id}")

            job_ids = [int(job_id) for job_id in assignment[vehicle_key]]
            if not job_ids:
                continue

            unknown = [job_id for job_id in job_ids if not jobs.contains(job_id)]
            if unknown:
                raise ValueError(f"assignment references unknown jobs {unknown}")

            repeated = assigned.intersection(job_ids)
            if repeated:
                raise ValueError(f"assignment repeats jobs {sorted(repeated)}")

            solution = services.vroom.solve([jobs.by_id(job_id) for job_id in job_ids], [vehicles.by_id(vehicle_id)], depot=depot, clock=clock)
            if solution is None or not solution.routes:
                raise ValueError(f"VROOM returned no route for vehicle {vehicle_id}")
            if solution.unassigned_ids:
                raise ValueError(f"vehicle {vehicle_id} cannot fit jobs {sorted(solution.unassigned_ids)} before shift end")

            routes.append(solution.routes[0])
            assigned.update(job_ids)

        return RoutingState(routes=routes, unassigned_ids={job.id for job in jobs} - assigned)

    def pools(self, jobs, vehicles):
        """Parse job and vehicle dicts into EntityPools; raises ValueError when either list is empty."""
        if not jobs or not vehicles:
            raise ValueError("scenario needs at least one job and one vehicle")

        return EntityPool([Job.from_dict(job) for job in jobs]), EntityPool([Vehicle.from_dict(vehicle) for vehicle in vehicles])

    def clock(self, vehicles):
        """Return the scenario start clock: the earliest shift start, in seconds of day."""
        return min(vehicle.time_window[0] for vehicle in vehicles)

    def initial_state(self, jobs, vehicles, assignment, depot, clock):
        """Return the starting RoutingState of a scenario.

        Args:
            jobs       : EntityPool of jobs.
            vehicles   : EntityPool of vehicles.
            assignment : Mapping {vehicle_id: [job_id]} that pins jobs to vehicles, or None or empty to let VROOM solve freely.
            depot      : Depot as (lon, lat).
            clock      : Clock in seconds of day.

        Returns:
            The RoutingState.

        Raises:
            ValueError : The assignment is inconsistent or VROOM finds no solution.
        """
        if depot is None:
            raise ValueError("scenario needs a depot")

        if assignment:
            return self._pinned_state(jobs, vehicles, assignment, depot, clock)

        state = services.vroom.solve(list(jobs), list(vehicles), depot=depot, clock=clock)
        if state is None:
            raise ValueError("VROOM returned no solution for this scenario")

        return state


class TraceSummary:
    """Reduces the frames of a trace to episode-level behaviour counts."""

    def _decisions(self, trace):
        return [frame for frame in trace["frames"] if frame["kind"] == "decision"]

    def _plain_reward(self, frame):
        return frame["decision"]["reward"] - frame["decision"]["rewards"]["holding_reward"]

    def _removal_outcomes(self, decisions):
        relocations  = []
        abandonments = []

        for position, frame in enumerate(decisions):
            action = frame["decision"]["action"]
            if action["operator"] != 1:
                continue

            reinsertion = None
            for later in decisions[position + 1:]:
                if later["round"] != frame["round"]:
                    break
                if later["decision"]["action"]["operator"] == 0 and later["decision"]["action"]["job_id"] == action["job_id"] and not later["decision"]["wasted"]:
                    reinsertion = later
                    break

            if reinsertion is None:
                abandonments.append({"seq": frame["seq"], "job_id": action["job_id"]})
            else:
                relocations.append({"remove_seq": frame["seq"], "insert_seq": reinsertion["seq"], "job_id": action["job_id"], "net_reward": self._plain_reward(frame) + self._plain_reward(reinsertion)})

        return relocations, abandonments

    def holding_share(self, reward_by_term):
        """Return the absolute holding sum as a share of the summed absolute term sums, 0 when every term is 0."""
        total = sum(abs(value) for value in reward_by_term.values())
        return abs(reward_by_term["holding_reward"]) / total if total > 0 else 0.0

    def summarise(self, trace, stopped_reason):
        """Return the summary block of a trace.

        Args:
            trace          : Trace dict whose frames are complete.
            stopped_reason : "complete", "cancelled" or "failed".

        Returns:
            Dict with the episode return and its six-term split, operator counts, edits per round, unassigned jobs at each round end, wasted edits, relocation chains, abandoned removals, execution totals and the final plan size.
        """
        decisions = self._decisions(trace)
        rounds    = sorted({frame["round"] for frame in decisions})
        final     = trace["frames"][-1]["plan"]["kpis"] if trace["frames"] else None

        relocations, abandonments = self._removal_outcomes(decisions)

        return {
            "total_reward"         : sum(frame["decision"]["reward"] for frame in decisions),
            "reward_by_term"       : {term: sum(frame["decision"]["rewards"][term] for frame in decisions) for term in REWARD_TERMS},
            "decisions"            : len(decisions),
            "operator_counts"      : [sum(1 for frame in decisions if frame["decision"]["action"]["operator"] == operator) for operator in range(3)],
            "edits_per_round"      : [sum(1 for frame in decisions if frame["round"] == round_index) for round_index in rounds],
            "unassigned_at_commit" : [frame["plan"]["kpis"]["unassigned"] for frame in decisions if frame["decision"]["ends_round"]],
            "wasted_edits"         : sum(1 for frame in decisions if frame["decision"]["wasted"]),
            "relocations"          : relocations,
            "abandonments"         : abandonments,
            "served"               : final["served"] if final else 0,
            "failed"               : final["failed"] if final else 0,
            "dropped"              : final["dropped"] if final else 0,
            "final_unassigned"     : final["unassigned"] if final else 0,
            "final_drive_s"        : final["drive_s"] if final else 0,
            "stopped_reason"       : stopped_reason,
        }

    def digest(self, trace):
        """Return the per-agent entry of a diagnostics index for one trace.

        Args:
            trace : Finished trace dict.

        Returns:
            Dict with the summary fields the index keeps plus "holding_share", "operator_by_edit" (one [INSERT, REMOVE, DO_NOTHING] count per edit position), "commits_with_work_left", "profitable_left" (None when no decision carries a verdict) and "pickups" ({"served", "failed"} over repossessions).
        """
        summary   = trace["summary"]
        decisions = self._decisions(trace)
        by_edit   = [[0, 0, 0] for _ in range(trace["meta"]["caps"]["edits"])]

        for frame in decisions:
            by_edit[frame["edit"]][frame["decision"]["action"]["operator"]] += 1

        verdicts = [frame["decision"]["verdict"] for frame in decisions if frame["decision"]["ends_round"] and frame["decision"]["verdict"] is not None]
        opens    = [frame["round_open"]["executed"] for frame in trace["frames"] if frame["kind"] == "round_open"]

        return {
            "total_reward"           : summary["total_reward"],
            "reward_by_term"         : summary["reward_by_term"],
            "holding_share"          : self.holding_share(summary["reward_by_term"]),
            "decisions"              : summary["decisions"],
            "operator_counts"        : summary["operator_counts"],
            "operator_by_edit"       : by_edit,
            "edits_per_round"        : summary["edits_per_round"],
            "unassigned_at_commit"   : summary["unassigned_at_commit"],
            "commits_with_work_left" : sum(1 for frame in decisions if frame["decision"]["is_commit"] and frame["decision"]["edits_remaining"] > 1 and frame["plan"]["kpis"]["unassigned"] > 0),
            "wasted_edits"           : summary["wasted_edits"],
            "relocations"            : len(summary["relocations"]),
            "abandonments"           : len(summary["abandonments"]),
            "profitable_left"        : sum(verdict["profitable_left"]["count"] for verdict in verdicts) if verdicts else None,
            "served"                 : summary["served"],
            "failed"                 : summary["failed"],
            "dropped"                : summary["dropped"],
            "final_unassigned"       : summary["final_unassigned"],
            "final_drive_s"          : summary["final_drive_s"],
            "pickups"                : {
                "served" : sum(1 for executed in opens for entry in executed["served"] if entry["kind"] == "repossession"),
                "failed" : sum(len(executed["failed"]) for executed in opens),
            },
        }


class ExpectationCheck:
    """Checks a scenario template's declared expectation on the plan an agent leaves at the end of round 0."""

    KINDS = ("job_on_vehicle", "job_assigned", "job_unassigned", "job_not_on_vehicle", "max_edits")

    def _holds(self, check, vehicle_by_job, unassigned, edits):
        kind = check["kind"]

        if kind == "job_on_vehicle":
            return vehicle_by_job.get(check["job_id"]) == check["vehicle_id"]
        if kind == "job_assigned":
            return check["job_id"] in vehicle_by_job
        if kind == "job_unassigned":
            return check["job_id"] in unassigned
        if kind == "job_not_on_vehicle":
            return vehicle_by_job.get(check["job_id"]) != check["vehicle_id"]
        if kind == "max_edits":
            return edits <= check["count"]

        raise ValueError(f"Unknown expectation kind {kind!r}; expected one of {self.KINDS}")

    def evaluate(self, checks, frames):
        """Evaluate expectation checks against a trace.

        Args:
            checks : List of {"kind", "job_id", "vehicle_id", "count"} declared by the template.
            frames : Frames of the trace; the plan after the last decision of round 0 is the one judged.

        Returns:
            {"status": "PASS" or "FAIL", "checks": [{"kind", "job_id", "vehicle_id", "count", "ok"}]}, or None when round 0 holds no decision.
        """
        opening = [frame for frame in frames if frame["kind"] == "decision" and frame["round"] == 0]
        if not opening:
            return None

        plan           = opening[-1]["plan"]
        vehicle_by_job = {stop["job_id"]: route["vehicle_id"] for route in plan["routes"] for stop in route["stops"] if stop["kind"] in ("job", "pickup")}
        unassigned     = {entry["job_id"] for entry in plan["unassigned"]}
        edits          = sum(1 for frame in opening if frame["decision"]["action"]["operator"] != 2)

        results = [{**check, "ok": self._holds(check, vehicle_by_job, unassigned, edits)} for check in checks]
        return {"status": "PASS" if all(result["ok"] for result in results) else "FAIL", "checks": results}


class EpisodeRecorder:
    """Runs one agent through one episode of the trained MDP and records a trace of every frame.

    Attributes:
        config      : Config of the episode (reward weights, environment, caps).
        environment : Environment the episode runs on; it is reseeded or reloaded for every recording.
        trace       : Trace dict of the recording in progress or just finished.
    """

    def __init__(self, config, environment, logger=None, guard=None, step_guard=None):
        self.config      = config
        self.environment = environment
        self.logger      = logger or NullLogger()
        self.guard       = guard or nullcontext()
        self.step_guard  = step_guard or nullcontext()

        self.metrics  = PlanMetrics(config.reward)
        self.actions  = ActionReader()
        self.features = FeatureReader()
        self.planner  = ScenarioPlanner()
        self.verdicts = VerdictBuilder(config, self.metrics)

        self.trace          = None
        self.ledger         = None
        self.last_decision  = None
        self.policy_readers = {}

    def _begin(self, meta, source):
        public_source = {key: value for key, value in source.items() if key not in ("jobs", "vehicles", "assignment", "expectation")}

        self.ledger        = EpisodeLedger()
        self.last_decision = None

        with self.guard:
            self.trace = {
                "meta" : {
                    **meta,
                    "seed"        : int(source["seed"]),
                    "source"      : public_source,
                    "reward"      : {spec.name: getattr(self.config.reward, spec.name) for spec in fields(self.config.reward)},
                    "environment" : {spec.name: getattr(self.config.env, spec.name) for spec in fields(self.config.env)},
                    "caps"        : {"rounds": self.config.training.max_disruptions_per_episode, "edits": self.config.training.max_edits_per_round},
                    "gamma"       : self.config.ppo.gamma,
                    "complete"    : False,
                },
                "depot"       : None,
                "entities"    : {"jobs": {}, "vehicles": {}},
                "frames"      : [],
                "snapshots"   : {},
                "summary"     : None,
                "expectation" : None,
            }

    def _open_scenario(self, source):
        environment = self.environment
        seed        = int(source["seed"])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        environment.seed_streams(seed)

        jobs, vehicles = self.planner.pools(source["jobs"], source["vehicles"])
        depot          = (float(source["depot"][0]), float(source["depot"][1]))
        clock          = self.planner.clock(vehicles)
        state          = self.planner.initial_state(jobs, vehicles, source.get("assignment"), depot, clock)

        environment.jobs          = jobs
        environment.vehicles      = vehicles
        environment.depot         = depot
        environment.clock         = clock
        environment.current_state = state
        environment.initial_state = state.copy()

        services.vroom.clear_cache()
        environment.open_episode(disrupt=bool(source["initial_disruption"]))

    def _open(self, source):
        if source["kind"] == "scenario":
            self._open_scenario(source)
        else:
            self.environment.sample_episode(int(source["seed"]))

        with self.guard:
            self.trace["depot"] = [float(self.environment.depot[0]), float(self.environment.depot[1])]

    def _see_entities(self):
        environment = self.environment
        self.ledger.see_jobs(environment.jobs, environment.clock)

        entities = self.trace["entities"]
        for job in environment.jobs:
            if str(job.id) not in entities["jobs"]:
                entities["jobs"][str(job.id)] = {**job.to_dict(), "first_seen_clock": self.ledger.first_seen[job.id]}

        for vehicle in environment.vehicles:
            if str(vehicle.id) not in entities["vehicles"]:
                entry = vehicle.to_dict()
                entry.pop("progress_credit")
                entities["vehicles"][str(vehicle.id)] = entry

    def _append(self, kind, round_index, edit_index, round_open, decision):
        environment = self.environment

        with self.guard:
            self._see_entities()

            frame = {
                "seq"        : len(self.trace["frames"]),
                "kind"       : kind,
                "round"      : int(round_index),
                "edit"       : None if edit_index is None else int(edit_index),
                "clock"      : int(environment.clock),
                "round_open" : round_open,
                "decision"   : decision,
                "plan"       : self.metrics.plan(environment.jobs, environment.vehicles, environment.current_state, environment.clock, self.ledger),
            }
            self.trace["frames"].append(frame)

        self.ledger.remember_drive(environment.current_state)
        return frame

    def _executed(self, record):
        served = [{"job_id": int(job_id), **record.plan_before[job_id]} for job_id in record.executed["served"]]
        failed = [{"job_id": int(job_id), **record.plan_before[job_id]} for job_id in record.executed["failed"]]

        picked = {}
        for entry in served:
            if entry["kind"] == "repossession":
                picked[entry["vehicle_id"]] = picked.get(entry["vehicle_id"], 0) + 1

        dropped = []
        for vehicle_id, before in sorted(record.onboard_before.items()):
            count = before + picked.get(vehicle_id, 0) - record.onboard_advanced.get(vehicle_id, before + picked.get(vehicle_id, 0))
            if count > 0:
                dropped.append({"vehicle_id": int(vehicle_id), "count": int(count)})

        return {"served": served, "failed": failed, "dropped": dropped}

    def _event(self, record):
        if record.event is None:
            return None

        advanced  = record.pools["advanced"]
        disrupted = record.pools["disrupted"]
        removed   = advanced["vehicles"] - disrupted["vehicles"]

        return {
            "type"                : record.event["type"],
            "count"               : int(record.event["count"]),
            "added_job_ids"       : sorted(disrupted["jobs"] - advanced["jobs"]),
            "removed_job_ids"     : sorted(advanced["jobs"] - disrupted["jobs"]),
            "added_vehicle_ids"   : sorted(disrupted["vehicles"] - advanced["vehicles"]),
            "removed_vehicle_ids" : sorted(removed),
            "orphaned_job_ids"    : sorted(job_id for job_id, vehicle_id in record.plan_advanced.items() if vehicle_id in removed and job_id in disrupted["jobs"]),
        }

    def _round_open_frame(self, record):
        executed = self._executed(record)
        self.ledger.execute(executed["served"], executed["failed"], executed["dropped"])

        round_open = {
            "gap_s"           : int(record.gap),
            "holding_charged" : -float(record.holding),
            "holding_for_seq" : self.last_decision if record.round_index > 0 else None,
            "executed"        : executed,
            "event"           : self._event(record),
        }

        self._append("round_open", record.round_index, None, round_open, None)
        self.ledger.last_executed = {}

    def _policy_block(self, agent, mask_info, action):
        output = getattr(agent, "last_output", None)
        if output is None:
            return None

        masker = agent.policy.masker
        if id(masker) not in self.policy_readers:
            self.policy_readers[id(masker)] = PolicyReader(self.config, masker)

        return self.policy_readers[id(masker)].read(output, mask_info, self.environment, action)

    def _snapshot(self, step):
        environment = self.environment
        return {
            "jobs"            : [job.to_dict() for job in environment.jobs],
            "vehicles"        : [vehicle.to_dict() for vehicle in environment.vehicles],
            "depot"           : [float(environment.depot[0]), float(environment.depot[1])],
            "clock"           : int(environment.clock),
            "state"           : environment.current_state.to_payload(),
            "round"           : int(step.round_index),
            "edits_remaining" : int(step.remaining),
        }

    def _break_even(self, action_entry, old_state, costs):
        if action_entry["operator"] != 0:
            return None

        priority   = self.environment.jobs.by_id(action_entry["job_id"]).priority
        wakes_idle = old_state.route_of_vehicle(action_entry["vehicle_id"]) is None
        return self.metrics.break_even(priority, costs["disruption"], wakes_idle)

    def _decision_frame(self, step, agent, verdict_scope):
        environment = self.environment
        features    = self.features.read(environment, step.graph)
        mask        = self.actions.mask(environment, step.mask_info)
        snapshot    = self._snapshot(step)

        action       = agent.act(environment, step.graph, step.mask_info, step.remaining)
        action_entry = self.actions.describe(environment, action)
        policy       = self._policy_block(agent, step.mask_info, action)

        old_state, new_state, rewards, costs = step.commit(action)

        verdict = None
        if verdict_scope != "none":
            verdict = self.verdicts.build(environment, old_state, step.remaining, verdict_scope, action_entry, rewards)

        self.ledger.reward(rewards)

        decision = {
            "action"                  : action_entry,
            "edits_remaining"         : int(step.remaining),
            "is_commit"               : action.operator == 2,
            "ends_round"              : action.operator == 2 or step.remaining == 1,
            "wasted"                  : action.operator == 0 and new_state is old_state,
            "rewards"                 : {term: float(rewards[term]) for term in REWARD_TERMS},
            "costs"                   : {key: float(value) for key, value in costs.items()},
            "reward"                  : float(sum(rewards.values())),
            "cumulative"              : float(sum(self.ledger.returns.values())),
            "cumulative_terms"        : dict(self.ledger.returns),
            "return_to_go"            : None,
            "return_to_go_discounted" : None,
            "added_drive_s"           : int(new_state.cost) - int(old_state.cost),
            "break_even_s"            : self._break_even(action_entry, old_state, costs),
            "mask"                    : mask,
            "policy"                  : policy,
            "features"                : features,
            "verdict"                 : verdict,
        }

        frame = self._append("decision", step.round_index, step.edit_index, None, decision)

        with self.guard:
            self.trace["snapshots"][str(frame["seq"])] = snapshot

        self.last_decision = frame["seq"]

    def _fill_returns(self):
        gamma      = self.config.ppo.gamma
        plain      = 0.0
        discounted = 0.0

        for frame in reversed(self.trace["frames"]):
            if frame["kind"] != "decision":
                continue

            plain      = frame["decision"]["reward"] + plain
            discounted = frame["decision"]["reward"] + gamma * discounted

            frame["decision"]["return_to_go"]            = plain
            frame["decision"]["return_to_go_discounted"] = discounted

    def _finish(self, stopped_reason, source):
        with self.guard:
            self._fill_returns()
            self.trace["summary"] = TraceSummary().summarise(self.trace, stopped_reason)

            if source.get("expectation"):
                self.trace["expectation"] = ExpectationCheck().evaluate(source["expectation"], self.trace["frames"])

            self.trace["meta"]["complete"] = True

    def record(self, agent, meta, source, should_stop=None, verdict_scope="none", on_frame=None):
        """Record one agent on one episode.

        Args:
            agent         : Object with `act(environment, graph, mask_info, remaining_steps)`; a ModelAgent also contributes its policy output.
            meta          : Dict with "agent_key", "agent", "run_name" and "greedy" describing the agent.
            source        : Episode source. {"kind": "seed" or "evaluation", "seed"} samples the episode from the seed; {"kind": "scenario", "jobs", "vehicles", "depot", "assignment", "seed", "initial_disruption", "expectation"} loads a hand-made one.
            should_stop   : Callable checked before every decision; a true value ends the recording as cancelled.
            verdict_scope : "none", "insertions" or "full": compute the teacher verdict of every decision while recording.
            on_frame      : Callable invoked with each appended frame, after the step guard is released.

        Returns:
            The trace dict, including the on-disk "snapshots" block.
        """
        self._begin(meta, source)

        with self.step_guard:
            self._open(source)

        driver   = EpisodeDriver(self.environment, self.config)
        steps    = driver.rounds()
        opened   = 0
        reason   = "complete"
        finished = False

        while not finished:
            emitted = len(self.trace["frames"])

            with self.step_guard:
                if should_stop is not None and should_stop():
                    reason   = "cancelled"
                    finished = True
                else:
                    step = next(steps, None)

                    if step is None:
                        finished = True
                    else:
                        while opened < len(driver.records):
                            self._round_open_frame(driver.records[opened])
                            opened += 1

                        self._decision_frame(step, agent, verdict_scope)

            if on_frame is not None:
                for frame in self.trace["frames"][emitted:]:
                    on_frame(frame)

        self._finish(reason, source)
        return self.trace


class TraceFile:
    """Reads and writes traces as gzip-compressed JSON."""

    def write(self, path, trace):
        """Write a trace dict to `path` atomically; non-finite floats are not expected and raise."""
        path      = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(trace, handle, allow_nan=False)

        os.replace(temporary, path)

    def read(self, path):
        """Return the trace dict stored at `path`."""
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
