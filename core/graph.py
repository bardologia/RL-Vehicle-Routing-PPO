import math
import torch
from torch_geometric.data import HeteroData # type: ignore
from collections import defaultdict
from tools.auxiliary import haversine_distance


class NodeBuilder:
    def __init__(self, graph):
        self.graph = graph

    def add_node(self, identifier, node_type, longitude, latitude, metadata):
        idx = len(self.graph.nodes)

        self.graph.nodes.append(
            {
                "index": idx,
                "identifier": identifier,
                "node_type": node_type,
                "longitude": float(longitude),
                "latitude": float(latitude),
                "metadata": metadata,
            }
        )

        self.graph.node_index_by_key[(node_type, identifier)] = idx
        self.graph.nodes_by_type[node_type].append(idx)

    def vehicle_nodes(self):
        for vehicle in self.graph.vehicles:
            metadata = {
                "vehicle_id": vehicle.id,
                "time_window": [int(vehicle.time_window[0]), int(vehicle.time_window[1])],
                "speed_factor": float(vehicle.speed_factor),
                "return_to_depot": bool(vehicle.return_to_depot),
            }

            self.add_node(f"veh:{vehicle.id}", "vehicle", float(vehicle.start[0]), float(vehicle.start[1]), metadata)

    def job_nodes(self):
        state = self.graph.state

        assigned_vehicle_by_job = {}
        for route in state.routes:
            for job_id in route.job_ids:
                assigned_vehicle_by_job[job_id] = route.vehicle_id

        for job in self.graph.jobs:
            metadata = {
                "job_id": job.id,
                "priority": int(job.priority),
                "service": int(job.service),
                "setup": int(job.setup),
                "is_unassigned": job.id in state.unassigned_ids,
                "in_route": job.id in assigned_vehicle_by_job,
                "assigned_vehicle_id": assigned_vehicle_by_job.get(job.id),
            }

            self.add_node(f"job:{job.id}", "job", float(job.location[0]), float(job.location[1]), metadata)

    def build(self):
        self.vehicle_nodes()
        self.job_nodes()


class EdgeBuilder:
    def __init__(self, graph):
        self.graph = graph

    def job_sequence(self, routes):
        for route in routes:
            for current_stop, next_stop in zip(route.stops, route.stops[1:]):
                current_index = self.graph.node_index_by_key.get(("job", f"job:{current_stop.job_id}"))
                next_index    = self.graph.node_index_by_key.get(("job", f"job:{next_stop.job_id}"))
                if current_index is None or next_index is None:
                    continue

                dist_m, dur_s = haversine_distance(current_stop.location[0], current_stop.location[1], next_stop.location[0], next_stop.location[1])
                self.add_edge(current_index, next_index, dist_m, dur_s, etype=0, is_same_route=1.0, is_assigned=1.0, is_bidirectional=True)

    def vehicle_assigned(self, routes):
        for route in routes:
            vehicle_index = self.graph.node_index_by_key.get(("vehicle", f"veh:{route.vehicle_id}"))
            if vehicle_index is None:
                continue

            vehicle_node = self.graph.nodes[vehicle_index]

            for stop in route.stops:
                job_index = self.graph.node_index_by_key.get(("job", f"job:{stop.job_id}"))
                if job_index is None:
                    continue

                dist_m, dur_s = haversine_distance(vehicle_node["longitude"], vehicle_node["latitude"], stop.location[0], stop.location[1])
                self.add_edge(vehicle_index, job_index, dist_m, dur_s, etype=1, is_same_route=1.0, is_assigned=1.0, is_bidirectional=True)

    def job_vehicle_proximity(self, job_nodes, vehicle_nodes):
        for job_node in job_nodes:
            assigned_vehicle_id = job_node["metadata"].get("assigned_vehicle_id")

            for vehicle_node in vehicle_nodes:
                dist_m, dur_s = haversine_distance(job_node["longitude"], job_node["latitude"], vehicle_node["longitude"], vehicle_node["latitude"])

                is_assigned_flag = float(
                    assigned_vehicle_id is not None
                    and int(vehicle_node["metadata"]["vehicle_id"]) == int(assigned_vehicle_id)
                )

                self.add_edge(job_node["index"], vehicle_node["index"], dist_m, dur_s, etype=2, is_same_route=0.0, is_assigned=is_assigned_flag, is_bidirectional=True)

    def add_edge(self, src_idx, dst_idx, distance_meters, duration_seconds, *, etype, is_same_route, is_assigned, is_bidirectional=False):
        distance_km = max(float(distance_meters) / 1000.0, 0.0)
        duration_h  = max(float(duration_seconds) / 3600.0, 0.0)

        if is_bidirectional:
            pairs = [(src_idx, dst_idx), (dst_idx, src_idx)]
        else:
            pairs = [(src_idx, dst_idx)]

        for source, target in pairs:
            self.graph.edge_sources.append(int(source))
            self.graph.edge_targets.append(int(target))
            self.graph.edge_attributes.append(
                [
                    float(math.log1p(distance_km)),
                    float(math.log1p(duration_h)),
                    float(is_same_route),
                    float(is_assigned),
                ]
            )
            self.graph.edge_types.append(int(etype))

    def build(self):
        routes = self.graph.state.routes

        job_nodes     = [self.graph.nodes[idx] for idx in self.graph.nodes_by_type["job"]]
        vehicle_nodes = [self.graph.nodes[idx] for idx in self.graph.nodes_by_type["vehicle"]]

        self.job_sequence(routes)
        self.vehicle_assigned(routes)
        self.job_vehicle_proximity(job_nodes, vehicle_nodes)


class Graph:
    edge_names = {
        0: "job_sequence",
        1: "vehicle_assigned",
        2: "job_vehicle_proximity",
    }

    def __init__(self, config):
        self.jobs     = []
        self.vehicles = []
        self.state    = None

        self.nodes             = []
        self.node_index_by_key = {}
        self.nodes_by_type     = {"job": [], "vehicle": []}

        self.edge_sources    = []
        self.edge_targets    = []
        self.edge_attributes = []
        self.edge_types      = []

        self.node_builder  = NodeBuilder(self)
        self.edge_builder  = EdgeBuilder(self)
        self.graph_handler = GraphHandler(config)

    def _reset(self):
        self.nodes.clear()
        self.node_index_by_key.clear()
        self.nodes_by_type = {"job": [], "vehicle": []}
        self.edge_sources.clear()
        self.edge_targets.clear()
        self.edge_attributes.clear()
        self.edge_types.clear()

    def _coordinate_stats(self):
        longitudes = [node["longitude"] for node in self.nodes]
        latitudes  = [node["latitude"]  for node in self.nodes]

        count    = max(len(self.nodes), 1)
        lon_mean = sum(longitudes) / count
        lat_mean = sum(latitudes) / count
        lon_std  = (sum((lon - lon_mean) ** 2 for lon in longitudes) / count) ** 0.5 + 1e-6
        lat_std  = (sum((lat - lat_mean) ** 2 for lat in latitudes) / count) ** 0.5 + 1e-6

        return lon_mean, lon_std, lat_mean, lat_std

    def _job_features(self, lon_mean, lon_std, lat_mean, lat_std, global_to_local):
        job_features = []
        for job_local_idx, job_global_idx in enumerate(self.nodes_by_type["job"]):
            node     = self.nodes[job_global_idx]
            metadata = node["metadata"]
            job_features.append(
                [
                    (node["longitude"] - lon_mean) / lon_std,
                    (node["latitude"] - lat_mean) / lat_std,
                    float(metadata["priority"]) / 100.0,
                    float(metadata["service"]) / 3600.0,
                    float(metadata["setup"]) / 3600.0,
                    float(metadata["is_unassigned"]),
                    0.0 if metadata["assigned_vehicle_id"] is None else 1.0,
                ]
            )
            global_to_local[job_global_idx] = ("job", job_local_idx)

        return job_features

    def _vehicle_features(self, lon_mean, lon_std, lat_mean, lat_std, global_to_local):
        vehicle_features = []
        for vehicle_local_idx, vehicle_global_idx in enumerate(self.nodes_by_type["vehicle"]):
            node        = self.nodes[vehicle_global_idx]
            metadata    = node["metadata"]
            time_window = metadata["time_window"]
            vehicle_features.append(
                [
                    (node["longitude"] - lon_mean) / lon_std,
                    (node["latitude"] - lat_mean) / lat_std,
                    float(metadata["speed_factor"]),
                    float((time_window[1] - time_window[0]) / 3600.0),
                    float(metadata["return_to_depot"]),
                ]
            )
            global_to_local[vehicle_global_idx] = ("vehicle", vehicle_local_idx)

        return vehicle_features

    def _edge_buffers(self, global_to_local):
        edge_buffers = defaultdict(lambda: {"src": [], "dst": [], "attr": []})
        for src_global, dst_global, edge_attr, edge_type in zip(
            self.edge_sources,
            self.edge_targets,
            self.edge_attributes,
            self.edge_types,
        ):
            src_type, src_local = global_to_local[src_global]
            dst_type, dst_local = global_to_local[dst_global]
            relation_name = self.edge_names[int(edge_type)]

            key = (src_type, relation_name, dst_type)
            edge_buffers[key]["src"].append(src_local)
            edge_buffers[key]["dst"].append(dst_local)
            edge_buffers[key]["attr"].append(edge_attr)

        return edge_buffers

    def pack_data(self):
        data = HeteroData()

        lon_mean, lon_std, lat_mean, lat_std = self._coordinate_stats()
        global_to_local = {}

        job_features     = self._job_features(lon_mean, lon_std, lat_mean, lat_std, global_to_local)
        vehicle_features = self._vehicle_features(lon_mean, lon_std, lat_mean, lat_std, global_to_local)

        data["job"].x     = torch.tensor(job_features, dtype=torch.float32) if job_features else torch.empty((0, 7), dtype=torch.float32)
        data["vehicle"].x = torch.tensor(vehicle_features, dtype=torch.float32) if vehicle_features else torch.empty((0, 5), dtype=torch.float32)

        for (src_type, relation_name, dst_type), buffer in self._edge_buffers(global_to_local).items():
            data[(src_type, relation_name, dst_type)].edge_index = torch.tensor([buffer["src"], buffer["dst"]], dtype=torch.long)
            data[(src_type, relation_name, dst_type)].edge_attr  = torch.tensor(buffer["attr"], dtype=torch.float32)

        return data

    def mappings(self):
        return {
            "index_to_node": [
                {
                    "index": n["index"],
                    "identifier": n["identifier"],
                    "node_type": n["node_type"],
                    "longitude": n["longitude"],
                    "latitude": n["latitude"],
                    **n["metadata"],
                }
                for n in self.nodes
            ],
            "edge_type_ids": dict(self.edge_names),
        }

    def build(self, jobs, vehicles, state):
        self._reset()
        self.jobs     = jobs
        self.vehicles = vehicles
        self.state    = state

        self.node_builder.build()
        self.edge_builder.build()

        data = self.graph_handler.build(self.pack_data())
        data.mappings = self.mappings()

        return data


class GraphHandler:
    required_relations = [
        ("job", "job_sequence", "job"),
        ("vehicle", "vehicle_assigned", "job"),
        ("job", "vehicle_assigned", "vehicle"),
        ("job", "job_vehicle_proximity", "vehicle"),
        ("vehicle", "job_vehicle_proximity", "job"),
    ]

    def __init__(self, config):
        self.config = config

    def ensure_relations(self, data: HeteroData) -> None:
        for relation in self.required_relations:
            store = data[relation]
            if "edge_index" not in store:
                store.edge_index = torch.empty((2, 0), dtype=torch.long)
            if "edge_attr" not in store:
                store.edge_attr = torch.empty((0, 4), dtype=torch.float32)

    def build(self, data: HeteroData) -> HeteroData:
        self.ensure_relations(data)
        return data
