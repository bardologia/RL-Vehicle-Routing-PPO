import numpy as np
import torch
from torch_geometric.data import HeteroData # type: ignore
from scipy.spatial import cKDTree
from tools.auxiliary import haversine_distance
from core.services import osrm
from collections import defaultdict


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

    def adaptive_sample(self, path_coords, min_angle_deg=15.0, max_points=None):
        sampled = [path_coords[0]]
        
        min_angle_rad = np.deg2rad(min_angle_deg)
        
        for i in range(1, len(path_coords) - 1):
            p1 = np.array(path_coords[i - 1], dtype=np.float64)
            p2 = np.array(path_coords[i], dtype=np.float64)
            p3 = np.array(path_coords[i + 1], dtype=np.float64)
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 < 1e-8 or norm2 < 1e-8:
                continue
            
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            
            if angle >= min_angle_rad:
                sampled.append(path_coords[i])
  
        sampled.append(path_coords[-1])
        
        if max_points and len(sampled) > max_points:
            step = len(sampled) / max_points
            indices = [int(i * step) for i in range(max_points - 1)]
            indices.append(len(sampled) - 1) 
            sampled = [sampled[i] for i in indices]
        
        return sampled

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
            self.graph.vehicles_with_routes.add(route.vehicle_id)
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

    def path_nodes(self):
        for route_index, route in enumerate(self.graph.state.routes):
            vehicle_id = route.vehicle_id
            path_coordinates = route.path_coords or []
            if not path_coordinates:
                continue
            
            max_path_nodes = max(10, len(path_coordinates) // 40) 
            sampled = self.adaptive_sample(
                path_coordinates, 
                min_angle_deg=10.0, 
                max_points=max_path_nodes
            )
            
            for path_sequence, (lat, lon) in enumerate(sampled):
                metadata = {
                    "route_index": route_index,
                    "vehicle_id": vehicle_id,
                    "path_sequence": path_sequence,
                }
                
                self.add_node(
                    f"path:{vehicle_id}:{route_index}:{path_sequence}",
                    "path",
                    float(lon),
                    float(lat),
                    metadata,
                )

    def build(self):
        self.vehicle_nodes()
        self.job_nodes()
        self.path_nodes()


class EdgeBuilder:
    def __init__(self, graph):
        self.graph = graph

    def _query_kdtree(self, tree, point, k):
        distances, indices = tree.query(point, k=k)
        if k == 1:
            distances = np.array([distances])
            indices = np.array([indices])
        return distances, indices

    def job_sequence(self, routes):
        for route in routes:
            for current_stop, next_stop in zip(route.stops, route.stops[1:]):
                current_index = self.graph.node_index_by_key.get(("job", f"job:{current_stop.job_id}"))
                next_index    = self.graph.node_index_by_key.get(("job", f"job:{next_stop.job_id}"))
                if current_index is None or next_index is None:
                    continue

                dist_m, dur_s = osrm.distance_duration(current_stop.location[0], current_stop.location[1], next_stop.location[0], next_stop.location[1])
                self.add_edge(current_index, next_index, dist_m, dur_s, etype=0, is_same_route=1.0, is_assigned=1.0, is_bidirectional=True)

    def get_sorted_path_nodes(self):
        return sorted(
            (self.graph.nodes[idx] for idx in self.graph.nodes_by_type["path"]),
            key=lambda n: (n["metadata"]["vehicle_id"], n["metadata"]["route_index"], n["metadata"]["path_sequence"]),
        )

    def path_sequence(self, path_nodes_sorted):
        for current_node, next_node in zip(path_nodes_sorted, path_nodes_sorted[1:]):
            current_metadata = current_node["metadata"]
            next_metadata = next_node["metadata"]
            
            if (current_metadata["vehicle_id"], current_metadata["route_index"]) != (next_metadata["vehicle_id"], next_metadata["route_index"]):
                continue
        
            dist_m, dur_s = haversine_distance(current_node["longitude"], current_node["latitude"], next_node["longitude"], next_node["latitude"])
            self.add_edge(current_node["index"], next_node["index"], dist_m, dur_s, etype=1, is_same_route=1.0, is_assigned=0.0, is_bidirectional=True)

    def vehicle_assigned(self, routes):
        for route in routes:
            vehicle_index = self.graph.node_index_by_key.get(("vehicle", f"veh:{route.vehicle_id}"))
            if vehicle_index is None:
                continue

            vehicle_node = self.graph.nodes[vehicle_index]
            vehicle_lon, vehicle_lat = vehicle_node["longitude"], vehicle_node["latitude"]

            for stop in route.stops:
                job_index = self.graph.node_index_by_key.get(("job", f"job:{stop.job_id}"))
                if job_index is None:
                    continue

                dist_m, dur_s = osrm.distance_duration(vehicle_lon, vehicle_lat, stop.location[0], stop.location[1])
                self.add_edge(vehicle_index, job_index, dist_m, dur_s, etype=2, is_same_route=1.0, is_assigned=1.0, is_bidirectional=True)

    def job_path_proximity(self, routes, job_nodes, path_nodes):
        if not self.graph.k_near_paths_for_job or not job_nodes or not path_nodes:
            return

        job_to_route = {}
        for route_index, route in enumerate(routes):
            for job_id in route.job_ids:
                job_to_route[job_id] = (route_index, route.vehicle_id)

        path_coords = np.array([[p["longitude"], p["latitude"]] for p in path_nodes], dtype=np.float64)
        tree = cKDTree(path_coords)

        for job_node in job_nodes:
            job_id = int(job_node["metadata"]["job_id"])
            job_lon, job_lat = job_node["longitude"], job_node["latitude"]

            k = min(self.graph.k_near_paths_for_job, len(path_nodes))
            distances, indices = self._query_kdtree(tree, [job_lon, job_lat], k)

            for _, path_index in zip(distances, indices):
                path_node = path_nodes[int(path_index)]

                if job_id in job_to_route:
                    route_index, vehicle_id_for_route = job_to_route[job_id]
                    if route_index == path_node["metadata"]["route_index"] and vehicle_id_for_route == path_node["metadata"]["vehicle_id"]:
                        continue

                dist_m, dur_s = osrm.distance_duration(job_lon, job_lat, path_node["longitude"], path_node["latitude"])
                same_route_flag = 0.0
                is_assigned_flag = float(job_node["metadata"]["assigned_vehicle_id"] is not None)
                self.add_edge(job_node["index"], path_node["index"], dist_m, dur_s, etype=3, is_same_route=same_route_flag, is_assigned=is_assigned_flag, is_bidirectional=True)

    def job_vehicle_proximity(self, job_nodes, vehicle_nodes):
        if not self.graph.k_near_vehicles_for_job or not job_nodes or not vehicle_nodes:
            return

        vehicle_coords = np.array([[v["longitude"], v["latitude"]] for v in vehicle_nodes], dtype=np.float64)
        tree = cKDTree(vehicle_coords)

        for job_node in job_nodes:
            job_lon, job_lat = job_node["longitude"], job_node["latitude"]
            k = min(self.graph.k_near_vehicles_for_job, len(vehicle_nodes))
            distances, indices = self._query_kdtree(tree, [job_lon, job_lat], k)

            assigned_vehicle_id = job_node["metadata"].get("assigned_vehicle_id")
            for _, vehicle_index in zip(distances, indices):
                vehicle_node = vehicle_nodes[int(vehicle_index)]
                dist_m, dur_s = osrm.distance_duration(job_lon, job_lat, vehicle_node["longitude"], vehicle_node["latitude"])
                is_assigned_flag = float(
                    assigned_vehicle_id is not None
                    and int(vehicle_node["metadata"]["vehicle_id"]) == int(assigned_vehicle_id)
                )
         
                self.add_edge(job_node["index"], vehicle_node["index"], dist_m, dur_s, etype=4, is_same_route=0.0, is_assigned=is_assigned_flag, is_bidirectional=True)

    def vehicle_near_job(self, job_nodes, vehicle_nodes):
        if not self.graph.k_near_jobs_for_unassigned_vehicle or not job_nodes or not vehicle_nodes:
            return

        jobs_coords = np.array([[j["longitude"], j["latitude"]] for j in job_nodes], dtype=np.float64)
        tree = cKDTree(jobs_coords)
        unassigned_vehicles = [v for v in vehicle_nodes if int(v["metadata"]["vehicle_id"]) not in self.graph.vehicles_with_routes]

        for vehicle_node in unassigned_vehicles:
            vehicle_lon, vehicle_lat = vehicle_node["longitude"], vehicle_node["latitude"]
            k = min(self.graph.k_near_jobs_for_unassigned_vehicle, len(job_nodes))
            distances, indices = self._query_kdtree(tree, [vehicle_lon, vehicle_lat], k)

            for _, job_index in zip(distances, indices):
                job_node = job_nodes[int(job_index)]
                dist_m, dur_s = osrm.distance_duration(vehicle_lon, vehicle_lat, job_node["longitude"], job_node["latitude"])
                self.add_edge(vehicle_node["index"], job_node["index"], dist_m, dur_s, etype=4, is_same_route=0.0, is_assigned=0.0, is_bidirectional=True)

    def add_edge(self, src_idx, dst_idx, distance_meters, duration_seconds, *, etype, is_same_route, is_assigned, is_bidirectional=False):
        distance_km = max(float(distance_meters) / 1000.0, 0.0)
        duration_h = max(float(duration_seconds) / 3600.0, 0.0)

        if is_bidirectional:
            pairs = [(src_idx, dst_idx), (dst_idx, src_idx)]
        else:
            pairs = [(src_idx, dst_idx)]

        for source, target in pairs:
            self.graph.edge_sources.append(int(source))
            self.graph.edge_targets.append(int(target))
            self.graph.edge_attributes.append(
                [
                    float(np.log1p(distance_km)),
                    float(np.log1p(duration_h)),
                    float(is_same_route),
                    float(is_assigned),
                ]
            )
            self.graph.edge_types.append(int(etype))

    def build(self):
        routes = self.graph.state.routes
        if not routes:
            print("[EdgeBuilder] No routes found in state")
            return
        
        job_nodes = [self.graph.nodes[idx] for idx in self.graph.nodes_by_type["job"]]
        path_nodes = [self.graph.nodes[idx] for idx in self.graph.nodes_by_type["path"]]
        vehicle_nodes = [self.graph.nodes[idx] for idx in self.graph.nodes_by_type["vehicle"]]

        self.job_sequence(routes)
        self.path_sequence(self.get_sorted_path_nodes())
        self.vehicle_assigned(routes)
        self.job_path_proximity(routes, job_nodes, path_nodes)
        self.job_vehicle_proximity(job_nodes, vehicle_nodes)
        self.vehicle_near_job(job_nodes, vehicle_nodes)


class Graph:
    edge_names = {
        0: "job_sequence",
        1: "path_sequence",
        2: "vehicle_assigned",
        3: "job_near_path",
        4: "job_vehicle_proximity",
    }

    def __init__(self, config):
        self.k_near_paths_for_job               = int(config.env.max_path_neighbors)
        self.k_near_vehicles_for_job            = int(config.env.max_vehicle_neighbors)
        self.k_near_jobs_for_unassigned_vehicle = int(config.env.max_vehicle_neighbors)

        self.jobs = []
        self.vehicles = []
        self.state = {}

        self.nodes = []
        self.node_index_by_key = {}
        self.nodes_by_type = {"job": [], "vehicle": [], "path": []}

        self.edge_sources    = []
        self.edge_targets    = []
        self.edge_attributes = []
        self.edge_types      = []

        self.vehicles_with_routes = set()

        self.node_builder  = NodeBuilder(self)
        self.edge_builder  = EdgeBuilder(self)
        self.graph_handler = GraphHandler(config)

    def _coordinate_stats(self):
        longitudes = np.array([n["longitude"] for n in self.nodes], dtype=np.float64)
        latitudes  = np.array([n["latitude"]  for n in self.nodes], dtype=np.float64)

        lon_mean = float(longitudes.mean())
        lon_std  = float(longitudes.std() + 1e-6)
        lat_mean = float(latitudes.mean())
        lat_std  = float(latitudes.std() + 1e-6)

        return lon_mean, lon_std, lat_mean, lat_std

    def _reset(self):
        self.nodes.clear()
        self.node_index_by_key.clear()
        self.nodes_by_type = {"job": [], "vehicle": [], "path": []}
        self.edge_sources.clear()
        self.edge_targets.clear()
        self.edge_attributes.clear()
        self.edge_types.clear()
        self.vehicles_with_routes.clear()

    def _job_features(self, lon_mean, lon_std, lat_mean, lat_std, global_to_local):
        job_features = []
        job_global_indices = self.nodes_by_type["job"]
        for job_local_idx, job_global_idx in enumerate(job_global_indices):
            node = self.nodes[job_global_idx]
            metadata = node["metadata"]
            job_features.append(
                [
                    (node["longitude"] - lon_mean) / lon_std,
                    (node["latitude"] - lat_mean) / lat_std,
                    float(metadata.get("priority", 0)) / 100.0,
                    float(metadata.get("service", 0)) / 3600.0,
                    float(metadata.get("setup", 0)) / 3600.0,
                    float(metadata.get("is_unassigned", False)),
                    0.0 if metadata.get("assigned_vehicle_id") is None else 1.0,
                ]
            )
            global_to_local[job_global_idx] = ("job", job_local_idx)

        return job_features

    def _vehicle_features(self, lon_mean, lon_std, lat_mean, lat_std, global_to_local):
        vehicle_features = []
        vehicle_global_indices = self.nodes_by_type["vehicle"]
        for vehicle_local_idx, vehicle_global_idx in enumerate(vehicle_global_indices):
            node = self.nodes[vehicle_global_idx]
            metadata = node["metadata"]
            time_window = metadata.get("time_window", [])
            time_window_hours = float((time_window[1] - time_window[0]) / 3600.0) if len(time_window) == 2 else 0.0
            vehicle_features.append(
                [
                    (node["longitude"] - lon_mean) / lon_std,
                    (node["latitude"] - lat_mean) / lat_std,
                    float(metadata.get("speed_factor", 1.0)),
                    time_window_hours,
                    float(metadata.get("return_to_depot", False)),
                ]
            )
            global_to_local[vehicle_global_idx] = ("vehicle", vehicle_local_idx)

        return vehicle_features

    def _path_features(self, lon_mean, lon_std, lat_mean, lat_std, global_to_local):
        path_features = []
        path_global_indices = self.nodes_by_type["path"]
        for path_local_idx, path_global_idx in enumerate(path_global_indices):
            node = self.nodes[path_global_idx]
            metadata = node["metadata"]
            path_features.append(
                [
                    (node["longitude"] - lon_mean) / lon_std,
                    (node["latitude"] - lat_mean) / lat_std,
                    float(metadata.get("route_index", 0)),
                    float(metadata.get("vehicle_id", -1)),
                    float(metadata.get("path_sequence", 0)),
                ]
            )
            global_to_local[path_global_idx] = ("path", path_local_idx)

        return path_features

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

    def _apply_edge_buffers(self, data, edge_buffers):
        for (src_type, relation_name, dst_type), buffer in edge_buffers.items():
            if len(buffer["src"]) == 0:
                data[(src_type, relation_name, dst_type)].edge_index = torch.empty((2, 0), dtype=torch.long)
                data[(src_type, relation_name, dst_type)].edge_attr = torch.empty((0, 4), dtype=torch.float32)
            else:
                data[(src_type, relation_name, dst_type)].edge_index = torch.tensor(
                    [buffer["src"], buffer["dst"]], dtype=torch.long
                )
                data[(src_type, relation_name, dst_type)].edge_attr = torch.tensor(
                    buffer["attr"], dtype=torch.float32
                )

    def pack_data(self):
        data = HeteroData()

        lon_mean, lon_std, lat_mean, lat_std = self._coordinate_stats()
        global_to_local = {}

        job_features     = self._job_features(lon_mean, lon_std, lat_mean, lat_std, global_to_local)
        vehicle_features = self._vehicle_features(lon_mean, lon_std, lat_mean, lat_std, global_to_local)
        path_features    = self._path_features(lon_mean, lon_std, lat_mean, lat_std, global_to_local)

        data["job"].x = (
            torch.tensor(job_features, dtype=torch.float32)
            if job_features
            else torch.empty((0, 7), dtype=torch.float32)
        )
        
        data["vehicle"].x = (
            torch.tensor(vehicle_features, dtype=torch.float32)
            if vehicle_features
            else torch.empty((0, 5), dtype=torch.float32)
        )
        
        data["path"].x = (
            torch.tensor(path_features, dtype=torch.float32)
            if path_features
            else torch.empty((0, 5), dtype=torch.float32)
        )

        edge_buffers = self._edge_buffers(global_to_local)
        self._apply_edge_buffers(data, edge_buffers)

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
        
        raw_data     = self.pack_data()
        cleaned_data = self.graph_handler.build(raw_data)
        cleaned_data.mappings = self.mappings()

        return cleaned_data


class GraphHandler:
    def __init__(self, config):
        self.config = config
        self.device = config.device.device
    
    required_relations = [
        ("job", "job_sequence", "job"),
        ("path", "path_sequence", "path"),
        ("vehicle", "vehicle_assigned", "job"),
        ("job", "vehicle_assigned", "vehicle"),
        ("job", "job_near_path", "path"),
        ("path", "job_near_path", "job"),
        ("job", "job_vehicle_proximity", "vehicle"),
        ("vehicle", "job_vehicle_proximity", "job"),
    ]

    def ensure_relations(self, data: HeteroData) -> None:
        for relation in self.required_relations:
            store = data[relation]
            if "edge_index" not in store:
                store.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            if "edge_attr" not in store:
                store.edge_attr = torch.empty((0, 4), dtype=torch.float32, device=self.device)
    
    def populate_node(self, data: HeteroData) -> None:
        for node_type, features in data.x_dict.items():
            data[node_type].x = features
    
    def populate_edge(self, data: HeteroData) -> None:
        for edge_type, edge_index in data.edge_index_dict.items():
            data[edge_type].edge_index = edge_index if edge_index.numel() else torch.empty((2, 0), dtype=torch.long, device=self.device)
        
        for edge_type, edge_attr in {etype: data[etype].edge_attr for etype in data.edge_types}.items():
            data[edge_type].edge_attr = edge_attr if edge_attr.numel() else torch.empty((0, 4), dtype=torch.float32, device=self.device)
    
    def build(self, raw_data) -> HeteroData:  
        data = raw_data
        self.ensure_relations(data)
        self.populate_node(data)
        self.populate_edge(data)
        return data

