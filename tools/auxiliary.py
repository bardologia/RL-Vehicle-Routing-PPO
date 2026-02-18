import math
import numpy as np
import requests # type: ignore
import pandas as pd
import polyline as polyline_module # type: ignore
import time
from functools import wraps, lru_cache
from requests.exceptions import RequestException, Timeout, ConnectionError # type: ignore
from tools.config import config

def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    duration = distance / (40000 / 3600)  
    return distance, duration

def retry_api_call(max_retries=3, backoff_factor=0.5, timeout=10, verbose=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RequestException, Timeout, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        if verbose:
                            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        if verbose:
                            print(f"API call failed after {max_retries} attempts: {e}")
            raise last_exception
        return wrapper
    return decorator

def generate_coords_batch(center, radius_km, num_coords, p=0.1, m=5.0, bounds=((-46.75, -46.45), (-23.7, -23.4))):
    lon_center, lat_center = center
    lat_rad = np.radians(lat_center)
    cos_lat = np.maximum(1e-6, np.abs(np.cos(lat_rad)))
    
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * cos_lat
    
    is_outlier = np.random.random(num_coords) < p
    R = radius_km * np.where(is_outlier, m, 1.0)
    
    theta = np.random.random(num_coords) * 2.0 * np.pi
    r = np.sqrt(np.random.random(num_coords)) * R
    dx_km = r * np.cos(theta)
    dy_km = r * np.sin(theta)
    
    dlon = dx_km / km_per_deg_lon
    dlat = dy_km / km_per_deg_lat
    
    lons = np.round(lon_center + dlon, 6)
    lats = np.round(lat_center + dlat, 6)
    
    if bounds is not None:
        (min_lon, max_lon), (min_lat, max_lat) = bounds
        lons = np.clip(lons, min_lon, max_lon)
        lats = np.clip(lats, min_lat, max_lat)
    
    return np.column_stack([lons, lats])

def add_jobs(jobs, num_jobs, config):
    center = config.center
    radius = config.radius
    p      = config.outlier_probability
    m      = config.outlier_multiplier
    
    used_ids = [int(job.get("id")) for job in jobs]
    new_id = int(max(used_ids) + 1) if used_ids else 0
    coords_batch = generate_coords_batch(center, radius, num_jobs, p, m)
    priorities = np.random.choice([1, 2, 3, 4, 5], size=num_jobs)

    for i in range(num_jobs):
        new_job = {
            "id": new_id + i,
            "location": coords_batch[i].tolist(),
            "setup": 0,
            "service": 300,
            "amount": [1],
            "priority": int(priorities[i]),
            "description": f"Job {new_id + i}"
        }
        jobs.append(new_job)

    return jobs

def add_vehicles(vehicles, num_vehicles, config):
    used_ids = [int(veh.get("id")) for veh in vehicles]
    new_id = int(max(used_ids) + 1) if used_ids else 0

    center = config.center
    radius = config.radius
    p      = config.outlier_probability
    m      = config.outlier_multiplier

    coords_batch = generate_coords_batch(center, radius, num_vehicles, 0, m)
    capacities = np.random.choice([1, 2, 3], size=num_vehicles)
    speeds = np.random.choice([0.9, 1.0, 1.1], size=num_vehicles)

    for i in range(num_vehicles):
        vehicle = {
            "id": new_id + i,
            "start": coords_batch[i].tolist(),
            "capacity": [int(capacities[i])],
            "time_window": [int(8 * 3600), int(20 * 3600)],
            "speed_factor": float(speeds[i]),
            "return_to_depot": False,
            "description": f"Vehicle {new_id + i}"
        }
        vehicles.append(vehicle)

    return vehicles

@retry_api_call(max_retries=3, backoff_factor=0.5, timeout=20, verbose=False)
def run_vroom(jobs_list, vehicles_list):
    payload = {"jobs": jobs_list, "vehicles": vehicles_list, "options": config.service.options}
    response = requests.post(config.service.vroom_url, json=payload, timeout=20)
    if response.status_code != 200:
        print("VROOM error:", response.text)
        return None
    return response.json()

@lru_cache(maxsize=10000)
def _distance_duration_cached(lon_a, lat_a, lon_b, lat_b):
    url = (
        f"{config.service.osrm_url}/route/v1/driving/"
        f"{lon_a},{lat_a};{lon_b},{lat_b}"
        f"?overview=false&steps=false&alternatives=false&annotations=false"
    )
    response = config.service.http_session.get(url, timeout=20)
    response.raise_for_status()
    data = response.json()
    route = (data.get("routes") or [{}])[0]
    return float(route.get("distance", 0.0)), float(route.get("duration", 0.0))

@retry_api_call(max_retries=3, backoff_factor=0.5, timeout=20, verbose=False)
def distance_duration(longitude_a, latitude_a, longitude_b, latitude_b):
    lon_a = round(float(longitude_a), 5)
    lat_a = round(float(latitude_a), 5)
    lon_b = round(float(longitude_b), 5)
    lat_b = round(float(latitude_b), 5)
    return _distance_duration_cached(lon_a, lat_a, lon_b, lat_b)

def add_osrm_polylines(state):
    routes = state.get("routes") or []
    if not routes:
        return state

    if all(
        isinstance(route, dict)
        and route.get("geometry") is not None
        and route.get("path_coords") is not None
        for route in routes
    ):
        return state

    routes_to_enrich = [(i, r) for i, r in enumerate(routes) if r.get("geometry") is None or r.get("path_coords") is None]
    
    if not routes_to_enrich:
        return state

    for idx, route in routes_to_enrich:
        steps = route.get("steps", []) or []
        if not steps:
            continue
        
        coords = ";".join(f"{step['location'][0]},{step['location'][1]}" for step in steps)
        url = f"{config.service.osrm_url}/route/v1/driving/{coords}?overview=full&geometries=polyline"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = config.service.http_session.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                osrm_routes = data.get("routes") or []
                
                if osrm_routes:
                    route_data = osrm_routes[0]
                    geometry = route_data.get("geometry")
                    if geometry:
                        path_coords = polyline_module.decode(geometry)
                        routes[idx]["geometry"] = geometry
                        routes[idx]["path_coords"] = path_coords
                break  
            except (RequestException, Timeout, ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
                    time.sleep(wait_time)
            except Exception as e:
                break

    return state


