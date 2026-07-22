import math
import time
import numpy as np
from functools import wraps
from requests.exceptions import RequestException, Timeout, ConnectionError # type: ignore


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
    r     = np.sqrt(np.random.random(num_coords)) * R
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
