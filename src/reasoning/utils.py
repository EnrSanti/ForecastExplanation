import numpy as np
import xarray as xr


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Return great-circle distance(s) in km between point(s) (lat1, lon1) and (lat2, lon2)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def get_compass_direction(degrees: float) -> str | float:
    """Convert a bearing in degrees to an 8-point compass label (e.g. 'NE'). Returns nan for nan input."""
    if np.isnan(degrees):
        return np.nan
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int((degrees + 22.5) // 45) % 8
    return directions[idx]


def get_heights(data: xr.Dataset) -> list[str]:
    heights = set()
    for var in data.data_vars:
        if "wind_direction_at_" in var:
            heights.add(var.split("wind_direction_at_")[1])
    sorted_heights = sorted(
        heights,
        key=lambda x: int(x.replace("m", "")) if x.replace("m", "").isdigit() else x,
    )

    return sorted_heights
