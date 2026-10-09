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


def get_heights(data: xr.Dataset) -> list[str]:
    """Level suffixes ("0300m", ...) present in the data, ascending."""
    prefix = "wind_direction_at_"
    return sorted(
        str(v).removeprefix(prefix) for v in data.data_vars if str(v).startswith(prefix)
    )
