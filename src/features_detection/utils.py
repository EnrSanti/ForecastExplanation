import logging

import matplotlib
import tobac
import xarray as xr

matplotlib.use("Agg")

from features_detection.constants import (
    DEFAULT_DT,
    DEFAULT_DXY,
)

logger = logging.getLogger(__name__)


def get_grid_spacings(
    referenced_data: xr.DataArray,
    default_dxy: float = DEFAULT_DXY,
    default_dt: float = DEFAULT_DT,
) -> tuple[float, float]:
    """
    Determines grid spacing dxy and dt dynamically from DataArray,
    falling back to provided defaults when unit dimensions are missing or 1.
    """
    try:
        dxy, dt = tobac.get_spacings(referenced_data)
        if dxy is None or dxy <= 1.0:
            dxy = default_dxy
        if dt is None or dt <= 0:
            dt = default_dt
        return float(dxy), float(dt)
    except Exception:  # noqa: BLE001
        return default_dxy, default_dt


def build_referenced_data_from_xarray(
    da: xr.DataArray,
    times: list,
    region_bounds=None,
) -> xr.DataArray:
    """
    Build tobac-compatible DataArray directly from a xarray DataArray.
    No PNG reading needed.
    """
    import numpy as np
    import pandas as pd

    # da already has (time, y, x) dims and proper coordinates
    referenced_data = xr.DataArray(
        da.values,
        dims=("time", "y", "x"),
        coords={
            "time": pd.to_datetime(times),
            "y": ("y", np.arange(da.sizes["y"]), {"units": "m"}),
            "x": ("x", np.arange(da.sizes["x"]), {"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )

    if "latitude" in da.coords and "longitude" in da.coords:
        referenced_data = referenced_data.assign_coords(
            latitude=(("y", "x"), da["latitude"].values),
            longitude=(("y", "x"), da["longitude"].values),
        )
    elif region_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = region_bounds
        lat = np.linspace(lat_min, lat_max, da.sizes["y"])
        lon = np.linspace(lon_min, lon_max, da.sizes["x"])
        longitude = np.tile(lon[np.newaxis, :], (da.sizes["y"], 1))
        latitude = np.tile(lat[:, np.newaxis], (1, da.sizes["x"]))
        referenced_data = referenced_data.assign_coords(
            latitude=(("y", "x"), latitude),
            longitude=(("y", "x"), longitude),
        )

    return referenced_data
