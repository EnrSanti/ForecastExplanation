import logging
from typing import Tuple

import matplotlib
import tobac
import xarray as xr

matplotlib.use("Agg")

from image_processing.constants import (
    DEFAULT_DT,
    DEFAULT_DXY,
)

logger = logging.getLogger(__name__)


def get_grid_spacings(
    referenced_data: xr.DataArray,
    default_dxy: float = DEFAULT_DXY,
    default_dt: float = DEFAULT_DT,
) -> Tuple[float, float]:
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
    except Exception:
        return default_dxy, default_dt


def normalize_referenced_data(referenced_data: xr.DataArray) -> xr.DataArray:
    """Normalizes DataArray values to the [0, 1] range."""
    vmin = float(referenced_data.min())
    vmax = float(referenced_data.max())
    if vmax == vmin:
        return xr.zeros_like(referenced_data)
    return (referenced_data - vmin) / (vmax - vmin)


def _latlon_to_px(
    lat: float,
    lon: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    frame_height: int,
    frame_width: int,
):
    """
    Convert a lat/lon into pixel (x, y) using the same linear mapping used
    to build the frame's coordinate grid elsewhere in the pipeline (row 0 /
    top of image = lat_min, matching the existing np.linspace(lat_min,
    lat_max, frame_height) convention — not the usual north-up mapping).
    """
    px_x = (lon - lon_min) / (lon_max - lon_min) * (frame_width - 1)
    px_y = (lat_max - lat) / (lat_max - lat_min) * (frame_height - 1)
    return px_x, px_y


def build_referenced_data_from_xarray(
    da: xr.DataArray,
    times: list,
    region_bounds=None,
) -> xr.DataArray:
    """
    Build tobac-compatible DataArray directly from an xarray DataArray.
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
