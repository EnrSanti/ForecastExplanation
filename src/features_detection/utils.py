import logging

import numpy as np
import pandas as pd
import xarray as xr

from utils import haversine

logger = logging.getLogger(__name__)


def grid_spacing(data: xr.Dataset | xr.DataArray) -> float:
    """Mean distance (m) between neighbouring grid points (x/y are pixel
    indices, so it comes from latitude/longitude)."""
    lat = data["latitude"].values
    lon = data["longitude"].values
    dx = haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dy = haversine(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    return 1000 * float(np.nanmean(np.concatenate([dx.ravel(), dy.ravel()])))


def build_referenced_data_from_xarray(da: xr.DataArray, times: list) -> xr.DataArray:
    """tobac-compatible (time, y, x) DataArray with latitude/longitude."""
    return xr.DataArray(
        da.values,
        dims=("time", "y", "x"),
        coords={
            "time": pd.to_datetime(times),
            "y": ("y", np.arange(da.sizes["y"]), {"units": "m"}),
            "x": ("x", np.arange(da.sizes["x"]), {"units": "m"}),
            "latitude": (("y", "x"), da["latitude"].values),
            "longitude": (("y", "x"), da["longitude"].values),
        },
        attrs={"units": "m s-1"},
    )
