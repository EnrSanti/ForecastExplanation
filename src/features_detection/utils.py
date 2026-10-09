import logging

import numpy as np
import pandas as pd
import xarray as xr

from features_detection.constants import DEFAULT_DT, DEFAULT_DXY

logger = logging.getLogger(__name__)


def to_compressed_netcdf(ds: xr.Dataset, path) -> None:
    encoding = {
        name: {"zlib": True, "complevel": 4}
        for name, var in ds.data_vars.items()
        if var.dtype.kind in "fiu"
    }
    ds.to_netcdf(path, encoding=encoding)


def get_grid_spacings(referenced_data: xr.DataArray) -> tuple[float, float]:
    """dxy (m) from latitude/longitude (x/y are pixel indices), dt of the
    hourly frames (s)."""
    return float(latlon_spacing(referenced_data) or DEFAULT_DXY), float(DEFAULT_DT)


def latlon_spacing(da: xr.DataArray) -> float | None:
    """Mean distance (m) between neighbouring grid points, or None."""
    if "latitude" not in da.coords or "longitude" not in da.coords:
        return None
    lat = np.radians(da["latitude"].values)
    lon = np.radians(da["longitude"].values)

    def dist(lat1, lon1, lat2, lon2):
        a = (
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
        return 2 * 6.371e6 * np.arcsin(np.sqrt(a))

    dx = dist(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    dy = dist(lat[:-1, :], lon[:-1, :], lat[1:, :], lon[1:, :])
    spacing = float(np.nanmean(np.concatenate([dx.ravel(), dy.ravel()])))
    return spacing if np.isfinite(spacing) and spacing > 1.0 else None


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
