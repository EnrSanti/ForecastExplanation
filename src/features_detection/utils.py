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


def to_compressed_netcdf(ds: xr.Dataset, path) -> None:
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
        }
        for name, var in ds.data_vars.items()
        if var.dtype.kind in "fiu"
    }
    ds.to_netcdf(path, encoding=encoding)


def get_grid_spacings(
    referenced_data: xr.DataArray,
    default_dxy: float = DEFAULT_DXY,
    default_dt: float = DEFAULT_DT,
) -> tuple[float, float]:
    """
    Determines grid spacing dxy and dt dynamically from DataArray,
    falling back to provided defaults when unit dimensions are missing or 1.
    x/y are pixel indices, so dxy comes from latitude/longitude when present.
    """
    try:
        dxy, dt = tobac.get_spacings(referenced_data)
    except Exception:  # noqa: BLE001
        dxy, dt = None, None
    if dxy is None or dxy <= 1.0:
        dxy = latlon_spacing(referenced_data) or default_dxy
    if dt is None or dt <= 0:
        dt = default_dt
    return float(dxy), float(dt)


def latlon_spacing(da: xr.DataArray) -> float | None:
    """Mean distance (m) between neighbouring grid points, or None."""
    import numpy as np

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
