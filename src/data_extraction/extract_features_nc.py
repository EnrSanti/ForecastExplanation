import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader
from scipy.ndimage import gaussian_filter

from . import FOLDERS, FRONT_LEVELS, LEVELS, LimitValues, Region

logger = logging.getLogger(__name__)

REQUIRED_VARIABLES = ["ccl", "t", "u", "v", "r"]

FrameCallback = Callable[[Any, str, str], None]
CsvCallback = Callable[[str, str, str], None]


@dataclass(frozen=True)
class FeatureSpec:
    var: str | tuple[str, ...]
    cmap: str
    limits: dict[int, tuple[int | float, int | float]] | None
    prefix: str
    min_range: dict[int, float] | None = None
    change_hours: int | None = None
    levels: list[int] = field(kw_only=True)

    @property
    def folder_key(self) -> str:
        return self.prefix


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "cloud": FeatureSpec(
        "ccl",
        "viridis",
        LimitValues.CLOUD,
        "cloud",
        levels=LEVELS,
    ),
    "temp": FeatureSpec(
        "t",
        "OrRd",
        LimitValues.TEMP,
        "temp",
        min_range=LimitValues.TEMP_MIN_RANGE,
        levels=LEVELS,
    ),
    "humidity": FeatureSpec(
        ("r", "rhum"),
        "YlGnBu",
        LimitValues.HUMIDITY,
        "humidity",
        levels=LEVELS,
    ),
    "wind": FeatureSpec(
        "wind_speed",
        "viridis",
        LimitValues.WIND_SPEED,
        "wind",
        levels=LEVELS,
    ),
    "wind_direction": FeatureSpec(
        "wind_direction",
        "viridis",
        LimitValues.WIND_SPEED,
        "wind_direction",
        levels=LEVELS,
    ),
    "front": FeatureSpec(
        "front",
        "magma",
        None,
        "front",
        levels=FRONT_LEVELS,
    ),
    "tadv": FeatureSpec(
        "tadv",
        "RdBu_r",
        None,
        "tadv",
        levels=FRONT_LEVELS,
    ),
    "te_change": FeatureSpec(
        "theta_e",
        "RdBu_r",
        None,
        "te_change",
        change_hours=3,
        levels=FRONT_LEVELS,
    ),
}

RAW_PREFIXES = ["temp", "humidity"]
FRONT_SMOOTH_PX = 4  # ~22 km on the 5.5 km CERRA grid

LEGEND_SPECS = {
    "cloud": {
        "cmap": "viridis",
        "limits": LimitValues.CLOUD,
        "label": "Cloud cover [%]",
    },
    "temp": {
        "cmap": "OrRd",
        "limits": {lvl: (0, 1) for lvl in LimitValues.TEMP},
        "label": "Relative temperature (0 = coldest point of the hour)",
    },
    "wind": {
        "cmap": "viridis",
        "limits": LimitValues.WIND_SPEED,
        "label": "Wind speed [m/s]",
    },
    "humidity": {
        "cmap": "YlGnBu",
        "limits": LimitValues.HUMIDITY,
        "label": "Relative humidity [%]",
    },
}


def _resolve_var(ds: xr.Dataset, var: str | tuple[str, ...]) -> xr.DataArray:
    names = (var,) if isinstance(var, str) else var
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(names)


def _with_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    ws = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    wd = (270 - np.arctan2(ds["v"], ds["u"]) * 180 / np.pi) % 360
    return ds.assign(wind_speed=ws, wind_direction=wd)


def _smooth_frames(arr: np.ndarray) -> np.ndarray:
    """Gaussian smoothing over the last two (y, x) axes."""
    sigma = (0,) * (arr.ndim - 2) + (FRONT_SMOOTH_PX, FRONT_SMOOTH_PX)
    return gaussian_filter(arr, sigma, mode="mirror")


def _gradient(lat: np.ndarray, lon: np.ndarray) -> Callable:
    """d/dx, d/dy (per metre) on the lat/lon grid."""
    # non usiamo np.gradient perché x y cambiano dimensione in metri
    x = 6.371e6 * np.cos(np.radians(lat.mean())) * np.radians(lon)
    y = 6.371e6 * np.radians(lat)
    xj, xi = np.gradient(x)
    yj, yi = np.gradient(y)
    det = xi * yj - xj * yi

    def grad(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fj, fi = np.gradient(f, axis=(-2, -1))
        return (fi * yj - fj * yi) / det, (fj * xi - fi * xj) / det

    return grad


def _with_front_fields(ds: xr.Dataset) -> xr.Dataset:
    """Adds theta_e (K), front = |grad theta_e| (K/100 km) and
    tadv = temperature advection (K/h)."""
    t = ds["t"].transpose(..., "y", "x")
    rh = _resolve_var(ds, ("r", "rhum")).transpose(*t.dims)
    p = ds["isobaricInhPa"].broadcast_like(t).transpose(*t.dims)
    tc = t - 273.15
    e = rh.clip(1, 100) / 100 * 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    mixing = 0.622 * e / (p - e)
    theta_e = t * (1000.0 / p) ** 0.2854 * np.exp(2.5e6 * mixing / (1004.0 * t))

    grad = _gradient(ds["latitude"].values, ds["longitude"].values)
    te = _smooth_frames(theta_e.values)
    gx, gy = grad(te)
    tx, ty = grad(_smooth_frames(t.values))
    u = ds["u"].transpose(*t.dims).values
    v = ds["v"].transpose(*t.dims).values
    return ds.assign(
        theta_e=(t.dims, te),
        front=(t.dims, np.hypot(gx, gy) * 1e5),
        tadv=(t.dims, -(u * tx + v * ty) * 3600),
    )


def _valid_times(coord_var: xr.DataArray) -> Iterator[tuple[int, int, pd.Timestamp]]:
    for i in range(coord_var.sizes["time"]):
        base_time = pd.to_datetime(str(coord_var["time"].isel(time=i).values))
        day_start = base_time.normalize() + pd.Timedelta(hours=1)
        day_end = day_start + pd.Timedelta(days=1)

        for j in range(coord_var.sizes["step"]):
            step_val = int(coord_var["step"].isel(step=j).values)
            valid_time = base_time + pd.Timedelta(hours=step_val)

            if not pd.isna(valid_time) and day_start <= valid_time <= day_end:
                yield i, j, valid_time


def create_one_time_images(coordinates: Region, output_base: Path) -> None:
    save_borders_png(output_base, coordinates)
    create_legends(output_base)


def save_borders_png(output_base: Path, coordinates: Region) -> None:
    fig, ax = plt.subplots(
        figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())

    ax.coastlines(resolution="10m", linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="black")

    shpfilename = shapereader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_1_states_provinces",
    )
    reader = shapereader.Reader(shpfilename)
    for record in reader.records():
        if record.attributes.get("adm0_a3") == "ITA":
            ax.add_geometries(
                [record.geometry],
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="gray",
                linewidth=0.6,
                linestyle="--",
            )

    ax.axis("off")
    fig.savefig(
        output_base / "borders.png",
        dpi=130,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)


def create_legends(output_base: Path) -> None:
    for key, props in LEGEND_SPECS.items():
        for lvl in LEVELS:
            vmin, vmax = props["limits"][lvl]

            fig, ax = plt.subplots(figsize=(6, 1))
            norm = plt.Normalize(vmin=float(vmin), vmax=float(vmax))

            cb = plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=str(props["cmap"])),
                cax=ax,
                orientation="horizontal",
            )
            cb.set_label(str(props["label"]))

            png_path = output_base / f"legend_{key}_{lvl}.png"
            plt.savefig(png_path, dpi=130, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def build_feature_dataarrays(
    input_path: Path,
) -> xr.Dataset:
    """
    Extract per-variable, per-level, time-series DataArrays from the NC file.

    Returns a dict keyed by folder name (e.g. "cloud_at_100m") with values
    being xr.DataArray of shape (time, y, x) with normalized [0, 1] values
    ready for tobac consumption.
    """
    with xr.open_dataset(input_path, engine="h5netcdf", decode_cf=False) as ds:
        if "dtype" in ds["step"].attrs:
            del ds["step"].attrs["dtype"]
        ds = xr.decode_cf(ds)
        ds = _with_wind_speed(ds)
        ds = _with_front_fields(ds)

        result = {}
        for spec in FEATURE_SPECS.values():
            field = _resolve_var(ds, spec.var)
            folders = {k: spec.folder_key + v for k, v in FOLDERS.items()}

            for lvl in spec.levels:
                level_field = field.sel(isobaricInhPa=lvl)

                frames = []
                times = []
                for i, j, valid_time in _valid_times(level_field):
                    frame = level_field.isel(time=i, step=j)
                    if not np.isfinite(frame).any():
                        continue
                    frames.append(frame)
                    times.append(valid_time)

                if not frames:
                    continue

                stacked = xr.concat(
                    frames,
                    dim=pd.Index(times, name="time"),
                    coords="minimal",
                    compat="override",
                )
                stacked = stacked.sortby("time")
                if spec.change_hours:
                    change = stacked - stacked.shift(time=spec.change_hours)
                    stacked = change.fillna(0.0)

                if spec.limits is None:
                    result[f"raw_{folders[lvl]}"] = stacked.drop_vars(
                        ["valid_time", "step", "isobaricInhPa", "number", "surface"],
                        errors="ignore",
                    )
                    continue
                if "wind" in spec.prefix:
                    normalized = stacked.fillna(0.0)
                elif spec.min_range is not None:
                    vmin = stacked.min(dim=("y", "x"))
                    vmax = stacked.max(dim=("y", "x"))
                    vrange = np.maximum(vmax - vmin, spec.min_range[lvl])
                    normalized = ((stacked - vmin) / vrange).fillna(0.0)
                else:
                    vmin, vmax = spec.limits[lvl]
                    normalized = (stacked - vmin) / (vmax - vmin)
                    normalized = normalized.clip(0, 1)

                    normalized = normalized.fillna(0.0)

                normalized = normalized.drop_vars(
                    ["valid_time", "step", "isobaricInhPa", "number", "surface"],
                    errors="ignore",
                )

                result[folders[lvl]] = normalized

                if spec.prefix in RAW_PREFIXES:
                    raw = stacked.drop_vars(
                        ["valid_time", "step", "isobaricInhPa", "number", "surface"],
                        errors="ignore",
                    )
                    result[f"raw_{folders[lvl]}"] = raw

    return xr.Dataset(result)
