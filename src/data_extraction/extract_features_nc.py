import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader
from scipy.ndimage import gaussian_filter

from region import Region

from . import FOLDERS, FRONT_LEVELS, LEVELS

logger = logging.getLogger(__name__)


# smallest temperature range (K) an hour is normalized over, per level; the
# physical range is about 263-311 K at 1000 hPa down to 218-268 K at 300 hPa
TEMP_MIN_RANGE = {1000: 12.0, 925: 10.0, 850: 9.0, 700: 5.0, 500: 4.0, 300: 3.0}
# wind speed colour scale (m/s) of the legend images
WIND_SPEED_LEGEND = {
    1000: (0, 100),
    925: (0, 100),
    850: (0, 100),
    700: (0, 120),
    500: (0, 150),
    300: (0, 200),
}


@dataclass(frozen=True)
class FeatureSpec:
    var: str
    prefix: str
    normalize: Literal["percent", "hourly", "physical"] | None
    change_hours: int | None = None
    levels: list[int] = field(kw_only=True)


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "cloud": FeatureSpec("ccl", "cloud", "percent", levels=LEVELS),
    "temp": FeatureSpec("t", "temp", "hourly", levels=LEVELS),
    "humidity": FeatureSpec("r", "humidity", "percent", levels=LEVELS),
    "wind": FeatureSpec("wind_speed", "wind", "physical", levels=LEVELS),
    "wind_direction": FeatureSpec(
        "wind_direction", "wind_direction", "physical", levels=LEVELS
    ),
    "front": FeatureSpec("front", "front", None, levels=FRONT_LEVELS),
    "tadv": FeatureSpec("tadv", "tadv", None, levels=FRONT_LEVELS),
    "te_change": FeatureSpec(
        "theta_e", "te_change", None, change_hours=3, levels=FRONT_LEVELS
    ),
}

RAW_PREFIXES = ["cloud", "temp", "humidity"]
FRONT_SMOOTH_PX = 4  # ~22 km on the 5.5 km CERRA grid

LEGEND_SPECS = {
    "cloud": {
        "cmap": "viridis",
        "limits": {lvl: (0, 100) for lvl in LEVELS},
        "label": "Cloud cover [%]",
    },
    "temp": {
        "cmap": "OrRd",
        "limits": {lvl: (0, 1) for lvl in LEVELS},
        "label": "Relative temperature (0 = coldest point of the hour)",
    },
    "wind": {
        "cmap": "viridis",
        "limits": WIND_SPEED_LEGEND,
        "label": "Wind speed [m/s]",
    },
    "humidity": {
        "cmap": "YlGnBu",
        "limits": {lvl: (0, 100) for lvl in LEVELS},
        "label": "Relative humidity [%]",
    },
}


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
    rh = ds["r"].transpose(*t.dims)
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
        resolution="10m", category="cultural", name="admin_1_states_provinces"
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


def _drop_coords(da: xr.DataArray) -> xr.DataArray:
    return da.drop_vars(
        ["valid_time", "step", "isobaricInhPa", "number", "surface"], errors="ignore"
    )


def build_feature_dataarrays(input_path: Path) -> xr.Dataset:
    """
    Per-variable, per-level (time, y, x) fields from the cut NC file, named
    <prefix>_at_<level>m: normalized to [0, 1] for tobac (wind in m/s and
    degrees), plus raw_<prefix>_at_<level>m physical values for RAW_PREFIXES
    and for the specs without limits.
    """
    with xr.open_dataset(input_path, engine="h5netcdf", decode_cf=False) as ds:
        if "dtype" in ds["step"].attrs:
            del ds["step"].attrs["dtype"]
        ds = xr.decode_cf(ds)
        ds = _with_wind_speed(ds)
        ds = _with_front_fields(ds)

        result = {}
        for spec in FEATURE_SPECS.values():
            for lvl in spec.levels:
                level_field = ds[spec.var].sel(isobaricInhPa=lvl)
                name = spec.prefix + FOLDERS[lvl]

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

                if spec.normalize is None or spec.prefix in RAW_PREFIXES:
                    result[f"raw_{name}"] = _drop_coords(stacked)
                if spec.normalize is None:
                    continue

                if spec.normalize == "physical":
                    normalized = stacked
                elif spec.normalize == "hourly":
                    vmin = stacked.min(dim=("y", "x"))
                    vmax = stacked.max(dim=("y", "x"))
                    vrange = np.maximum(vmax - vmin, TEMP_MIN_RANGE[lvl])
                    normalized = (stacked - vmin) / vrange
                else:
                    normalized = (stacked / 100).clip(0, 1)
                result[name] = _drop_coords(normalized.fillna(0.0))

    return xr.Dataset(result)
