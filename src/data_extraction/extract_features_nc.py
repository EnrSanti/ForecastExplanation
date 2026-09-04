import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader

from . import Region, LimitValues, LEVELS, FOLDERS

logger = logging.getLogger(__name__)

REQUIRED_VARIABLES = ["ccl", "t", "u", "v", "r"]

FrameCallback = Callable[[Any, str, str], None]
CsvCallback = Callable[[str, str, str], None]


@dataclass(frozen=True)
class FeatureSpec:
    var: Union[str, Tuple[str, ...]]
    cmap: str
    limits: dict[int, tuple[int | float, int | float]]
    prefix: str

    @property
    def folder_key(self) -> str:
        return self.prefix


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "cloud": FeatureSpec(
        "ccl",
        "viridis",
        LimitValues.CLOUD,
        "cloud",
    ),
    "temp": FeatureSpec(
        "t",
        "OrRd",
        LimitValues.TEMP,
        "temp",
    ),
    "humidity": FeatureSpec(
        ("r", "rhum"),
        "YlGnBu",
        LimitValues.HUMIDITY,
        "humidity",
    ),
    "wind": FeatureSpec(
        "wind_speed",
        "viridis",
        LimitValues.WIND_SPEED,
        "wind",
    ),
    "wind_direction": FeatureSpec(
        "wind_direction",
        "viridis",
        LimitValues.WIND_SPEED,
        "wind_direction",
    ),
}

LEGEND_SPECS = {
    "cloud": {
        "cmap": "viridis",
        "limits": LimitValues.CLOUD,
        "label": "Cloud cover [%]",
    },
    "temp": {
        "cmap": "OrRd",
        "limits": LimitValues.TEMP,
        "label": "Temperature [K]",
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


def _resolve_var(ds: xr.Dataset, var: Union[str, Tuple[str, ...]]) -> xr.DataArray:
    names = (var,) if isinstance(var, str) else var
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(names)


def _with_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    ws = np.sqrt(ds["u"] ** 2 + ds["v"] ** 2)
    wd = (270 - np.arctan2(ds["v"], ds["u"]) * 180 / np.pi) % 360
    return ds.assign(wind_speed=ws, wind_direction=wd)


def _valid_times(coord_var: xr.DataArray) -> Iterator[Tuple[int, int, pd.Timestamp]]:
    for i in range(coord_var.sizes["time"]):
        base_time = pd.to_datetime(str(coord_var["time"].isel(time=i).values))
        day_start = base_time.normalize() + pd.Timedelta(hours=1)
        day_end = day_start + pd.Timedelta(days=1)

        for j in range(coord_var.sizes["step"]):
            step_val = int(coord_var["step"].isel(step=j).values)
            valid_time = base_time + pd.Timedelta(hours=step_val)

            if day_start <= valid_time <= day_end:
                yield i, j, valid_time


def create_one_time_images(coordinates: Region, output_base: str) -> None:
    save_borders_png(output_base, coordinates)
    create_legends(output_base)


def save_borders_png(output_base: str, coordinates: Region) -> None:
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
        os.path.join(output_base, "borders.png"),
        dpi=130,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)


def create_legends(output_base: str) -> None:

    for key, props in LEGEND_SPECS.items():

        for lvl in LEVELS:
            vmin, vmax = props["limits"][lvl]

            fig, ax = plt.subplots(figsize=(6, 1))
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            cb = plt.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=props["cmap"]),
                cax=ax,
                orientation="horizontal",
            )
            cb.set_label(props["label"])

            png_path = os.path.join(output_base, f"legend_{key}_{lvl}.png")
            plt.savefig(png_path, dpi=130, bbox_inches="tight", pad_inches=0)
            plt.close(fig)


def build_feature_dataarrays(
    input_path: str,
) -> xr.Dataset:
    """
    Extract per-variable, per-level, time-series DataArrays from the NC file.

    Returns a dict keyed by folder name (e.g. "cloud_at_100m") with values
    being xr.DataArray of shape (time, y, x) with normalized [0, 1] values
    ready for tobac consumption.
    """
    with xr.open_dataset(input_path, decode_cf=False) as ds:
        if "dtype" in ds["step"].attrs:
            del ds["step"].attrs["dtype"]
        ds = xr.decode_cf(ds)
        ds = _with_wind_speed(ds)

        result = {}
        for spec in FEATURE_SPECS.values():
            field = _resolve_var(ds, spec.var)
            folders = {k: spec.folder_key + v for k, v in FOLDERS.items()}

            for lvl in LEVELS:
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

                if "wind" in spec.prefix:
                    normalized = stacked.fillna(0.0)
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

                if spec.prefix == "temp":
                    raw = stacked.fillna(0.0)
                    raw = raw.drop_vars(
                        ["valid_time", "step", "isobaricInhPa", "number", "surface"],
                        errors="ignore",
                    )
                    result[f"raw_{folders[lvl]}"] = raw

    return xr.Dataset(result)
