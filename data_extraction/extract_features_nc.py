import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from . import Region, LimitValues

logger = logging.getLogger(__name__)

LEVELS = [1000, 925, 850, 700, 500, 300]
FOLDERS = {
    1000: "_at_100m",
    925: "_at_750m",
    850: "_at_1_4km",
    700: "_at_3km",
    500: "_at_5_5km",
    300: "_at_9km",
}

REQUIRED_VARIABLES = ["ccl", "t", "u", "v", "r"]

FrameCallback = Callable[[Any, str, str], None]
CsvCallback = Callable[[str, str, str], None]


@dataclass(frozen=True)
class FeatureSpec:
    var: Union[str, Tuple[str, ...]]
    cmap: str
    limits: dict[int, tuple[int | float, int | float]]
    prefix: str
    folder_prefix: Optional[str] = None
    axis_off: bool = False

    @property
    def folder_key(self) -> str:
        return self.folder_prefix or self.prefix


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
        folder_prefix="winds",
        axis_off=True,
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
    return ds.assign(wind_speed=np.sqrt(ds["u"] ** 2 + ds["v"] ** 2))


def _valid_times(coord_var: xr.DataArray) -> Iterator[Tuple[int, int, pd.Timestamp]]:
    for i in range(coord_var.sizes["time"]):
        base_time = pd.to_datetime(str(coord_var["time"].isel(time=i).values))
        day_start = base_time.normalize() + pd.Timedelta(hours=1)
        day_end = day_start + pd.Timedelta(days=1)

        for j in range(coord_var.sizes["step"]):
            step_val = int(coord_var["step"].isel(step=j).values)
            valid_time = base_time + pd.Timedelta(hours=step_val)
            logger.debug(
                f"Checking valid_time: {valid_time}, day_start: {day_start}, day_end: {day_end}, step_val: {step_val}"
            )

            if day_start <= valid_time <= day_end:
                yield i, j, valid_time


def _file_frame_writer(output_base: str) -> FrameCallback:
    def on_frame(fig: Any, folder: str, fname: str) -> None:
        out_dir = os.path.join(output_base, folder)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(
            os.path.join(out_dir, fname), dpi=130, bbox_inches="tight", pad_inches=0
        )

    return on_frame


def _render_scalar_frames(
    ds: xr.Dataset,
    coordinates: Region,
    spec: FeatureSpec,
) -> Iterator[Tuple[str, str, Any]]:

    field = _resolve_var(ds, spec.var)

    folders = {k: spec.folder_key + v for k, v in FOLDERS.items()}

    fig, ax = plt.subplots(
        figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    if spec.axis_off:
        ax.axis("off")

    try:
        for lvl in LEVELS:
            level_field = field.sel(isobaricInhPa=lvl)

            for i, j, valid_time in _valid_times(level_field):
                frame = level_field.isel(time=i, step=j)
                if not np.isfinite(frame).any():
                    continue

                vmin, vmax = spec.limits[lvl]

                mesh = ax.pcolormesh(
                    frame["longitude"],
                    frame["latitude"],
                    frame,
                    cmap=spec.cmap,
                    shading="auto",
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                )

                fname = f"{spec.prefix}_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png"
                yield folders[lvl], fname, fig
                mesh.remove()
    finally:
        plt.close(fig)


def _save_scalar_maps(
    ds: xr.Dataset, coordinates: Region, spec: FeatureSpec, output_base: str
) -> None:
    frame_writer = _file_frame_writer(output_base)
    for folder, fname, fig in _render_scalar_frames(ds, coordinates, spec):
        frame_writer(fig, folder, fname)


def create_one_time_images(coordinates: Region, output_base: str) -> None:
    save_borders_png(output_base, coordinates)
    create_legends(output_base)


def save_feature_maps(input_path: str, coordinates: Region, output_base: str) -> None:
    # todo add flag to call this one
    with xr.open_dataset(input_path, decode_cf=False) as ds:
        # todo Here xarray becomes png images after extracting the needed data
        if "dtype" in ds["step"].attrs:
            del ds["step"].attrs["dtype"]

        ds = xr.decode_cf(ds)

        if any(var not in ds for var in REQUIRED_VARIABLES):
            logger.error("Error: One or more required variables not found in dataset.")
            return

        ds = _with_wind_speed(ds)
        for spec in FEATURE_SPECS.values():
            _save_scalar_maps(ds, coordinates, spec, output_base)

        ds.close()

    logger.debug(f"Feature maps saved for {input_path} in {output_base}")


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

                stacked = xr.concat(frames, dim=pd.Index(times, name="time"))
                stacked = stacked.sortby("time")

                vmin, vmax = spec.limits[lvl]
                normalized = (stacked - vmin) / (vmax - vmin)
                normalized = normalized.clip(0, 1)

                normalized = normalized.fillna(0.0)
                # todo guardare quanti nan ci sono e se usare media
                
                normalized = normalized.drop_vars(["valid_time", "step", "isobaricInhPa", "number", "surface"], errors="ignore")

                result[folders[lvl]] = normalized

    return xr.Dataset({name: da for name, da in result.items()})
