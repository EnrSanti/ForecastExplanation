import io
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.io import shapereader

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
    vmin: float
    vmax: float
    prefix: str
    folder_prefix: Optional[str] = None
    axis_off: bool = False

    @property
    def folder_key(self) -> str:
        return self.folder_prefix or self.prefix


FEATURE_SPECS: Dict[str, FeatureSpec] = {
    "cloud": FeatureSpec("ccl", "viridis", LimitValues.MINIMUM_CLOUD_VALUE, LimitValues.MAXIMUM_CLOUD_VALUE, "cloud"),
    "temp": FeatureSpec("t", "OrRd", LimitValues.MINIMUM_TEMP_VALUE, LimitValues.MAXIMUM_TEMP_VALUE, "temp"),
    "humidity": FeatureSpec(("r", "rhum"), "YlGnBu", LimitValues.MINIMUM_HUMIDITY_VALUE,
                             LimitValues.MAXIMUM_HUMIDITY_VALUE, "humidity"),
    "wind": FeatureSpec("wind_speed", "viridis", LimitValues.MINIMUM_WIND_SPEED_VALUE,
                         LimitValues.MAXIMUM_WIND_SPEED_VALUE, "wind", folder_prefix="winds", axis_off=True),
}

LEGEND_SPECS = {
    "cloud": {"cmap": "viridis", "vmin": LimitValues.MINIMUM_CLOUD_VALUE, "vmax": LimitValues.MAXIMUM_CLOUD_VALUE,
              "label": "Cloud cover [%]", "txt_prefix": "Cloud cover", "txt_unit": "%"},
    "temp": {"cmap": "OrRd", "vmin": LimitValues.MINIMUM_TEMP_VALUE, "vmax": LimitValues.MAXIMUM_TEMP_VALUE,
             "label": "Temperature [K]", "txt_prefix": "Temperature", "txt_unit": "K"},
    "wind": {"cmap": "viridis", "vmin": LimitValues.MINIMUM_WIND_SPEED_VALUE,
             "vmax": LimitValues.MAXIMUM_WIND_SPEED_VALUE, "label": "Wind speed [m/s]",
             "txt_prefix": "Wind Speed", "txt_unit": "m/s"},
    "humidity": {"cmap": "YlGnBu", "vmin": LimitValues.MINIMUM_HUMIDITY_VALUE,
                 "vmax": LimitValues.MAXIMUM_HUMIDITY_VALUE, "label": "Relative humidity [%]",
                 "txt_prefix": "Humidity", "txt_unit": "%"},
}


def _fig_to_bgr_array(fig: Any, **savefig_kwargs: Any) -> np.ndarray:
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", **savefig_kwargs)
        buf.seek(0)
        img_rgba = plt.imread(buf)
        img_rgb = (img_rgba[:, :, :3] * 255).astype(np.uint8)
        return img_rgb[:, :, ::-1]


def _resolve_var(ds: xr.Dataset, var: Union[str, Tuple[str, ...]]) -> xr.DataArray:
    names = (var,) if isinstance(var, str) else var
    for name in names:
        if name in ds:
            return ds[name]
    raise KeyError(names)


def _has_required_variables(ds: xr.Dataset) -> bool:
    if any(var not in ds for var in REQUIRED_VARIABLES):
        logger.error("Error: One or more required variables not found in dataset.")
        return False
    return True


def _with_wind_speed(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(wind_speed=np.sqrt(ds["u"] ** 2 + ds["v"] ** 2))


def _valid_times(coord_var: xr.DataArray) -> Iterator[Tuple[int, int, pd.Timestamp]]:
    for i in range(coord_var.sizes["time"]):
        base_time = pd.to_datetime(str(coord_var["time"].isel(time=i).values))
        day_start = base_time.normalize() + pd.Timedelta(hours=1)
        day_end = day_start + pd.Timedelta(days=1)

        for j in range(coord_var.sizes["step"]):
            step_val = int(coord_var["step"].isel(step=j).values)
            valid_time = base_time + pd.Timedelta(nanoseconds=step_val)

            if day_start <= valid_time <= day_end:
                yield i, j, valid_time


def _file_frame_writer(output_base: str) -> FrameCallback:
    def on_frame(fig: Any, folder: str, fname: str) -> None:
        out_dir = os.path.join(output_base, folder)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, fname), dpi=130, bbox_inches="tight", pad_inches=0)

    return on_frame


def _file_csv_writer(output_base: str) -> CsvCallback:
    def on_csv(folder: str, fname: str, content: str) -> None:
        out_dir = os.path.join(output_base, folder)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, fname), "w") as f:
            f.write(content)

    return on_csv


def _render_scalar_frames(
        ds: xr.Dataset,
        coordinates: Region,
        spec: FeatureSpec,
) -> Iterator[Tuple[str, str, Any]]:
    field = _resolve_var(ds, spec.var)
    folders = {k: spec.folder_key + v for k, v in FOLDERS.items()}

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
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

                mesh = ax.pcolormesh(
                    frame["longitude"],
                    frame["latitude"],
                    frame,
                    cmap=spec.cmap,
                    shading="auto",
                    vmin=spec.vmin,
                    vmax=spec.vmax,
                    transform=ccrs.PlateCarree(),
                )

                fname = f"{spec.prefix}_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.png"
                yield folders[lvl], fname, fig
                mesh.remove()
    finally:
        plt.close(fig)


def _wind_pixel_transform(
        coordinates: Region,
) -> Tuple[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], Any]:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=130, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(coordinates.value, crs=ccrs.PlateCarree())
    ax.axis("off")
    fig.canvas.draw()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, y0, height_in = bbox.x0, bbox.y0, bbox.height

    def to_pixel(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        disp = ax.transData.transform(np.column_stack((lon, lat)))
        px = np.round(disp[:, 0] - x0 * fig.dpi).astype(int)
        py = np.round(int(height_in * fig.dpi) - (disp[:, 1] - y0 * fig.dpi)).astype(int)
        return px, py

    return to_pixel, fig


def _render_wind_vectors(ds: xr.Dataset, coordinates: Region) -> Iterator[Tuple[str, str, str]]:
    wind_folders = {k: "winds" + v for k, v in FOLDERS.items()}
    u_var = ds["u"]
    v_var = ds["v"]
    lon_min, lon_max, lat_min, lat_max = coordinates.value
    vector_step = 10

    to_pixel, fig = _wind_pixel_transform(coordinates)
    try:
        for lvl in LEVELS:
            u_lvl = u_var.sel(isobaricInhPa=lvl)
            v_lvl = v_var.sel(isobaricInhPa=lvl)

            for i, j, valid_time in _valid_times(u_lvl):
                u_slice = u_lvl.isel(time=i, step=j)
                v_slice = v_lvl.isel(time=i, step=j)

                if not np.isfinite(np.sqrt(u_slice ** 2 + v_slice ** 2)).any():
                    continue

                lon2d = u_slice["longitude"].broadcast_like(u_slice).values
                lat2d = u_slice["latitude"].broadcast_like(u_slice).values
                lon_subset = lon2d[::vector_step, ::vector_step]
                lat_subset = lat2d[::vector_step, ::vector_step]
                u_subset = u_slice.values[::vector_step, ::vector_step]
                v_subset = v_slice.values[::vector_step, ::vector_step]

                mask = (
                        (lon_subset >= lon_min)
                        & (lon_subset <= lon_max)
                        & (lat_subset >= lat_min)
                        & (lat_subset <= lat_max)
                )

                lon_visible = lon_subset[mask]
                lat_visible = lat_subset[mask]
                u_visible = u_subset[mask]
                v_visible = v_subset[mask]

                px, py = to_pixel(lon_visible, lat_visible)
                mags = np.sqrt(u_visible ** 2 + v_visible ** 2)
                alphas = np.degrees(np.arctan2(v_visible, u_visible))
                alphas[~np.isfinite(alphas)] = 0
                mags[~np.isfinite(mags)] = 0

                rows = "\n".join(
                    f"{idx},{px[idx]},{py[idx]},{lat_visible[idx]:.6f},{lon_visible[idx]:.6f},"
                    f"{mags[idx]:.6f},{alphas[idx]:.2f}"
                    for idx in range(len(lon_visible))
                )
                csv_content = "vector_id,pixel_x,pixel_y,latitude,longitude,magnitude,alpha_deg\n" + rows + "\n"
                csv_fname = f"wind_{lvl}_{valid_time.strftime('%Y%m%d_%H%M')}.csv"
                yield wind_folders[lvl], csv_fname, csv_content
    finally:
        plt.close(fig)


def _save_scalar_maps(ds: xr.Dataset, coordinates: Region, spec: FeatureSpec, output_base: str) -> None:
    frame_writer = _file_frame_writer(output_base)
    for folder, fname, fig in _render_scalar_frames(ds, coordinates, spec):
        frame_writer(fig, folder, fname)


def create_one_time_images(coordinates: Region, output_base: str) -> None:
    save_borders_png(output_base, coordinates)
    create_legends(output_base)


def save_feature_maps(input_path: str, coordinates: Region, output_base: str) -> None:
    with xr.open_dataset(input_path, decode_cf=False) as ds:
        if "dtype" in ds["step"].attrs:
            del ds["step"].attrs["dtype"]

        ds = xr.decode_cf(ds)

        if not _has_required_variables(ds):
            return

        save_cloud_maps(ds, coordinates, output_base)
        save_temperature_maps(ds, coordinates, output_base)
        save_wind_maps(ds, coordinates, output_base)
        save_humidity_maps(ds, coordinates, output_base)

        ds.close()

    logger.debug(f"Feature maps saved for {input_path} in {output_base}")


def render_feature_maps(ds: xr.Dataset, coordinates: Region) -> Dict[str, Dict[str, np.ndarray]]:
    if not _has_required_variables(ds):
        return {}

    ds = _with_wind_speed(ds)
    images: Dict[str, Dict[str, np.ndarray]] = {}

    for spec in FEATURE_SPECS.values():
        for folder, fname, fig in _render_scalar_frames(ds, coordinates, spec):
            images.setdefault(folder, {})[fname] = _fig_to_bgr_array(
                fig, dpi=130, bbox_inches="tight", pad_inches=0)

    logger.debug(f"Feature maps rendered in memory: {sum(len(v) for v in images.values())} images")
    return images


def save_borders_png(output_base: str, coordinates: Region) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
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
        fig, ax = plt.subplots(figsize=(6, 1))
        norm = plt.Normalize(vmin=props["vmin"], vmax=props["vmax"])

        cb = plt.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=props["cmap"]),
            cax=ax,
            orientation="horizontal",
        )
        cb.set_label(props["label"])

        png_path = os.path.join(output_base, f"legend_{key}.png")
        plt.savefig(png_path, dpi=130, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        cmap_obj = plt.get_cmap(props["cmap"])
        rgb255_min = tuple(int(255 * c) for c in cmap_obj(norm(props["vmin"]))[:3])
        rgb255_max = tuple(int(255 * c) for c in cmap_obj(norm(props["vmax"]))[:3])

        txt_path = os.path.join(output_base, f"legend_{key}.txt")
        with open(txt_path, "w") as ftxt:
            prefix, unit = props["txt_prefix"], props["txt_unit"]
            vmin, vmax = props["vmin"], props["vmax"]
            ftxt.write(f"{prefix} range: {vmin:.2f} {unit} to {vmax:.2f} {unit}\n")
            ftxt.write(f"Respective colors: {rgb255_min}, {rgb255_max}\n")


def save_cloud_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    _save_scalar_maps(ds, coordinates, FEATURE_SPECS["cloud"], output_base)


def save_temperature_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    _save_scalar_maps(ds, coordinates, FEATURE_SPECS["temp"], output_base)


def save_humidity_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    _save_scalar_maps(ds, coordinates, FEATURE_SPECS["humidity"], output_base)


def save_wind_maps(ds: xr.Dataset, coordinates: Region, output_base: str) -> None:
    ds = _with_wind_speed(ds)
    _save_scalar_maps(ds, coordinates, FEATURE_SPECS["wind"], output_base)

    csv_writer = _file_csv_writer(output_base)
    for folder, fname, content in _render_wind_vectors(ds, coordinates):
        csv_writer(folder, fname, content)