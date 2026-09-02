import os
import re
from typing import List, Optional, Tuple, Union

import imageio
import numpy as np
import pandas as pd
import tobac
import xarray as xr
import logging
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from image_processing.constants import (
    DEFAULT_DT,
    DEFAULT_DXY,
    DEFAULT_TIME_OFFSET_HOURS,
    CITIES,
    Region,
)

logger = logging.getLogger(__name__)


def extract_keys(filename: str) -> Tuple[int, int]:
    """
    Extracts date and number from filename for sorting purposes.

    Parameters
    ----------
    filename: file name string

    Returns
    -------
    tuple (date as int YYYYMMDD, number as int)
    """
    m = re.search(r"_(\d{8})_(\d+)\.(?:png|jpg|jpeg)$", filename, re.IGNORECASE)
    if m:
        date = int(m.group(1))
        num = int(m.group(2))
        return date, num
    return 0, 0


def extract_times(
    image_files: List[str],
    time_offset_hours: int = DEFAULT_TIME_OFFSET_HOURS,
) -> List[pd.Timestamp]:
    """
    Extracts datetimes from a list of image filenames.

    Parameters
    ----------
    image_files: list of file paths
    time_offset_hours: optional time offset to apply (in hours)

    Returns
    -------
    list of pandas Timestamps
    """
    times = []
    for filename in image_files:
        basename = os.path.basename(filename)
        match = re.search(
            r"_(\d{8})_(\d{4})\.(?:png|jpg|jpeg)$", basename, re.IGNORECASE
        )
        if match:
            date_str, time_str = match.groups()
            dt = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M")
        else:
            parts = os.path.splitext(basename)[0].split("_")
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                dt = pd.to_datetime(parts[-2] + parts[-1], format="%Y%m%d%H%M")
            else:
                raise ValueError(
                    f"Could not extract date/time from filename: {filename}"
                )

        if time_offset_hours:
            dt = dt + pd.Timedelta(hours=time_offset_hours)
        times.append(dt)
    return times


def load_image_frames(image_files: List[str]) -> xr.DataArray:
    """Reads raw image files into an xarray DataArray."""
    if not image_files:
        return xr.DataArray()
    imgs = [imageio.v2.imread(f) for f in image_files]
    arr = np.stack(imgs)
    if arr.ndim == 4:  # colors
        dims = ("frame", "y", "x", "channel")
    elif arr.ndim == 3:  # binanco nero
        dims = ("frame", "y", "x")
    else:
        logger.error(f"Unsupported image shape: {arr.shape}")
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return xr.DataArray(arr, dims=dims)


def convert_frames_to_grayscale(
    frames: xr.DataArray,
    is_temperature: bool = False,
) -> Union[xr.DataArray, List[np.ndarray]]:
    """
    Converts image frames to grayscale.
    For temperature data, inverts pixel intensity values.
    """
    if "channel" in frames.dims:
        if frames.sizes["channel"] >= 3:
            gray = frames.isel(channel=slice(0, 3)).mean(dim="channel").astype(float)
        else:
            gray = frames.mean(dim="channel").astype(float)
    elif frames.ndim >= 4:
        gray = frames[..., :3].mean(dim=frames.dims[-1]).astype(float)
    else:
        gray = frames.astype(float)

    if is_temperature:
        gray = 255.0 - gray
    return gray


def build_referenced_data(
    data: xr.DataArray,
    times: List[pd.Timestamp],
    region_bounds: Optional[
        Union[Tuple[float, float, float, float], List[float], List[int]]
    ] = None,
) -> xr.DataArray:
    """
    Constructs an xarray.DataArray with spatial (y, x) and time coordinates,
    optionally setting latitude and longitude coordinates.
    """
    frame_height = data.sizes.get("y", data.shape[-2])
    frame_width = data.sizes.get("x", data.shape[-1])

    x_coords = np.arange(frame_width)
    y_coords = np.arange(frame_height)

    referenced_data = xr.DataArray(
        data.values,
        dims=("time", "y", "x"),
        coords={
            "time": pd.to_datetime(times),
            "y": ("y", y_coords, {"units": "m"}),
            "x": ("x", x_coords, {"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )

    if region_bounds is not None:
        lon_min, lon_max, lat_min, lat_max = region_bounds
        lat = np.linspace(lat_min, lat_max, frame_height)
        lon = np.linspace(lon_min, lon_max, frame_width)
        longitude = np.tile(lon[np.newaxis, :], (frame_height, 1))
        latitude = np.tile(lat[:, np.newaxis], (1, frame_width))
        referenced_data = referenced_data.assign_coords(
            latitude=(("y", "x"), latitude),
            longitude=(("y", "x"), longitude),
        )

    return referenced_data


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


def overlay_cities(axs: plt.Axes, region: Region, frame_height: int, frame_width: int):
    """
    Overlay city markers + labels on top of whatever's already drawn on axs.
    Call this after imshow() (and after overlay_image(), if used), since
    later draw calls layer on top.

    cities : {city_name: {"lat": float, "lon": float}}
    lat_min/lat_max/lon_min/lon_max, frame_height/frame_width : the same
        region bounds and frame shape used to build this plot's data grid,
        so city positions line up with the underlying image.
    text_offset : (dx, dy) in points, offsetting the label from its dot so
        text doesn't sit directly on top of the marker.
    """
    lon_min, lon_max, lat_min, lat_max = region.value
    for name, coords in CITIES.items():
        px_x, px_y = _latlon_to_px(
            coords["lat"],
            coords["lon"],
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            frame_height,
            frame_width,
        )
        axs.plot(
            px_x,
            px_y,
            marker="o",
            markersize=3,
            color="red",
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=10,
        )
        axs.annotate(
            name,
            xy=(px_x, px_y),
            xytext=(4, -4),
            textcoords="offset points",
            color="red",
            fontsize=6,
            zorder=11,
        )

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
    
    if region_bounds is not None:
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
