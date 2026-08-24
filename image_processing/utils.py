import os
import re
from typing import List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tobac
import xarray as xr

from image_processing.constants import (
    DEFAULT_DT,
    DEFAULT_DXY,
    DEFAULT_TIME_OFFSET_HOURS,
)


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
        match = re.search(r"_(\d{8})_(\d{4})\.(?:png|jpg|jpeg)$", basename, re.IGNORECASE)
        if match:
            date_str, time_str = match.groups()
            dt = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M")
        else:
            parts = os.path.splitext(basename)[0].split("_")
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                dt = pd.to_datetime(parts[-2] + parts[-1], format="%Y%m%d%H%M")
            else:
                raise ValueError(f"Could not extract date/time from filename: {filename}")

        if time_offset_hours:
            dt = dt + pd.Timedelta(hours=time_offset_hours)
        times.append(dt)
    return times


def load_image_frames(image_files: List[str]) -> List[np.ndarray]:
    """Reads raw image files into numpy arrays."""
    return [imageio.v2.imread(f) for f in image_files]


def convert_frames_to_grayscale(
        frames: List[np.ndarray],
        is_temperature: bool = False,
) -> List[np.ndarray]:
    """
    Converts list of image frames to 2D grayscale arrays.
    For temperature data, inverts pixel intensity values.
    """
    if is_temperature:
        return [255.0 - (np.mean(f[:, :, :3], axis=2) if f.ndim >= 3 else f.astype(float)) for f in frames]
    return [np.mean(f[:, :, :3], axis=2) if f.ndim >= 3 else f.astype(float) for f in frames]


def build_referenced_data(
        data: np.ndarray,
        times: List[pd.Timestamp],
        region_bounds: Optional[Union[Tuple[float, float, float, float], List[float]]] = None,
) -> xr.DataArray:
    """
    Constructs an xarray.DataArray with spatial (y, x) and time coordinates,
    optionally setting latitude and longitude coordinates.
    """
    _, frame_height, frame_width = data.shape
    x_coords = np.arange(frame_width)
    y_coords = np.arange(frame_height)

    referenced_data = xr.DataArray(
        data,
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


def overlay_image(path_borders: Optional[str], axs: plt.Axes, temp_da: xr.DataArray):
    """Overlays border image onto matplotlib axes if file exists."""
    if path_borders and os.path.exists(path_borders):
        img = plt.imread(path_borders)
        axs.imshow(img, extent=(0, temp_da.sizes["x"], temp_da.sizes["y"], 0), alpha=0.6)
